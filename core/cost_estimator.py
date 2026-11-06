"""
Dry-Run maliyet hesaplayıcı: API'ye tek istek atmadan kelime, token ve maliyet tahmini.

Girdi tokenları, gerçekte gönderilecek sistem promptu ve kullanıcı mesajlarının
karakter sayısından hesaplanır (main.py istekleri birebir aynı fonksiyonla kurar).
Çıktı tokenları, Türkçenin İngilizceden biraz uzun olması ve eklemeli yapısı
nedeniyle daha fazla token tüketmesi hesaba katılarak tahmin edilir.

Tahminler sağlayıcının kendi tokenizer'ı kullanılmadığı için yaklaşık ±%25'tir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from config import ModelPricing
from core.regex_shield import TAG_RE

CHARS_PER_TOKEN_INPUT = 3.6  # İngilizce + JSON karışımı
CHARS_PER_TOKEN_TR_OUTPUT = 2.7  # Türkçe metin, İngilizceye göre daha çok token
TR_LENGTH_RATIO = 1.15  # Türkçe çeviri, kaynaktan ortalama bu kadar uzun
OUTPUT_TOKENS_PER_ITEM = 12  # {"id":"L1","text":"..."} iskeleti
RETRY_OVERHEAD = 0.08  # yeniden denemeler için pay

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*")


@dataclass(frozen=True)
class RequestShape:
    """Tek bir API isteğinin boyutu (maliyet hesabı için)."""

    system_chars: int
    user_chars: int
    item_count: int
    source_chars: int  # çevrilecek maskeli metinlerin toplam karakteri


@dataclass(frozen=True)
class TextStats:
    total_entries: int
    translatable: int
    code_only: int
    skipped_by_reason: Mapping[str, int]
    words: int
    characters: int
    protected_tags: int
    batches: int


@dataclass(frozen=True)
class ModelCostEstimate:
    model: str
    input_tokens: int
    output_tokens: int
    usd: float
    try_amount: float
    note: str = ""


@dataclass(frozen=True)
class CostReport:
    stats: TextStats
    input_tokens: int
    output_tokens: int
    estimates: list[ModelCostEstimate] = field(default_factory=list)
    usd_try_rate: float = 0.0


def count_words(masked_text: str) -> int:
    return len(_WORD_RE.findall(TAG_RE.sub(" ", masked_text)))


def estimate_input_tokens(chars: int) -> int:
    return int(round(chars / CHARS_PER_TOKEN_INPUT))


def estimate_output_tokens(source_chars: int, item_count: int) -> int:
    text_tokens = source_chars * TR_LENGTH_RATIO / CHARS_PER_TOKEN_TR_OUTPUT
    return int(round(text_tokens + item_count * OUTPUT_TOKENS_PER_ITEM))


class CostEstimator:
    def __init__(self, pricing: Mapping[str, ModelPricing], usd_try_rate: float) -> None:
        self.pricing = pricing
        self.usd_try_rate = usd_try_rate

    def estimate(self, requests: Sequence[RequestShape], stats: TextStats,
                 models: Sequence[str] | None = None) -> CostReport:
        base_input = sum(estimate_input_tokens(r.system_chars + r.user_chars) for r in requests)
        base_output = sum(estimate_output_tokens(r.source_chars, r.item_count) for r in requests)
        estimates = [
            self._estimate_model(name, self.pricing[name], requests)
            for name in (models or list(self.pricing))
            if name in self.pricing
        ]
        return CostReport(stats, base_input, base_output, estimates, self.usd_try_rate)

    def actual_cost_usd(self, model_key: str, input_tokens: int, output_tokens: int,
                        cached_input_tokens: int = 0) -> float | None:
        """Gerçek kullanım (usage) verisinden maliyet; fiyat tablosunda yoksa None."""
        price = self.pricing.get(model_key)
        if price is None:
            return None
        cached_price = price.cached_input_per_mtok if price.cached_input_per_mtok is not None else price.input_per_mtok
        return (
            input_tokens * price.input_per_mtok
            + cached_input_tokens * cached_price
            + output_tokens * price.output_per_mtok
        ) / 1_000_000

    def _estimate_model(self, name: str, price: ModelPricing,
                        requests: Sequence[RequestShape]) -> ModelCostEstimate:
        input_usd = 0.0
        input_tokens = 0
        output_tokens = 0
        for position, request in enumerate(requests):
            system_tokens = estimate_input_tokens(request.system_chars)
            user_tokens = estimate_input_tokens(request.user_chars)
            input_tokens += system_tokens + user_tokens
            output_tokens += estimate_output_tokens(request.source_chars, request.item_count)

            # Prompt caching: sistem promptu ilk istekte önbelleğe yazılır, sonra ucuza okunur.
            cacheable = price.cached_input_per_mtok is not None and system_tokens >= price.cache_min_tokens
            if cacheable and position == 0:
                system_usd = system_tokens * price.input_per_mtok * price.cache_write_multiplier
            elif cacheable:
                system_usd = system_tokens * price.cached_input_per_mtok
            else:
                system_usd = system_tokens * price.input_per_mtok
            input_usd += (system_usd + user_tokens * price.input_per_mtok) / 1_000_000

        output_tokens = int(output_tokens * price.output_overhead)
        overhead = 1 + RETRY_OVERHEAD
        usd = (input_usd + output_tokens * price.output_per_mtok / 1_000_000) * overhead
        return ModelCostEstimate(
            model=name,
            input_tokens=int(input_tokens * overhead),
            output_tokens=int(output_tokens * overhead),
            usd=usd,
            try_amount=usd * self.usd_try_rate,
            note=price.note,
        )


# --------------------------------------------------------------------------- rapor


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def format_usd(value: float) -> str:
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:,.2f}"


def format_try(value: float) -> str:
    if value == 0:
        return "0 TL"
    text = f"{value:,.2f}" if value >= 0.01 else f"{value:.4f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") + " TL"


def format_report(report: CostReport, *, source_name: str, active_model: str,
                  chunk_size: int, context_size: int) -> str:
    stats = report.stats
    skipped_total = sum(stats.skipped_by_reason.values())
    skipped_detail = ", ".join(f"{reason}: {count}" for reason, count in stats.skipped_by_reason.items())
    width = 78
    lines = [
        "=" * width,
        " DRY-RUN  |  KELİME VE MALİYET ANALİZİ (API'ye istek atılmadı)".ljust(width),
        "=" * width,
        f" Dosya                : {source_name}",
        f" Toplam metin alanı   : {format_int(stats.total_entries)}",
        f" Çevrilecek satır     : {format_int(stats.translatable)}"
        f"  ({format_int(stats.words)} kelime, {format_int(stats.characters)} karakter)",
        f" Korunan kod/etiket   : {format_int(stats.protected_tags)} adet -> [TAG_n] olarak donduruldu",
        f" Sadece kod içeren    : {format_int(stats.code_only)} (aynen kopyalanır)",
        f" Atlanan              : {format_int(skipped_total)}" + (f"  ({skipped_detail})" if skipped_detail else ""),
        f" İstek (batch) sayısı : {format_int(stats.batches)}"
        f"  ({chunk_size} satırlık bloklar + {context_size} satır bağlam)",
        f" Tahmini token        : girdi ~{format_int(report.input_tokens)}  |  çıktı ~{format_int(report.output_tokens)}",
        "-" * width,
        f" {'Model':<26}{'Girdi tok.':>11}{'Çıktı tok.':>11}{'USD':>11}{'TL':>14}",
    ]
    for estimate in report.estimates:
        marker = " *" if estimate.model == active_model else ""
        lines.append(
            f" {(estimate.model + marker):<26}{format_int(estimate.input_tokens):>11}"
            f"{format_int(estimate.output_tokens):>11}{format_usd(estimate.usd):>11}{format_try(estimate.try_amount):>14}"
        )
        if estimate.note:
            lines.append(f"   {estimate.note}")
    lines += [
        "-" * width,
        " * = aktif model. Tahminler ±%25; yeniden deneme payı ve prompt önbelleği dahildir.",
        f" Kur: 1 USD = {format_try(report.usd_try_rate)} (L10N_USD_TRY ile güncelleyin).",
        "=" * width,
    ]
    return "\n".join(lines)
