"""
Merkezi yapılandırma.

Tüm ayarlar ortam değişkenleriyle ezilebilir; API anahtarları asla koda yazılmaz:

    GEMINI_API_KEY / GOOGLE_API_KEY   -> Gemini
    ANTHROPIC_API_KEY                 -> Claude (ya da `ant auth login` profili)
    L10N_PROVIDER=gemini|claude|ollama|mock
    L10N_GEMINI_MODEL, L10N_CLAUDE_MODEL, L10N_OLLAMA_MODEL, L10N_OLLAMA_HOST
    L10N_USD_TRY                      -> Dry-Run raporundaki TL karşılığı için kur
    L10N_GEMINI_MIN_INTERVAL          -> Gemini istekleri arası en az bekleme (sn, varsayılan 13); 429 alırsanız artırın
    L10N_REQUEST_INTERVAL             -> tüm sağlayıcılar için istek arası bekleme (sn); GUI'den de ayarlanır
    L10N_CHUNK_SIZE / L10N_MAX_BATCH_CHARS -> bir istekteki en fazla satır / karakter (mega-batch)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Provider(str, Enum):
    GEMINI = "gemini"
    CLAUDE = "claude"
    OLLAMA = "ollama"
    MOCK = "mock"  # API'siz uçtan uca test için sahte çevirmen


@dataclass(frozen=True)
class ModelPricing:
    """1 milyon token başına USD fiyatları (Dry-Run maliyet hesabı için)."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None  # prompt caching okuma fiyatı
    cache_write_multiplier: float = 1.25  # önbelleğe yazma = girdi fiyatı x çarpan
    cache_min_tokens: int = 1024  # bu boyutun altındaki promptlar önbelleğe alınmaz
    output_overhead: float = 1.0  # düşünme (thinking) token'ları için çıktı çarpanı
    note: str = ""


# Fiyatlar değişebilir; faturalandırma kararı vermeden önce sağlayıcının
# güncel fiyat sayfasından doğrulayın.
MODEL_PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-flash": ModelPricing(
        0.30, 2.50, note="Ücretsiz katmanda $0 (dakika/gün istek limitleri dahilinde)"
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        0.10, 0.40, note="Ücretsiz katmanda $0 (dakika/gün istek limitleri dahilinde)"
    ),
    "claude-haiku-5-5": ModelPricing(
        0.10, 0.50, cached_input_per_mtok=0.01, cache_min_tokens=512,
        output_overhead=1.3, note="100K token altı prompt fiyatı; hızlı taslak",
    ),
    "claude-sonnet-5-5": ModelPricing(
        2.00, 10.00, cached_input_per_mtok=0.20, cache_min_tokens=512,
        output_overhead=1.3, note="Final yama kalitesi",
    ),
    "ollama": ModelPricing(0.0, 0.0, note="Yerel model; API ücreti yok"),
    "mock": ModelPricing(0.0, 0.0, note="Sahte çevirmen; boru hattı testi"),
}


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else None


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "evet", "on"}


@dataclass
class Settings:
    # --- Sağlayıcı seçimi -------------------------------------------------
    provider: Provider = Provider.GEMINI
    gemini_model: str = "gemini-2.5-flash"
    claude_model: str = "claude-sonnet-5-5"  # hızlı/ucuz taslak için: claude-haiku-5-5
    ollama_model: str = "gemma3:12b"
    ollama_host: str = "http://localhost:11434"

    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None  # None ise SDK ortamdan/profilden çözer

    # --- Dil ---------------------------------------------------------------
    source_lang: str = "English"
    target_lang: str = "Turkish"

    # --- Batch ve bağlam ---------------------------------------------------
    # Mega-batch: 340 satırlık SSpk_Eng.asr tek istekte, 1688 satırlık English.asr 5 istekte gider.
    # İki sınır birlikte uygulanır; hangisine önce ulaşılırsa blok orada kapanır.
    chunk_size: int = 350  # bir istekteki en fazla satır (GUI: 10-500)
    max_batch_chars: int = 30000  # bir istekteki en fazla kaynak karakter (çıktı token sınırını korur)
    context_size: int = 3  # "sadece bağlam" olarak eklenen önceki satır sayısı

    # --- Dayanıklılık ------------------------------------------------------
    max_retries: int = 3  # sağlayıcı seviyesi: API/JSON hatalarında deneme sayısı
    sanity_rounds: int = 3  # SanityChecker seviyesi: hatalı satırları yeniden isteme turu
    request_timeout: float = 300.0  # saniye; 350 satırlık tek istek 1-2 dakika sürebilir

    # --- İstek hızı (rate limit) ------------------------------------------
    # Ücretsiz Gemini katmanı dakikada 5 istekle sınırlı (60/5 = 12 sn) -> 13 sn güvenli aralık.
    # Ücretli katmanda 0 yapın. Güncel sınırlar: AI Studio > kota sayfası.
    gemini_min_interval: float = 13.0
    # Tüm sağlayıcılar için elle ayar (GUI'deki "İstek arası bekleme"). None = sağlayıcı varsayılanı.
    request_interval: float | None = None

    # --- Model davranışı ---------------------------------------------------
    temperature: float = 0.3  # Gemini/Ollama; Claude 5.x modelleri temperature kabul etmez
    gemini_thinking_budget: int | None = 0  # 0 = düşünme kapalı (ucuz); None = model varsayılanı
    # Düşünme tokenları da bu sınıra dahildir. 350 satır ≈ 7.000 çıktı token; 32768 bol pay bırakır.
    gemini_max_output_tokens: int = 32768
    claude_effort: str = "low"  # low | medium | high; final cilada "medium" deneyin
    claude_max_tokens: int = 16000
    claude_refusal_fallback: bool = True  # destekleyen modellerde sunucu tarafı yedek model

    # --- Maliyet raporu ----------------------------------------------------
    usd_try_rate: float = 42.0  # örnek kur; L10N_USD_TRY ile güncel kuru verin

    # --- Dosyalar ----------------------------------------------------------
    glossary_path: Path = field(default_factory=lambda: BASE_DIR / "glossary.json")
    output_dir: Path = field(default_factory=lambda: BASE_DIR / "output")

    @property
    def active_model(self) -> str:
        return {
            Provider.GEMINI: self.gemini_model,
            Provider.CLAUDE: self.claude_model,
            Provider.OLLAMA: self.ollama_model,
            Provider.MOCK: "mock",
        }[self.provider]

    @property
    def effective_request_interval(self) -> float:
        """Seçili sağlayıcı için iki istek arası bekleme (sn). Mock hiç beklemez."""
        if self.provider is Provider.MOCK:
            return 0.0
        if self.request_interval is not None:
            return max(0.0, self.request_interval)
        return self.gemini_min_interval if self.provider is Provider.GEMINI else 0.0

    @property
    def pricing_key(self) -> str:
        """Maliyet tablosundaki anahtar (Ollama modelleri tek satırda toplanır)."""
        if self.provider is Provider.OLLAMA:
            return "ollama"
        return self.active_model

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=Provider(_env("L10N_PROVIDER", Provider.GEMINI.value).lower()),
            gemini_model=_env("L10N_GEMINI_MODEL", cls.gemini_model),
            claude_model=_env("L10N_CLAUDE_MODEL", cls.claude_model),
            ollama_model=_env("L10N_OLLAMA_MODEL", cls.ollama_model),
            ollama_host=_env("L10N_OLLAMA_HOST", cls.ollama_host),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            source_lang=_env("L10N_SOURCE_LANG", cls.source_lang),
            target_lang=_env("L10N_TARGET_LANG", cls.target_lang),
            chunk_size=_env_int("L10N_CHUNK_SIZE", cls.chunk_size),
            context_size=_env_int("L10N_CONTEXT_SIZE", cls.context_size),
            max_batch_chars=_env_int("L10N_MAX_BATCH_CHARS", cls.max_batch_chars),
            max_retries=_env_int("L10N_MAX_RETRIES", cls.max_retries),
            sanity_rounds=_env_int("L10N_SANITY_ROUNDS", cls.sanity_rounds),
            request_timeout=_env_float("L10N_REQUEST_TIMEOUT", cls.request_timeout),
            temperature=_env_float("L10N_TEMPERATURE", cls.temperature),
            claude_effort=_env("L10N_CLAUDE_EFFORT", cls.claude_effort),
            gemini_min_interval=_env_float("L10N_GEMINI_MIN_INTERVAL", cls.gemini_min_interval),
            request_interval=_env_optional_float("L10N_REQUEST_INTERVAL"),
            gemini_max_output_tokens=_env_int("L10N_GEMINI_MAX_OUTPUT_TOKENS", cls.gemini_max_output_tokens),
            claude_refusal_fallback=_env_bool("L10N_CLAUDE_REFUSAL_FALLBACK", cls.claude_refusal_fallback),
            usd_try_rate=_env_float("L10N_USD_TRY", cls.usd_try_rate),
        )
