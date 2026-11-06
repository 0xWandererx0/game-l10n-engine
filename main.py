#!/usr/bin/env python3
"""
Yeni Nesil Otonom Oyun Yerelleştirme Derleyicisi: CLI ve orkestrasyon.

Oyun çalışmadan önce bir kez çalışır: ham dil dosyasını okur, kodları dondurur,
bağlamlı bloklar halinde çevirtir, her satırı doğrular ve oyun klasörüne
atılmaya hazır Türkçe dosyayı üretir. Oyuna çalışma zamanı yükü eklemez.

Kullanım:
    python main.py                               # yerleşik örnek oyun verisi, config'deki sağlayıcı
    python main.py --provider mock               # API anahtarı olmadan uçtan uca test
    python main.py --provider mock --chaos 0.3   # kasıtlı model hatalarıyla retry zincirini izle
    python main.py lang/en.json --dry-run        # yalnızca kelime/token/maliyet raporu
    python main.py lang/en.json -o lang/tr.json --provider claude --yes
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from config import MODEL_PRICING, Provider, Settings
from core.context_batcher import (
    Batch,
    ContextBatcher,
    TranslationUnit,
    group_key,
    key_hint,
    split_edges,
)
from core.cost_estimator import (
    CostEstimator,
    CostReport,
    RequestShape,
    TextStats,
    count_words,
    format_int,
    format_report,
    format_try,
    format_usd,
)
from core.glossary import Glossary
from core.parser import CsvLocalizationFile, FlatEntry, LocalizationFile, load_localization_file
from core.regex_shield import TAG_RE, RegexShield
from core.sanity_checker import SanityChecker
from providers import create_translator
from providers.base import (
    BaseTranslator,
    ContextItem,
    FatalProviderError,
    SourceItem,
    TranslationFailedError,
    build_system_prompt,
    build_user_message,
)

logger = logging.getLogger("l10n")

# Uçtan uca test için karmaşık örnek oyun dili: iç içe yapı, Unity etiketleri,
# {player}, %d/%s, \n, tuş atamaları, asset yolları, sayılar ve askeri argo.
SAMPLE_GAME_DATA: dict = {
    "meta": {
        "language": "en",
        "version": "1.4.2",
        "font_asset": "Fonts/NotoSans-Bold.asset",
        "rtl": False,
        "string_count": 27,
    },
    "ui": {
        "main_menu": {
            "title": "<size=64><b>Operation Nightfall</b></size>",
            "new_game": "New Game",
            "continue": "Continue",
            "settings": "Settings",
            "quit": "Quit to Desktop",
        },
        "hud": {
            "health": "Health: {0}/{1}",
            "ammo": "Ammo: %d / %d",
            "interact": "Press [E] to interact",
            "reload_hint": "Hold <color=#FFD700>[R]</color> to reload",
            "objective": "Objective:\n<i>{objective_name}</i>",
            "score_label": "Score: ",
        },
        "notifications": {
            "level_up": "<color=#00FF00>LEVEL UP!</color>\nYou reached level {level}.",
            "loot": "{player_name} picked up {count} {item_name}.",
            "cooldown": "Ability on cooldown ({0:F1}s)",
            "achievement": "Achievement unlocked: <b>{achievement}</b>",
            "low_hp": "%s's health is critical!",
        },
    },
    "dialogues": {
        "chapter_1": {
            "briefing": [
                {"speaker_id": "SGT_MILLER", "voice_file": "audio/vo/miller_01.wav",
                 "text": "Listen up, {player}! The bridge is crawling with hostiles."},
                {"speaker_id": "SGT_MILLER", "voice_file": "audio/vo/miller_02.wav",
                 "text": "We go in hot, we come out clean. Roger that?"},
                {"speaker_id": "PLAYER", "voice_file": "audio/vo/player_01.wav",
                 "text": "Roger that, Sarge."},
                {"speaker_id": "SGT_MILLER", "voice_file": "audio/vo/miller_03.wav",
                 "text": "Fire in the hole!"},
                {"speaker_id": "CPL_REYES", "voice_file": "audio/vo/reyes_01.wav",
                 "text": "Cover me, I'm reloading!\nThis piece of junk keeps jamming."},
                {"speaker_id": "SGT_MILLER", "voice_file": "audio/vo/miller_04.wav",
                 "text": "Enemy down! Hold the line, people!"},
                {"speaker_id": "CPL_REYES", "voice_file": "audio/vo/reyes_02.wav",
                 "text": "That was a piece of cake. Let's loot the bodies before they respawn."},
            ]
        }
    },
    "items": [
        {
            "id": "wpn_rifle_01",
            "icon": "Textures/Icons/rifle_01.png",
            "name": "Rusty Rifle",
            "description": "An old rifle. Deals {damage} damage.\n<color=#FF0000>Overheats after %d shots.</color>",
            "stack": 1,
            "price": 250.5,
            "tradeable": True,
        },
        {
            "id": "cons_medkit",
            "icon": "Textures/Icons/medkit.png",
            "name": "Field Medkit",
            "description": "Restores 50% health over 5 seconds. Buffs armor by {armor_bonus}.",
            "stack": 5,
            "price": 75,
            "tradeable": True,
        },
    ],
    "patch_notes": "Nerf: Sniper rifle damage reduced by 10%.\nBuff: Shotgun spread tightened.",
    "credits_url": "https://example.com/credits",
    "placeholders_only": "{0}",
    "empty": "",
}


# --------------------------------------------------------------------------- pipeline


@dataclass
class PreparedJob:
    source_name: str
    loc_file: LocalizationFile
    entries: list[FlatEntry]
    units: list[TranslationUnit]
    code_only: list[FlatEntry]
    skipped: list[FlatEntry]
    batches: list[Batch]


@dataclass
class RunStats:
    translated: int = 0
    fallback: int = 0
    rerequested_lines: int = 0
    bisections: int = 0
    failures: dict[str, list[str]] = field(default_factory=dict)


def _plain(masked: str) -> str:
    """Bağlam satırlarında etiketleri gizler; model bağlamdaki etiketleri çeviri satırlarıyla karıştırmaz."""
    return TAG_RE.sub("…", masked)


class LocalizationPipeline:
    def __init__(self, settings: Settings, glossary: Glossary, shield: RegexShield | None = None) -> None:
        self.settings = settings
        self.glossary = glossary
        self.shield = shield or RegexShield()
        self.checker = SanityChecker(self.shield)
        self.batcher = ContextBatcher(settings.chunk_size, settings.context_size, settings.max_batch_chars)
        self.system_prompt = build_system_prompt(settings.source_lang, settings.target_lang, glossary.style_rules)
        self._accepted: dict[str, str] = {}  # uid -> kabul edilmiş maskeli çeviri (bağlam için)

    # ---- hazırlık -----------------------------------------------------------

    def prepare(self, loc_file: LocalizationFile, source_name: str) -> PreparedJob:
        entries = loc_file.entries()
        units: list[TranslationUnit] = []
        code_only: list[FlatEntry] = []
        skipped: list[FlatEntry] = []
        for entry in entries:
            if not entry.translatable:
                skipped.append(entry)
                continue
            lead, core, trail = split_edges(entry.value)
            masked, tag_map = self.shield.mask(core)
            if not self.shield.has_translatable_text(masked):
                code_only.append(entry)
                continue
            units.append(TranslationUnit(
                uid=entry.uid, source=entry.value, masked=masked, tag_map=tag_map,
                key_hint=key_hint(entry.path), group=group_key(entry.path),
                lead_ws=lead, trail_ws=trail,
            ))
        return PreparedJob(source_name, loc_file, entries, units, code_only, skipped, self.batcher.build(units))

    # ---- dry-run ------------------------------------------------------------

    def estimate(self, job: PreparedJob) -> CostReport:
        shapes = []
        for batch in job.batches:
            items = self._source_items(batch.units)
            message = build_user_message(items, self._context_items(batch.context),
                                         self.glossary.match(u.masked for u in batch.units))
            shapes.append(RequestShape(
                system_chars=len(self.system_prompt), user_chars=len(message),
                item_count=len(items), source_chars=sum(len(u.masked) for u in batch.units),
            ))
        skipped = Counter(entry.skip_reason.value for entry in job.skipped if entry.skip_reason)
        stats = TextStats(
            total_entries=len(job.entries),
            translatable=len(job.units),
            code_only=len(job.code_only),
            skipped_by_reason=dict(skipped),
            words=sum(count_words(u.masked) for u in job.units),
            characters=sum(len(TAG_RE.sub("", u.masked)) for u in job.units),
            protected_tags=sum(len(u.tag_map) for u in job.units),
            batches=len(job.batches),
        )
        models = [name for name in MODEL_PRICING if name != "mock" or self.settings.provider is Provider.MOCK]
        return CostEstimator(MODEL_PRICING, self.settings.usd_try_rate).estimate(shapes, stats, models)

    # ---- çeviri -------------------------------------------------------------

    def run(self, job: PreparedJob, translator: BaseTranslator,
            progress: Callable[[str], None] = print) -> tuple[dict[str, str], RunStats]:
        stats = RunStats()
        accepted_masked: dict[str, str] = {}

        for batch in job.batches:
            local_units = {f"L{i + 1}": unit for i, unit in enumerate(batch.units)}
            context = self._context_items(batch.context)
            accepted, failed = self._translate_units(translator, local_units, context, stats)
            for local_id, unit in local_units.items():
                if local_id in accepted:
                    accepted_masked[unit.uid] = accepted[local_id]
                    self._accepted[unit.uid] = accepted[local_id]
                else:
                    stats.failures[unit.uid] = failed.get(local_id, ["no valid translation"])
            missing = len(local_units) - len(accepted)
            suffix = f"  ({missing} satır İngilizce bırakıldı)" if missing else ""
            progress(f"  [{batch.index + 1}/{len(job.batches)}] {len(accepted)}/{len(local_units)} satır doğrulandı{suffix}")

        translations: dict[str, str] = {}
        for unit in job.units:
            masked_translation = accepted_masked.get(unit.uid)
            if masked_translation is not None:
                final = unit.lead_ws + self.shield.unmask(masked_translation, unit.tag_map) + unit.trail_ws
                # Son güvenlik kapısı: geri açılmış metindeki kodlar kaynağınkiyle birebir aynı olmalı.
                if self.shield.signature(final) == self.shield.signature(unit.source):
                    translations[unit.uid] = final
                    stats.translated += 1
                    continue
                stats.failures[unit.uid] = ["code signature of the final text differs from the source"]
            translations[unit.uid] = unit.source  # güvenli yedek: oyun çökmez, satır İngilizce kalır
            stats.fallback += 1
        return translations, stats

    def _translate_units(
        self,
        translator: BaseTranslator,
        pending: dict[str, TranslationUnit],
        context: list[ContextItem],
        stats: RunStats,
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        """Bloğu çevirir; SanityChecker'ın reddettiği satırları gerekçeleriyle yeniden ister.

        Sağlayıcı bloğu hiç çeviremezse (güvenlik reddi, kesilen çıktı, tükenen denemeler)
        blok ikiye bölünür; sorunlu satır yalnız kalana kadar diğerleri kurtarılır.
        """
        accepted: dict[str, str] = {}
        feedback: dict[str, list[str]] = {}
        rounds = self.settings.sanity_rounds

        for round_number in range(1, rounds + 1):
            terms = self.glossary.match(unit.masked for unit in pending.values())
            try:
                received = translator.translate(self._source_items(pending), context, terms, feedback or None)
            except TranslationFailedError as exc:
                if len(pending) == 1:
                    return accepted, {local_id: [f"provider error: {exc}"] for local_id in pending}
                stats.bisections += 1
                logger.warning("Blok çevrilemedi (%s); %d satır ikiye bölünerek yeniden denenecek", exc, len(pending))
                local_ids = list(pending)
                middle = len(local_ids) // 2
                failed: dict[str, list[str]] = {}
                for part in (local_ids[:middle], local_ids[middle:]):
                    part_ok, part_failed = self._translate_units(
                        translator, {local_id: pending[local_id] for local_id in part}, context, stats)
                    accepted.update(part_ok)
                    failed.update(part_failed)
                return accepted, failed

            verdict = self.checker.check(pending, received)
            accepted.update(verdict.accepted)
            if verdict.ok:
                return accepted, {}
            for local_id, problems in verdict.rejected.items():
                logger.info("%s reddedildi (tur %d/%d): %s", pending[local_id].uid, round_number, rounds,
                            "; ".join(problems))
            pending = {local_id: pending[local_id] for local_id in verdict.rejected}
            feedback = verdict.rejected
            if round_number < rounds:
                stats.rerequested_lines += len(pending)
        return accepted, feedback

    @staticmethod
    def _source_items(units: dict[str, TranslationUnit] | list[TranslationUnit]) -> list[SourceItem]:
        if isinstance(units, dict):
            return [SourceItem(local_id, unit.masked, unit.key_hint) for local_id, unit in units.items()]
        return [SourceItem(f"L{i + 1}", unit.masked, unit.key_hint) for i, unit in enumerate(units)]

    def _context_items(self, units: Sequence[TranslationUnit]) -> list[ContextItem]:
        return [
            ContextItem(_plain(unit.masked), _plain(self._accepted[unit.uid]) if unit.uid in self._accepted else None)
            for unit in units
        ]


# --------------------------------------------------------------------------- CLI yardımcıları


def _configure_console() -> None:
    """Windows konsolunda (cp1254/cp857) Türkçe karakter yazarken çökmeyi önler."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        print("Etkileşimsiz ortam algılandı: onay vermek için --yes kullanın.")
        return False
    try:
        return input(question).strip().lower() in {"e", "evet", "y", "yes"}
    except EOFError:
        return False


def _atomic_write(path: Path, content: str) -> None:
    """Yarım yazılmış dosya bırakmaz: önce geçici dosyaya yazar, sonra tek adımda yerine koyar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _default_output(input_path: Path, target_column: str) -> Path:
    return input_path.with_name(f"{target_column}{input_path.suffix}")


def _write_sample(output_dir: Path) -> Path:
    path = output_dir / "sample_en.json"
    _atomic_write(path, json.dumps(SAMPLE_GAME_DATA, ensure_ascii=False, indent=2) + "\n")
    return path


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.provider:
        settings.provider = Provider(args.provider)
    if args.model:
        attribute = {Provider.GEMINI: "gemini_model", Provider.CLAUDE: "claude_model",
                     Provider.OLLAMA: "ollama_model"}.get(settings.provider)
        if attribute:
            setattr(settings, attribute, args.model)
    if args.chunk_size:
        settings.chunk_size = args.chunk_size
    if args.context_size is not None:
        settings.context_size = args.context_size


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oyun dil dosyasını doğrulanmış Türkçe yamaya derler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="Kaynak dil dosyası (.json/.csv/.tsv). Verilmezse yerleşik örnek kullanılır.")
    parser.add_argument("-o", "--output", help="Çıktı dosyası (varsayılan: kaynağın yanında tr.<uzantı>)")
    parser.add_argument("--provider", choices=[p.value for p in Provider], help="config.py'deki sağlayıcıyı ez")
    parser.add_argument("--model", help="Seçili sağlayıcının modelini ez (örn. claude-haiku-5-5)")
    parser.add_argument("--glossary", help="Sözlük dosyası (varsayılan: glossary.json)")
    parser.add_argument("--dry-run", action="store_true", help="Yalnızca kelime/token/maliyet raporu; API çağrısı yok")
    parser.add_argument("-y", "--yes", action="store_true", help="Onay sormadan çeviriye başla")
    parser.add_argument("--chunk-size", type=int, help="Blok başına satır sayısı")
    parser.add_argument("--context-size", type=int, help="Bağlam olarak eklenen önceki satır sayısı")
    parser.add_argument("--target-column", default="tr", help="CSV hedef sütunu (varsayılan: tr)")
    parser.add_argument("--chaos", type=float, default=0.0, help="Yalnızca mock: kasıtlı hata oranı (0-1)")
    parser.add_argument("--preview", type=int, default=6, help="Sonuçta gösterilecek örnek satır sayısı")
    parser.add_argument("-v", "--verbose", action="store_true", help="Reddedilen satırları ve gerekçelerini göster")
    return parser.parse_args(argv)


def _print_preview(job: PreparedJob, translations: dict[str, str], limit: int) -> None:
    if limit <= 0:
        return
    print("\n Örnek satırlar:")
    for unit in job.units[:limit]:
        print(f"   {unit.uid}")
        print(f"     EN: {unit.source!r}")
        print(f"     TR: {translations[unit.uid]!r}")


# --------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    _configure_console()
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="  ! %(message)s")

    settings = Settings.from_env()
    _apply_overrides(settings, args)

    try:
        glossary = Glossary.load(args.glossary or settings.glossary_path)
        input_path = Path(args.input) if args.input else _write_sample(settings.output_dir)
        loader_options = {"target_column": args.target_column} if input_path.suffix.lower() in {".csv", ".tsv"} else {}
        loc_file = load_localization_file(input_path, **loader_options)
    except (OSError, ValueError) as exc:
        print(f"HATA: {exc}")
        return 2

    output_path = Path(args.output) if args.output else _default_output(input_path, args.target_column)
    if output_path.resolve() == input_path.resolve() and not isinstance(loc_file, CsvLocalizationFile):
        print("HATA: Çıktı dosyası kaynak dosyanın üzerine yazılamaz.")
        return 2

    pipeline = LocalizationPipeline(settings, glossary)
    job = pipeline.prepare(loc_file, input_path.name)
    print(format_report(pipeline.estimate(job), source_name=str(input_path),
                        active_model=settings.pricing_key,
                        chunk_size=settings.chunk_size, context_size=settings.context_size))

    if args.dry_run:
        return 0
    print(f"\n Sağlayıcı: {settings.provider.value}  |  Model: {settings.active_model}")
    if job.units and not args.yes and not _confirm(" Çeviriye başlansın mı? [e/H]: "):
        print(" İptal edildi; API'ye istek atılmadı.")
        return 1

    try:
        translator = create_translator(settings, pipeline.system_prompt, chaos_rate=args.chaos)
        translations, stats = pipeline.run(job, translator)
        rendered = loc_file.render(translations)
        loc_file.verify_rendered(rendered, translations)
    except FatalProviderError as exc:
        print(f"\nHATA: {exc}\nÇıktı dosyası yazılmadı.")
        return 2
    except KeyboardInterrupt:
        print("\nKullanıcı durdurdu; çıktı dosyası yazılmadı.")
        return 130
    except ValueError as exc:  # StructureMismatchError dahil: bozuk dosya asla yazılmaz
        print(f"\nHATA: Çıktı doğrulanamadı: {exc}\nÇıktı dosyası yazılmadı.")
        return 2

    _atomic_write(output_path, rendered)
    usage = translator.usage
    cost = CostEstimator(MODEL_PRICING, settings.usd_try_rate).actual_cost_usd(
        settings.pricing_key, usage.input_tokens, usage.output_tokens, usage.cached_input_tokens)
    report_path = output_path.with_name(f"{output_path.stem}.report.json")
    _atomic_write(report_path, json.dumps({
        "source": str(input_path),
        "output": str(output_path),
        "provider": settings.provider.value,
        "model": settings.active_model,
        "translated": stats.translated,
        "fallback_to_source": stats.fallback,
        "code_only_copied": len(job.code_only),
        "skipped": dict(Counter(e.skip_reason.value for e in job.skipped if e.skip_reason)),
        "rerequested_lines": stats.rerequested_lines,
        "bisections": stats.bisections,
        "api_requests": translator.request_count,
        "usage": asdict(usage),
        "cost_usd": cost,
        "failures": [
            {"id": uid, "source": next(u.source for u in job.units if u.uid == uid), "problems": problems}
            for uid, problems in stats.failures.items()
        ],
    }, ensure_ascii=False, indent=2) + "\n")

    total = len(job.units)
    print("\n" + "=" * 78)
    print(f" SONUÇ: {stats.translated}/{total} satır çevrildi ve doğrulandı")
    if stats.fallback:
        print(f" Yedeğe düşen: {stats.fallback} satır orijinal haliyle bırakıldı (ayrıntı: {report_path.name})")
    print(f" Yeniden istenen satır: {stats.rerequested_lines}  |  Bölünen blok: {stats.bisections}"
          f"  |  API isteği: {translator.request_count}")
    cost_text = f" -> {format_usd(cost)} ({format_try(cost * settings.usd_try_rate)})" if cost is not None else ""
    print(f" Kullanım: {format_int(usage.input_tokens)} girdi + {format_int(usage.cached_input_tokens)} önbellek"
          f" / {format_int(usage.output_tokens)} çıktı token{cost_text}")
    print(f" Çıktı : {output_path}")
    print(f" Rapor : {report_path}")
    _print_preview(job, translations, args.preview)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
