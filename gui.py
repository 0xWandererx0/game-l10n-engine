#!/usr/bin/env python3
"""
Game L10n Engine: CustomTkinter masaüstü arayüzü (Faz 2 + Faz 3 Paket Stüdyosu).

Kurulum : pip install customtkinter tkinterdnd2
Çalıştır: python gui.py                 (isteğe bağlı: python gui.py lang/en.json | English.asr)

Sekmeler
    1. Çeviri Motoru (JSON / CSV)         -> Faz 2 ekranı, değişmeden
    2. Oyun Paket Stüdyosu (Binary)       -> .asr/.vdf -> JSON (+ .meta) -> Sekme 1 -> geri paketleme

Mimari
    App (yalnızca ana thread, yalnızca widget)
       |  komut: analyze / translate / cancel / inspect_package / extract_package / repack_package
       v
    EngineService (tkinter bilmez; motoru ve parsers/ paketini arka plan thread'inde çalıştırır)
       |  olay: analysis / log / progress / done / studio_info / studio_extracted / studio_repacked /
       |        error / cancelled / idle
       v
    queue.Queue  ->  App._poll_events() her 80 ms'de kuyruğu boşaltır

Motor (main.LocalizationPipeline, core/*, providers/*) ve parsers/ hiç değiştirilmeden kullanılır.
Arayüz onlara yalnızca EngineService üzerinden dokunur; motor arayüzü hiç tanımaz.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

try:
    import customtkinter as ctk
except ImportError:  # pragma: no cover - kurulum yönlendirmesi
    sys.exit("CustomTkinter kurulu değil: pip install customtkinter")

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _DND_BASES: tuple[type, ...] = (TkinterDnD.DnDWrapper,)
except ImportError:  # sürükle-bırak isteğe bağlıdır; "Dosya Seç" her zaman çalışır
    DND_FILES, TkinterDnD, _DND_BASES = None, None, ()

from config import MODEL_PRICING, Provider, Settings
from core.cost_estimator import CostReport, CostEstimator, format_int, format_try, format_usd
from core.glossary import Glossary
from core.parser import LocalizationFile, TranslatableFilter, load_localization_file
from main import LocalizationPipeline, PreparedJob, _atomic_write, _default_output, _write_sample
from parsers import GameParserError, PackageStudio, is_studio_json, translated_path_for
from parsers.studio import ExtractedPackage, PackageInfo, RepackResult
from providers import create_translator
from providers.base import FatalProviderError

SUPPORTED_SUFFIXES = (".json", ".csv", ".tsv")
STUDIO_SUFFIXES = (".asr", ".vdf", ".txt", ".res", ".arc")
TARGET_COLUMN = "tr"

TAB_ENGINE = "Çeviri Motoru (JSON / CSV)"
TAB_STUDIO = "Oyun Paket Stüdyosu (Binary / Archive)"
FORMAT_OPTIONS = {"Otomatik": None, "Rebellion Asura (.asr)": "asura", "Valve KeyValues (.vdf / .txt)": "vdf"}
ENCODING_OPTIONS = {"Otomatik": None, "UTF-16 LE": "utf-16-le", "UTF-8": "utf-8", "CP1254": "cp1254"}
FONT_OPTIONS = {"Tam Türkçe (ğ ş ı korunur)": False, "Latin-1 uyumlu (ğ→g ş→s ı→i)": True}
BATCH_MIN, BATCH_MAX, BATCH_STEP = 10, 500, 10  # "Batch boyutu" kaydırıcısı (satır/istek)

COLORS = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "tile": "#1c2330",
    "border": "#30363d",
    "accent": "#2f81f7",
    "accent_hover": "#1f6feb",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
    "danger_hover": "#da3633",
    "text": "#e6edf3",
    "muted": "#8b949e",
}

PROVIDER_LABELS = {
    Provider.MOCK: "Mock (test)",
    Provider.GEMINI: "Gemini",
    Provider.CLAUDE: "Claude",
    Provider.OLLAMA: "Ollama (yerel)",
}
API_KEY_ENV = {Provider.GEMINI: "GEMINI_API_KEY", Provider.CLAUDE: "ANTHROPIC_API_KEY"}

if sys.platform.startswith("win"):
    MONO_FAMILY = "Consolas"
elif sys.platform == "darwin":
    MONO_FAMILY = "Menlo"
else:
    MONO_FAMILY = "DejaVu Sans Mono"


def _patch_tk9_menu_index() -> None:
    """Python 3.10/3.11'in tkinter'ı, Tk 9'un boş menü için döndürdüğü '' yanıtını tanımaz
    (CPython GH-103685) ve CustomTkinter'ın açılır menüleri kurulurken çöker. Yeni CPython
    sürümlerindeki düzeltmenin aynısı; yalnızca Tk 9'da uygulanır, Tk 8.6'da hiçbir şeyi değiştirmez."""
    if tk.TkVersion < 9.0:
        return

    def index(self: tk.Menu, index: Any) -> int | None:
        value = self.tk.call(self._w, "index", index)
        return None if value in ("", "none") else self.tk.getint(value)

    tk.Menu.index = index  # type: ignore[method-assign]


_patch_tk9_menu_index()


def provider_models(settings: Settings) -> dict[Provider, list[str]]:
    """Açılır listede önerilen modeller. Model kutusu düzenlenebilir; listede olmayan bir ad da yazılabilir."""
    def with_default(default: str, options: list[str]) -> list[str]:
        return [default] + [option for option in options if option != default]

    return {
        Provider.MOCK: ["mock"],
        Provider.GEMINI: with_default(settings.gemini_model, ["gemini-3.5-flash", "gemini-3.5-flash-lite"]),
        Provider.CLAUDE: with_default(settings.claude_model, ["claude-sonnet-5-5", "claude-haiku-5-5"]),
        Provider.OLLAMA: with_default(settings.ollama_model, ["gemma3:12b", "qwen2.5:14b"]),
    }


# =========================================================================== motor köprüsü


@dataclass(frozen=True)
class Analysis:
    path: Path
    report: CostReport


@dataclass(frozen=True)
class RunResult:
    output_path: Path
    report_path: Path
    total: int
    translated: int
    fallback: int
    rerequested_lines: int
    bisections: int
    api_requests: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float | None


ENGINE_LOGGERS = ("l10n", "providers")  # main.py ve providers/* logger adları


class _Cancelled(Exception):
    """Kullanıcı durdurdu; motor bir sonraki blok sınırında duruyor."""


class QueueLogHandler(logging.Handler):
    """Motorun logging kayıtlarını (reddedilen satırlar, yeniden denemeler) arayüz kuyruğuna taşır."""

    def __init__(self, events: queue.Queue) -> None:
        super().__init__(level=logging.INFO)
        self.events = events
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.events.put(("log", (record.levelname, self.format(record))))
        except Exception:  # log yüzünden motor asla durmamalı
            self.handleError(record)


class EngineService:
    """Arayüz ile motor arasındaki tek köprü.

    tkinter'a hiç dokunmaz: her işi daemon bir thread'de yapar ve sonucu
    (tür, veri) olayı olarak kuyruğa bırakır. Aynı anda tek iş çalışır.
    """

    def __init__(self, events: queue.Queue, studio: PackageStudio | None = None) -> None:
        self.events = events
        self.studio = studio or PackageStudio(Settings.from_env().output_dir / "studio")
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def analyze(self, path: Path, settings: Settings) -> None:
        self._start(self._analyze, path, settings)

    def translate(self, path: Path, output_path: Path, settings: Settings) -> None:
        self._cancel.clear()
        self._start(self._translate, path, output_path, settings)

    def cancel(self) -> None:
        self._cancel.set()

    # ---- Paket Stüdyosu (parsers/) ----------------------------------------

    def inspect_package(self, path: Path, format_id: str | None, encoding: str | None) -> None:
        self._start(lambda: self._emit("studio_info", self.studio.inspect(
            path, format_override=format_id, encoding=encoding)))

    def extract_package(self, path: Path, format_id: str | None, encoding: str | None) -> None:
        self._start(lambda: self._emit("studio_extracted", self.studio.extract(
            path, format_override=format_id, encoding=encoding)))

    def repack_package(self, original: Path, translated_json: Path, *, latin1_fallback: bool,
                       format_id: str | None, encoding: str | None) -> None:
        self._start(lambda: self._emit("studio_repacked", self.studio.repack(
            original, translated_json, latin1_fallback=latin1_fallback,
            format_override=format_id, encoding=encoding)))

    # ---- iç işleyiş -------------------------------------------------------

    def _emit(self, kind: str, payload: Any = None) -> None:
        self.events.put((kind, payload))

    def _start(self, target: Any, *args: Any) -> None:
        if self.busy:
            raise RuntimeError("Motor şu anda başka bir iş yürütüyor")
        self._worker = threading.Thread(target=self._guard, args=(target, *args), daemon=True)
        self._worker.start()

    def _guard(self, target: Any, *args: Any) -> None:
        # İş süresince motorun logları (reddedilen satırlar, yeniden denemeler) bu kuyruğa akar.
        handler = QueueLogHandler(self.events)
        root = logging.getLogger()
        previous_levels = {name: logging.getLogger(name).level for name in ENGINE_LOGGERS}
        for name in ENGINE_LOGGERS:
            logging.getLogger(name).setLevel(logging.INFO)
        root.addHandler(handler)
        try:
            target(*args)
        except _Cancelled:
            self._emit("cancelled")
        except (FatalProviderError, GameParserError) as exc:
            self._emit("error", str(exc))
        except Exception as exc:  # beklenmeyen her hata arayüzde görünür, uygulama kapanmaz
            logging.getLogger("l10n.gui").debug("Motor hatası", exc_info=True)
            self._emit("error", f"{type(exc).__name__}: {exc}")
        finally:
            root.removeHandler(handler)
            for name, level in previous_levels.items():
                logging.getLogger(name).setLevel(level)
            self._emit("idle")

    @staticmethod
    def _load(path: Path, settings: Settings) -> tuple[LocalizationPipeline, LocalizationFile, PreparedJob]:
        glossary = Glossary.load(settings.glossary_path)
        options: dict[str, Any] = {"target_column": TARGET_COLUMN} if path.suffix.lower() in (".csv", ".tsv") else {}
        if is_studio_json(path):
            # Paket Stüdyosu JSON'unda anahtarlar oyun ID'sidir ("Weapon_Key", "0042"); motorun
            # "sistem anahtarı" kuralları bunları yanlışlıkla atlamasın.
            options["text_filter"] = TranslatableFilter(use_key_rules=False)
        loc_file = load_localization_file(path, **options)
        pipeline = LocalizationPipeline(settings, glossary)
        return pipeline, loc_file, pipeline.prepare(loc_file, path.name)

    def _analyze(self, path: Path, settings: Settings) -> None:
        pipeline, _, job = self._load(path, settings)
        self._emit("analysis", Analysis(path, pipeline.estimate(job)))

    def _translate(self, path: Path, output_path: Path, settings: Settings) -> None:
        pipeline, loc_file, job = self._load(path, settings)
        total_batches = len(job.batches)
        self._emit("progress", (0, total_batches))
        translator = create_translator(settings, pipeline.system_prompt)
        done = 0

        def on_batch(message: str) -> None:
            nonlocal done
            done += 1
            self._emit("log", ("SUCCESS" if "satır İngilizce" not in message else "WARNING", message.strip()))
            self._emit("progress", (done, total_batches))
            if self._cancel.is_set():
                raise _Cancelled()

        translations, stats = pipeline.run(job, translator, progress=on_batch)
        rendered = loc_file.render(translations)
        loc_file.verify_rendered(rendered, translations)  # iskelet bozuksa dosya asla yazılmaz
        _atomic_write(output_path, rendered)

        usage = translator.usage
        cost = CostEstimator(MODEL_PRICING, settings.usd_try_rate).actual_cost_usd(
            settings.pricing_key, usage.input_tokens, usage.output_tokens, usage.cached_input_tokens)
        report_path = output_path.with_name(f"{output_path.stem}.report.json")
        source_by_uid = {unit.uid: unit.source for unit in job.units}
        _atomic_write(report_path, json.dumps({
            "source": str(path),
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
            "failures": [{"id": uid, "source": source_by_uid.get(uid), "problems": problems}
                         for uid, problems in stats.failures.items()],
        }, ensure_ascii=False, indent=2) + "\n")

        self._emit("done", RunResult(
            output_path=output_path, report_path=report_path, total=len(job.units),
            translated=stats.translated, fallback=stats.fallback,
            rerequested_lines=stats.rerequested_lines, bisections=stats.bisections,
            api_requests=translator.request_count, input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cached_input_tokens, output_tokens=usage.output_tokens,
            cost_usd=cost,
        ))


# =========================================================================== arayüz


class StatTile(ctk.CTkFrame):
    """Bilgi kartındaki tek bir sayı kutucuğu."""

    def __init__(self, master: Any, title: str) -> None:
        super().__init__(master, fg_color=COLORS["tile"], corner_radius=10)
        self.value = ctk.CTkLabel(self, text="—", font=ctk.CTkFont(size=24, weight="bold"),
                                  text_color=COLORS["text"])
        self.value.pack(anchor="w", padx=14, pady=(10, 0))
        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=12), text_color=COLORS["muted"]).pack(
            anchor="w", padx=14, pady=(0, 10))

    def set(self, value: str) -> None:
        self.value.configure(text=value)


class App(ctk.CTk, *_DND_BASES):
    def __init__(self, initial_file: Path | None = None) -> None:
        super().__init__(fg_color=COLORS["bg"])
        self.title("Game L10n Engine · Türkçe Yama Derleyicisi")
        self.geometry("1180x960")
        self.minsize(1000, 760)

        self.base_settings = Settings.from_env()
        self.models = provider_models(self.base_settings)
        self.events: queue.Queue = queue.Queue()
        self.engine = EngineService(self.events)
        self.source_path: Path | None = None
        self.analysis: Analysis | None = None
        self.last_output: Path | None = None
        self.mode = "idle"  # idle | analyzing | translating | studio
        # Paket Stüdyosu durumu
        self.studio_path: Path | None = None
        self.studio_info: PackageInfo | None = None
        self.studio_output: Path | None = None
        self.pending_engine_load: Path | None = None  # extract bitince Sekme 1'e yüklenecek JSON

        self.dnd_enabled = self._enable_dnd()
        self._build_layout()
        self._on_provider_change(PROVIDER_LABELS[self.base_settings.provider])
        self._apply_mode()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._poll_events)
        self._log("Hazır. Bir dil dosyası seçin ya da pencereye sürükleyin.", "MUTED")
        if not self.dnd_enabled:
            self._log("Sürükle-bırak kapalı: etkinleştirmek için `pip install tkinterdnd2`.", "WARNING")
        if initial_file:
            self.after(150, lambda: self._route_file(initial_file))

    # ------------------------------------------------------------------ kurulum

    def _enable_dnd(self) -> bool:
        if TkinterDnD is None:
            return False
        try:
            self.TkdndVersion = TkinterDnD._require(self)
            return True
        except (tk.TclError, RuntimeError):
            return False

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)

        self._build_header()
        self.tabs = ctk.CTkTabview(
            self, height=560, fg_color=COLORS["bg"], corner_radius=12,
            segmented_button_fg_color=COLORS["panel"], segmented_button_selected_color=COLORS["accent"],
            segmented_button_selected_hover_color=COLORS["accent_hover"],
            segmented_button_unselected_color=COLORS["panel"], segmented_button_unselected_hover_color=COLORS["tile"],
            segmented_button_font=ctk.CTkFont(size=14, weight="bold"), text_color=COLORS["text"])
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 4))

        engine_tab = self.tabs.add(TAB_ENGINE)
        engine_tab.grid_columnconfigure(0, weight=0, minsize=380)
        engine_tab.grid_columnconfigure(1, weight=1)
        left = ctk.CTkFrame(engine_tab, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 10), pady=(0, 12))
        self._build_source_panel(left)
        self._build_provider_panel(left)
        self._build_analysis_panel(engine_tab)
        self._build_action_bar(engine_tab)

        self._build_studio_tab(self.tabs.add(TAB_STUDIO))
        self.tabs.set(TAB_ENGINE)
        self._build_log_panel()

    def _panel(self, master: Any, step: str, title: str) -> ctk.CTkFrame:
        panel = ctk.CTkFrame(master, fg_color=COLORS["panel"], corner_radius=12,
                             border_width=1, border_color=COLORS["border"])
        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 8))
        ctk.CTkLabel(header, text=step, width=24, height=24, corner_radius=12, fg_color=COLORS["accent"],
                     text_color="white", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        ctk.CTkLabel(header, text=title, font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=COLORS["text"]).pack(side="left", padx=10)
        return panel

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(16, 12))
        ctk.CTkLabel(header, text="Game L10n Engine", font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=COLORS["text"]).pack(side="left")
        ctk.CTkLabel(header, text="  ön-derleme Türkçe yama üretici", font=ctk.CTkFont(size=14),
                     text_color=COLORS["muted"]).pack(side="left", pady=(6, 0))
        self.status_pill = ctk.CTkLabel(header, text="Hazır", corner_radius=12, height=28, width=150,
                                        fg_color=COLORS["tile"], text_color=COLORS["muted"],
                                        font=ctk.CTkFont(size=13, weight="bold"))
        self.status_pill.pack(side="right")

    def _build_source_panel(self, master: Any) -> None:
        panel = self._panel(master, "1", "Kaynak dosya")
        panel.pack(fill="x", pady=(0, 12))

        self.drop_zone = ctk.CTkFrame(panel, height=132, fg_color=COLORS["tile"], corner_radius=10,
                                      border_width=2, border_color=COLORS["border"])
        self.drop_zone.pack(fill="x", padx=16)
        self.drop_zone.pack_propagate(False)
        ctk.CTkLabel(self.drop_zone, text="⇩", font=ctk.CTkFont(size=26),
                     text_color=COLORS["accent"]).pack(pady=(8, 0))
        drop_text = "Dil dosyasını buraya sürükleyin" if self.dnd_enabled else "Dil dosyası seçin"
        ctk.CTkLabel(self.drop_zone, text=drop_text, font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).pack()
        ctk.CTkLabel(self.drop_zone, text=".json  ·  .csv  ·  .tsv", font=ctk.CTkFont(size=12),
                     text_color=COLORS["muted"]).pack()
        self.browse_button = ctk.CTkButton(self.drop_zone, text="Dosya Seç", width=120, height=30,
                                           fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                                           command=self._browse)
        self.browse_button.pack(pady=(6, 0))

        if self.dnd_enabled:
            # Pencerenin herhangi bir yerine bırakmak da çalışır; dosya türüne göre doğru sekmeye gider.
            for widget in (self, self.drop_zone):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
            self.drop_zone.dnd_bind("<<DropEnter>>", lambda e: self._highlight_drop(True))
            self.drop_zone.dnd_bind("<<DropLeave>>", lambda e: self._highlight_drop(False))

        row = ctk.CTkFrame(panel, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(10, 14))
        self.file_label = ctk.CTkLabel(row, text="Henüz dosya seçilmedi", anchor="w", justify="left",
                                       text_color=COLORS["muted"], font=ctk.CTkFont(size=12), wraplength=250)
        self.file_label.pack(side="left", fill="x", expand=True)
        self.sample_button = ctk.CTkButton(row, text="Örnek veri", width=90, height=26,
                                           fg_color="transparent", border_width=1,
                                           border_color=COLORS["border"], hover_color=COLORS["tile"],
                                           text_color=COLORS["muted"], command=self._load_sample)
        self.sample_button.pack(side="right")

    def _build_provider_panel(self, master: Any) -> None:
        panel = self._panel(master, "2", "Sağlayıcı ve model")
        panel.pack(fill="x")
        form = ctk.CTkFrame(panel, fg_color="transparent")
        form.pack(fill="x", padx=16, pady=(0, 16))
        form.grid_columnconfigure(1, weight=1)

        def label(row: int, text: str) -> None:
            ctk.CTkLabel(form, text=text, text_color=COLORS["muted"], font=ctk.CTkFont(size=13)).grid(
                row=row, column=0, sticky="w", pady=6, padx=(0, 12))

        label(0, "Sağlayıcı")
        self.provider_menu = ctk.CTkOptionMenu(
            form, values=list(PROVIDER_LABELS.values()), command=self._on_provider_change,
            fg_color=COLORS["tile"], button_color=COLORS["border"], button_hover_color=COLORS["accent"])
        self.provider_menu.grid(row=0, column=1, sticky="ew", pady=6)

        label(1, "Model")
        self.model_box = ctk.CTkComboBox(
            form, values=[], command=lambda _value: self._refresh_cost(),
            fg_color=COLORS["tile"], border_color=COLORS["border"], button_color=COLORS["border"])
        self.model_box.grid(row=1, column=1, sticky="ew", pady=6)
        self.model_box.bind("<FocusOut>", lambda _e: self._refresh_cost())

        label(2, "API anahtarı")
        self.api_key_entry = ctk.CTkEntry(form, show="•", fg_color=COLORS["tile"], border_color=COLORS["border"])
        self.api_key_entry.grid(row=2, column=1, sticky="ew", pady=6)

        label(3, "Çıktı")
        output_row = ctk.CTkFrame(form, fg_color="transparent")
        output_row.grid(row=3, column=1, sticky="ew", pady=6)
        self.output_entry = ctk.CTkEntry(output_row, fg_color=COLORS["tile"], border_color=COLORS["border"],
                                         placeholder_text="kaynağın yanında tr.json")
        self.output_entry.pack(side="left", fill="x", expand=True)
        self.output_button = ctk.CTkButton(output_row, text="…", width=34, fg_color=COLORS["tile"],
                                           hover_color=COLORS["border"], command=self._browse_output)
        self.output_button.pack(side="left", padx=(6, 0))

        # Mega-batch: bir istekte kaç satır gideceği (dry-run kartı anında yenilenir)
        label(4, "Batch boyutu")
        batch_row = ctk.CTkFrame(form, fg_color="transparent")
        batch_row.grid(row=4, column=1, sticky="ew", pady=6)
        self.batch_slider = ctk.CTkSlider(
            batch_row, from_=BATCH_MIN, to=BATCH_MAX, number_of_steps=(BATCH_MAX - BATCH_MIN) // BATCH_STEP,
            command=self._on_batch_change, progress_color=COLORS["accent"], button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"], fg_color=COLORS["tile"])
        self.batch_slider.pack(side="left", fill="x", expand=True)
        self.batch_label = ctk.CTkLabel(batch_row, text="", width=78, anchor="e", text_color=COLORS["text"],
                                        font=ctk.CTkFont(size=13, weight="bold"))
        self.batch_label.pack(side="left", padx=(8, 0))
        self.batch_slider.set(self._snap_batch(self.base_settings.chunk_size))
        self.batch_label.configure(text=f"{self._batch_size()} satır")
        self._batch_reanalyze_job: str | None = None

        # Rate limit: iki API isteği arası en az bekleme (Gemini ücretsiz katmanı: 5 istek/dk -> 13 sn)
        label(5, "İstek arası")
        interval_row = ctk.CTkFrame(form, fg_color="transparent")
        interval_row.grid(row=5, column=1, sticky="ew", pady=6)
        self.interval_entry = ctk.CTkEntry(interval_row, width=64, justify="center", fg_color=COLORS["tile"],
                                           border_color=COLORS["border"])
        self.interval_entry.pack(side="left")
        self.interval_hint = ctk.CTkLabel(interval_row, text="sn", anchor="w", text_color=COLORS["muted"],
                                          font=ctk.CTkFont(size=12))
        self.interval_hint.pack(side="left", padx=(8, 0))

    def _build_analysis_panel(self, master: Any) -> None:
        panel = self._panel(master, "3", "Dry-run analizi")
        panel.grid(row=0, column=1, sticky="nsew", padx=(10, 10), pady=(0, 12))
        self.analysis_hint = ctk.CTkLabel(panel, text="Dosya seçildiğinde API'ye istek atmadan analiz edilir.",
                                          text_color=COLORS["muted"], font=ctk.CTkFont(size=12))
        self.analysis_hint.pack(anchor="w", padx=16)

        grid = ctk.CTkFrame(panel, fg_color="transparent")
        grid.pack(fill="x", padx=16, pady=(8, 0))
        titles = [("total", "Toplam metin alanı"), ("translatable", "Çevrilecek satır"), ("words", "Kelime"),
                  ("tags", "Korunan kod/etiket"), ("skipped", "Atlanan alan"), ("batches", "İstek (batch)")]
        self.tiles: dict[str, StatTile] = {}
        for index, (key, title) in enumerate(titles):
            tile = StatTile(grid, title)
            tile.grid(row=index // 3, column=index % 3, sticky="ew", padx=4, pady=4)
            grid.grid_columnconfigure(index % 3, weight=1, uniform="tiles")
            self.tiles[key] = tile

        cost = ctk.CTkFrame(panel, fg_color=COLORS["tile"], corner_radius=10)
        cost.pack(fill="x", padx=20, pady=(8, 0))
        cost.grid_columnconfigure(1, weight=1)
        self.cost_title = ctk.CTkLabel(cost, text="Tahmini maliyet", text_color=COLORS["muted"],
                                       font=ctk.CTkFont(size=12))
        self.cost_title.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))
        self.cost_value = ctk.CTkLabel(cost, text="—", font=ctk.CTkFont(size=28, weight="bold"),
                                       text_color=COLORS["success"])
        self.cost_value.grid(row=1, column=0, sticky="w", padx=14)
        self.cost_detail = ctk.CTkLabel(cost, text="", text_color=COLORS["muted"], font=ctk.CTkFont(size=12),
                                        justify="left", anchor="w")
        self.cost_detail.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 10))
        self.compare_frame = ctk.CTkFrame(cost, fg_color="transparent")
        self.compare_frame.grid(row=0, column=1, rowspan=3, sticky="ne", padx=14, pady=10)

        self.skipped_label = ctk.CTkLabel(panel, text="", text_color=COLORS["muted"], font=ctk.CTkFont(size=12),
                                          anchor="w", justify="left", wraplength=640)
        self.skipped_label.pack(fill="x", padx=20, pady=(8, 14))

    def _build_action_bar(self, master: Any) -> None:
        bar = ctk.CTkFrame(master, fg_color=COLORS["panel"], corner_radius=12, border_width=1,
                           border_color=COLORS["border"])
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 4))
        bar.grid_columnconfigure(2, weight=1)
        self.start_button = ctk.CTkButton(bar, text="▶  Çeviriyi Başlat", width=190, height=40,
                                          font=ctk.CTkFont(size=15, weight="bold"),
                                          fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                                          command=self._start_translation)
        self.start_button.grid(row=0, column=0, padx=(14, 8), pady=12)
        self.stop_button = ctk.CTkButton(bar, text="■  Durdur", width=110, height=40,
                                         fg_color="transparent", border_width=1, border_color=COLORS["danger"],
                                         text_color=COLORS["danger"], hover_color="#2a1215",
                                         command=self._stop_translation)
        self.stop_button.grid(row=0, column=1, padx=(0, 14), pady=12)
        self.progress = ctk.CTkProgressBar(bar, height=14, progress_color=COLORS["accent"],
                                           fg_color=COLORS["tile"])
        self.progress.set(0)
        self.progress.grid(row=0, column=2, sticky="ew", padx=4)
        self.progress_label = ctk.CTkLabel(bar, text="0 / 0 batch", width=120, text_color=COLORS["muted"],
                                           font=ctk.CTkFont(size=13))
        self.progress_label.grid(row=0, column=3, padx=8)
        self.open_button = ctk.CTkButton(bar, text="Çıktı Klasörünü Aç", width=150, height=40,
                                         fg_color=COLORS["tile"], hover_color=COLORS["border"],
                                         command=self._open_output_folder)
        self.open_button.grid(row=0, column=4, padx=(8, 14))

    def _build_log_panel(self) -> None:
        panel = ctk.CTkFrame(self, fg_color=COLORS["panel"], corner_radius=12, border_width=1,
                             border_color=COLORS["border"])
        panel.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 18))
        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(10, 4))
        ctk.CTkLabel(header, text="Canlı log",font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=COLORS["text"]).pack(side="left")
        ctk.CTkButton(header, text="Temizle", width=70, height=24, fg_color="transparent",
                      hover_color=COLORS["tile"], text_color=COLORS["muted"],
                      command=self._clear_log).pack(side="right")
        self.log_box = ctk.CTkTextbox(panel, fg_color=COLORS["bg"], text_color=COLORS["text"],
                                      font=ctk.CTkFont(family=MONO_FAMILY, size=12), wrap="word",
                                      corner_radius=8, border_width=0)
        self.log_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        for tag, color in (("INFO", COLORS["text"]), ("SUCCESS", COLORS["success"]),
                           ("WARNING", COLORS["warning"]), ("ERROR", COLORS["danger"]),
                           ("MUTED", COLORS["muted"]), ("TIME", COLORS["muted"])):
            self.log_box.tag_config(tag, foreground=color)
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------------ Sekme 2: Paket Stüdyosu

    def _build_studio_tab(self, tab: Any) -> None:
        tab.grid_columnconfigure(0, weight=0, minsize=380)
        tab.grid_columnconfigure(1, weight=1)

        # (1) Oyun dosyası + elle seçimler
        source = self._panel(tab, "1", "Oyun dosyası")
        source.grid(row=0, column=0, sticky="nsew", padx=(10, 10), pady=(0, 12))
        self.studio_drop = ctk.CTkFrame(source, height=118, fg_color=COLORS["tile"], corner_radius=10,
                                        border_width=2, border_color=COLORS["border"])
        self.studio_drop.pack(fill="x", padx=16)
        self.studio_drop.pack_propagate(False)
        drop_text = "Oyun dosyasını buraya sürükleyin" if self.dnd_enabled else "Oyun dosyası seçin"
        ctk.CTkLabel(self.studio_drop, text=drop_text, font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).pack(pady=(14, 0))
        ctk.CTkLabel(self.studio_drop, text=".asr  ·  .vdf  ·  .txt (Valve)", font=ctk.CTkFont(size=12),
                     text_color=COLORS["muted"]).pack()
        self.studio_browse = ctk.CTkButton(self.studio_drop, text="Dosya Seç", width=120, height=30,
                                           fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                                           command=self._studio_browse)
        self.studio_browse.pack(pady=(8, 0))
        if self.dnd_enabled:
            self.studio_drop.drop_target_register(DND_FILES)
            self.studio_drop.dnd_bind("<<Drop>>", self._on_drop)
            self.studio_drop.dnd_bind("<<DropEnter>>", lambda e: self.studio_drop.configure(border_color=COLORS["accent"]))
            self.studio_drop.dnd_bind("<<DropLeave>>", lambda e: self.studio_drop.configure(border_color=COLORS["border"]))

        self.studio_file_label = ctk.CTkLabel(source, text="Henüz oyun dosyası seçilmedi", anchor="w", justify="left",
                                              text_color=COLORS["muted"], font=ctk.CTkFont(size=12), wraplength=330)
        self.studio_file_label.pack(fill="x", padx=16, pady=(10, 4))

        form = ctk.CTkFrame(source, fg_color="transparent")
        form.pack(fill="x", padx=16, pady=(4, 16))
        form.grid_columnconfigure(1, weight=1)
        menus: dict[str, ctk.CTkOptionMenu] = {}
        for row, (label, options, callback) in enumerate((
                ("Format", FORMAT_OPTIONS, self._studio_reinspect),
                ("Kodlama", ENCODING_OPTIONS, self._studio_reinspect),
                ("Font", FONT_OPTIONS, None))):
            ctk.CTkLabel(form, text=label, text_color=COLORS["muted"], font=ctk.CTkFont(size=13)).grid(
                row=row, column=0, sticky="w", pady=5, padx=(0, 12))
            menu = ctk.CTkOptionMenu(form, values=list(options), command=callback,
                                     fg_color=COLORS["tile"], button_color=COLORS["border"],
                                     button_hover_color=COLORS["accent"])
            menu.grid(row=row, column=1, sticky="ew", pady=5)
            menus[label] = menu
        self.studio_format_menu, self.studio_encoding_menu, self.studio_font_menu = (
            menus["Format"], menus["Kodlama"], menus["Font"])

        # (2) Algılama kartı
        detect = self._panel(tab, "2", "Otomatik algılama (magic bytes)")
        detect.grid(row=0, column=1, sticky="nsew", padx=(10, 10), pady=(0, 12))
        self.studio_format_label = ctk.CTkLabel(
            detect, text="Algılanan Format: —", anchor="w", justify="left", wraplength=640,
            font=ctk.CTkFont(size=15, weight="bold"), text_color=COLORS["text"])
        self.studio_format_label.pack(fill="x", padx=16, pady=(0, 6))
        tiles = ctk.CTkFrame(detect, fg_color="transparent")
        tiles.pack(fill="x", padx=16)
        self.studio_tiles: dict[str, StatTile] = {}
        for column, (key, title) in enumerate((("count", "Toplam metin"), ("glyphs", "Buton/ikon glifi"),
                                               ("printf", "printf yer tutucu"), ("size", "Dosya boyutu"))):
            tile = StatTile(tiles, title)
            tile.grid(row=0, column=column, sticky="ew", padx=4, pady=4)
            tiles.grid_columnconfigure(column, weight=1, uniform="studio")
            self.studio_tiles[key] = tile
        self.studio_detail_label = ctk.CTkLabel(detect, text="", anchor="w", justify="left", wraplength=640,
                                                text_color=COLORS["muted"], font=ctk.CTkFont(size=12))
        self.studio_detail_label.pack(fill="x", padx=20, pady=(6, 14))

        # (3) Tek tıkla iş akışı
        flow = self._panel(tab, "3", "İş akışı")
        flow.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 4))
        body =ctk.CTkFrame(flow, fg_color="transparent")
        body.pack(fill="x", padx=16, pady=(0, 14))
        body.grid_columnconfigure(1, weight=1)

        self.extract_button = ctk.CTkButton(
            body, text="1. Paketi Çöz ve Çeviriye Gönder (Extract)", height=40, width=380,
            font=ctk.CTkFont(size=14, weight="bold"), fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"], command=self._studio_extract)
        self.extract_button.grid(row=0, column=0, sticky="w", pady=(0, 8))
        ctk.CTkLabel(body, text="JSON + .meta yedeği oluşturur ve JSON'u otomatik olarak Çeviri Motoru sekmesine yükler.",
                     text_color=COLORS["muted"], font=ctk.CTkFont(size=12), anchor="w").grid(
            row=0, column=1, columnspan=2, sticky="w", padx=14, pady=(0, 8))

        self.repack_button = ctk.CTkButton(
            body, text="2. Çeviriyi Oyun Dosyasına Göm (Repack / Inject)", height=40, width=380,
            font=ctk.CTkFont(size=14, weight="bold"), fg_color=COLORS["success"], hover_color="#2ea043",
            text_color="#0d1117", command=self._studio_repack)
        self.repack_button.grid(row=1, column=0, sticky="w")
        self.studio_translated_entry = ctk.CTkEntry(body, fg_color=COLORS["tile"], border_color=COLORS["border"],
                                                    placeholder_text="Çevrilmiş JSON (…/<ad>.tr.json)")
        self.studio_translated_entry.grid(row=1, column=1, sticky="ew", padx=(14, 6))
        self.studio_translated_button = ctk.CTkButton(body, text="…", width=34, fg_color=COLORS["tile"],
                                                      hover_color=COLORS["border"], command=self._studio_browse_json)
        self.studio_translated_button.grid(row=1, column=2)

        result_row = ctk.CTkFrame(body, fg_color="transparent")
        result_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.studio_output_label = ctk.CTkLabel(result_row, text="Çıktı: —", anchor="w", text_color=COLORS["muted"],
                                                font=ctk.CTkFont(size=12))
        self.studio_output_label.pack(side="left", fill="x", expand=True)
        self.studio_open_button = ctk.CTkButton(result_row, text="Klasörü Aç", width=110, height=30,
                                                fg_color=COLORS["tile"], hover_color=COLORS["border"],
                                                command=lambda: self._open_folder(self.studio_output))
        self.studio_open_button.pack(side="right")

    def _studio_options(self) -> tuple[str | None, str | None, bool]:
        return (FORMAT_OPTIONS[self.studio_format_menu.get()], ENCODING_OPTIONS[self.studio_encoding_menu.get()],
                FONT_OPTIONS[self.studio_font_menu.get()])

    def _studio_browse(self) -> None:
        selected = filedialog.askopenfilename(
            title="Oyun dil dosyası seçin",
            filetypes=[("Oyun dil dosyaları", "*.asr *.vdf *.txt *.res *.arc"), ("Tüm dosyalar", "*.*")])
        if selected:
            self.studio_load(Path(selected))

    def _studio_browse_json(self) -> None:
        selected = filedialog.askopenfilename(title="Çevrilmiş JSON", filetypes=[("JSON", "*.json")])
        if selected:
            self._set_entry(self.studio_translated_entry, selected)

    def studio_load(self, path: Path) -> None:
        if self.engine.busy:
            self._log("Motor meşgulken yeni dosya yüklenemez.", "WARNING")
            return
        if not path.is_file():
            self._log(f"Dosya bulunamadı: {path}", "ERROR")
            return
        self.studio_path = path
        self.studio_info = None
        self.studio_output = None
        self.studio_file_label.configure(text=f"{path.name}\n{path.parent}", text_color=COLORS["text"])
        self._set_entry(self.studio_translated_entry, "")
        self.tabs.set(TAB_STUDIO)
        self._log(f"Oyun dosyası seçildi: {path}", "INFO")
        self._studio_reinspect()

    def _studio_reinspect(self, _choice: str | None = None) -> None:
        if self.studio_path is None or self.engine.busy:
            return
        format_id, encoding, _ = self._studio_options()
        self.mode = "studio"
        self._set_status("Paket analiz ediliyor…", COLORS["warning"])
        self._apply_mode()
        self.engine.inspect_package(self.studio_path, format_id, encoding)

    def _studio_extract(self) -> None:
        if self.studio_path is None or self.engine.busy:
            return
        format_id, encoding, _ = self._studio_options()
        self.mode = "studio"
        self._set_status("Paket çözülüyor…", COLORS["warning"])
        self._apply_mode()
        self.engine.extract_package(self.studio_path, format_id, encoding)

    def _studio_repack(self) -> None:
        if self.studio_path is None or self.engine.busy:
            return
        translated = Path(self.studio_translated_entry.get().strip())
        if not translated.name or not translated.is_file():
            messagebox.showerror("Çevrilmiş JSON", "Önce Çeviri Motoru sekmesinde çeviriyi tamamlayın "
                                                    "ya da çevrilmiş JSON dosyasını seçin.")
            return
        format_id, encoding, latin1 = self._studio_options()
        self.mode = "studio"
        self._set_status("Paketleniyor…", COLORS["warning"])
        self._apply_mode()
        self._log(f"Repack: {self.studio_path.name} + {translated.name}"
                  + (" (Latin-1 font uyumu: ğ→g ş→s ı→i)" if latin1 else ""), "INFO")
        self.engine.repack_package(self.studio_path, translated, latin1_fallback=latin1,
                                   format_id=format_id, encoding=encoding)

    def _show_studio_info(self, package: PackageInfo) -> None:
        self.studio_info = package
        info = package.info
        self.studio_format_label.configure(text=package.describe(),
                                           text_color=COLORS["text"] if info.supported else COLORS["warning"])
        details = info.details or {}
        self.studio_tiles["count"].set(format_int(info.entry_count) if info.entry_count is not None else "—")
        self.studio_tiles["glyphs"].set(format_int(details["glyphs"]) if "glyphs" in details else "—")
        self.studio_tiles["printf"].set(format_int(details["printf"]) if "printf" in details else "—")
        self.studio_tiles["size"].set(f"{package.size / 1024:,.1f} KB".replace(",", "X").replace(".", ",").replace("X", "."))
        lines = [f"SHA-256: {package.sha256[:32]}…"]
        if details.get("chunks"):
            lines.append("Chunk'lar: " + ", ".join(details["chunks"]))
        if details.get("skipped"):
            skipped = {key: value for key, value in details["skipped"].items() if value}
            if skipped:
                lines.append("Çıkarılmayanlar: " + ", ".join(f"{k}: {v}" for k, v in skipped.items()))
        if not info.supported:
            lines.append("Bu format tanındı ama extract/repack desteği henüz yok.")
        self.studio_detail_label.configure(text="\n".join(lines))
        self._set_status("Paket analiz edildi", COLORS["success"] if info.supported else COLORS["warning"])
        self._log(package.describe(), "SUCCESS" if info.supported else "WARNING")

    def _show_studio_extract(self, package: ExtractedPackage) -> None:
        self._set_entry(self.studio_translated_entry, str(package.translated_path))
        self.pending_engine_load = package.json_path
        self._set_status("Paket çözüldü", COLORS["success"])
        self._log(f"Extract tamam: {package.info.entry_count} metin -> {package.json_path}", "SUCCESS")
        self._log(f"Meta yedeği: {package.meta_path}", "MUTED")
        self._log("JSON, Çeviri Motoru sekmesine yükleniyor; çeviri bitince 2. adıma dönün.", "INFO")

    def _show_studio_repack(self, result: RepackResult) -> None:
        self.studio_output = result.output_path
        self.studio_output_label.configure(text=f"Çıktı: {result.output_path}", text_color=COLORS["success"])
        self._set_status("✓ Oyun dosyası hazır", COLORS["success"])
        self._log(f"Repack tamam: {result.changed} metin gömüldü, {result.unchanged} orijinal kaldı · "
                  f"{format_int(result.size_before)} -> {format_int(result.size_after)} bayt", "SUCCESS")
        for warning in result.warnings[:20]:
            self._log(warning, "WARNING")
        if len(result.warnings) > 20:
            self._log(f"... ve {len(result.warnings) - 20} uyarı daha", "WARNING")
        self._log(f"Oyuna atılmaya hazır dosya: {result.output_path}", "SUCCESS")

    # ------------------------------------------------------------------ durum

    def _apply_mode(self) -> None:
        busy = self.mode != "idle"
        translating = self.mode == "translating"
        ready = not busy and self.analysis is not None
        self.start_button.configure(state="normal" if ready else "disabled")
        self.stop_button.configure(state="normal" if translating else "disabled")
        for widget in (self.browse_button, self.sample_button, self.provider_menu, self.model_box,
                       self.api_key_entry, self.output_entry, self.output_button, self.batch_slider,
                       self.interval_entry):
            widget.configure(state="disabled" if busy else "normal")
        self.open_button.configure(state="normal" if self.last_output and not busy else "disabled")
        if not busy:
            self._update_api_key_field()
            if self._selected_provider() is Provider.MOCK:
                self.interval_entry.configure(state="disabled")  # mock hiç beklemez

        supported = self.studio_info is not None and self.studio_info.info.supported
        for widget in (self.studio_browse, self.studio_format_menu, self.studio_encoding_menu, self.studio_font_menu,
                       self.studio_translated_entry, self.studio_translated_button):
            widget.configure(state="disabled" if busy else "normal")
        self.extract_button.configure(state="normal" if supported and not busy else "disabled")
        self.repack_button.configure(state="normal" if supported and not busy else "disabled")
        self.studio_open_button.configure(state="normal" if self.studio_output and not busy else "disabled")

    def _set_status(self, text: str, color: str) -> None:
        self.status_pill.configure(text=text, text_color=color)

    def _highlight_drop(self, active: bool) -> None:
        self.drop_zone.configure(border_color=COLORS["accent"] if active else COLORS["border"])

    # ------------------------------------------------------------------ dosya seçimi

    def _on_drop(self, event: Any) -> None:
        self._highlight_drop(False)
        self.studio_drop.configure(border_color=COLORS["border"])
        paths = [Path(p) for p in self.tk.splitlist(event.data)]
        if paths:
            self._route_file(paths[0])

    def _route_file(self, path: Path) -> None:
        """Dosya türüne göre doğru sekmeye gönderir: oyun dosyası -> Stüdyo, JSON/CSV -> Motor."""
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            self.tabs.set(TAB_ENGINE)
            self.load_file(path)
        else:
            self.studio_load(path)

    def _browse(self) -> None:
        selected = filedialog.askopenfilename(
            title="Oyun dil dosyası seçin",
            filetypes=[("Dil dosyaları", "*.json *.csv *.tsv"), ("Tüm dosyalar", "*.*")])
        if selected:
            self.load_file(Path(selected))

    def _browse_output(self) -> None:
        suffix = self.source_path.suffix if self.source_path else ".json"
        selected = filedialog.asksaveasfilename(title="Çıktı dosyası", defaultextension=suffix,
                                                initialfile=f"{TARGET_COLUMN}{suffix}")
        if selected:
            self._set_entry(self.output_entry, selected)

    def _load_sample(self) -> None:
        self.load_file(_write_sample(self.base_settings.output_dir))

    def load_file(self, path: Path) -> None:
        if self.engine.busy:
            self._log("Motor meşgulken yeni dosya yüklenemez.", "WARNING")
            return
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            self._log(f"Desteklenmeyen dosya türü: {path.name} (.json, .csv, .tsv)", "ERROR")
            return
        if not path.is_file():
            self._log(f"Dosya bulunamadı: {path}", "ERROR")
            return
        self.source_path = path
        self.analysis = None
        self.file_label.configure(text=f"{path.name}\n{path.parent}", text_color=COLORS["text"])
        output = translated_path_for(path) if is_studio_json(path) else _default_output(path, TARGET_COLUMN)
        self._set_entry(self.output_entry, str(output))
        self._log(f"Dosya yüklendi: {path}", "INFO")
        self._run_analysis()

    def _run_analysis(self) -> None:
        assert self.source_path is not None
        self.mode = "analyzing"
        self._set_status("Analiz ediliyor…", COLORS["warning"])
        self._apply_mode()
        self.engine.analyze(self.source_path, self._settings_from_ui())

    # ------------------------------------------------------------------ sağlayıcı / model

    def _selected_provider(self) -> Provider:
        label = self.provider_menu.get()
        return next(provider for provider, text in PROVIDER_LABELS.items() if text == label)

    def _on_provider_change(self, label: str) -> None:
        self.provider_menu.set(label)
        provider = self._selected_provider()
        values = self.models[provider]
        self.model_box.configure(values=values)
        self.model_box.set(values[0])
        self._update_api_key_field()
        self._update_interval_field(provider)
        self._refresh_cost()

    def _update_interval_field(self, provider: Provider) -> None:
        """Sağlayıcı değişince istek aralığını o sağlayıcının güvenli varsayılanına getirir."""
        defaults = replace(self.base_settings, provider=provider)
        hints = {
            Provider.GEMINI: "sn · ücretsiz katman 5 istek/dk → 13 sn; ücretli: 0",
            Provider.CLAUDE: "sn · ücretli API, genelde 0",
            Provider.OLLAMA: "sn · yerel model, 0",
            Provider.MOCK: "sn · mock beklemez",
        }
        self.interval_entry.configure(state="normal")
        self._set_entry(self.interval_entry, f"{defaults.effective_request_interval:g}")
        self.interval_hint.configure(text=hints[provider])
        if provider is Provider.MOCK:
            self.interval_entry.configure(state="disabled")

    @staticmethod
    def _snap_batch(value: float) -> int:
        step = round((value - BATCH_MIN) / BATCH_STEP)
        return int(min(BATCH_MAX, max(BATCH_MIN, BATCH_MIN + step * BATCH_STEP)))

    def _batch_size(self) -> int:
        return self._snap_batch(self.batch_slider.get())

    def _on_batch_change(self, value: float) -> None:
        self.batch_label.configure(text=f"{self._snap_batch(value)} satır")
        if self._batch_reanalyze_job is not None:  # kaydırırken her adımda değil, bırakınca analiz et
            self.after_cancel(self._batch_reanalyze_job)
        self._batch_reanalyze_job = self.after(500, self._reanalyze_after_batch_change)

    def _reanalyze_after_batch_change(self) -> None:
        self._batch_reanalyze_job = None
        if self.source_path is None:
            return
        if self.engine.busy:
            self._batch_reanalyze_job = self.after(300, self._reanalyze_after_batch_change)
            return
        self._run_analysis()

    def _request_interval(self, provider: Provider) -> float | None:
        if provider is Provider.MOCK:
            return 0.0
        raw = self.interval_entry.get().strip().replace(",", ".")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            self._log(f"'İstek arası' sayı olmalı ({raw!r}); sağlayıcı varsayılanı kullanılıyor.", "WARNING")
            return None

    def _update_api_key_field(self) -> None:
        provider = self._selected_provider()
        env_name = API_KEY_ENV.get(provider)
        if env_name is None:
            self._set_entry(self.api_key_entry, "")
            self.api_key_entry.configure(state="disabled", placeholder_text="Bu sağlayıcı anahtar istemez")
            return
        self.api_key_entry.configure(state="normal")
        has_env = bool(os.getenv(env_name) or (provider is Provider.GEMINI and os.getenv("GOOGLE_API_KEY")))
        hint = f"{env_name} ortam değişkeninden okunuyor" if has_env else f"Anahtarı yapıştırın ({env_name})"
        self.api_key_entry.configure(placeholder_text=hint)

    def _settings_from_ui(self) -> Settings:
        """Her işte ortamdan taze ayar alınır; arayüz yalnızca seçilen alanları ezer."""
        settings = Settings.from_env()
        settings.provider = self._selected_provider()
        model = self.model_box.get().strip()
        attribute = {Provider.GEMINI: "gemini_model", Provider.CLAUDE: "claude_model",
                     Provider.OLLAMA: "ollama_model"}.get(settings.provider)
        if attribute and model:
            setattr(settings, attribute, model)
        key = self.api_key_entry.get().strip() if self.api_key_entry.cget("state") == "normal" else ""
        if key and settings.provider is Provider.GEMINI:
            settings.gemini_api_key = key
        elif key and settings.provider is Provider.CLAUDE:
            settings.anthropic_api_key = key
        settings.chunk_size = self._batch_size()
        interval = self._request_interval(settings.provider)
        if interval is not None:
            settings.request_interval = interval
        return settings

    # ------------------------------------------------------------------ bilgi kartı

    def _show_analysis(self, analysis: Analysis) -> None:
        self.analysis = analysis
        stats = analysis.report.stats
        skipped_total = sum(stats.skipped_by_reason.values())
        values = {
            "total": stats.total_entries, "translatable": stats.translatable, "words": stats.words,
            "tags": stats.protected_tags, "skipped": skipped_total + stats.code_only, "batches": stats.batches,
        }
        for key, value in values.items():
            self.tiles[key].set(format_int(value))
        self.analysis_hint.configure(text=f"{analysis.path.name}  ·  {format_int(stats.characters)} karakter  ·  "
                                          f"tahmini ~{format_int(analysis.report.input_tokens)} girdi / "
                                          f"~{format_int(analysis.report.output_tokens)} çıktı token")
        details = ", ".join(f"{reason}: {count}" for reason, count in stats.skipped_by_reason.items())
        if stats.code_only:
            details += f"{', ' if details else ''}sadece kod: {stats.code_only}"
        self.skipped_label.configure(text=f"Atlananlar → {details}" if details else "Atlanan alan yok.")
        self._refresh_cost()
        interval = self._settings_from_ui().effective_request_interval
        waiting = max(stats.batches - 1, 0) * interval
        wait_note = f" İstek arası {interval:g} sn → toplam en az ~{waiting:.0f} sn bekleme." if waiting else ""
        self._log(f"Dry-run: {stats.translatable} satır, {stats.words} kelime, {stats.protected_tags} korunan "
                  f"kod, {stats.batches} batch ({self._batch_size()} satır/istek).{wait_note}", "SUCCESS")

    def _refresh_cost(self) -> None:
        for child in self.compare_frame.winfo_children():
            child.destroy()
        if self.analysis is None:
            self.cost_value.configure(text="—")
            self.cost_detail.configure(text="Analizden sonra seçili modele göre hesaplanır.")
            return

        provider = self._selected_provider()
        key = "ollama" if provider is Provider.OLLAMA else self.model_box.get().strip()
        estimates = {estimate.model: estimate for estimate in self.analysis.report.estimates}
        selected = estimates.get(key)
        self.cost_title.configure(text=f"Tahmini maliyet · {key}")
        if selected is not None:
            self.cost_value.configure(text=format_usd(selected.usd))
            self.cost_detail.configure(text=f"{format_try(selected.try_amount)}  ·  ~{format_int(selected.input_tokens)}"
                                            f" girdi / ~{format_int(selected.output_tokens)} çıktı token"
                                            + (f"\n{selected.note}" if selected.note else ""))
        elif provider is Provider.MOCK:
            self.cost_value.configure(text="$0")
            self.cost_detail.configure(text="Test modu: API'ye istek gitmez, ücret yok.")
        else:
            self.cost_value.configure(text="—")
            self.cost_detail.configure(text="Bu model fiyat tablosunda yok (config.py → MODEL_PRICING).")

        ctk.CTkLabel(self.compare_frame, text="Karşılaştırma", text_color=COLORS["muted"],
                     font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, columnspan=3, sticky="e")
        rows = [e for e in self.analysis.report.estimates if e.model != "mock"]
        for row, estimate in enumerate(rows, start=1):
            color = COLORS["text"] if estimate.model == key else COLORS["muted"]
            font = ctk.CTkFont(size=12, weight="bold" if estimate.model == key else "normal")
            for column, text in enumerate((estimate.model, format_usd(estimate.usd), format_try(estimate.try_amount))):
                ctk.CTkLabel(self.compare_frame, text=text, text_color=color, font=font, height=18).grid(
                    row=row, column=column, sticky="e", padx=(12, 0))

    # ------------------------------------------------------------------ çeviri

    def _start_translation(self) -> None:
        if self.source_path is None or self.analysis is None or self.engine.busy:
            return
        output_text = self.output_entry.get().strip()
        output_path = Path(output_text) if output_text else _default_output(self.source_path, TARGET_COLUMN)
        if output_path.resolve() == self.source_path.resolve() and self.source_path.suffix.lower() == ".json":
            messagebox.showerror("Geçersiz çıktı", "Çıktı dosyası kaynak dosyanın üzerine yazılamaz.")
            return
        settings = self._settings_from_ui()
        if settings.provider is Provider.GEMINI and not settings.gemini_api_key:
            messagebox.showerror("API anahtarı", "Gemini anahtarı yok. Alana yapıştırın ya da "
                                                 "GEMINI_API_KEY ortam değişkenini ayarlayın.")
            return
        if settings.provider is Provider.CLAUDE and not settings.anthropic_api_key:
            # SDK anahtarı `ant auth login` profilinden de çözebilir; kullanıcıya soralım.
            if not messagebox.askyesno("API anahtarı", "ANTHROPIC_API_KEY bulunamadı.\n"
                                                       "Kayıtlı `ant auth login` profiliyle denensin mi?"):
                return

        self.mode = "translating"
        self.last_output = None
        self.progress.set(0)
        self._set_progress_busy(True)  # mega-batch'te ilk blok 1-2 dk sürebilir; çubuk hareketli kalsın
        self.progress_label.configure(text="başlatılıyor…")
        self._set_status("Çevriliyor…", COLORS["accent"])
        self._apply_mode()
        self._log(f"Çeviri başladı · {PROVIDER_LABELS[settings.provider]} · {settings.active_model} · "
                  f"{settings.chunk_size} satır/istek · istek arası {settings.effective_request_interval:g} sn "
                  f"→ {output_path}", "INFO")
        self.engine.translate(self.source_path, output_path, settings)

    def _set_progress_busy(self, busy: bool) -> None:
        if busy and not getattr(self, "_progress_busy", False):
            self.progress.configure(mode="indeterminate")
            self.progress.start()
        elif not busy and getattr(self, "_progress_busy", False):
            self.progress.stop()
            self.progress.configure(mode="determinate")
        self._progress_busy = busy

    def _stop_translation(self) -> None:
        if self.mode == "translating":
            self.engine.cancel()
            self.stop_button.configure(state="disabled")
            self._log("Durdurma istendi; mevcut batch bitince duracak. Çıktı dosyası yazılmayacak.", "WARNING")

    def _show_result(self, result: RunResult) -> None:
        self.last_output = result.output_path
        success = result.fallback == 0
        self._set_status("✓ Tamamlandı" if success else "! Kısmen tamamlandı",
                         COLORS["success"] if success else COLORS["warning"])
        self._log(f"SONUÇ: {result.translated}/{result.total} satır çevrildi ve doğrulandı.",
                  "SUCCESS" if success else "WARNING")
        if result.fallback:
            self._log(f"{result.fallback} satır orijinal haliyle bırakıldı (ayrıntı: {result.report_path.name}).",
                      "WARNING")
        cost = (f" → {format_usd(result.cost_usd)} ({format_try(result.cost_usd * self.base_settings.usd_try_rate)})"
                if result.cost_usd is not None else "")
        self._log(f"Yeniden istenen satır: {result.rerequested_lines} · Bölünen blok: {result.bisections} · "
                  f"API isteği: {result.api_requests}", "INFO")
        self._log(f"Kullanım: {format_int(result.input_tokens)} girdi + {format_int(result.cached_input_tokens)} "
                  f"önbellek / {format_int(result.output_tokens)} çıktı token{cost}", "INFO")
        self._log(f"Çıktı: {result.output_path}", "SUCCESS")
        self._log(f"Rapor: {result.report_path}", "MUTED")
        if self.source_path is not None and is_studio_json(self.source_path) and self.studio_path is not None:
            self._set_entry(self.studio_translated_entry, str(result.output_path))
            self._log("Çeviri hazır. 'Oyun Paket Stüdyosu' sekmesinde 2. adımla oyun dosyasına gömebilirsiniz.",
                      "INFO")

    def _open_output_folder(self) -> None:
        self._open_folder(self.last_output)

    def _open_folder(self, file_path: Path | None) -> None:
        if not file_path:
            return
        folder = str(file_path.parent)
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except OSError as exc:
            self._log(f"Klasör açılamadı: {exc}", "ERROR")

    # ------------------------------------------------------------------ olay döngüsü

    def _poll_events(self) -> None:
        try:
            for _ in range(200):  # tek seferde çok olay gelirse arayüz donmasın
                kind, payload = self.events.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        self.after(80, self._poll_events)

    def _handle_event(self, kind: str, payload: Any) -> None:
        if kind == "log":
            level, message = payload
            if level == "INFO" and "reddedildi" in message:
                level = "WARNING"
            self._log(message, level)
        elif kind == "progress":
            done, total = payload
            if done == 0:
                self.progress_label.configure(text=f"0 / {total} batch  ·  çevriliyor…")
                return
            self._set_progress_busy(False)
            fraction = done / total if total else 1.0
            self.progress.set(fraction)
            self.progress_label.configure(text=f"{done} / {total} batch  ·  %{int(fraction * 100)}")
        elif kind == "analysis":
            self._show_analysis(payload)
            self._set_status("Analiz hazır", COLORS["success"])
        elif kind == "done":
            self._show_result(payload)
        elif kind == "studio_info":
            self._show_studio_info(payload)
        elif kind == "studio_extracted":
            self._show_studio_extract(payload)
        elif kind == "studio_repacked":
            self._show_studio_repack(payload)
        elif kind == "cancelled":
            self._set_status("Durduruldu", COLORS["warning"])
            self._log("Çeviri durduruldu; çıktı dosyası yazılmadı.", "WARNING")
        elif kind == "error":
            self._set_status("✗ Hata", COLORS["danger"])
            self._log(f"HATA: {payload}", "ERROR")
            if self.mode == "translating":
                self._log("Çıktı dosyası yazılmadı.", "ERROR")
        elif kind == "idle":
            self._set_progress_busy(False)
            self.mode = "idle"
            self._apply_mode()
            if self.pending_engine_load is not None:  # extract bitti: JSON'u Sekme 1'e aktar
                path, self.pending_engine_load = self.pending_engine_load, None
                self.tabs.set(TAB_ENGINE)
                self.load_file(path)

    # ------------------------------------------------------------------ yardımcılar

    def _log(self, message: str, level: str = "INFO") -> None:
        tag = level if level in {"INFO", "SUCCESS", "WARNING", "ERROR", "MUTED"} else "INFO"
        self.log_box.configure(state="normal")
        self.log_box.insert("end", time.strftime("%H:%M:%S  "), "TIME")
        self.log_box.insert("end", message + "\n", tag)
        self.log_box.configure(state="disabled")
        self.log_box.see("end")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    @staticmethod
    def _set_entry(entry: ctk.CTkEntry, text: str) -> None:
        previous = entry.cget("state")
        entry.configure(state="normal")
        entry.delete(0, "end")
        if text:
            entry.insert(0, text)
        entry.configure(state=previous)

    def _on_close(self) -> None:
        if self.mode == "translating":
            if not messagebox.askyesno("Çeviri sürüyor", "Çeviri devam ediyor. Durdurup çıkılsın mı?\n"
                                                         "Yarım çıktı dosyası yazılmaz."):
                return
            self.engine.cancel()
        self.destroy()


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    initial = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    App(initial).mainloop()


if __name__ == "__main__":
    main()
