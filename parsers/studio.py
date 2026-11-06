"""
Oyun Paket Stüdyosu iş akışı: arayüzün (gui.py) ve terminal komutlarının (python -m parsers) tek kapısı.

    inspect(oyun_dosyası)               -> algılanan format + metin sayısı
    extract(oyun_dosyası)               -> <iş_klasörü>/<ad>.source.json + <ad>.meta.json
    [çeviri motoru: <ad>.source.json    -> <ad>.tr.json]
    repack(oyun_dosyası, <ad>.tr.json)  -> <iş_klasörü>/<ad>_TR.asr (vb.)

Her dosya için ayrı bir iş klasörü açılır (output/studio/<ad>/); oyunun kendi klasörüne
hiçbir şey yazılmaz.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from parsers.base import (
    LATIN1_TURKISH_FALLBACK,
    DetectResult,
    GameParserError,
    UnsupportedFormatError,
    atomic_write_bytes,
)
from parsers.detector import AutoDetector

SOURCE_SUFFIX = ".source.json"
TRANSLATED_SUFFIX = ".tr.json"
META_SUFFIX = ".meta.json"


def _base_name(json_path: Path) -> str | None:
    for suffix in (SOURCE_SUFFIX, TRANSLATED_SUFFIX):
        if json_path.name.endswith(suffix):
            return json_path.name[: -len(suffix)]
    return None


def meta_path_for(json_path: Path) -> Path | None:
    base = _base_name(json_path)
    return json_path.with_name(base + META_SUFFIX) if base else None


def is_studio_json(json_path: Path) -> bool:
    """Bu JSON Paket Stüdyosu'ndan mı çıktı? (Anahtarlar oyun ID'si; motorun anahtar kuralları kapatılır.)"""
    meta = meta_path_for(json_path)
    return meta is not None and meta.exists()


def translated_path_for(json_path: Path) -> Path:
    base = _base_name(json_path) or json_path.stem
    return json_path.with_name(base + TRANSLATED_SUFFIX)


@dataclass(frozen=True)
class PackageInfo:
    path: Path
    parser_id: str
    info: DetectResult
    size: int
    sha256: str

    def describe(self) -> str:
        return f"Algılanan Format: {self.info.describe()}"


@dataclass(frozen=True)
class ExtractedPackage:
    source: Path
    work_dir: Path
    json_path: Path
    meta_path: Path
    translated_path: Path
    info: DetectResult


@dataclass(frozen=True)
class RepackResult:
    output_path: Path
    changed: int
    unchanged: int
    size_before: int
    size_after: int
    warnings: list[str] = field(default_factory=list)


class PackageStudio:
    def __init__(self, work_root: Path, detector: AutoDetector | None = None) -> None:
        self.work_root = Path(work_root)
        self.detector = detector or AutoDetector()

    def work_dir_for(self, source: Path) -> Path:
        safe = re.sub(r"[^\w.\-]+", "_", source.stem) or "paket"
        return self.work_root / safe

    def inspect(self, path: Path, *, format_override: str | None = None,
                encoding: str | None = None) -> PackageInfo:
        parser = self.detector.detect(path, format_override=format_override)()
        data = path.read_bytes()
        info = parser.inspect(data, encoding=encoding)
        return PackageInfo(path, parser.parser_id, info, len(data), parser.fingerprint(data))

    def extract(self, path: Path, *, format_override: str | None = None,
                encoding: str | None = None) -> ExtractedPackage:
        parser = self.detector.detect(path, format_override=format_override)()
        if not parser.implemented:
            raise UnsupportedFormatError(f"{parser.format_name} için extract henüz desteklenmiyor.")
        data = path.read_bytes()
        result = parser.extract(data, encoding=encoding)

        work_dir = self.work_dir_for(path)
        base = path.stem
        json_path = work_dir / f"{base}{SOURCE_SUFFIX}"
        meta_path = work_dir / f"{base}{META_SUFFIX}"
        meta = {
            **result.meta,
            "source_name": path.name,
            "source_path": str(path.resolve()),
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "json_file": json_path.name,
            "encoding_override": encoding,
            "studio_version": 1,
        }
        _write_json(json_path, result.strings)
        _write_json(meta_path, meta)
        return ExtractedPackage(path, work_dir, json_path, meta_path, translated_path_for(json_path), result.info)

    def repack(self, original: Path, translated_json: Path, *, meta_path: Path | None = None,
               output_path: Path | None = None, latin1_fallback: bool = False,
               format_override: str | None = None, encoding: str | None = None) -> RepackResult:
        candidates = [meta_path, meta_path_for(translated_json), self.work_dir_for(original) / f"{original.stem}{META_SUFFIX}"]
        found = next((path for path in candidates if path is not None and path.exists()), None)
        if meta_path is not None and found != meta_path:
            raise GameParserError(f"Meta dosyası bulunamadı: {meta_path}")
        meta: Mapping[str, Any] | None = json.loads(found.read_text(encoding="utf-8")) if found else None
        parser_id = format_override or (meta or {}).get("parser")
        parser = (self.detector.get(parser_id) if parser_id else self.detector.detect(original))()

        try:
            translations = json.loads(translated_json.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise GameParserError(f"{translated_json.name} geçersiz JSON (satır {exc.lineno}): {exc.msg}") from exc
        if not isinstance(translations, dict):
            raise GameParserError(f"{translated_json.name} bir JSON nesnesi olmalı")

        data = original.read_bytes()
        result = parser.inject(data, translations, meta, encoding=encoding,
                               char_map=LATIN1_TURKISH_FALLBACK if latin1_fallback else None)
        output_path = output_path or self.work_dir_for(original) / parser.output_name(original)
        if output_path.resolve() == original.resolve():
            raise GameParserError("Çıktı orijinal oyun dosyasının üzerine yazılamaz")
        atomic_write_bytes(output_path, result.data)
        return RepackResult(output_path, result.changed, result.unchanged, len(data), len(result.data), result.warnings)


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
