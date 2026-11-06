"""
Dil dosyası ayrıştırıcı: iç içe JSON (ve düz CSV) -> benzersiz ID'li metin listesi -> geri birleştirme.

JSON'da her yaprağın kimliği bir JSON Pointer'dır (RFC 6901), örn.
"/dialogues/chapter_1/briefing/3/text". Anahtarında "/" veya "." geçen dosyalarda
bile benzersizdir ve geri dönüştürülebilir.

Geri birleştirme orijinalin derin kopyası üzerinde yalnızca çevrilen metinleri
değiştirir; anahtar sırası, sayılar, boolean'lar, null'lar ve çevrilmeyen her şey
olduğu gibi kalır. `verify_rendered` çıktının iskeletini orijinalle karşılaştırır.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, Union

PathPart = Union[str, int]
JsonPath = tuple[PathPart, ...]


class SkipReason(str, Enum):
    NON_STRING = "sayı/bool/null"
    EMPTY = "boş metin"
    SKIP_KEY = "sistem anahtarı"
    NO_LETTERS = "harf içermiyor"
    URL = "URL/e-posta"
    PATH = "dosya yolu"
    IDENTIFIER = "kimlik/kod"


@dataclass(frozen=True)
class FlatEntry:
    uid: str
    path: JsonPath
    value: Any
    skip_reason: SkipReason | None = None

    @property
    def translatable(self) -> bool:
        return self.skip_reason is None


# Bu anahtarların altındaki metinler oyun sistemine aittir; çevrilirse asset bulunamaz.
DEFAULT_SKIP_KEYS = frozenset({
    "id", "uid", "uuid", "guid", "key", "keys", "ref", "icon", "image", "img", "sprite",
    "texture", "material", "sound", "sfx", "audio", "voice", "voice_file", "music", "clip",
    "path", "file", "filename", "dir", "folder", "prefab", "asset", "scene", "font",
    "font_asset", "color", "colour", "anim", "animation", "speaker_id", "type", "class",
    "tag", "tags", "url", "uri", "link", "href", "email", "version", "locale", "language",
    "lang", "event", "trigger", "script", "shader", "layer", "slot",
})
DEFAULT_SKIP_KEY_SUFFIXES = (
    "_id", "_ids", "_key", "_path", "_file", "_url", "_icon", "_sprite", "_texture",
    "_sound", "_sfx", "_audio", "_prefab", "_asset", "_guid", "_color", "_anim", "_event",
)

_ASSET_EXTENSIONS = (
    "png|jpe?g|tga|psd|dds|bmp|gif|webp|svg|exr|hdr|wav|mp3|ogg|flac|bnk|wem|fbx|obj|gltf|glb|"
    "blend|prefab|asset|mat|anim|controller|unity|json|xml|ya?ml|csv|tsv|txt|ini|cfg|lua|cs|js|py|"
    "uasset|umap|pak|tscn|tres|res|gd|shader|hlsl|glsl|ttf|otf|fnt|mp4|webm|bik|usm|bytes|dat|bin|"
    "dll|exe|so|zip|locres"
)
_URL_RE = re.compile(r"^(?:[a-z][a-z0-9+.\-]*://\S+|www\.\S+|[\w.+\-]+@[\w\-]+\.[\w.\-]+)$", re.IGNORECASE)
_FILE_RE = re.compile(rf"^\S+\.(?:{_ASSET_EXTENSIONS})$", re.IGNORECASE)
_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_DOTTED_OR_SNAKE_RE = re.compile(r"^[A-Za-z0-9]+(?:[_.:][A-Za-z0-9]+)+$")
_CAMEL_RE = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]*)+$")
_HEX_COLOR_RE = re.compile(r"^#(?:[0-9A-Fa-f]{3,4}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$")
_GUID_RE = re.compile(r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")


class TranslatableFilter:
    """Bir değerin çevrilebilir metin mi, yoksa motorun kullandığı sistem değeri mi olduğuna karar verir.

    Tereddütte "sistem değeri" tarafında kalır: çevrilmeyen bir metin yalnızca
    İngilizce görünür, çevrilen bir asset yolu ise oyunu bozar.
    """

    def __init__(
        self,
        skip_keys: Iterable[str] = DEFAULT_SKIP_KEYS,
        skip_key_suffixes: Sequence[str] = DEFAULT_SKIP_KEY_SUFFIXES,
        *,
        use_key_rules: bool = True,
    ) -> None:
        self.skip_keys = frozenset(k.lower() for k in skip_keys)
        self.skip_key_suffixes = tuple(s.lower() for s in skip_key_suffixes)
        self.use_key_rules = use_key_rules

    def classify(self, path: JsonPath, value: Any) -> SkipReason | None:
        if not isinstance(value, str):
            return SkipReason.NON_STRING
        text = value.strip()
        if not text:
            return SkipReason.EMPTY
        if self.use_key_rules and self._is_system_key(path):
            return SkipReason.SKIP_KEY
        if not any(ch.isalpha() for ch in text):
            return SkipReason.NO_LETTERS
        if _URL_RE.match(text):
            return SkipReason.URL
        if self._looks_like_path(text):
            return SkipReason.PATH
        if self._looks_like_identifier(text):
            return SkipReason.IDENTIFIER
        return None

    def _is_system_key(self, path: JsonPath) -> bool:
        key = next((part for part in reversed(path) if isinstance(part, str)), None)
        if key is None:
            return False
        key = key.lower()
        return key in self.skip_keys or key.endswith(self.skip_key_suffixes)

    @staticmethod
    def _looks_like_path(text: str) -> bool:
        if any(ch.isspace() for ch in text):
            return False
        if _FILE_RE.match(text) or _WINDOWS_PATH_RE.match(text) or text.startswith(("res://", "user://", "/Game/")):
            return True
        # "Yes/No" veya "and/or" metindir; en az iki ayraç bir klasör yolunu işaret eder.
        return text.count("/") >= 2 or text.count("\\") >= 2

    @staticmethod
    def _looks_like_identifier(text: str) -> bool:
        if any(ch.isspace() for ch in text):
            return False
        return bool(
            _DOTTED_OR_SNAKE_RE.match(text)
            or _CAMEL_RE.match(text)
            or _HEX_COLOR_RE.match(text)
            or _GUID_RE.match(text)
        )


class LocalizationFile(Protocol):
    """main.py'nin dosya türünden bağımsız çalışması için ortak arayüz."""

    source_path: Path | None

    def entries(self) -> list[FlatEntry]: ...

    def render(self, translations: Mapping[str, str]) -> str: ...

    def verify_rendered(self, rendered: str, translations: Mapping[str, str]) -> None: ...


class StructureMismatchError(ValueError):
    """Yeniden oluşturulan dosyanın iskeleti orijinalden farklı (asla yazılmamalı)."""


# --------------------------------------------------------------------------- JSON


def pointer_from_path(path: JsonPath) -> str:
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)


class JsonLocalizationFile:
    """İç içe JSON dil dosyası."""

    def __init__(
        self,
        data: Any,
        *,
        source_path: Path | None = None,
        had_bom: bool = False,
        indent: int | None = 2,
        text_filter: TranslatableFilter | None = None,
    ) -> None:
        self.data = data
        self.source_path = source_path
        self.had_bom = had_bom
        self.indent = indent
        self.filter = text_filter or TranslatableFilter()
        self._entries = self._flatten()
        self._path_by_uid = {entry.uid: entry.path for entry in self._entries}

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "JsonLocalizationFile":
        path = Path(path)
        raw = path.read_bytes()
        had_bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig")
        return cls.from_string(text, source_path=path, had_bom=had_bom, **kwargs)

    @classmethod
    def from_string(cls, text: str, **kwargs: Any) -> "JsonLocalizationFile":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Geçersiz JSON (satır {exc.lineno}, sütun {exc.colno}): {exc.msg}") from exc
        kwargs.setdefault("indent", _detect_indent(text))
        return cls(data, **kwargs)

    def entries(self) -> list[FlatEntry]:
        """Tüm yapraklar (çevrilmeyecekler dahil, skip_reason ile işaretli)."""
        return list(self._entries)

    def translatable_entries(self) -> list[FlatEntry]:
        return [entry for entry in self._entries if entry.translatable]

    def unflatten(self, translations: Mapping[str, str]) -> Any:
        """Çevirileri orijinal iskeletin derin kopyasına yerleştirir."""
        result = copy.deepcopy(self.data)
        for uid, text in translations.items():
            path = self._path_by_uid.get(uid)
            if path is None:
                raise KeyError(f"Bilinmeyen ID: {uid!r}")
            if not isinstance(text, str):
                raise TypeError(f"{uid}: çeviri str olmalı, {type(text).__name__} geldi")
            if not path:
                result = text
                continue
            parent = result
            for part in path[:-1]:
                parent = parent[part]
            if not isinstance(parent[path[-1]], str):
                raise TypeError(f"{uid}: orijinal değer metin değil, üzerine yazılamaz")
            parent[path[-1]] = text
        return result

    def render(self, translations: Mapping[str, str]) -> str:
        rendered = json.dumps(self.unflatten(translations), ensure_ascii=False, indent=self.indent)
        if self.indent is not None:
            rendered += "\n"
        return ("﻿" if self.had_bom else "") + rendered

    def verify_rendered(self, rendered: str, translations: Mapping[str, str]) -> None:
        """Çıktı dosyasını yeniden okuyup orijinalle yaprak yaprak karşılaştırır."""
        reparsed = json.loads(rendered.lstrip("﻿"))
        _assert_same_shape(self.data, reparsed, (), translations)

    def _flatten(self) -> list[FlatEntry]:
        return [
            FlatEntry(pointer_from_path(path), path, value, self.filter.classify(path, value))
            for path, value in _walk(self.data, ())
        ]


def _walk(node: Any, path: JsonPath) -> Iterator[tuple[JsonPath, Any]]:
    if isinstance(node, dict):
        for key, child in node.items():
            yield from _walk(child, (*path, key))
    elif isinstance(node, list):
        for index, child in enumerate(node):
            yield from _walk(child, (*path, index))
    else:
        yield path, node


def _assert_same_shape(original: Any, rendered: Any, path: JsonPath, translations: Mapping[str, str]) -> None:
    where = pointer_from_path(path) or "/"
    if isinstance(original, dict):
        if not isinstance(rendered, dict) or list(original) != list(rendered):
            raise StructureMismatchError(f"{where}: anahtarlar değişmiş")
        for key in original:
            _assert_same_shape(original[key], rendered[key], (*path, key), translations)
    elif isinstance(original, list):
        if not isinstance(rendered, list) or len(original) != len(rendered):
            raise StructureMismatchError(f"{where}: liste uzunluğu değişmiş")
        for index, (a, b) in enumerate(zip(original, rendered)):
            _assert_same_shape(a, b, (*path, index), translations)
    else:
        uid = pointer_from_path(path)
        expected = translations.get(uid, original)
        if type(expected) is not type(rendered) or expected != rendered:
            raise StructureMismatchError(f"{where}: değer beklenenden farklı")


def _detect_indent(text: str) -> int | None:
    match = re.search(r"\n( +)\S", text)
    if match:
        return len(match.group(1))
    return 2 if "\n" in text.strip() else None


# --------------------------------------------------------------------------- CSV


class CsvLocalizationFile:
    """Satır başına bir anahtar içeren düz CSV/TSV (Unity Localization, I2 vb.).

    Çeviri hedef sütuna yazılır (yoksa eklenir); diğer sütunlar korunur.
    """

    SOURCE_COLUMN_CANDIDATES = ("en", "english", "en-us", "en_us", "source", "text", "value")

    def __init__(
        self,
        text: str,
        *,
        source_path: Path | None = None,
        had_bom: bool = False,
        key_column: str | None = None,
        source_column: str | None = None,
        target_column: str = "tr",
        text_filter: TranslatableFilter | None = None,
    ) -> None:
        self.source_path = source_path
        self.had_bom = had_bom
        sample = text[:8192]
        try:
            self.dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            self.dialect = csv.excel_tab if source_path and source_path.suffix.lower() == ".tsv" else csv.excel
        reader = csv.DictReader(io.StringIO(text, newline=""), dialect=self.dialect)
        self.fieldnames: list[str] = list(reader.fieldnames or [])
        self.rows: list[dict[str, str]] = [dict(row) for row in reader]
        if len(self.fieldnames) < 2:
            raise ValueError("CSV en az iki sütun içermeli (anahtar + kaynak metin)")

        self.key_column = key_column or self.fieldnames[0]
        self.source_column = source_column or self._guess_source_column()
        self.target_column = target_column
        for column in (self.key_column, self.source_column):
            if column not in self.fieldnames:
                raise ValueError(f"CSV'de '{column}' sütunu yok. Mevcut: {self.fieldnames}")

        self.filter = text_filter or TranslatableFilter(use_key_rules=False)
        self._entries, self._row_by_uid = self._build_entries()

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "CsvLocalizationFile":
        path = Path(path)
        raw = path.read_bytes()
        return cls(raw.decode("utf-8-sig"), source_path=path, had_bom=raw.startswith(b"\xef\xbb\xbf"), **kwargs)

    def entries(self) -> list[FlatEntry]:
        return list(self._entries)

    def render(self, translations: Mapping[str, str]) -> str:
        unknown = set(translations) - set(self._row_by_uid)
        if unknown:
            raise KeyError(f"Bilinmeyen ID'ler: {sorted(unknown)[:5]}")
        fieldnames = list(self.fieldnames)
        if self.target_column not in fieldnames:
            fieldnames.append(self.target_column)
        translated_rows = {row_index: translations[uid] for uid, row_index in self._row_by_uid.items()
                           if uid in translations}
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, dialect=self.dialect)
        writer.writeheader()
        for index, row in enumerate(self.rows):
            out = dict(row)
            source_value = row.get(self.source_column) or ""
            out[self.target_column] = translated_rows.get(index, row.get(self.target_column) or source_value)
            writer.writerow(out)
        return ("﻿" if self.had_bom else "") + buffer.getvalue()

    def verify_rendered(self, rendered: str, translations: Mapping[str, str]) -> None:
        reader = csv.DictReader(io.StringIO(rendered.lstrip("﻿"), newline=""), dialect=self.dialect)
        rows = list(reader)
        if len(rows) != len(self.rows):
            raise StructureMismatchError(f"Satır sayısı değişmiş: {len(self.rows)} -> {len(rows)}")
        for uid, row_index in self._row_by_uid.items():
            if rows[row_index].get(self.key_column) != self.rows[row_index].get(self.key_column):
                raise StructureMismatchError(f"{uid}: anahtar sütunu değişmiş")
            if uid in translations and rows[row_index].get(self.target_column) != translations[uid]:
                raise StructureMismatchError(f"{uid}: çeviri yazılamamış")

    def _guess_source_column(self) -> str:
        lowered = {name.lower(): name for name in self.fieldnames}
        for candidate in self.SOURCE_COLUMN_CANDIDATES:
            if candidate in lowered:
                return lowered[candidate]
        return self.fieldnames[1]

    def _build_entries(self) -> tuple[list[FlatEntry], dict[str, int]]:
        entries: list[FlatEntry] = []
        row_by_uid: dict[str, int] = {}
        for index, row in enumerate(self.rows):
            key = (row.get(self.key_column) or "").strip() or f"row{index + 1}"
            uid = key if key not in row_by_uid else f"{key}#{index + 1}"
            path: JsonPath = tuple(part for part in re.split(r"[./]", key) if part) or (key,)
            value = row.get(self.source_column)
            entries.append(FlatEntry(uid, path, value, self.filter.classify(path, value)))
            row_by_uid[uid] = index
        return entries, row_by_uid


def load_localization_file(path: str | Path, **kwargs: Any) -> LocalizationFile:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return JsonLocalizationFile.load(path, **kwargs)
    if suffix in {".csv", ".tsv"}:
        return CsvLocalizationFile.load(path, **kwargs)
    raise ValueError(f"Desteklenmeyen dosya türü: {suffix} (.json, .csv, .tsv destekleniyor)")
