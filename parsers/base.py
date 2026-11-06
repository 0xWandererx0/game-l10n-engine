"""
Oyun dosyası ayrıştırıcıları için ortak altyapı (Strategy / Adapter deseni).

    Oyun dosyası --extract()--> {saf JSON metinler} + {.meta yapı bilgisi}
                                        |
                              L10nShield çeviri motoru (değişmeden)
                                        |
    Türkçe oyun dosyası <--inject()-- {çevrilmiş JSON} + orijinal dosya + .meta

Çeviri motoru yalnızca JSON görür. Binary ayrıntılar (başlıklar, uzunluk alanları,
hash'ler, dolgu baytları) parser'da ve .meta dosyasında kalır.
"""

from __future__ import annotations

import hashlib
import os
import re
import zlib
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

from core.regex_shield import RegexShield


class GameParserError(ValueError):
    """Dosya beklenen yapıda değil ya da güvenle işlenemiyor."""


class UnsupportedFormatError(GameParserError):
    """Format tanındı ama (henüz) desteklenmiyor ya da hiç tanınmadı."""


class SourceMismatchError(GameParserError):
    """Repack için verilen orijinal dosya, extract edilen dosyayla aynı değil."""


@dataclass(frozen=True)
class DetectResult:
    parser_id: str
    format_name: str
    variant: str = ""
    encoding: str = ""
    entry_count: int | None = None
    supported: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """Örn. 'Rebellion Asura Engine (LTXT v3 - UTF-16 LE) | Toplam Metin: 1688 adet'."""
        qualifiers = " - ".join(part for part in (self.variant, self.encoding) if part)
        text = self.format_name + (f" ({qualifiers})" if qualifiers else "")
        if self.entry_count is not None:
            text += f" | Toplam Metin: {self.entry_count} adet"
        if not self.supported:
            text += " | henüz desteklenmiyor"
        return text


@dataclass
class ExtractResult:
    strings: dict[str, Any]  # çeviri motoruna gidecek saf JSON
    meta: dict[str, Any]  # .meta dosyasına yazılacak yapı bilgisi
    info: DetectResult


@dataclass
class InjectResult:
    data: bytes
    changed: int = 0  # çevirisi gömülen metin sayısı
    unchanged: int = 0  # çevirisi olmayan ya da aynı kalan metin sayısı
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- özel glifler

# Unicode Özel Kullanım Alanı (PUA): oyunlar buton ve harita ikonlarını buraya koyar
# (Asura: U+E801 = kontrolcü tuşu, U+EA0E = harita ikonu). RegexShield bunları harf saymaz,
# model silebilir; bu yüzden JSON'da korunan bir yer tutucu olarak taşınırlar.
_PUA_RE = re.compile("[-]")
GLYPH_TOKEN_RE = re.compile(r"\{glyph:(E[0-9A-F]{3}|F[0-8][0-9A-F]{2})\}")
_GLYPH_PREFIX = "{glyph:"


def protect_glyphs(text: str) -> str:
    """PUA karakterlerini '{glyph:E801}' yer tutucusuna çevirir (RegexShield bunu [TAG_n] yapar)."""
    if _GLYPH_PREFIX in text:
        raise GameParserError(f"Metin zaten '{_GLYPH_PREFIX}' içeriyor; glif kodlaması geri döndürülemez: {text[:60]!r}")
    return _PUA_RE.sub(lambda m: "{glyph:%04X}" % ord(m.group()), text)


def restore_glyphs(text: str) -> str:
    return GLYPH_TOKEN_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


def glyph_census(texts: list[str]) -> dict[str, int]:
    counts = Counter(f"{ord(ch):04X}" for text in texts for ch in _PUA_RE.findall(text))
    return dict(sorted(counts.items()))


# Oyun fontunda ğ/ş/ı/İ glifi yoksa (Latin-1 fontlar) bunlar kutu olarak görünür.
# ç ö ü Latin-1'de bulunduğu için korunur.
LATIN1_TURKISH_FALLBACK = str.maketrans({"ğ": "g", "Ğ": "G", "ı": "i", "İ": "I", "ş": "s", "Ş": "S"})

_SHIELD = RegexShield()


def code_signature_matches(original: str, translated: str) -> bool:
    """Elle düzenlenmiş tr.json'a karşı son kontrol: kodlar ve glifler birebir korunmuş mu?"""
    return (
        _SHIELD.signature(original) == _SHIELD.signature(translated)
        and Counter(_PUA_RE.findall(original)) == Counter(_PUA_RE.findall(translated))
    )


def choose_text(original: str, candidate: Any, *, char_map: Mapping[int, str] | None,
                label: str, warnings: list[str]) -> tuple[str, bool]:
    """Bir kaydın yazılacak son metnini seçer: (metin, değişti_mi).

    Çeviri yoksa, boşsa ya da kodları bozulmuşsa orijinal korunur (oyun çökmez).
    """
    if not isinstance(candidate, str) or (candidate == "" and original != ""):
        return original, False
    text = restore_glyphs(candidate).replace("\0", "")
    if char_map:
        text = text.translate(char_map)
    if text == original:
        return original, False
    if not code_signature_matches(original, text):
        warnings.append(f"{label}: kod/glif imzası farklı, orijinal metin korundu")
        return original, False
    return text, True


# --------------------------------------------------------------------------- temel sınıf


class BaseGameParser(ABC):
    """Her oyun formatı bu sözleşmeyi uygular. Yeni format = yeni alt sınıf + detector kaydı."""

    parser_id: ClassVar[str]
    format_name: ClassVar[str]
    extensions: ClassVar[tuple[str, ...]] = ()
    implemented: ClassVar[bool] = True

    @classmethod
    @abstractmethod
    def detect(cls, head: bytes, path: Path | None = None) -> float:
        """Dosyanın ilk baytlarına (magic bytes) ve uzantısına bakarak 0-1 arası güven puanı."""

    @abstractmethod
    def inspect(self, data: bytes, *, encoding: str | None = None) -> DetectResult:
        """Dosyayı tamamen okuyup format/sürüm/kodlama/metin sayısı bilgisini döndürür."""

    @abstractmethod
    def extract(self, data: bytes, *, encoding: str | None = None) -> ExtractResult:
        """Metinleri saf JSON'a, metin dışı her şeyi meta'ya ayırır."""

    @abstractmethod
    def inject(self, data: bytes, translations: Mapping[str, Any], meta: Mapping[str, Any] | None = None,
               *, encoding: str | None = None, char_map: Mapping[int, str] | None = None) -> InjectResult:
        """Çevirileri orijinal yapıya gömer; tüm uzunluk/boyut alanlarını yeniden hesaplar."""

    def output_name(self, source: Path) -> str:
        return f"{source.stem}_TR{source.suffix}"

    @staticmethod
    def fingerprint(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def verify_source(self, data: bytes, meta: Mapping[str, Any] | None) -> None:
        if not meta:
            return
        if meta.get("parser") not in (None, self.parser_id):
            raise SourceMismatchError(f".meta '{meta.get('parser')}' formatına ait, bu dosya '{self.parser_id}'")
        expected = meta.get("sha256")
        if expected and expected != self.fingerprint(data):
            raise SourceMismatchError(
                "Orijinal dosya, extract edilen dosyayla aynı değil (SHA-256 farklı). "
                "Paketi çözdüğünüz dosyanın aynısını seçin."
            )


# --------------------------------------------------------------------------- yardımcılar


def zlib_inflate(data: bytes, expected_size: int | None = None) -> bytes:
    """Sıkıştırılmış arşivler (Capcom ARC, AsuraZlb) için; boyut tutmazsa hata verir."""
    output = zlib.decompress(data)
    if expected_size is not None and len(output) != expected_size:
        raise GameParserError(f"zlib çıktısı {len(output)} bayt, beklenen {expected_size}")
    return output


def zlib_deflate(data: bytes, level: int = 9) -> bytes:
    return zlib.compress(data, level)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Yarım dosya bırakmaz: geçici dosyaya yazar, sonra tek adımda yerine koyar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
