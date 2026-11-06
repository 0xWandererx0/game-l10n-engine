"""Oyun Paket Stüdyosu: oyun dil dosyalarını saf JSON'a çeviren ve geri paketleyen adaptör katmanı.

Çeviri motoru (core/, providers/, main.py) bu paketi tanımaz; köprü tek yönlüdür.
"""

from parsers.asura import AsuraParser
from parsers.base import (
    BaseGameParser,
    DetectResult,
    ExtractResult,
    GameParserError,
    InjectResult,
    SourceMismatchError,
    UnsupportedFormatError,
)
from parsers.capcom_arc import CapcomArcParser
from parsers.detector import PARSERS, AutoDetector
from parsers.studio import PackageStudio, is_studio_json, meta_path_for, translated_path_for
from parsers.vdf import VdfParser

__all__ = [
    "PARSERS", "AsuraParser", "AutoDetector", "BaseGameParser", "CapcomArcParser", "DetectResult",
    "ExtractResult", "GameParserError", "InjectResult", "PackageStudio", "SourceMismatchError",
    "UnsupportedFormatError", "VdfParser", "is_studio_json", "meta_path_for", "translated_path_for",
]
