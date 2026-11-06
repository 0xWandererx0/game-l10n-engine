"""
AutoDetector: dosyanın ilk baytlarına (magic bytes) ve uzantısına bakarak doğru parser'ı seçer.

    "Asura   " / "AsuraZlb"  -> AsuraParser
    "ARC\\0"                  -> CapcomArcParser (iskelet)
    "lang" { ... }  (metin)   -> VdfParser

Yeni bir format eklemek için: BaseGameParser alt sınıfı yaz, PARSERS'a ekle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from parsers.asura import AsuraParser
from parsers.base import BaseGameParser, UnsupportedFormatError
from parsers.capcom_arc import CapcomArcParser
from parsers.vdf import VdfParser

PARSERS: tuple[type[BaseGameParser], ...] = (AsuraParser, VdfParser, CapcomArcParser)


class AutoDetector:
    HEAD_SIZE = 4096  # ikili formatlar için 16 bayt yeter; metin formatları başta yorum içerebilir
    MIN_CONFIDENCE = 0.5

    def __init__(self, parsers: Sequence[type[BaseGameParser]] = PARSERS) -> None:
        self.parsers = {parser.parser_id: parser for parser in parsers}

    def get(self, parser_id: str) -> type[BaseGameParser]:
        try:
            return self.parsers[parser_id]
        except KeyError:
            raise UnsupportedFormatError(f"Bilinmeyen format: {parser_id}") from None

    def scores(self, head: bytes, path: Path | None = None) -> list[tuple[float, type[BaseGameParser]]]:
        ranked = [(parser.detect(head, path), parser) for parser in self.parsers.values()]
        return sorted(ranked, key=lambda item: item[0], reverse=True)

    def detect(self, path: Path, *, format_override: str | None = None) -> type[BaseGameParser]:
        if format_override:
            return self.get(format_override)
        with open(path, "rb") as handle:
            head = handle.read(self.HEAD_SIZE)
        best_score, best = self.scores(head, path)[0]
        if best_score < self.MIN_CONFIDENCE:
            preview = head[:16].hex(" ")
            raise UnsupportedFormatError(f"Format tanınamadı (ilk 16 bayt: {preview}). Formatı elle seçin.")
        return best
