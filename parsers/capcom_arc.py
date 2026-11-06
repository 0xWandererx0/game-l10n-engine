"""
Capcom MT Framework ARC arşivleri: genişletme iskeleti (Faz 4).

Bilinen yapı (little-endian sürüm):
    "ARC\\0" | sürüm u16 | dosya_sayısı u16 | giriş tablosu | zlib ile sıkıştırılmış veriler
    Giriş   : yol (64 bayt) | tür hash u32 | sıkıştırılmış boyut u32 | açık boyut + bayraklar u32 | ofset u32

Şimdilik yalnızca algılama ve başlık bilgisi vardır. Extract/inject, gerçek örnek
dosyalarla doğrulanınca eklenecek; doğrulanmamış bir yapıya yazmak oyunu bozar.
Dosya içi .msg/.gmd metin formatları bu sınıfın içinden ayrı parser'lara devredilecek.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, ClassVar, Mapping

from parsers.base import BaseGameParser, DetectResult, ExtractResult, InjectResult, UnsupportedFormatError

ARC_MAGIC_LE = b"ARC\x00"
ARC_MAGIC_BE = b"\x00CRA"


class CapcomArcParser(BaseGameParser):
    parser_id: ClassVar[str] = "capcom_arc"
    format_name: ClassVar[str] = "Capcom MT Framework ARC"
    extensions: ClassVar[tuple[str, ...]] = (".arc",)
    implemented: ClassVar[bool] = False

    @classmethod
    def detect(cls, head: bytes, path: Path | None = None) -> float:
        if head[:4] in (ARC_MAGIC_LE, ARC_MAGIC_BE):
            return 0.95
        return 0.1 if path is not None and path.suffix.lower() == ".arc" else 0.0

    def inspect(self, data: bytes, *, encoding: str | None = None) -> DetectResult:
        endian = "<" if data[:4] == ARC_MAGIC_LE else ">"
        try:
            version, file_count = struct.unpack_from(f"{endian}HH", data, 4)
        except struct.error:
            version, file_count = 0, 0
        return DetectResult(self.parser_id, self.format_name, f"v{version} · {file_count} dosya",
                            "zlib", supported=False, details={"endianness": "LE" if endian == "<" else "BE"})

    def extract(self, data: bytes, *, encoding: str | None = None) -> ExtractResult:
        raise UnsupportedFormatError("Capcom ARC extract desteği henüz eklenmedi (Faz 4).")

    def inject(self, data: bytes, translations: Mapping[str, Any], meta: Mapping[str, Any] | None = None,
               *, encoding: str | None = None, char_map: Mapping[int, str] | None = None) -> InjectResult:
        raise UnsupportedFormatError("Capcom ARC repack desteği henüz eklenmedi (Faz 4).")
