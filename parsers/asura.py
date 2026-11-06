"""
Rebellion Asura Engine metin dosyaları (.asr): Sniper Elite ve aynı motoru kullanan oyunlar.

Gerçek English.asr (LTXT v3) ve SSpk_Eng.asr (PTXT v5) üzerinden çözülen yapı
(tüm sayılar little-endian):

    Dosya   : "Asura   " (8 bayt) + chunk* + bitiş bloğu (16 sıfır bayt)
    Chunk   : tür (4 bayt ASCII) | boyut u32 (16 baytlık başlık DAHİL) | sürüm u32 | bilinmeyen u32 | gövde
    LTXT v3 : kayıt_sayısı u32 | kayıt*
    PTXT v5 : ad_hash u32 | kayıt_sayısı u32 | ayrılmış u32 | ad (ASCII, NUL ile biter) | kayıt*
    Kayıt   : karakter_sayısı u32 (NUL dahil, UTF-16 birimi) | UTF-16 LE metin (NUL ile biter)

    ad_hash = küçük harfli ad üzerinde h = h*31 + c (SNIPE_SPOKEN_ID -> 0xBB3D55BC, doğrulandı)
    Hizalama dolgusu yoktur (PTXT chunk boyutu 15662, 4'ün katı değil); okunamayan
    son baytlar olursa chunk "kuyruğu" olarak aynen korunur.

Repack, dosyayı sıfırdan inşa eder: her kaydın uzunluk alanı, kayıt sayısı ve chunk
boyutu struct.pack ile yeniden hesaplanır. Yerinde üzerine yazma yapılmaz. Metin dışı
her bayt (imza, sürüm, bilinmeyen alanlar, hash, ad, dolgu, bitiş bloğu) aynen korunur.
"""

from __future__ import annotations

import base64
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

from parsers.base import (
    BaseGameParser,
    DetectResult,
    ExtractResult,
    GameParserError,
    InjectResult,
    UnsupportedFormatError,
    choose_text,
    glyph_census,
    protect_glyphs,
)

ASURA_MAGIC = b"Asura   "
COMPRESSED_MAGICS = (b"AsuraZlb", b"AsuraCmp")
CHUNK_HEADER = struct.Struct("<4sIII")  # tür, boyut, sürüm, bilinmeyen
U32 = struct.Struct("<I")
PTXT_PREFIX = struct.Struct("<III")  # ad_hash, kayıt_sayısı, ayrılmış
TEXT_CHUNK_TYPES = (b"LTXT", b"PTXT")
KNOWN_VERSIONS = {b"LTXT": {3}, b"PTXT": {5}}
_PRINTF_RE = re.compile(r"%[-+0#]*\d*(?:\.\d+)?[diouxXeEfFgGsScp]")


def asura_name_hash(name: str) -> int:
    value = 0
    for byte in name.lower().encode("ascii"):
        value = (value * 31 + byte) & 0xFFFFFFFF
    return value


@dataclass
class TextEntry:
    text: str  # sondaki NUL hariç
    terminated: bool = True


@dataclass
class Chunk:
    offset: int
    type: bytes
    size: int
    version: int
    unknown: int
    body: bytes
    label: str | None = None  # metin chunk'ı ise JSON'daki üst anahtar
    ptxt: dict[str, Any] | None = None
    entries: list[TextEntry] | None = None
    tail: bytes = b""

    @property
    def is_text(self) -> bool:
        return self.entries is not None


@dataclass
class AsuraFile:
    magic: bytes
    chunks: list[Chunk] = field(default_factory=list)
    trailer: bytes = b""

    def text_chunks(self) -> list[Chunk]:
        return [chunk for chunk in self.chunks if chunk.is_text]


# --------------------------------------------------------------------------- okuma


def parse_asura(data: bytes) -> AsuraFile:
    if data[:8] in COMPRESSED_MAGICS:
        raise UnsupportedFormatError(
            f"Sıkıştırılmış Asura arşivi ({data[:8].decode('ascii')}) henüz desteklenmiyor; "
            "önce arşivden çıkarılmış .asr dosyasını kullanın."
        )
    if not data.startswith(ASURA_MAGIC):
        raise GameParserError("Asura imzası ('Asura   ') bulunamadı")

    asura = AsuraFile(magic=data[:8])
    position = 8
    while position < len(data):
        if len(data) - position < CHUNK_HEADER.size:
            asura.trailer = data[position:]
            break
        chunk_type, size, version, unknown = CHUNK_HEADER.unpack_from(data, position)
        if size == 0:  # bitiş bloğu: 16 sıfır bayt (ve varsa sonrası) aynen korunur
            asura.trailer = data[position:]
            break
        if size < CHUNK_HEADER.size or position + size > len(data):
            raise GameParserError(f"{position}. bayttaki {chunk_type!r} chunk'ının boyutu geçersiz: {size}")
        chunk = Chunk(position, chunk_type, size, version, unknown, data[position + CHUNK_HEADER.size:position + size])
        if chunk_type in TEXT_CHUNK_TYPES:
            _parse_text_chunk(chunk)
        asura.chunks.append(chunk)
        position += size
    _assign_labels(asura)
    return asura


def _parse_text_chunk(chunk: Chunk) -> None:
    body = chunk.body
    name = chunk.type.decode("ascii")
    known = KNOWN_VERSIONS.get(chunk.type, set())
    if chunk.version not in known:
        raise UnsupportedFormatError(
            f"{name} sürüm {chunk.version} henüz doğrulanmadı (desteklenen: {sorted(known)}). "
            "Bozuk dosya üretmemek için işlem durduruldu."
        )
    try:
        if chunk.type == b"LTXT":
            (count,) = U32.unpack_from(body, 0)
            position = U32.size
        else:
            name_hash, count, reserved = PTXT_PREFIX.unpack_from(body, 0)
            position = PTXT_PREFIX.size
            end = body.index(b"\0", position)
            chunk_name = body[position:end].decode("ascii")
            position = end + 1
            chunk.ptxt = {"name_hash": name_hash, "reserved": reserved, "name": chunk_name,
                          "name_hash_verified": asura_name_hash(chunk_name) == name_hash}

        entries: list[TextEntry] = []
        for index in range(count):
            (units,) = U32.unpack_from(body, position)
            position += U32.size
            raw = body[position:position + units * 2]
            if len(raw) != units * 2:
                raise GameParserError(f"{name}: {index}. kayıt chunk sınırını aşıyor")
            position += units * 2
            text = raw.decode("utf-16-le", errors="surrogatepass")
            terminated = text.endswith("\0")
            entries.append(TextEntry(text[:-1] if terminated else text, terminated))
    except (struct.error, ValueError, UnicodeDecodeError) as exc:
        if isinstance(exc, GameParserError):
            raise
        raise GameParserError(f"{name} chunk'ı okunamadı ({chunk.offset}. bayt): {exc}") from exc

    chunk.entries = entries
    chunk.tail = body[position:]


def _assign_labels(asura: AsuraFile) -> None:
    seen: dict[str, int] = {}
    for chunk in asura.text_chunks():
        base = "LTXT" if chunk.type == b"LTXT" else f"PTXT:{chunk.ptxt['name']}"  # type: ignore[index]
        seen[base] = seen.get(base, 0) + 1
        chunk.label = base if seen[base] == 1 else f"{base}#{seen[base]}"


def entry_keys(count: int) -> list[str]:
    width = max(4, len(str(max(count - 1, 0))))
    return [str(index).zfill(width) for index in range(count)]


# --------------------------------------------------------------------------- yazma


def build_asura(asura: AsuraFile, texts: Mapping[str, list[str]] | None = None) -> bytes:
    """Dosyayı sıfırdan inşa eder. texts: {etiket: [yeni metinler]}; verilmeyen chunk'lar aynen yazılır."""
    output = bytearray(asura.magic)
    for chunk in asura.chunks:
        body = chunk.body
        if chunk.is_text and texts is not None and chunk.label in texts:
            body = _encode_text_chunk(chunk, texts[chunk.label])
        output += CHUNK_HEADER.pack(chunk.type, CHUNK_HEADER.size + len(body), chunk.version, chunk.unknown)
        output += body
    output += asura.trailer
    return bytes(output)


def _encode_text_chunk(chunk: Chunk, new_texts: list[str]) -> bytes:
    assert chunk.entries is not None
    if len(new_texts) != len(chunk.entries):
        raise GameParserError(f"{chunk.label}: {len(chunk.entries)} kayıt bekleniyordu, {len(new_texts)} geldi")
    parts: list[bytes] = []
    if chunk.type == b"LTXT":
        parts.append(U32.pack(len(new_texts)))
    else:
        info = chunk.ptxt or {}
        # Ad değişmediği için hash orijinal haliyle yazılır (algoritma doğrulanamasa bile güvenli).
        parts.append(PTXT_PREFIX.pack(info["name_hash"], len(new_texts), info["reserved"]))
        parts.append(info["name"].encode("ascii") + b"\0")
    for text, entry in zip(new_texts, chunk.entries):
        payload = (text + ("\0" if entry.terminated else "")).encode("utf-16-le", errors="surrogatepass")
        parts.append(U32.pack(len(payload) // 2))  # UTF-16 birimi; vekil çiftler doğru sayılır
        parts.append(payload)
    parts.append(chunk.tail)
    return b"".join(parts)


# --------------------------------------------------------------------------- parser


class AsuraParser(BaseGameParser):
    parser_id: ClassVar[str] = "asura"
    format_name: ClassVar[str] = "Rebellion Asura Engine"
    extensions: ClassVar[tuple[str, ...]] = (".asr", ".pc", ".ps2")

    @classmethod
    def detect(cls, head: bytes, path: Path | None = None) -> float:
        if head.startswith(ASURA_MAGIC):
            return 1.0
        if head[:8] in COMPRESSED_MAGICS:
            return 0.9  # tanınır; inspect "desteklenmiyor" der
        return 0.1 if path is not None and path.suffix.lower() == ".asr" else 0.0

    def inspect(self, data: bytes, *, encoding: str | None = None) -> DetectResult:
        if data[:8] in COMPRESSED_MAGICS:
            return DetectResult(self.parser_id, self.format_name, data[:8].decode("ascii"), "zlib",
                                supported=False)
        asura = parse_asura(data)
        texts = [entry.text for chunk in asura.text_chunks() for entry in chunk.entries or []]
        variants = ", ".join(f"{c.type.decode()} v{c.version}" for c in asura.text_chunks()) or "metin chunk'ı yok"
        return DetectResult(
            self.parser_id, self.format_name, variants, "UTF-16 LE", entry_count=len(texts),
            details={
                "chunks": [f"{c.type.decode('ascii', 'replace')} v{c.version} ({c.size} bayt)" for c in asura.chunks],
                "glyphs": sum(glyph_census(texts).values()),
                "printf": sum(len(_PRINTF_RE.findall(text)) for text in texts),
                "empty": sum(1 for text in texts if not text),
            },
        )

    def extract(self, data: bytes, *, encoding: str | None = None) -> ExtractResult:
        self._check_encoding(encoding)
        asura = parse_asura(data)
        strings: dict[str, Any] = {}
        chunk_meta: list[dict[str, Any]] = []
        all_texts: list[str] = []
        for index, chunk in enumerate(asura.chunks):
            info: dict[str, Any] = {"index": index, "offset": chunk.offset, "type": chunk.type.decode("ascii", "replace"),
                                    "size": chunk.size, "version": chunk.version, "unknown": chunk.unknown}
            if chunk.is_text:
                assert chunk.entries is not None and chunk.label is not None
                keys = entry_keys(len(chunk.entries))
                strings[chunk.label] = {key: protect_glyphs(entry.text) for key, entry in zip(keys, chunk.entries)}
                all_texts += [entry.text for entry in chunk.entries]
                info.update({
                    "text": True, "label": chunk.label, "entry_count": len(chunk.entries),
                    "unterminated": [i for i, entry in enumerate(chunk.entries) if not entry.terminated],
                    "tail_b64": base64.b64encode(chunk.tail).decode("ascii"),
                })
                if chunk.ptxt:
                    info["ptxt"] = dict(chunk.ptxt)
            else:
                info.update({"text": False, "body_b64": base64.b64encode(chunk.body).decode("ascii")})
            chunk_meta.append(info)

        meta = {
            "parser": self.parser_id,
            "format": self.format_name,
            "encoding": "utf-16-le",
            "sha256": self.fingerprint(data),
            "size": len(data),
            "magic_b64": base64.b64encode(asura.magic).decode("ascii"),
            "trailer_b64": base64.b64encode(asura.trailer).decode("ascii"),
            "chunks": chunk_meta,
            "glyphs": glyph_census(all_texts),
            "glyph_token": "{glyph:XXXX} = U+XXXX (Özel Kullanım Alanı buton/harita ikonu)",
        }
        return ExtractResult(strings, meta, self.inspect(data))

    def inject(self, data: bytes, translations: Mapping[str, Any], meta: Mapping[str, Any] | None = None,
               *, encoding: str | None = None, char_map: Mapping[int, str] | None = None) -> InjectResult:
        self._check_encoding(encoding)
        self.verify_source(data, meta)
        asura = parse_asura(data)
        result = InjectResult(b"")
        new_texts: dict[str, list[str]] = {}

        for chunk in asura.text_chunks():
            assert chunk.entries is not None and chunk.label is not None
            table = translations.get(chunk.label)
            if not isinstance(table, Mapping):
                result.warnings.append(f"{chunk.label}: çeviri JSON'unda yok, orijinal korundu")
                table = {}
            texts: list[str] = []
            for key, entry in zip(entry_keys(len(chunk.entries)), chunk.entries):
                text, changed = choose_text(entry.text, table.get(key), char_map=char_map,
                                            label=f"{chunk.label}/{key}", warnings=result.warnings)
                texts.append(text)
                result.changed += changed
                result.unchanged += not changed
            new_texts[chunk.label] = texts

        output = build_asura(asura, new_texts)
        self._verify_output(asura, output, new_texts)
        result.data = output
        return result

    @staticmethod
    def _verify_output(original: AsuraFile, output: bytes, new_texts: Mapping[str, list[str]]) -> None:
        """Yazılan dosyayı tekrar okuyup her alanı karşılaştırır; tutarsızlıkta dosya asla döndürülmez."""
        rebuilt = parse_asura(output)
        if len(rebuilt.chunks) != len(original.chunks) or rebuilt.trailer != original.trailer:
            raise GameParserError("Doğrulama: chunk listesi ya da bitiş bloğu değişti")
        position = len(rebuilt.magic)
        for before, after in zip(original.chunks, rebuilt.chunks):
            same_header = (before.type, before.version, before.unknown) == (after.type, after.version, after.unknown)
            if not same_header or after.offset != position or after.size != CHUNK_HEADER.size + len(after.body):
                raise GameParserError(f"Doğrulama: {before.type!r} chunk başlığı tutarsız")
            position += after.size
            if before.is_text:
                if [e.text for e in after.entries or []] != list(new_texts[before.label or ""]):
                    raise GameParserError(f"Doğrulama: {before.label} metinleri geri okunamadı")
                if after.ptxt and before.ptxt and after.ptxt["name_hash"] != before.ptxt["name_hash"]:
                    raise GameParserError("Doğrulama: PTXT ad hash'i değişti")
                if after.tail != before.tail:
                    raise GameParserError(f"Doğrulama: {before.label} dolgu baytları değişti")
            elif after.body != before.body:
                raise GameParserError(f"Doğrulama: metin dışı {before.type!r} chunk'ı değişti")
        if position + len(rebuilt.trailer) != len(output):
            raise GameParserError("Doğrulama: dosya boyutu chunk toplamıyla tutmuyor")

    @staticmethod
    def _check_encoding(encoding: str | None) -> None:
        if encoding not in (None, "utf-16-le"):
            raise GameParserError("Asura metinleri her zaman UTF-16 LE'dir; kodlama seçimi 'Otomatik' olmalı")
