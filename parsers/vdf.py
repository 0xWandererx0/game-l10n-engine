"""
Valve KeyValues dil dosyaları (Source / Source 2: *_english.txt, *.vdf).

    "lang"
    {
        "Language"  "English"
        "Tokens"
        {
            // yorum
            "Menu_Quit"        "Quit Game"
            "Hint_Use"         "Press %use% to \"use\"\n"     [$WIN32]
            "[english]Old_Key" "kaynak metin kopyası (çevrilmez)"
        }
    }

Extract yalnızca lang > Tokens altındaki anahtar/değer çiftlerini alır. Repack
dosyayı yeniden üretmez: orijinal metinde yalnızca değerlerin tırnak içi
aralıkları değiştirilir. Girinti, yorumlar, satır sonları, koşul etiketleri ve
kodlama (BOM dahil) birebir korunur.

Değerler ham haliyle taşınır: "\\n" ve "\\"" kaçışları metinde kalır, RegexShield
onları [TAG_n] olarak korur. Repack, çeviride kaçışsız kalan çift tırnakları
kaçışlar; aksi halde dosya bozulurdu.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

from parsers.base import (
    BaseGameParser,
    DetectResult,
    ExtractResult,
    GameParserError,
    InjectResult,
    choose_text,
    protect_glyphs,
)

_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xef\xbb\xbf", "utf-8"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)
_ENCODING_LABELS = {"utf-8": "UTF-8", "utf-16-le": "UTF-16 LE", "utf-16-be": "UTF-16 BE",
                    "cp1252": "CP1252", "cp1254": "CP1254"}
_LANG_HEAD_RE = re.compile(r'^\s*(?://[^\n]*\n\s*)*"?lang"?\s*\{', re.IGNORECASE)
_TOKEN_REFERENCE_RE = re.compile(r"^#[A-Za-z0-9_]+$")  # başka bir anahtara gönderme; çevrilmez


@dataclass(frozen=True)
class VdfToken:
    kind: str  # "str" | "{" | "}" | "cond"
    value: str
    start: int  # tırnaklıysa içeriğin başı
    end: int
    quoted: bool = False
    line: int = 0


@dataclass(frozen=True)
class VdfEntry:
    json_key: str
    key: str
    value: str
    start: int
    end: int
    line: int


def detect_encoding(data: bytes, override: str | None = None) -> tuple[str, bytes]:
    """(codec, bom). Önce BOM, sonra NUL deseni (BOM'suz UTF-16), sonra UTF-8 denenir."""
    for bom, codec in _BOMS:
        if data.startswith(bom):
            return (override or codec), bom
    if override:
        return override, b""
    sample = data[:4096]
    if len(sample) >= 4 and sample[1] == 0 and sample[3] == 0 and sample[0] != 0:
        return "utf-16-le", b""
    if len(sample) >= 4 and sample[0] == 0 and sample[2] == 0 and sample[1] != 0:
        return "utf-16-be", b""
    try:
        data.decode("utf-8")
        return "utf-8", b""
    except UnicodeDecodeError:
        return "cp1252", b""


def decode(data: bytes, override: str | None = None) -> tuple[str, str, bytes]:
    codec, bom = detect_encoding(data, override)
    try:
        return data[len(bom):].decode(codec), codec, bom
    except (UnicodeDecodeError, LookupError) as exc:
        raise GameParserError(f"Dosya '{codec}' ile çözülemedi: {exc}. Kodlamayı elle seçin.") from exc


def tokenize(text: str) -> list[VdfToken]:
    tokens: list[VdfToken] = []
    position, line, length = 0, 1, len(text)
    while position < length:
        ch = text[position]
        if ch == "\n":
            line += 1
            position += 1
        elif ch.isspace():
            position += 1
        elif text.startswith("//", position):
            end = text.find("\n", position)
            position = length if end == -1 else end
        elif ch in "{}":
            tokens.append(VdfToken(ch, ch, position, position + 1, line=line))
            position += 1
        elif ch == '"':
            start = position + 1
            cursor = start
            start_line = line
            while cursor < length and text[cursor] != '"':
                if text[cursor] == "\\" and cursor + 1 < length:
                    cursor += 1
                if text[cursor] == "\n":
                    line += 1
                cursor += 1
            if cursor >= length:
                raise GameParserError(f"{start_line}. satırda kapanmayan tırnak")
            tokens.append(VdfToken("str", text[start:cursor], start, cursor, quoted=True, line=start_line))
            position = cursor + 1
        elif ch == "[":
            end = text.find("]", position)
            if end == -1:
                raise GameParserError(f"{line}. satırda kapanmayan '[' koşul etiketi")
            tokens.append(VdfToken("cond", text[position:end + 1], position, end + 1, line=line))
            position = end + 1
        else:
            start = position
            while position < length and not text[position].isspace() and text[position] not in '{}"':
                position += 1
            tokens.append(VdfToken("str", text[start:position], start, position, line=line))
    return tokens


def collect_entries(text: str) -> tuple[list[VdfEntry], dict[str, int]]:
    """lang > Tokens altındaki çevrilebilir değerleri, dosyadaki sırayla döndürür."""
    stack: list[str] = []
    pending: VdfToken | None = None
    entries: list[VdfEntry] = []
    used_keys: dict[str, int] = {}
    skipped = {"english_reference": 0, "token_reference": 0, "empty": 0, "unquoted": 0}

    for token in tokenize(text):
        if token.kind == "cond":
            continue
        if token.kind == "{":
            if pending is None:
                raise GameParserError(f"{token.line}. satırda anahtarsız blok")
            stack.append(pending.value.lower())
            pending = None
        elif token.kind == "}":
            if pending is not None or not stack:
                raise GameParserError(f"{token.line}. satırda dengesiz '}}'")
            stack.pop()
        elif pending is None:
            pending = token
        else:
            if stack == ["lang", "tokens"]:
                key = pending.value
                if key.lower().startswith("[english]"):
                    skipped["english_reference"] += 1
                elif not token.quoted:
                    skipped["unquoted"] += 1
                elif not token.value.strip():
                    skipped["empty"] += 1
                elif _TOKEN_REFERENCE_RE.match(token.value):
                    skipped["token_reference"] += 1
                else:
                    used_keys[key] = used_keys.get(key, 0) + 1
                    json_key = key if used_keys[key] == 1 else f"{key}#{used_keys[key]}"
                    entries.append(VdfEntry(json_key, key, token.value, token.start, token.end, token.line))
            pending = None
    if stack or pending is not None:
        raise GameParserError("Dosya sonunda kapanmamış blok ya da değersiz anahtar var")
    return entries, skipped


def escape_value(value: str, *, keep_raw_newlines: bool = False) -> str:
    """Çeviriyi tırnak içine güvenle yerleştirir: kaçışsız '"' ve (orijinalde yoksa) ham satır sonları kaçışlanır."""
    output: list[str] = []
    index = 0
    while index < len(value):
        ch = value[index]
        if ch == "\\":
            if index + 1 < len(value):
                output.append(value[index:index + 2])
                index += 2
                continue
            output.append("\\\\")  # sondaki tek ters bölü kapanış tırnağını yutmasın
        elif ch == '"':
            output.append('\\"')
        elif ch in "\r\n" and not keep_raw_newlines:
            if ch == "\n":
                output.append("\\n")
        else:
            output.append(ch)
        index += 1
    return "".join(output)


class VdfParser(BaseGameParser):
    parser_id: ClassVar[str] = "vdf"
    format_name: ClassVar[str] = "Valve KeyValues (Source Engine)"
    extensions: ClassVar[tuple[str, ...]] = (".vdf", ".txt", ".res")

    @classmethod
    def detect(cls, head: bytes, path: Path | None = None) -> float:
        try:
            codec, bom = detect_encoding(head)
            text = head[len(bom):].decode(codec, errors="ignore")
        except LookupError:
            return 0.0
        suffix = path.suffix.lower() if path is not None else ""
        if _LANG_HEAD_RE.match(text):
            return 0.95
        if '"tokens"' in text.lower() and suffix in cls.extensions:
            return 0.6
        return 0.2 if suffix == ".vdf" else 0.0

    def inspect(self, data: bytes, *, encoding: str | None = None) -> DetectResult:
        text, codec, bom = decode(data, encoding)
        entries, skipped = collect_entries(text)
        language = _language_name(text)
        label = _ENCODING_LABELS.get(codec, codec.upper()) + (" + BOM" if bom else "")
        return DetectResult(self.parser_id, self.format_name, f"lang/Tokens{f' · {language}' if language else ''}",
                            label, entry_count=len(entries), details={"skipped": skipped})

    def extract(self, data: bytes, *, encoding: str | None = None) -> ExtractResult:
        text, codec, bom = decode(data, encoding)
        entries, skipped = collect_entries(text)
        strings = {"Tokens": {entry.json_key: protect_glyphs(entry.value) for entry in entries}}
        meta = {
            "parser": self.parser_id,
            "format": self.format_name,
            "encoding": codec,
            "bom_hex": bom.hex(),
            "sha256": self.fingerprint(data),
            "size": len(data),
            "newline": "\\r\\n" if "\r\n" in text else "\\n",
            "language": _language_name(text),
            "entries": [{"json_key": e.json_key, "key": e.key, "line": e.line} for e in entries],
            "skipped": skipped,
        }
        return ExtractResult(strings, meta, self.inspect(data, encoding=encoding))

    def inject(self, data: bytes, translations: Mapping[str, Any], meta: Mapping[str, Any] | None = None,
               *, encoding: str | None = None, char_map: Mapping[int, str] | None = None) -> InjectResult:
        self.verify_source(data, meta)
        override = encoding or (meta or {}).get("encoding")
        text, codec, bom = decode(data, override)
        entries, _ = collect_entries(text)
        table = translations.get("Tokens") if isinstance(translations.get("Tokens"), Mapping) else {}
        result = InjectResult(b"")

        pieces: list[str] = []
        expected: list[str] = []
        cursor = 0
        for entry in entries:
            value, changed = choose_text(entry.value, table.get(entry.json_key), char_map=char_map,
                                         label=entry.json_key, warnings=result.warnings)
            raw = escape_value(value, keep_raw_newlines="\n" in entry.value) if changed else entry.value
            pieces += [text[cursor:entry.start], raw]
            expected.append(raw)
            cursor = entry.end
            result.changed += changed
            result.unchanged += not changed
        pieces.append(text[cursor:])
        output_text = "".join(pieces)

        rebuilt, _ = collect_entries(output_text)  # yapı bozulduysa burada yakalanır
        if [entry.value for entry in rebuilt] != expected or [e.json_key for e in rebuilt] != [e.json_key for e in entries]:
            raise GameParserError("Doğrulama: yeniden okunan değerler beklenenle aynı değil")
        try:
            result.data = bom + output_text.encode(codec)
        except UnicodeEncodeError as exc:
            bad = output_text[exc.start:exc.end]
            raise GameParserError(
                f"'{codec}' kodlaması {bad!r} karakterini yazamıyor. Kodlamayı UTF-8/UTF-16 LE seçin "
                "ya da 'Latin-1 uyumlu' font seçeneğini kullanın."
            ) from exc
        return result

    def output_name(self, source: Path) -> str:
        if "english" in source.name.lower():
            return re.sub("english", "turkish", source.name, flags=re.IGNORECASE)
        return super().output_name(source)


def _language_name(text: str) -> str | None:
    match = re.search(r'"Language"\s+"([^"]*)"', text, re.IGNORECASE)
    return match.group(1) if match else None
