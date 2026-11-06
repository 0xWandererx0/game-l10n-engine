"""Paket Stüdyosu testleri.

Asura testleri sentetik dosyalar üretir; gerçek oyun dosyaları için:
    L10N_GAME_SAMPLES=/oyun/dosyalari/klasoru pytest tests/test_parsers.py
(klasörde English.asr ve/veya SSpk_Eng.asr bulunmalı; yoksa bu testler atlanır).
"""

import json
import os
import struct
from pathlib import Path

import pytest

import main
from config import Provider, Settings
from core.glossary import Glossary
from core.parser import JsonLocalizationFile, TranslatableFilter
from parsers import AutoDetector, PackageStudio, is_studio_json, translated_path_for
from parsers.asura import AsuraParser, asura_name_hash, entry_keys, parse_asura
from parsers.base import (
    LATIN1_TURKISH_FALLBACK,
    GameParserError,
    SourceMismatchError,
    UnsupportedFormatError,
    protect_glyphs,
    restore_glyphs,
)
from parsers.vdf import VdfParser, escape_value
from providers.mock_provider import MockTranslator

# --------------------------------------------------------------------------- Asura yardımcıları


def asura_entry(text: str) -> bytes:
    payload = (text + "\0").encode("utf-16-le")
    return struct.pack("<I", len(payload) // 2) + payload


def asura_chunk(kind: bytes, version: int, body: bytes) -> bytes:
    return struct.pack("<4sIII", kind, 16 + len(body), version, 0) + body


def make_ltxt(texts: list[str], extra_chunk: bytes = b"") -> bytes:
    body = struct.pack("<I", len(texts)) + b"".join(asura_entry(t) for t in texts)
    return b"Asura   " + extra_chunk + asura_chunk(b"LTXT", 3, body) + b"\0" * 16


def make_ptxt(texts: list[str], name: str = "SNIPE_SPOKEN_ID") -> bytes:
    body = struct.pack("<III", asura_name_hash(name), len(texts), 0) + name.encode() + b"\0"
    body += b"".join(asura_entry(t) for t in texts)
    return b"Asura   " + asura_chunk(b"PTXT", 5, body) + b"\0" * 16


LTXT_TEXTS = ["\ue801Accept", "", "Ammo: %d / %d", "Exit - {\uec21}", "Line one\nLine two", "Fire in the hole!"]


def roundtrip(parser, data: bytes, transform=lambda v: v):
    extracted = parser.extract(data)
    changed = {label: {k: transform(v) for k, v in table.items()} for label, table in extracted.strings.items()}
    return extracted, parser.inject(data, changed, extracted.meta)


# --------------------------------------------------------------------------- Asura


def test_asura_unchanged_roundtrip_is_byte_identical():
    for data in (make_ltxt(LTXT_TEXTS), make_ptxt(["Medic!", "Pull back!"])):
        _, result = roundtrip(AsuraParser(), data)
        assert result.data == data and result.changed == 0


def test_asura_glyphs_become_protected_tokens():
    extracted = AsuraParser().extract(make_ltxt(LTXT_TEXTS))
    table = extracted.strings["LTXT"]
    assert table["0000"] == "{glyph:E801}Accept"
    assert table["0003"] == "Exit - {{glyph:EC21}}"
    assert extracted.meta["glyphs"] == {"E801": 1, "EC21": 1}
    assert restore_glyphs(protect_glyphs("\ue801X\uea0e")) == "\ue801X\uea0e"


def test_asura_lengths_and_sizes_are_recalculated():
    data = make_ltxt(LTXT_TEXTS)
    turkish = {"0000": "{glyph:E801}Kabul Et", "0002": "Mermi: %d / %d", "0005": "Siper alın! Bomba geliyor!"}
    extracted = AsuraParser().extract(data)
    result = AsuraParser().inject(data, {"LTXT": {**extracted.strings["LTXT"], **turkish}}, extracted.meta)
    rebuilt = parse_asura(result.data)
    chunk = rebuilt.chunks[0]
    assert [e.text for e in chunk.entries][:3] == ["\ue801Kabul Et", "", "Mermi: %d / %d"]
    assert chunk.size == 16 + len(chunk.body)
    assert len(result.data) == 8 + chunk.size + 16
    # ham uzunluk alanı: "Siper alın! Bomba geliyor!" = 26 karakter + NUL
    raw = result.data
    offset = raw.index("Siper".encode("utf-16-le")) - 4
    assert struct.unpack_from("<I", raw, offset)[0] == 27
    assert result.changed == 3


def test_ptxt_hash_and_name_are_preserved():
    data = make_ptxt(["Medic!", "Pull back!"])
    _, result = roundtrip(AsuraParser(), data, lambda v: "Sıhhiye!" if v == "Medic!" else v)
    chunk = parse_asura(result.data).chunks[0]
    assert chunk.ptxt["name"] == "SNIPE_SPOKEN_ID"
    assert chunk.ptxt["name_hash"] == 0xBB3D55BC == asura_name_hash("SNIPE_SPOKEN_ID")
    assert chunk.entries[0].text == "Sıhhiye!"


def test_unknown_chunks_and_trailer_are_kept_verbatim():
    extra = asura_chunk(b"XTRA", 7, b"\x01\x02\x03\x04binary")
    data = make_ltxt(["Hello"], extra_chunk=extra)
    _, result = roundtrip(AsuraParser(), data, lambda v: "Merhaba dünya")
    assert result.data.startswith(b"Asura   " + extra)
    assert result.data.endswith(b"\0" * 16)


def test_broken_hand_edit_keeps_original_text():
    data = make_ltxt(["\ue801Accept", "Ammo: %d"])
    extracted = AsuraParser().extract(data)
    result = AsuraParser().inject(data, {"LTXT": {"0000": "Kabul Et", "0001": "Mermi: %s"}}, extracted.meta)
    assert [e.text for e in parse_asura(result.data).chunks[0].entries] == ["\ue801Accept", "Ammo: %d"]
    assert len(result.warnings) == 2


def test_latin1_fallback_maps_turkish_letters():
    data = make_ltxt(["Fire"])
    result = AsuraParser().inject(data, {"LTXT": {"0000": "Ateş ışığı İleri"}}, char_map=LATIN1_TURKISH_FALLBACK)
    assert parse_asura(result.data).chunks[0].entries[0].text == "Ates isigi Ileri"


@pytest.mark.parametrize("data, error", [
    (b"AsuraZlb" + b"\0" * 32, UnsupportedFormatError),
    (b"Asura   " + struct.pack("<4sIII", b"LTXT", 16 + 8, 9, 0) + b"\0" * 8 + b"\0" * 16, UnsupportedFormatError),
    (b"Asura   " + struct.pack("<4sIII", b"LTXT", 9999, 3, 0), GameParserError),
    (make_ltxt(["Hello"])[:-20], GameParserError),
    (b"NotAsura" + b"\0" * 16, GameParserError),
])
def test_asura_rejects_unsafe_input(data, error):
    with pytest.raises(error):
        AsuraParser().extract(data)


def test_inject_refuses_a_different_source_file():
    meta = AsuraParser().extract(make_ltxt(["A"])).meta
    with pytest.raises(SourceMismatchError):
        AsuraParser().inject(make_ltxt(["B"]), {"LTXT": {"0000": "x"}}, meta)


def test_entry_keys_are_zero_padded():
    assert entry_keys(3) == ["0000", "0001", "0002"]
    assert entry_keys(12000)[-1] == "11999"


SAMPLES = Path(os.environ.get("L10N_GAME_SAMPLES", "/nonexistent"))


@pytest.mark.parametrize("name, variant, count", [("English.asr", "LTXT v3", 1688), ("SSpk_Eng.asr", "PTXT v5", 340)])
def test_real_game_files_roundtrip(name, variant, count):
    path = SAMPLES / name
    if not path.exists():
        pytest.skip(f"gerçek örnek yok: {path}")
    data = path.read_bytes()
    info = AsuraParser().inspect(data)
    assert info.variant == variant and info.entry_count == count
    _, unchanged = roundtrip(AsuraParser(), data)
    assert unchanged.data == data
    _, longer = roundtrip(AsuraParser(), data, lambda v: f"ÇĞİÖŞÜ {v} (çevrildi)" if v else v)
    rebuilt = parse_asura(longer.data)
    assert sum(len(c.entries) for c in rebuilt.text_chunks()) == count
    assert len(longer.data) == 8 + sum(c.size for c in rebuilt.chunks) + 16


# --------------------------------------------------------------------------- VDF

VDF_SOURCE = (
    '// Oyun metinleri\r\n'
    '"lang"\r\n'
    '{\r\n'
    '\t"Language"\t"English"\r\n'
    '\t"Tokens"\r\n'
    '\t{\r\n'
    '\t\t"Menu_Quit"\t\t"Quit Game"\r\n'
    '\t\t"Hint_Use"\t\t"Press %use% to \\"use\\" it\\n"\t[$WIN32]\r\n'
    '\t\t"Hint_Use"\t\t"Press A to \\"use\\" it\\n"\t[$X360]\r\n'
    '\t\t"[english]Menu_Quit"\t"Quit Game"\r\n'
    '\t\t"Menu_Alias"\t\t"#Menu_Quit"\r\n'
    '\t\t"Empty_One"\t\t""\r\n'
    '\t\t// ara yorum\r\n'
    '\t\t"Weapon_Key"\t\t"Fire in the hole!"\r\n'
    '\t}\r\n'
    '}\r\n'
)


def vdf_bytes(encoding: str) -> bytes:
    bom = {"utf-8": b"", "utf-16-le": b"\xff\xfe"}[encoding]
    return bom + VDF_SOURCE.encode(encoding)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le"])
def test_vdf_extract_only_translatable_tokens(encoding):
    extracted = VdfParser().extract(vdf_bytes(encoding))
    assert extracted.strings == {"Tokens": {
        "Menu_Quit": "Quit Game",
        "Hint_Use": 'Press %use% to \\"use\\" it\\n',
        "Hint_Use#2": 'Press A to \\"use\\" it\\n',
        "Weapon_Key": "Fire in the hole!",
    }}
    assert extracted.meta["skipped"] == {"english_reference": 1, "token_reference": 1, "empty": 1, "unquoted": 0}
    assert extracted.info.encoding == ("UTF-16 LE + BOM" if encoding == "utf-16-le" else "UTF-8")


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le"])
def test_vdf_unchanged_roundtrip_is_byte_identical(encoding):
    data = vdf_bytes(encoding)
    _, result = roundtrip(VdfParser(), data)
    assert result.data == data


def test_vdf_inject_preserves_layout_and_escapes_quotes():
    data = vdf_bytes("utf-16-le")
    translations = {"Tokens": {
        "Menu_Quit": 'Oyundan "Çık"',
        "Hint_Use": 'Kullanmak için %use% tuşuna \\"bas\\"\\n',
        "Weapon_Key": "Siper alın!",
    }}
    result = VdfParser().inject(data, translations, VdfParser().extract(data).meta)
    text = result.data[2:].decode("utf-16-le")
    assert result.data.startswith(b"\xff\xfe")
    assert '\t\t"Menu_Quit"\t\t"Oyundan \\"Çık\\""\r\n' in text
    assert '"Kullanmak için %use% tuşuna \\"bas\\"\\n"\t[$WIN32]' in text
    assert '"[english]Menu_Quit"\t"Quit Game"' in text and "// ara yorum" in text
    assert text.count("\r\n") == VDF_SOURCE.count("\r\n")
    assert VdfParser().extract(result.data).strings["Tokens"]["Weapon_Key"] == "Siper alın!"


def test_vdf_reports_unencodable_characters():
    data = VDF_SOURCE.encode("cp1252")
    with pytest.raises(GameParserError, match="Latin-1"):
        VdfParser().inject(data, {"Tokens": {"Menu_Quit": "Oyundan çıkış"}}, encoding="cp1252")
    result = VdfParser().inject(data, {"Tokens": {"Menu_Quit": "Oyundan çıkış"}}, encoding="cp1254")
    assert "çıkış".encode("cp1254") in result.data


def test_escape_value_never_breaks_the_string():
    assert escape_value('a "b" \\"c\\"') == 'a \\"b\\" \\"c\\"'
    assert escape_value("ends with \\") == "ends with \\\\"
    assert escape_value("two\nlines") == "two\\nlines"


@pytest.mark.parametrize("source", ['"lang" { "Tokens" { "A" "x" ', '"lang" { "Tokens" { "A" "unterminated } }'])
def test_vdf_rejects_broken_files(source):
    with pytest.raises(GameParserError):
        VdfParser().extract(source.encode())


def test_vdf_output_name():
    assert VdfParser().output_name(Path("valve_english.txt")) == "valve_turkish.txt"
    assert VdfParser().output_name(Path("70_loc.vdf")) == "70_loc_TR.vdf"


# --------------------------------------------------------------------------- algılama ve iş akışı


def test_auto_detector(tmp_path):
    cases = {
        "a.asr": make_ltxt(["x"]),
        "b.bin": make_ptxt(["x"]),
        "lang.txt": "\ufeff// c\n\"lang\" { }".encode("utf-8"),
        "c.vdf": b'"lang"\n{\n}',
        "d.arc": b"ARC\x00\x07\x00\x02\x00" + b"\0" * 32,
    }
    expected = {"a.asr": "asura", "b.bin": "asura", "lang.txt": "vdf", "c.vdf": "vdf", "d.arc": "capcom_arc"}
    for name, data in cases.items():
        (tmp_path / name).write_bytes(data)
        assert AutoDetector().detect(tmp_path / name).parser_id == expected[name], name
    (tmp_path / "e.dat").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    with pytest.raises(UnsupportedFormatError):
        AutoDetector().detect(tmp_path / "e.dat")


def test_studio_full_workflow_with_engine(tmp_path):
    game_file = tmp_path / "English.asr"
    game_file.write_bytes(make_ltxt(LTXT_TEXTS))
    studio = PackageStudio(tmp_path / "studio")

    info = studio.inspect(game_file)
    assert info.describe() == "Algılanan Format: Rebellion Asura Engine (LTXT v3 - UTF-16 LE) | Toplam Metin: 6 adet"

    package = studio.extract(game_file)
    assert package.json_path.name == "English.source.json" and is_studio_json(package.json_path)
    assert translated_path_for(package.json_path).name == "English.tr.json"

    # Çeviri motoru: oyun ID'li JSON için anahtar kuralları kapalı (GUI de böyle yükler)
    settings = Settings(provider=Provider.MOCK)
    pipeline = main.LocalizationPipeline(settings, Glossary.load(settings.glossary_path))
    loc = JsonLocalizationFile.load(package.json_path, text_filter=TranslatableFilter(use_key_rules=False))
    job = pipeline.prepare(loc, package.json_path.name)
    translations, stats = pipeline.run(job, MockTranslator(system_prompt="x"), progress=lambda _m: None)
    package.translated_path.write_text(loc.render(translations), encoding="utf-8")
    assert stats.fallback == 0

    result = studio.repack(game_file, package.translated_path)
    assert result.output_path.name == "English_TR.asr"
    texts = [e.text for e in parse_asura(result.output_path.read_bytes()).chunks[0].entries]
    assert texts[0] == "TR» \ue801Accept" and texts[3] == "TR» Exit - {\uec21}"
    assert texts[5] == "TR» Siper alın!"
    assert not result.warnings


def test_studio_refuses_to_overwrite_original(tmp_path):
    game_file = tmp_path / "English.asr"
    game_file.write_bytes(make_ltxt(["Hello"]))
    studio = PackageStudio(tmp_path)
    package = studio.extract(game_file)
    with pytest.raises(GameParserError):
        studio.repack(game_file, package.json_path, output_path=game_file)
