import json

import pytest

from core.parser import (
    CsvLocalizationFile,
    JsonLocalizationFile,
    SkipReason,
    StructureMismatchError,
    TranslatableFilter,
    load_localization_file,
)

NESTED = {
    "ui": {"title": "Start Game", "icon": "Textures/UI/start.png", "scale": 1.5, "visible": True},
    "dialogues": [{"speaker_id": "SGT_MILLER", "text": "Move out!"}, {"text": "Roger."}],
    "a/b": {"c~d": "Weird keys survive"},
    "nothing": None,
    "url": "https://example.com",
}


@pytest.fixture()
def loc() -> JsonLocalizationFile:
    return JsonLocalizationFile.from_string(json.dumps(NESTED, indent=4))


def test_flatten_marks_only_real_text_as_translatable(loc):
    translatable = {e.uid: e.value for e in loc.translatable_entries()}
    assert translatable == {
        "/ui/title": "Start Game",
        "/dialogues/0/text": "Move out!",
        "/dialogues/1/text": "Roger.",
        "/a~1b/c~0d": "Weird keys survive",
    }
    reasons = {e.uid: e.skip_reason for e in loc.entries() if not e.translatable}
    assert reasons["/ui/icon"] is SkipReason.SKIP_KEY
    assert reasons["/ui/scale"] is SkipReason.NON_STRING
    assert reasons["/ui/visible"] is SkipReason.NON_STRING
    assert reasons["/nothing"] is SkipReason.NON_STRING
    assert reasons["/dialogues/0/speaker_id"] is SkipReason.SKIP_KEY


def test_unflatten_keeps_skeleton_and_types(loc):
    translations = {"/ui/title": "Oyunu Başlat", "/a~1b/c~0d": "Tuhaf anahtarlar"}
    rendered = loc.render(translations)
    loc.verify_rendered(rendered, translations)
    data = json.loads(rendered)
    assert data["ui"] == {"title": "Oyunu Başlat", "icon": "Textures/UI/start.png", "scale": 1.5, "visible": True}
    assert data["a/b"]["c~d"] == "Tuhaf anahtarlar"
    assert list(data) == list(NESTED)
    assert rendered.startswith('{\n    "ui"')  # orijinal girinti korunur


def test_verify_detects_structure_drift(loc):
    rendered = loc.render({})
    tampered = json.loads(rendered)
    tampered["dialogues"].pop()
    with pytest.raises(StructureMismatchError):
        loc.verify_rendered(json.dumps(tampered), {})


def test_unknown_or_non_string_targets_are_refused(loc):
    with pytest.raises(KeyError):
        loc.unflatten({"/does/not/exist": "x"})
    with pytest.raises(TypeError):
        loc.unflatten({"/ui/scale": "bir buçuk"})


def test_bom_is_preserved(tmp_path):
    path = tmp_path / "en.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"a": "Hello"}).encode())
    loc = load_localization_file(path)
    assert loc.render({"/a": "Merhaba"}).startswith("\ufeff")


@pytest.mark.parametrize("value, reason", [
    ("Yes/No", None),
    ("and/or", None),
    ("OK", None),
    ("Re-roll", None),
    ("Assets/UI/Icons/a.png", SkipReason.PATH),
    ("sfx_click.wav", SkipReason.PATH),
    ("res://scenes/main.tscn", SkipReason.URL),
    ("C:\\Games\\save.dat", SkipReason.PATH),
    ("item_sword_01", SkipReason.IDENTIFIER),
    ("com.studio.game", SkipReason.IDENTIFIER),
    ("itemSwordId", SkipReason.IDENTIFIER),
    ("#FF00AA", SkipReason.IDENTIFIER),
    ("player@example.com", SkipReason.URL),
    ("1.4.2", SkipReason.NO_LETTERS),
    ("   ", SkipReason.EMPTY),
])
def test_filter_classification(value, reason):
    assert TranslatableFilter().classify(("value",), value) is reason


def test_csv_roundtrip_adds_target_column():
    text = "key;en;notes\nui.start;Start Game;menu\nui.icon;Textures/a/b.png;asset\n"
    loc = CsvLocalizationFile(text)
    entries = {e.uid: e for e in loc.entries()}
    assert entries["ui.start"].translatable
    assert entries["ui.icon"].skip_reason is SkipReason.PATH
    translations = {"ui.start": "Oyunu Başlat"}
    rendered = loc.render(translations)
    loc.verify_rendered(rendered, translations)
    lines = rendered.splitlines()
    assert lines[0] == "key;en;notes;tr"
    assert lines[1] == "ui.start;Start Game;menu;Oyunu Başlat"
    assert lines[2].endswith(";Textures/a/b.png")  # çevrilmeyen satır kaynakla doldurulur
