import pytest

from core.regex_shield import RegexShield

shield = RegexShield()

ROUNDTRIP_CASES = [
    "Hello {player}!\nPress [E]",
    "<color=#00FF00>LEVEL UP!</color>\nYou reached level {level}.",
    "Objective:\n<i>{objective_name}</i>",
    "Ammo: %d / %d",
    "Crit chance %d%% and %.2f multiplier",
    "{count, plural, one {# item} other {# items}}",
    "Ability on cooldown ({0:F1}s)",
    "Line1\\nLine2 \\\"quoted\\\" &amp; §aGreen $PLAYER$ ${x} %NAME% %(n)s %1$s",
    "<RichText.Bold>Hi</> [color=red]x[/color] [url=https://a.com/b c]link[/url]",
    "{{mustache}} and {} and {:.1f}",
    "Tab\there\r\nWindows newline",
    "",
]


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_mask_unmask_roundtrip(text):
    masked, tags = shield.mask(text)
    assert shield.unmask(masked, tags) == text
    assert shield.validate(masked, masked, tags)


def test_mask_replaces_codes_with_numbered_tags():
    masked, tags = shield.mask("Hello {player}!\nPress [E]")
    assert masked == "Hello [TAG_0]![TAG_1]Press [TAG_2]"
    assert tags == {"[TAG_0]": "{player}", "[TAG_1]": "\n", "[TAG_2]": "[E]"}


@pytest.mark.parametrize("text", ["50% chance", "Yes/No", "a < b and c > d", "Costs $100", "100% sure"])
def test_plain_text_is_not_masked(text):
    masked, tags = shield.mask(text)
    assert masked == text and tags == {}


def test_icu_message_is_frozen_whole():
    masked, tags = shield.mask("{count, plural, one {# item} other {# items}}")
    assert masked == "[TAG_0]"
    assert not RegexShield.has_translatable_text(masked)


def test_unclosed_brace_does_not_hide_later_placeholder():
    masked, tags = shield.mask("Use { carefully, {0} times")
    assert "{0}" in tags.values()


def test_placeholder_is_not_merged_with_line_break():
    masked, tags = shield.mask("Welcome {0}\nPress start")
    assert masked == "Welcome [TAG_0][TAG_1]Press start"


def test_adjacent_markup_is_merged():
    masked, tags = shield.mask("Hold <color=#FFD700>[R]</color> to reload")
    assert masked == "Hold [TAG_0] to reload"


def test_placeholder_reorder_is_allowed():
    masked, tags = shield.mask("Hello {name}, you have {count} coins")
    assert shield.validate(masked, "[TAG_1] altının var, [TAG_0]", tags)


@pytest.mark.parametrize("translation, expected_problem", [
    ("Merhaba [TAG_0]!", "missing [TAG_1]"),
    ("[TAG_0] [TAG_0] [TAG_1]", "[TAG_0] appears 2 times"),
    ("[TAG_0] [TAG_1] [TAG_2]", "invented tag"),
    ("[TAG 0] [TAG_1]", "malformed"),
    ("[tag_0] [TAG_1]", "malformed"),
    ("TAG_0 [TAG_1]", "malformed"),
    ("[TAG_0] {oyuncu} [TAG_1]", "code-like text"),
    ("[TAG_0] <b>[TAG_1]</b>", "code-like text"),
    ("[TAG_0]\n[TAG_1]", "code-like text"),
    ("[TAG_0] } [TAG_1]", "unexpected '}'"),
])
def test_corrupted_translations_are_rejected(translation, expected_problem):
    masked, tags = shield.mask("Hello {name}, you have {count} coins")
    problems = shield.diagnose(masked, translation, tags)
    assert any(expected_problem in p for p in problems), problems


def test_sequential_printf_order_is_enforced():
    masked, tags = shield.mask("%s has %d coins")
    assert shield.validate(masked, "[TAG_0] adlı oyuncunun [TAG_1] altını var", tags)
    problems = shield.diagnose(masked, "[TAG_1] altın: [TAG_0]", tags)
    assert any("sequential" in p for p in problems)


def test_positional_printf_may_be_reordered():
    masked, tags = shield.mask("%1$s has %2$d coins")
    assert shield.validate(masked, "[TAG_1] altın: [TAG_0]", tags)


def test_markup_pair_must_open_before_it_closes():
    masked, tags = shield.mask("<b>Gold</b> and <i>Silver</i>")
    assert shield.validate(masked, "[TAG_2]Gümüş[TAG_3] ve [TAG_0]Altın[TAG_1]", tags)
    problems = shield.diagnose(masked, "[TAG_1]Altın[TAG_0] ve [TAG_2]Gümüş[TAG_3]", tags)
    assert any("must stay a pair" in p for p in problems)


def test_crossed_markup_is_rejected():
    masked, tags = shield.mask("<b>bold <i>both</i> end</b>")
    assert masked == "[TAG_0]bold [TAG_1]both[TAG_2] end[TAG_3]"
    assert shield.validate(masked, "[TAG_0]kalın [TAG_1]ikisi[TAG_2] son[TAG_3]", tags)
    problems = shield.diagnose(masked, "[TAG_0]kalın [TAG_1]ikisi[TAG_3] son[TAG_2]", tags)
    assert any("must stay a pair" in p for p in problems)


def test_turkish_percent_style_allowed_without_printf():
    masked, tags = shield.mask("Restores 50% health. Buffs {stat}.")
    assert shield.validate(masked, "%50 can yeniler. [TAG_0] güçlendirilir.", tags)


def test_free_percent_rejected_in_printf_string():
    # "%[TAG_0]" geri açılınca "%%d" olur: printf argümanı tüketmez, sonraki argümanlar kayar.
    masked, tags = shield.mask("Boosts damage by %d percent")
    problems = shield.diagnose(masked, "Hasarı %[TAG_0] artırır", tags)
    assert any("unexpected '%'" in p for p in problems)
    assert shield.validate(masked, "Hasarı yüzde [TAG_0] artırır", tags)


def test_bracket_words_can_be_left_translatable():
    relaxed = RegexShield(protect_bracket_words=False)
    masked, tags = relaxed.mask("[sighs] Press [E]. [b]Bold[/b]")
    assert masked == "[sighs] Press [TAG_0]. [TAG_1]Bold[TAG_2]"


def test_signature_counts_codes():
    assert shield.signature("Hi {0} {0} <b>x</b>") == shield.signature("<b>y</b> {0} - {0}")
    assert shield.signature("Hi {0}") != shield.signature("Hi {1}")
