import json

import pytest

import main
from config import Provider, Settings
from core.context_batcher import ContextBatcher, TranslationUnit
from core.glossary import Glossary, GlossaryTerm
from core.parser import JsonLocalizationFile
from core.regex_shield import TAG_RE, RegexShield
from core.sanity_checker import SanityChecker
from providers.base import ContentBlockedError, FatalProviderError, TokenUsage
from providers.mock_provider import MockTranslator


def make_pipeline(**overrides) -> main.LocalizationPipeline:
    settings = Settings(provider=Provider.MOCK, **overrides)
    return main.LocalizationPipeline(settings, Glossary.load(settings.glossary_path))


def run_sample(translator, **overrides):
    pipeline = make_pipeline(**overrides)
    loc = JsonLocalizationFile(main.SAMPLE_GAME_DATA)
    job = pipeline.prepare(loc, "sample")
    translations, stats = pipeline.run(job, translator, progress=lambda _msg: None)
    rendered = loc.render(translations)
    loc.verify_rendered(rendered, translations)
    return job, translations, stats, json.loads(rendered)


def assert_codes_preserved(job, translations):
    shield = RegexShield()
    for unit in job.units:
        assert shield.signature(translations[unit.uid]) == shield.signature(unit.source), unit.uid


def test_mock_end_to_end_translates_everything():
    job, translations, stats, data = run_sample(MockTranslator(system_prompt="x"))
    assert stats.translated == len(job.units) and stats.fallback == 0
    assert data["dialogues"]["chapter_1"]["briefing"][3]["text"] == "TR» Siper alın!"
    assert data["ui"]["hud"]["score_label"].endswith(": ")  # kenar boşluğu korunur
    assert data["items"][0]["icon"] == "Textures/Icons/rifle_01.png"
    assert data["meta"] == main.SAMPLE_GAME_DATA["meta"]
    assert_codes_preserved(job, translations)


@pytest.mark.parametrize("seed", range(5))
def test_chaotic_model_never_corrupts_output(seed):
    translator = MockTranslator(system_prompt="x", chaos_rate=0.6, seed=seed)
    job, translations, stats, _ = run_sample(translator)
    assert stats.translated + stats.fallback == len(job.units)
    assert_codes_preserved(job, translations)


class TagDroppingTranslator(MockTranslator):
    """Belirli bir satırda her seferinde etiketi siler."""

    def _generate(self, system_prompt, user_message):
        raw, usage = super()._generate(system_prompt, user_message)
        payload = json.loads(raw)
        for entry in payload["translations"]:
            if "You reached level" in entry["text"]:
                entry["text"] = TAG_RE.sub("", entry["text"])
        return json.dumps(payload, ensure_ascii=False), usage


def test_persistently_broken_line_falls_back_to_source():
    job, translations, stats, data = run_sample(TagDroppingTranslator(system_prompt="x"))
    assert stats.fallback == 1
    assert data["ui"]["notifications"]["level_up"] == main.SAMPLE_GAME_DATA["ui"]["notifications"]["level_up"]
    assert "/ui/notifications/level_up" in stats.failures
    assert stats.rerequested_lines == 2  # yalnızca hatalı satır, iki kez yeniden istendi


class BlockingTranslator(MockTranslator):
    """'Fire in the hole' içeren her bloğu reddeder (güvenlik filtresi benzetimi)."""

    def _generate(self, system_prompt, user_message):
        if "Fire in the hole" in user_message.split('"items"', 1)[1]:
            raise ContentBlockedError("blocked")
        return super()._generate(system_prompt, user_message)


def test_blocked_batch_is_bisected_and_only_offender_falls_back():
    job, translations, stats, data = run_sample(BlockingTranslator(system_prompt="x"))
    assert stats.bisections >= 1
    assert stats.fallback == 1
    assert data["dialogues"]["chapter_1"]["briefing"][3]["text"] == "Fire in the hole!"
    assert data["dialogues"]["chapter_1"]["briefing"][2]["text"].startswith("TR» ")


class FatalTranslator(MockTranslator):
    def _generate(self, system_prompt, user_message):
        raise FatalProviderError("invalid api key")


def test_fatal_error_aborts_run():
    with pytest.raises(FatalProviderError):
        run_sample(FatalTranslator(system_prompt="x"))


def test_context_lines_come_from_same_group_and_carry_translations():
    pipeline = make_pipeline(chunk_size=3, context_size=2)
    job = pipeline.prepare(JsonLocalizationFile(main.SAMPLE_GAME_DATA), "sample")
    seen_contexts = []

    class Spy(MockTranslator):
        def translate(self, items, context=(), glossary=(), feedback=None):
            seen_contexts.append(list(context))
            return super().translate(items, context, glossary, feedback)

    pipeline.run(job, Spy(system_prompt="x"), progress=lambda _msg: None)
    dialogue_batches = [b for b in job.batches if b.units[0].group == "dialogues/chapter_1/briefing"]
    later = dialogue_batches[1]
    assert [u.group for u in later.context] == ["dialogues/chapter_1/briefing"] * 2
    context_for_later = seen_contexts[job.batches.index(later)]
    assert all(item.translation and item.translation.startswith("TR» ") for item in context_for_later)


def test_batcher_respects_chunk_size_and_char_limit():
    units = [TranslationUnit(f"/{i}", "x" * 50, "x" * 50, {}, str(i), "g") for i in range(45)]
    batches = ContextBatcher(chunk_size=20, context_size=3, max_batch_chars=10_000).build(units)
    assert [len(b.units) for b in batches] == [20, 20, 5]
    assert [u.uid for u in batches[1].context] == ["/17", "/18", "/19"]
    batches = ContextBatcher(chunk_size=20, max_batch_chars=120).build(units)
    assert all(len(b.units) <= 2 for b in batches)


def test_glossary_matches_inflections_but_not_lookalikes():
    glossary = Glossary([GlossaryTerm("Nerf", "zayıflatma"), GlossaryTerm("Mana", "mana")], ["GG"])
    assert [t.source for t in glossary.match(["The rifle was nerfed. gg"])] == ["Nerf"]
    assert glossary.match(["Manage your inventory"]) == []
    assert [t.source for t in glossary.match(["GG WP"])] == ["GG"]


def test_sanity_checker_flags_drift_and_bloat():
    shield = RegexShield()
    masked, tags = shield.mask("Press {0} to continue the mission now")
    unit = TranslationUnit("/a", "", masked, tags, "a", "g")
    verdict = SanityChecker(shield).check(
        {"L1": unit, "L2": unit, "L3": unit},
        {"L1": "Görevi sürdürmek için [TAG_0] tuşuna bas", "L3": masked, "L9": "?"},
    )
    assert verdict.accepted == {"L1": "Görevi sürdürmek için [TAG_0] tuşuna bas"}
    assert "missing from your response" in verdict.rejected["L2"][0]
    assert verdict.rejected["L3"] == ["text was returned untranslated"]
    assert verdict.unexpected_ids == ["L9"]
    bloated = "[TAG_0] " + "çok uzun açıklama " * 20
    assert any("far longer" in p for p in SanityChecker(shield).check_item(unit, bloated))


def test_cli_dry_run_makes_no_requests(tmp_path, capsys, monkeypatch):
    source = tmp_path / "en.json"
    source.write_text(json.dumps(main.SAMPLE_GAME_DATA), encoding="utf-8")
    monkeypatch.setattr(main, "create_translator", lambda *a, **k: pytest.fail("API çağrısı yapılmamalı"))
    assert main.main([str(source), "--dry-run", "--provider", "claude"]) == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out and "claude-sonnet-5-5 *" in out
    assert not (tmp_path / "tr.json").exists()


def test_cli_full_run_with_mock(tmp_path, capsys):
    source = tmp_path / "en.json"
    source.write_text(json.dumps(main.SAMPLE_GAME_DATA), encoding="utf-8")
    assert main.main([str(source), "--provider", "mock", "--yes", "--preview", "0"]) == 0
    data = json.loads((tmp_path / "tr.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "tr.report.json").read_text(encoding="utf-8"))
    assert data["ui"]["main_menu"]["new_game"] == "TR» New Game"
    assert report["translated"] == 28 and report["failures"] == []
    assert TokenUsage(**report["usage"]).input_tokens > 0
