"""EngineService testleri: arayüz köprüsü pencere açmadan, yalnızca olay kuyruğu üzerinden sınanır."""

import json
import queue
import time

import pytest

pytest.importorskip("tkinter")
pytest.importorskip("customtkinter")

import gui  # noqa: E402
import main  # noqa: E402
from config import Provider, Settings  # noqa: E402
from providers.base import FatalProviderError  # noqa: E402
from providers.mock_provider import MockTranslator  # noqa: E402


def collect(events: queue.Queue, timeout: float = 15.0) -> list[tuple[str, object]]:
    """'idle' olayı gelene kadar kuyruğu okur."""
    received, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            kind, payload = events.get(timeout=0.1)
        except queue.Empty:
            continue
        received.append((kind, payload))
        if kind == "idle":
            return received
    pytest.fail(f"Motor zamanında bitmedi: {received}")


@pytest.fixture()
def sample(tmp_path):
    path = tmp_path / "en.json"
    path.write_text(json.dumps(main.SAMPLE_GAME_DATA), encoding="utf-8")
    return path


def settings() -> Settings:
    return Settings(provider=Provider.MOCK)


def test_analyze_emits_cost_report(sample):
    events: queue.Queue = queue.Queue()
    gui.EngineService(events).analyze(sample, settings())
    kinds = dict(collect(events))
    analysis = kinds["analysis"]
    assert analysis.report.stats.translatable == 28
    assert {e.model for e in analysis.report.estimates} >= {"gemini-2.5-flash", "claude-sonnet-5-5"}


def test_translate_streams_progress_and_writes_output(sample, tmp_path):
    events: queue.Queue = queue.Queue()
    small_batches = Settings(provider=Provider.MOCK, chunk_size=20, max_batch_chars=6000)
    gui.EngineService(events).translate(sample, tmp_path / "tr.json", small_batches)
    received = collect(events)
    progress = [payload for kind, payload in received if kind == "progress"]
    assert progress[0] == (0, 3) and progress[-1] == (3, 3)
    result = dict(received)["done"]
    assert result.translated == 28 and result.fallback == 0
    assert json.loads((tmp_path / "tr.json").read_text(encoding="utf-8"))["ui"]["main_menu"]["quit"].startswith("TR» ")
    assert (tmp_path / "tr.report.json").exists()


def test_rejected_lines_reach_the_log(sample, tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "create_translator",
                        lambda s, prompt, **kw: MockTranslator("mock", prompt, chaos_rate=0.6, seed=1))
    events: queue.Queue = queue.Queue()
    gui.EngineService(events).translate(sample, tmp_path / "tr.json", settings())
    logs = [message for kind, (level, message) in
            ((k, p) for k, p in collect(events) if k == "log")]
    assert any("reddedildi" in message for message in logs)


def test_cancel_stops_at_batch_boundary_without_output(sample, tmp_path, monkeypatch):
    service = gui.EngineService(queue.Queue())

    class CancelAfterFirst(MockTranslator):
        def translate(self, *args, **kwargs):
            result = super().translate(*args, **kwargs)
            service.cancel()  # ilk batch sırasında kullanıcı "Durdur"a basmış gibi
            return result

    monkeypatch.setattr(gui, "create_translator", lambda s, prompt, **kw: CancelAfterFirst("mock", prompt))
    service.translate(sample, tmp_path / "tr.json", settings())
    kinds = [kind for kind, _ in collect(service.events)]
    assert "cancelled" in kinds and "done" not in kinds
    assert not (tmp_path / "tr.json").exists()


def test_fatal_provider_error_is_reported_not_raised(sample, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise FatalProviderError("Gemini API anahtarı yok")

    monkeypatch.setattr(gui, "create_translator", broken)
    events: queue.Queue = queue.Queue()
    gui.EngineService(events).translate(sample, tmp_path / "tr.json", settings())
    received = dict(collect(events))
    assert received["error"] == "Gemini API anahtarı yok"
    assert not (tmp_path / "tr.json").exists()


def test_only_one_job_at_a_time(sample, monkeypatch):
    class Slow(MockTranslator):
        def translate(self, *args, **kwargs):
            time.sleep(0.3)
            return super().translate(*args, **kwargs)

    monkeypatch.setattr(gui, "create_translator", lambda s, prompt, **kw: Slow("mock", prompt))
    service = gui.EngineService(queue.Queue())
    service.translate(sample, sample.with_name("tr.json"), settings())
    with pytest.raises(RuntimeError):
        service.analyze(sample, settings())
    service.cancel()
    collect(service.events)


# --------------------------------------------------------------------------- Paket Stüdyosu köprüsü

VDF_SAMPLE = '"lang"\n{\n\t"Language" "English"\n\t"Tokens"\n\t{\n\t\t"Weapon_Key" "Fire in the hole!"\n\t\t"Menu_Quit" "Quit"\n\t}\n}\n'


def test_studio_extract_then_engine_keeps_game_keys(tmp_path):
    game_file = tmp_path / "valve_english.txt"
    game_file.write_text(VDF_SAMPLE, encoding="utf-8")
    service = gui.EngineService(queue.Queue(), studio=gui.PackageStudio(tmp_path / "studio"))

    service.extract_package(game_file, None, None)
    package = dict(collect(service.events))["studio_extracted"]
    assert package.json_path.name == "valve_english.source.json"
    assert gui.is_studio_json(package.json_path)

    # "Weapon_Key" motorun genel kurallarında "_key" ile biten sistem anahtarı sayılırdı;
    # stüdyo JSON'unda bu kural kapalı olmalı.
    service.analyze(package.json_path, settings())
    analysis = dict(collect(service.events))["analysis"]
    assert analysis.report.stats.translatable == 2


def test_studio_repack_event(tmp_path):
    game_file = tmp_path / "valve_english.txt"
    game_file.write_text(VDF_SAMPLE, encoding="utf-8")
    service = gui.EngineService(queue.Queue(), studio=gui.PackageStudio(tmp_path / "studio"))
    service.extract_package(game_file, None, None)
    package = dict(collect(service.events))["studio_extracted"]
    package.translated_path.write_text(json.dumps({"Tokens": {"Weapon_Key": "Siper alın!", "Menu_Quit": "Çık"}},
                                                  ensure_ascii=False), encoding="utf-8")

    service.repack_package(game_file, package.translated_path, latin1_fallback=True, format_id=None, encoding=None)
    result = dict(collect(service.events))["studio_repacked"]
    assert result.output_path.name == "valve_turkish.txt" and result.changed == 2
    assert '"Weapon_Key" "Siper alin!"' in result.output_path.read_text(encoding="utf-8")


def test_studio_errors_are_reported_not_raised(tmp_path):
    bogus = tmp_path / "image.asr"
    bogus.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    service = gui.EngineService(queue.Queue(), studio=gui.PackageStudio(tmp_path / "studio"))
    service.inspect_package(bogus, None, None)
    received = dict(collect(service.events))
    # Uzantı .asr olsa da magic bytes PNG: algılayıcı uzantıya güvenmez, elle seçim ister.
    assert "studio_info" not in received and "tanınamadı" in received["error"]
