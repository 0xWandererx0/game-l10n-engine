"""Mega-batch ve ücretsiz katman (rate limit) davranışı."""

import json

import pytest

import main
from config import Provider, Settings
from core.context_batcher import ContextBatcher, TranslationUnit
from core.glossary import Glossary
from core.parser import JsonLocalizationFile, TranslatableFilter
from providers.base import FatalProviderError, SourceItem

genai = pytest.importorskip("google.genai")
httpx2 = pytest.importorskip("httpx2")
from google.genai import types  # noqa: E402

from providers import create_translator  # noqa: E402
from providers.gemini_provider import GeminiTranslator  # noqa: E402

ITEMS = [SourceItem("L1", "Fire in the hole!")]
OK = {"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(
    {"translations": [{"id": "L1", "text": "Siper alın!"}]})}]}, "finishReason": "STOP"}],
      "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}


def quota(retry_delay=None, message="Quota exceeded for metric: generate_content_free_tier_requests, limit: 5"):
    error = {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED"}
    if retry_delay:
        error["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
    return {"error": error}


def gemini(responses, **kwargs):
    queue, sleeps, bodies = list(responses), [], []

    def handler(request):
        bodies.append(json.loads(request.content))
        status, body = queue.pop(0)
        return httpx2.Response(status, json=body)

    client = genai.Client(api_key="k", http_options=types.HttpOptions(
        httpx_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        retry_options=types.HttpRetryOptions(attempts=1)))
    translator = GeminiTranslator("gemini-2.5-flash", "S", client=client, sleep=sleeps.append, **kwargs)
    return translator, sleeps, bodies


# --------------------------------------------------------------------------- ayarlar


def test_defaults_are_free_tier_friendly():
    settings = Settings()
    assert (settings.chunk_size, settings.max_batch_chars) == (350, 30000)
    assert settings.gemini_max_output_tokens >= 16384
    assert Settings(provider=Provider.GEMINI).effective_request_interval == 13.0  # 5 istek/dk
    assert Settings(provider=Provider.CLAUDE).effective_request_interval == 0.0
    assert Settings(provider=Provider.MOCK, request_interval=30).effective_request_interval == 0.0
    assert Settings(provider=Provider.GEMINI, request_interval=0).effective_request_interval == 0.0


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("L10N_REQUEST_INTERVAL", "20")
    monkeypatch.setenv("L10N_CHUNK_SIZE", "500")
    settings = Settings.from_env()
    assert settings.request_interval == 20.0 and settings.chunk_size == 500


def test_factory_wires_interval_and_output_limit():
    translator = create_translator(Settings(provider=Provider.GEMINI, gemini_api_key="k"), "S")
    assert translator.min_interval == 13.0
    assert translator._config.max_output_tokens == 32768
    mock = create_translator(Settings(provider=Provider.MOCK), "S")
    assert mock.min_interval == 0.0


def test_340_lines_go_in_a_single_request():
    """SSpk_Eng.asr büyüklüğünde (340 satır, ~7.300 karakter) bir dosya tek istekte gider."""
    texts = {f"{i:04d}": f"Enemy plane number {i}, take cover!" for i in range(340)}
    loc = JsonLocalizationFile({"PTXT:SNIPE_SPOKEN_ID": texts}, text_filter=TranslatableFilter(use_key_rules=False))
    settings = Settings(provider=Provider.GEMINI)
    pipeline = main.LocalizationPipeline(settings, Glossary.load(settings.glossary_path))
    job = pipeline.prepare(loc, "SSpk_Eng.source.json")
    assert [len(batch.units) for batch in job.batches] == [340]


def test_char_limit_still_splits_huge_texts():
    units = [TranslationUnit(f"/{i}", "x" * 400, "x" * 400, {}, str(i), "g") for i in range(350)]
    batches = ContextBatcher(chunk_size=350, max_batch_chars=30000).build(units)
    assert all(sum(len(u.masked) for u in b.units) <= 30000 for b in batches) and len(batches) == 5


# --------------------------------------------------------------------------- 429 / bekleme


@pytest.mark.parametrize("response, expected_wait", [
    (quota(retry_delay="17s"), 20.0),  # RetryInfo + 3 sn pay
    (quota(retry_delay="17.43s"), 20.43),
    (quota(message="Quota exceeded ... limit: 5. Please retry in 17.43s."), 20.43),  # yalnızca mesaj
    (quota(), 20.0),  # hiçbir bilgi yok: ücretsiz katman için 20 sn
    (quota(retry_delay="300s"), 120.0),  # uzun öneriler 120 sn ile sınırlı
])
def test_429_waits_what_google_asks(response, expected_wait):
    translator, sleeps, _ = gemini([(429, response), (200, OK)])
    assert translator.translate(ITEMS)["L1"] == "Siper alın!"
    assert sleeps == [pytest.approx(expected_wait)]


def test_requests_are_spaced_out():
    translator, sleeps, _ = gemini([(200, OK)] * 3, min_interval=13.0)
    for _ in range(3):
        translator.translate(ITEMS)
    assert [round(s) for s in sleeps] == [13, 13]  # ilk istek beklemez


def test_persistent_quota_stops_with_clear_message():
    translator, _, _ = gemini([(429, quota(retry_delay="17s"))] * 3)
    with pytest.raises(FatalProviderError, match="kotası doldu"):
        translator.translate(ITEMS)


def test_request_uses_large_output_limit():
    translator, _, bodies = gemini([(200, OK)], max_output_tokens=32768)
    translator.translate(ITEMS)
    assert bodies[0]["generationConfig"]["maxOutputTokens"] == 32768
