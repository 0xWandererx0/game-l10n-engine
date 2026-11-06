"""Sağlayıcı testleri. Claude ve Gemini gerçek SDK'larla, sahte HTTP katmanı üzerinden sınanır:
istek gövdesi gerçekten serileştirilir, yanıt gerçekten ayrıştırılır, ağa çıkılmaz."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core.glossary import GlossaryTerm
from providers.base import (
    BaseTranslator,
    ContextItem,
    FatalProviderError,
    ResponseFormatError,
    SourceItem,
    TokenUsage,
    TranslationFailedError,
    build_system_prompt,
    build_user_message,
    parse_translation_payload,
)

ITEMS = [SourceItem("L1", "Fire in the hole!", "briefing/3/text"), SourceItem("L2", "Press [TAG_0]", "hud/interact")]
ANSWER = {"translations": [{"id": "L1", "text": "Siper alın!"}, {"id": "L2", "text": "[TAG_0] tuşuna bas"}]}
NO_SLEEP = {"sleep": lambda _seconds: None}


# --------------------------------------------------------------------------- ortak


@pytest.mark.parametrize("raw", [
    json.dumps(ANSWER),
    "```json\n" + json.dumps(ANSWER) + "\n```",
    "Sure! Here you go:\n" + json.dumps(ANSWER) + "\nHope this helps.",
    json.dumps(ANSWER["translations"]),
])
def test_parse_tolerates_wrappers(raw):
    assert parse_translation_payload(raw) == {"L1": "Siper alın!", "L2": "[TAG_0] tuşuna bas"}


def test_parse_drops_duplicates_and_malformed_entries():
    raw = json.dumps({"translations": [
        {"id": "L1", "text": "a"}, {"id": "L1", "text": "b"}, {"id": "L2"}, "junk", {"id": 3, "text": "c"},
    ]})
    assert parse_translation_payload(raw) == {"3": "c"}


@pytest.mark.parametrize("raw", ["", "no json here", '{"translations": "nope"}', '{"translations": [1, 2]}'])
def test_parse_rejects_garbage(raw):
    with pytest.raises(ResponseFormatError):
        parse_translation_payload(raw)


def test_user_message_layout():
    message = build_user_message(
        ITEMS, [ContextItem("Listen up!", "Dinleyin!")], [GlossaryTerm("Fire in the hole", "Siper alın!", "note")],
        {"L2": ["missing [TAG_0]"], "L9": ["stale"]},
    )
    payload = json.loads(message[message.index("{"):])
    assert list(payload) == ["glossary", "context", "feedback", "items"]
    assert payload["feedback"] == [{"id": "L2", "problems": ["missing [TAG_0]"]}]
    assert payload["items"][1] == {"id": "L2", "key": "hud/interact", "text": "Press [TAG_0]"}


def test_system_prompt_is_complete():
    prompt = build_system_prompt("English", "Turkish", ["Oyuncuya 'sen' diye hitap et."])
    assert "$" not in prompt
    assert "[TAG_0]" in prompt and "Delikteki ateş" in prompt
    assert "## Turkish rules for this game\n- Oyuncuya 'sen' diye hitap et." in prompt


class ScriptedTranslator(BaseTranslator):
    provider_name = "scripted"

    def __init__(self, script):
        super().__init__("m", "s", max_retries=3, **NO_SLEEP)
        self.script = list(script)

    def _generate(self, system_prompt, user_message):
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step, TokenUsage(10, 5)


def test_transient_errors_are_retried():
    translator = ScriptedTranslator([ConnectionError("reset"), "not json", json.dumps(ANSWER)])
    assert translator.translate(ITEMS)["L1"] == "Siper alın!"
    assert translator.request_count == 3 and translator.usage.input_tokens == 20


def test_retries_are_bounded():
    translator = ScriptedTranslator([ConnectionError("1"), ConnectionError("2"), ConnectionError("3")])
    with pytest.raises(TranslationFailedError):
        translator.translate(ITEMS)


def test_unknown_errors_are_fatal():
    with pytest.raises(FatalProviderError):
        ScriptedTranslator([KeyError("bug")]).translate(ITEMS)


# --------------------------------------------------------------------------- Claude

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")
from providers.claude_provider import ClaudeTranslator  # noqa: E402


def claude_message(text=None, stop_reason="end_turn", **extra):
    content = [{"type": "text", "text": text}] if text is not None else []
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 40,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 900},
        **extra,
    }


def make_claude(responses, model="claude-sonnet-5-5", **kwargs):
    requests = []
    queue = list(responses)

    def handler(request):
        requests.append(request)
        status, body = queue.pop(0)
        return httpx2.Response(status, json=body, headers={"retry-after": "0"})

    client = anthropic.Anthropic(
        api_key="test-key", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeTranslator(model, "SYSTEM PROMPT", client=client, **NO_SLEEP, **kwargs), requests


def test_claude_request_shape_and_parsing():
    translator, requests = make_claude([(200, claude_message(json.dumps(ANSWER)))])
    assert translator.translate(ITEMS) == {"L1": "Siper alın!", "L2": "[TAG_0] tuşuna bas"}
    body = json.loads(requests[0].content)
    assert body["model"] == "claude-sonnet-5-5"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in body
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in requests[0].headers["anthropic-beta"]
    assert translator.usage.cached_input_tokens == 900


def test_claude_haiku_uses_plain_endpoint_without_fallbacks():
    translator, requests = make_claude([(200, claude_message(json.dumps(ANSWER)))], model="claude-haiku-5-5")
    translator.translate(ITEMS)
    body = json.loads(requests[0].content)
    assert "fallbacks" not in body and "anthropic-beta" not in requests[0].headers


def test_claude_refusal_fails_the_batch_without_retry():
    refusal = claude_message(stop_reason="refusal", stop_details={"type": "refusal", "category": "general_harms"})
    translator, requests = make_claude([(200, refusal)])
    with pytest.raises(TranslationFailedError, match="general_harms"):
        translator.translate(ITEMS)
    assert len(requests) == 1


def test_claude_overload_is_retried():
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    translator, requests = make_claude([(529, error), (500, error), (200, claude_message(json.dumps(ANSWER)))])
    assert translator.translate(ITEMS)["L1"] == "Siper alın!"
    assert len(requests) == 3


def test_claude_bad_key_is_fatal():
    error = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
    translator, _ = make_claude([(401, error)])
    with pytest.raises(FatalProviderError):
        translator.translate(ITEMS)


# --------------------------------------------------------------------------- Gemini

genai = pytest.importorskip("google.genai")
from google.genai import types  # noqa: E402

from providers.gemini_provider import GeminiTranslator  # noqa: E402


def gemini_body(text, finish="STOP"):
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 50, "totalTokenCount": 350},
    }


def make_gemini(responses):
    requests = []
    queue = list(responses)

    def handler(request):
        requests.append(request)
        status, body = queue.pop(0)
        return httpx2.Response(status, json=body)

    client = genai.Client(api_key="test-key", http_options=types.HttpOptions(
        httpx_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        retry_options=types.HttpRetryOptions(attempts=1),
    ))
    return GeminiTranslator("gemini-2.5-flash", "SYSTEM PROMPT", client=client, **NO_SLEEP), requests


def test_gemini_request_shape_and_parsing():
    translator, requests = make_gemini([(200, gemini_body(json.dumps(ANSWER)))])
    assert translator.translate(ITEMS)["L2"] == "[TAG_0] tuşuna bas"
    body = json.loads(requests[0].content)
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["required"] == ["translations"]
    assert list(config["thinkingConfig"].values()) == [0]  # SDK snake_case serileştirir; API ikisini de kabul eder
    assert body["systemInstruction"]["parts"][0]["text"] == "SYSTEM PROMPT"
    assert {s["threshold"] for s in body["safetySettings"]} == {"BLOCK_ONLY_HIGH"}
    assert translator.usage.input_tokens == 300 and translator.usage.output_tokens == 50


def test_gemini_rate_limit_is_retried_and_bad_request_is_fatal():
    quota = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}
    translator, requests = make_gemini([(429, quota), (200, gemini_body(json.dumps(ANSWER)))])
    assert translator.translate(ITEMS)["L1"] == "Siper alın!"
    assert len(requests) == 2

    invalid = {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}}
    translator, _ = make_gemini([(400, invalid)])
    with pytest.raises(FatalProviderError):
        translator.translate(ITEMS)


@pytest.mark.parametrize("finish", ["MAX_TOKENS", "SAFETY"])
def test_gemini_truncation_and_safety_fail_the_batch(finish):
    translator, requests = make_gemini([(200, gemini_body('{"translations": [', finish))])
    with pytest.raises(TranslationFailedError):
        translator.translate(ITEMS)
    assert len(requests) == 1


def test_gemini_requires_api_key():
    with pytest.raises(FatalProviderError, match="GEMINI_API_KEY"):
        GeminiTranslator("gemini-2.5-flash", "s", api_key=None)


# --------------------------------------------------------------------------- Ollama

from providers.ollama_provider import OllamaTranslator  # noqa: E402


@pytest.fixture()
def ollama_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            if body["model"] == "missing":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'{"error":"model not found"}')
                return
            reply = {"message": {"role": "assistant", "content": json.dumps(ANSWER)},
                     "done_reason": "stop", "prompt_eval_count": 210, "eval_count": 33}
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", received
    server.shutdown()


def test_ollama_roundtrip(ollama_server):
    host, received = ollama_server
    translator = OllamaTranslator("gemma3:12b", "SYSTEM", host=host, **NO_SLEEP)
    assert translator.translate(ITEMS)["L1"] == "Siper alın!"
    assert received[0]["format"]["required"] == ["translations"]
    assert received[0]["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert translator.usage.output_tokens == 33


def test_ollama_missing_model_and_offline_server_are_fatal(ollama_server):
    host, _ = ollama_server
    with pytest.raises(FatalProviderError, match="ollama pull"):
        OllamaTranslator("missing", "s", host=host, **NO_SLEEP).translate(ITEMS)
    with pytest.raises(FatalProviderError, match="ollama serve"):
        OllamaTranslator("m", "s", host="http://127.0.0.1:9", **NO_SLEEP).translate(ITEMS)
