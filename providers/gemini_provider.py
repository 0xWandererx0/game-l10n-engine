"""
Google Gemini sağlayıcısı (google-genai SDK). Ücretsiz katmanla sıfır maliyetli testler için.

Kurulum : pip install google-genai
Anahtar : GEMINI_API_KEY (https://aistudio.google.com/apikey)
"""

from __future__ import annotations

import importlib
import re
from typing import Any, ClassVar

from providers.base import (
    BaseTranslator,
    ContentBlockedError,
    ErrorAction,
    FatalProviderError,
    OutputTruncatedError,
    ResponseFormatError,
    TokenUsage,
    output_schema,
)

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_FATAL_STATUS = {400, 401, 403, 404}
_BLOCKED_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION", "SPII", "LANGUAGE"}
_RETRY_DELAY_RE = re.compile(r"(\d+(?:\.\d+)?)s")
# Yedek: yapılandırılmış alan yoksa mesaj metni ("... Please retry in 17.43s.") okunur.
_RETRY_IN_MESSAGE_RE = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_RETRY_SAFETY_MARGIN = 3.0  # kota penceresinin kesin sıfırlanması için önerilen süreye eklenen pay
# 429 yanıtında bekleme süresi yoksa: ücretsiz katman sınırları dakikalık olduğu için kısa beklemek işe yaramaz.
_DEFAULT_RATE_LIMIT_WAIT = 20.0


def _transport_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [ConnectionError, TimeoutError]
    for module_name in ("httpx2", "httpx"):
        try:
            errors.append(importlib.import_module(module_name).TransportError)
        except (ImportError, AttributeError):
            continue
    return tuple(errors)


class GeminiTranslator(BaseTranslator):
    provider_name: ClassVar[str] = "gemini"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.3,
        thinking_budget: int | None = 0,
        max_output_tokens: int = 32768,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, system_prompt, **kwargs)
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:
            raise FatalProviderError("google-genai kurulu değil: pip install google-genai") from exc
        self._types = types
        self._errors = errors
        self._transport_errors = _transport_errors()
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens

        if client is not None:
            self._client = client
        else:
            if not api_key:
                raise FatalProviderError(
                    "Gemini API anahtarı yok. GEMINI_API_KEY ortam değişkenini ayarlayın "
                    "(ücretsiz anahtar: https://aistudio.google.com/apikey)."
                )
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.request_timeout * 1000),  # milisaniye
                    retry_options=types.HttpRetryOptions(attempts=1),  # tekrar denemeyi biz yönetiyoruz
                ),
            )
        self._config = self._build_config()

    def _build_config(self) -> Any:
        types = self._types
        # Oyun metinlerinde şiddet olağandır; yalnızca yüksek riskli içerik engellensin.
        safety = [
            types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH)
            for category in (
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            )
        ]
        thinking = (
            types.ThinkingConfig(thinking_budget=self.thinking_budget)
            if self.thinking_budget is not None else None
        )
        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=output_schema(strict=False),
            safety_settings=safety,
            thinking_config=thinking,
        )

    def _generate(self, system_prompt: str, user_message: str) -> tuple[str, TokenUsage]:
        response = self._client.models.generate_content(
            model=self.model, contents=user_message, config=self._config
        )

        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None) if feedback else None
        if block_reason:
            raise ContentBlockedError(f"istem engellendi: {_enum_name(block_reason)}")

        candidates = getattr(response, "candidates", None) or []
        finish = _enum_name(getattr(candidates[0], "finish_reason", None)) if candidates else None
        if finish == "MAX_TOKENS":
            raise OutputTruncatedError("çıktı max_output_tokens sınırında kesildi")
        if finish in _BLOCKED_FINISH_REASONS:
            raise ContentBlockedError(f"yanıt engellendi: {finish}")

        text = response.text
        if not text:
            raise ResponseFormatError(f"boş yanıt (finish_reason={finish})")
        return text, self._usage(response)

    @staticmethod
    def _usage(response: Any) -> TokenUsage:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return TokenUsage()
        cached = meta.cached_content_token_count or 0
        prompt = meta.prompt_token_count or 0
        output = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
        return TokenUsage(input_tokens=prompt - cached, output_tokens=output, cached_input_tokens=cached)

    def _classify_error(self, exc: Exception) -> ErrorAction:
        if isinstance(exc, self._errors.APIError):
            code = getattr(exc, "code", None) or 0
            if code in _RETRYABLE_STATUS or code >= 500:
                return ErrorAction.RETRY
            if code in _FATAL_STATUS:
                return ErrorAction.FATAL
            return ErrorAction.FAIL_BATCH
        if isinstance(exc, self._transport_errors):
            return ErrorAction.RETRY
        return super()._classify_error(exc)

    def _is_rate_limit(self, exc: Exception) -> bool:
        return isinstance(exc, self._errors.APIError) and getattr(exc, "code", None) == 429

    def _retry_after_seconds(self, exc: Exception) -> float | None:
        """429'da Gemini'nin önerdiği bekleme süresi + 3 sn pay.

        Önce yapılandırılmış alan (error.details[].RetryInfo.retryDelay = "17s"), yoksa mesaj
        metni ("Please retry in 17.43s"), o da yoksa 20 sn.
        """
        if not self._is_rate_limit(exc):
            return None
        payload = getattr(exc, "details", None)
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            for item in error.get("details", []) or []:
                if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
                    match = _RETRY_DELAY_RE.fullmatch(str(item.get("retryDelay", "")).strip())
                    if match:
                        return float(match.group(1)) + _RETRY_SAFETY_MARGIN
        message = " ".join(str(part) for part in (getattr(exc, "message", ""), error.get("message", "")
                                                   if isinstance(error, dict) else "", exc))
        match = _RETRY_IN_MESSAGE_RE.search(message)
        if match:
            return float(match.group(1)) + _RETRY_SAFETY_MARGIN
        return _DEFAULT_RATE_LIMIT_WAIT


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)
