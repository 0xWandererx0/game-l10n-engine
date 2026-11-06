"""Sağlayıcı fabrikası: config'deki tek ayarla model değiştirilir.

SDK'lar tembel (lazy) içe aktarılır; yalnızca Gemini kullanan biri `anthropic`
kurmak zorunda kalmaz.
"""

from __future__ import annotations

from config import Provider, Settings
from providers.base import BaseTranslator


def create_translator(settings: Settings, system_prompt: str, *, chaos_rate: float = 0.0) -> BaseTranslator:
    common = {
        "max_retries": settings.max_retries,
        "request_timeout": settings.request_timeout,
        "min_interval": settings.effective_request_interval,  # Gemini ücretsiz: 13 sn; mock: 0
    }

    if settings.provider is Provider.GEMINI:
        from providers.gemini_provider import GeminiTranslator

        return GeminiTranslator(
            settings.gemini_model, system_prompt,
            api_key=settings.gemini_api_key,
            temperature=settings.temperature,
            thinking_budget=settings.gemini_thinking_budget,
            max_output_tokens=settings.gemini_max_output_tokens,
            **common,
        )
    if settings.provider is Provider.CLAUDE:
        from providers.claude_provider import ClaudeTranslator

        return ClaudeTranslator(
            settings.claude_model, system_prompt,
            api_key=settings.anthropic_api_key,
            effort=settings.claude_effort,
            max_tokens=settings.claude_max_tokens,
            refusal_fallback=settings.claude_refusal_fallback,
            **common,
        )
    if settings.provider is Provider.OLLAMA:
        from providers.ollama_provider import OllamaTranslator

        return OllamaTranslator(
            settings.ollama_model, system_prompt,
            host=settings.ollama_host,
            temperature=settings.temperature,
            **common,
        )
    from providers.mock_provider import MockTranslator

    return MockTranslator("mock", system_prompt, chaos_rate=chaos_rate, **common)
