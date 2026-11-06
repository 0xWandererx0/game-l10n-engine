"""
Anthropic Claude sağlayıcısı (anthropic SDK). Final yama kalitesi için.

Kurulum : pip install anthropic
Anahtar : ANTHROPIC_API_KEY (ya da `ant auth login` profili)
Modeller: claude-sonnet-5-5 (final kalite), claude-haiku-5-5 (hızlı/ucuz taslak)

Notlar
* JSON çıktısı yapılandırılmış çıktı ile garanti edilir (output_config.format).
* Sistem promptu her istekte aynıdır; prompt önbelleği sayesinde ilk istekten
  sonra ~%90 daha ucuza okunur.
* Claude 5.x modelleri temperature kabul etmez; düşünme miktarı `effort` ile
  ayarlanır (low/medium/high).
* Destekleyen modellerde (Sonnet 5.5 vb.) güvenlik sınıflandırıcısı bir isteği
  reddederse sunucu tarafı yedek model (fallbacks="default") devreye girer.
"""

from __future__ import annotations

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

# Sunucu tarafı reddetme yedeğini destekleyen modeller ("default" biçimi).
_REFUSAL_FALLBACK_MODELS = frozenset({"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"})
_REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeTranslator(BaseTranslator):
    provider_name: ClassVar[str] = "claude"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        *,
        api_key: str | None = None,
        effort: str = "low",
        max_tokens: int = 16000,
        refusal_fallback: bool = True,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, system_prompt, **kwargs)
        try:
            import anthropic
        except ImportError as exc:
            raise FatalProviderError("anthropic kurulu değil: pip install anthropic") from exc
        self._anthropic = anthropic
        self.effort = effort
        self.max_tokens = max_tokens
        self.use_refusal_fallback = refusal_fallback and model in _REFUSAL_FALLBACK_MODELS
        # SDK'nın kendi tekrar denemesi kapalı: denemeleri BaseTranslator yönetir (çarpan etkisi olmasın).
        self._client = client or anthropic.Anthropic(
            api_key=api_key, max_retries=0, timeout=self.request_timeout
        )
        self._schema = output_schema(strict=True)

    def _generate(self, system_prompt: str, user_message: str) -> tuple[str, TokenUsage]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # Sabit sistem promptu önbelleğe alınır; değişken içerik kullanıcı mesajındadır.
            "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_message}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": self._schema},
            },
        }
        if self.use_refusal_fallback:
            response = self._client.beta.messages.create(
                **params, betas=[_REFUSAL_FALLBACK_BETA], fallbacks="default"
            )
        else:
            response = self._client.messages.create(**params)

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ContentBlockedError(f"model isteği reddetti (kategori: {category or 'belirtilmemiş'})")
        if response.stop_reason == "max_tokens":
            raise OutputTruncatedError("çıktı max_tokens sınırında kesildi")

        # Yanıt thinking/fallback blokları içerebilir; yalnızca metin blokları okunur.
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise ResponseFormatError(f"metin bloğu yok (stop_reason={response.stop_reason})")
        return text, self._usage(response)

    @staticmethod
    def _usage(response: Any) -> TokenUsage:
        usage = response.usage
        return TokenUsage(
            input_tokens=(usage.input_tokens or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0),
            output_tokens=usage.output_tokens or 0,
            cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    def _classify_error(self, exc: Exception) -> ErrorAction:
        a = self._anthropic
        if isinstance(exc, (a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError, a.BadRequestError)):
            return ErrorAction.FATAL  # yanlış anahtar/model/parametre: her blokta tekrarlanır
        retryable_extra = getattr(a, "RetryableError", a.APIConnectionError)  # SDK 2.x'te var
        if isinstance(exc, (a.RateLimitError, a.InternalServerError, a.APIConnectionError, retryable_extra)):
            return ErrorAction.RETRY
        if isinstance(exc, a.APIStatusError):
            if exc.status_code in (408, 409) or exc.status_code >= 500:
                return ErrorAction.RETRY
            return ErrorAction.FAIL_BATCH  # 413 vb.: blok bölünerek denenir
        return super()._classify_error(exc)

    def _retry_after_seconds(self, exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        header = response.headers.get("retry-after") if response is not None else None
        try:
            return float(header) if header is not None else None
        except ValueError:
            return None
