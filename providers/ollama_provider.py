"""
Yerel Ollama sağlayıcısı: internet ve API ücreti olmadan çeviri.

Kurulum : https://ollama.com -> `ollama pull gemma3:12b` -> `ollama serve`
Ek Python bağımlılığı yoktur (standart kütüphane HTTP istemcisi).
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any, ClassVar
from urllib.parse import urlparse

from providers.base import (
    BaseTranslator,
    FatalProviderError,
    OutputTruncatedError,
    ResponseFormatError,
    TokenUsage,
    TransientProviderError,
    output_schema,
)


class OllamaTranslator(BaseTranslator):
    provider_name: ClassVar[str] = "ollama"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        *,
        host: str = "http://localhost:11434",
        temperature: float = 0.3,
        num_ctx: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, system_prompt, **kwargs)
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        # Yerel sunucuya giden istek kurumsal proxy'den geçmesin.
        local = urlparse(self.host).hostname in {"localhost", "127.0.0.1", "::1"}
        handlers = [urllib.request.ProxyHandler({})] if local else []
        self._opener = urllib.request.build_opener(*handlers)

    def _generate(self, system_prompt: str, user_message: str) -> tuple[str, TokenUsage]:
        body = {
            "model": self.model,
            "stream": False,
            "format": output_schema(strict=True),
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code == 404:
                raise FatalProviderError(
                    f"Ollama modeli bulunamadı: {self.model}. Önce `ollama pull {self.model}` çalıştırın."
                ) from exc
            if exc.code == 429 or exc.code >= 500:
                raise TransientProviderError(f"Ollama HTTP {exc.code}: {detail}") from exc
            raise FatalProviderError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise TransientProviderError("Ollama zaman aşımı") from exc
            raise FatalProviderError(
                f"Ollama'ya bağlanılamadı ({self.host}): {exc.reason}. `ollama serve` çalışıyor mu?"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise TransientProviderError("Ollama zaman aşımı") from exc
        except json.JSONDecodeError as exc:
            raise ResponseFormatError("Ollama geçersiz HTTP yanıtı döndürdü") from exc

        if data.get("done_reason") == "length":
            raise OutputTruncatedError("çıktı bağlam/çıktı sınırında kesildi (num_ctx'i artırın)")
        text = (data.get("message") or {}).get("content", "")
        usage = TokenUsage(input_tokens=data.get("prompt_eval_count", 0), output_tokens=data.get("eval_count", 0))
        return text, usage
