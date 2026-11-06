"""
Sahte çevirmen: API anahtarı ve internet olmadan tüm boru hattını uçtan uca test eder.

Gerçek bir sağlayıcı gibi kullanıcı mesajını okur ve JSON döndürür. Sözlük
terimlerini uygular, metnin başına "TR» " ekler, [TAG_n] etiketlerini korur.

chaos_rate > 0 ise kasıtlı model hataları üretir (satır atlama, etiket silme,
bozuk etiket, geçersiz JSON). SanityChecker ve retry zincirinin bunları nasıl
yakaladığını görmek için: `python main.py --provider mock --chaos 0.3`
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, ClassVar

from core.regex_shield import TAG_RE
from providers.base import BaseTranslator, TokenUsage


class MockTranslator(BaseTranslator):
    provider_name: ClassVar[str] = "mock"

    def __init__(self, model: str = "mock", system_prompt: str = "", *,
                 chaos_rate: float = 0.0, seed: int = 7, **kwargs: Any) -> None:
        kwargs.setdefault("sleep", lambda _seconds: None)  # testte bekleme yok
        super().__init__(model, system_prompt, **kwargs)
        self.chaos_rate = max(0.0, min(chaos_rate, 1.0))
        self._rng = random.Random(seed)

    def _generate(self, system_prompt: str, user_message: str) -> tuple[str, TokenUsage]:
        payload = json.loads(user_message[user_message.index("{"):])
        glossary = payload.get("glossary", [])
        translations = []
        for item in payload["items"]:
            text = self._fake_translate(item["text"], glossary)
            roll = self._rng.random()
            if roll < self.chaos_rate * 0.25:
                continue  # satırı atla
            if roll < self.chaos_rate * 0.5 and TAG_RE.search(text):
                text = TAG_RE.sub("", text, count=1)  # bir etiketi unut
            elif roll < self.chaos_rate * 0.75 and TAG_RE.search(text):
                text = TAG_RE.sub(lambda m: f"[TAG {m.group(1)}]", text, count=1)  # etiketi boz
            translations.append({"id": item["id"], "text": text})

        raw = json.dumps({"translations": translations}, ensure_ascii=False)
        if self._rng.random() < self.chaos_rate * 0.2:
            raw = "Elbette! İşte çeviriler: " + raw[: len(raw) // 2]  # kesik, geçersiz JSON
        usage = TokenUsage(input_tokens=(len(system_prompt) + len(user_message)) // 4,
                           output_tokens=len(raw) // 3)
        return raw, usage

    @staticmethod
    def _fake_translate(text: str, glossary: list[dict[str, str]]) -> str:
        for term in glossary:
            pattern = re.compile(rf"(?<!\w){re.escape(term['source'])}[!.]?", re.IGNORECASE)
            text = pattern.sub(term["target"], text)
        return f"TR» {text}"
