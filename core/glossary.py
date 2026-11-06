"""
glossary.json yükleyici ve batch bazlı terim eşleştirici.

Tüm sözlük her istekte gönderilmez: yalnızca o batch'in metinlerinde geçen
terimler eklenir. Hem token tasarrufu sağlar hem de modelin dikkatini dağıtmaz.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Tek kelimelik terimler İngilizce çekim ekleriyle de eşleşir: nerf -> nerfed, loot -> looting.
# "mana" -> "manage" gibi yanlış eşleşmeleri önlemek için ekler sınırlıdır.
_ENGLISH_INFLECTIONS = r"(?:s|es|ed|d|ing|er|ers|'s)?"


@dataclass(frozen=True)
class GlossaryTerm:
    source: str
    target: str
    note: str = ""

    def as_prompt_dict(self) -> dict[str, str]:
        entry = {"source": self.source, "target": self.target}
        if self.note:
            entry["note"] = self.note
        return entry


class Glossary:
    def __init__(
        self,
        terms: Iterable[GlossaryTerm] = (),
        do_not_translate: Iterable[str] = (),
        style_rules: Iterable[str] = (),
    ) -> None:
        self.terms = list(terms)
        self.do_not_translate = list(do_not_translate)
        self.style_rules = list(style_rules)
        self._matchers: list[tuple[re.Pattern[str], GlossaryTerm]] = [
            (_term_pattern(term.source), term) for term in self.terms
        ]
        self._matchers += [
            (_term_pattern(word, inflect=False, case_sensitive=True),
             GlossaryTerm(word, word, "Do not translate; keep exactly as written."))
            for word in self.do_not_translate
        ]

    @classmethod
    def load(cls, path: str | Path) -> "Glossary":
        path = Path(path)
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Sözlük dosyası bulunamadı: {path}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} geçersiz JSON (satır {exc.lineno}): {exc.msg}") from exc

        terms: list[GlossaryTerm] = []
        for index, raw in enumerate(data.get("terms", [])):
            if not isinstance(raw, dict) or not raw.get("source") or "target" not in raw:
                raise ValueError(f"{path}: terms[{index}] 'source' ve 'target' içermeli")
            terms.append(GlossaryTerm(str(raw["source"]), str(raw["target"]), str(raw.get("note", ""))))
        return cls(terms, data.get("do_not_translate", []), data.get("style_rules", []))

    def match(self, texts: Iterable[str]) -> list[GlossaryTerm]:
        """Verilen metinlerde geçen terimleri sözlük sırasıyla döndürür."""
        joined = "\n".join(texts)
        return [term for pattern, term in self._matchers if pattern.search(joined)]


def _term_pattern(source: str, *, inflect: bool = True, case_sensitive: bool = False) -> re.Pattern[str]:
    words = source.split()
    body = r"\s+".join(re.escape(word) for word in words)
    if inflect and len(words) == 1:
        body += _ENGLISH_INFLECTIONS
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<!\w){body}(?!\w)", flags)
