"""
SanityChecker: modelden dönen bloğu satır satır denetler.

Gönderilen ID kümesi ile dönen küme birebir eşleşmeli, her satırın [TAG_n]
etiketleri RegexShield doğrulamasından geçmelidir. Geçenler kabul edilir;
eksik ya da hatalı olanlar gerekçeleriyle birlikte "rejected" olarak döner ve
yalnızca onlar yeniden istenir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from core.regex_shield import TAG_RE, RegexShield

_WORD_RE = re.compile(r"[^\W\d_]+")


class MaskedUnit(Protocol):
    masked: str
    tag_map: dict[str, str]


@dataclass
class SanityVerdict:
    accepted: dict[str, str] = field(default_factory=dict)
    rejected: dict[str, list[str]] = field(default_factory=dict)
    unexpected_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected


class SanityChecker:
    def __init__(
        self,
        shield: RegexShield,
        *,
        max_length_ratio: float = 3.0,
        length_slack: int = 40,
        untranslated_min_words: int = 4,
    ) -> None:
        """
        max_length_ratio / length_slack: çeviri kaynağın `ratio * uzunluk + slack`
            karakterini aşarsa model açıklama ya da halüsinasyon eklemiş sayılır.
        untranslated_min_words: bu kadar ve daha fazla kelimelik bir metin aynen geri
            döndüyse model çevirmeyi atlamış sayılır.
        """
        self.shield = shield
        self.max_length_ratio = max_length_ratio
        self.length_slack = length_slack
        self.untranslated_min_words = untranslated_min_words

    def check(self, expected: Mapping[str, MaskedUnit], received: Mapping[str, str]) -> SanityVerdict:
        verdict = SanityVerdict(unexpected_ids=sorted(set(received) - set(expected)))
        for item_id, unit in expected.items():
            if item_id not in received:
                verdict.rejected[item_id] = ["missing from your response; return every id exactly once"]
                continue
            problems = self.check_item(unit, received[item_id])
            if problems:
                verdict.rejected[item_id] = problems
            else:
                verdict.accepted[item_id] = received[item_id].strip()
        return verdict

    def check_item(self, unit: MaskedUnit, translation: str) -> list[str]:
        if not isinstance(translation, str):
            return ["translation must be a string"]
        candidate = translation.strip()
        if not candidate:
            return ["translation is empty"]

        problems = self.shield.diagnose(unit.masked, candidate, unit.tag_map)

        source_plain = TAG_RE.sub("", unit.masked)
        candidate_plain = TAG_RE.sub("", candidate)
        if len(candidate_plain) > self.max_length_ratio * len(source_plain) + self.length_slack:
            problems.append("translation is far longer than the source; return only the translation, no notes")
        if (
            candidate == unit.masked.strip()
            and len(_WORD_RE.findall(source_plain)) >= self.untranslated_min_words
        ):
            problems.append("text was returned untranslated")
        return problems
