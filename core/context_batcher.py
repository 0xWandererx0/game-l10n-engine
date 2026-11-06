"""
Bağlam korumalı batch'leyici.

Metinler dosyadaki sırayla (diyalog akışı korunarak) en fazla `chunk_size`
satırlık bloklara bölünür. Her bloğa, kendisinden hemen önce gelen ve aynı
gruba (aynı sahne/diyalog listesi) ait `context_size` satır "sadece bağlam"
olarak eklenir; model bunları çevirmez ama konuşmacının tonunu ve önceki
repliği bilerek çeviri yapar.

Grup değişince blok yarı doluysa erkenden kapatılır ki bir diyalog gereksiz yere
iki isteğe bölünmesin. `max_batch_chars`, uzun lore metinlerinde model çıktısının
token sınırında kesilmesini önler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.parser import JsonPath


@dataclass
class TranslationUnit:
    uid: str  # dosya genelinde benzersiz kimlik (JSON Pointer / CSV anahtarı)
    source: str  # orijinal metin (kenar boşlukları dahil)
    masked: str  # kenar boşlukları ayrılmış, kodları [TAG_n] ile maskelenmiş metin
    tag_map: dict[str, str]
    key_hint: str  # modele "bu metin nerede görünüyor" ipucu
    group: str
    lead_ws: str = ""
    trail_ws: str = ""


@dataclass
class Batch:
    index: int
    units: list[TranslationUnit]
    context: list[TranslationUnit]  # önceki satırlar; çevrilmez, döndürülmez


def group_key(path: JsonPath) -> str:
    """Aynı sahneye/listeye ait satırları gruplar.

    /dialogues/ch1/briefing/3/text -> "dialogues/ch1/briefing" (son liste indeksine kadar)
    /ui/main_menu/title            -> "ui/main_menu"
    """
    cut = len(path) - 1
    for position in range(len(path) - 1, -1, -1):
        if isinstance(path[position], int):
            cut = position
            break
    return "/".join(str(part) for part in path[:cut])


def key_hint(path: JsonPath, max_parts: int = 4) -> str:
    return "/".join(str(part) for part in path[-max_parts:])


def split_edges(text: str) -> tuple[str, str, str]:
    """Baştaki/sondaki boşlukları ayırır: "Score: " -> ("", "Score:", " ").

    Model kenar boşluklarını silebilir; kodda "Score: " + value birleştirmesi varsa
    Türkçe çıktıda "Skor:100" olur. Boşluklar modele hiç gönderilmez, sonra geri eklenir.
    """
    stripped = text.strip("  ")
    if not stripped:
        return text, "", ""
    start = text.index(stripped)
    return text[:start], stripped, text[start + len(stripped):]


class ContextBatcher:
    def __init__(
        self,
        chunk_size: int = 20,
        context_size: int = 3,
        max_batch_chars: int = 6000,
        soft_break_ratio: float = 0.5,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size en az 1 olmalı")
        self.chunk_size = chunk_size
        self.context_size = max(0, context_size)
        self.max_batch_chars = max_batch_chars
        self.soft_break_ratio = soft_break_ratio

    def build(self, units: Sequence[TranslationUnit]) -> list[Batch]:
        batches: list[Batch] = []
        current: list[int] = []
        current_chars = 0

        def flush() -> None:
            nonlocal current, current_chars
            if current:
                first = current[0]
                batches.append(Batch(
                    index=len(batches),
                    units=[units[i] for i in current],
                    context=self._context_for(units, first),
                ))
            current, current_chars = [], 0

        for index, unit in enumerate(units):
            size = len(unit.masked)
            if current:
                full = len(current) >= self.chunk_size or current_chars + size > self.max_batch_chars
                group_changed = unit.group != units[current[-1]].group
                soft_break = group_changed and len(current) >= self.chunk_size * self.soft_break_ratio
                if full or soft_break:
                    flush()
            current.append(index)
            current_chars += size
        flush()
        return batches

    def _context_for(self, units: Sequence[TranslationUnit], first_index: int) -> list[TranslationUnit]:
        if self.context_size == 0:
            return []
        group = units[first_index].group
        context: list[TranslationUnit] = []
        index = first_index - 1
        while index >= 0 and len(context) < self.context_size and units[index].group == group:
            context.append(units[index])
            index -= 1
        context.reverse()
        return context
