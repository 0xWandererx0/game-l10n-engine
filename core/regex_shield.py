"""
RegexShield: oyun kodlarını çeviri öncesi dondurur, çeviri sonrası doğrular ve geri açar.

Modele asla çıplak kod gitmez:

    >>> shield = RegexShield()
    >>> masked, tags = shield.mask("Hello {player}!\\nPress [E]")
    >>> masked
    'Hello [TAG_0]![TAG_1]Press [TAG_2]'
    >>> shield.unmask("Merhaba [TAG_0]![TAG_1][TAG_2] tuşuna bas", tags)
    'Merhaba {player}!\\n[E] tuşuna bas'

Doğrulama yalnızca "etiket sayısı tutuyor mu" diye bakmaz; oyunu çökerten daha
sinsi hataları da yakalar:

* `%s`, `%d`, `{}` gibi sıralı biçimleyicilerin yeri değişirse printf/format
  yanlış türde argüman okur (C/C++'ta çökme). Bu tokenların sırası korunmalıdır.
* `<b>...</b>` gibi etiket çiftleri çaprazlanırsa ya da kapanış açılıştan önce
  gelirse bazı zengin metin ayrıştırıcıları hata fırlatır.
* Model kendi kafasından `{`, `}` ya da `<b>` gibi kod üretirse C# `string.Format`
  `FormatException` fırlatır.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

TAG_RE = re.compile(r"\[TAG_(\d+)\]")

# Modelin bozduğu etiket biçimleri: "[TAG 0]", "[tag_0]", "( TAG-0 )", "【TAG_0】" ...
_LOOSE_TAG_RE = re.compile(r"[\[【［(<{]\s*TAG\s*[_\-\s]?\s*\d+\s*[\]】］)>}]", re.IGNORECASE)
_BARE_TAG_RE = re.compile(r"(?<![\[\w])TAG_\d+(?![\]\w])", re.IGNORECASE)

# Etiketleri silerken kelimelerin yanlışlıkla birleşip yeni bir kod oluşturmasını önler.
_SENTINEL = "\x00"


class TokenKind(str, Enum):
    MARKUP = "markup"  # <b>, </color>, [E], [b]...[/b], {i}...{/i}
    PLACEHOLDER = "placeholder"  # {0}, {name}, %1$s, %(name)s, $VAR$ -> yeri değişebilir
    SEQUENTIAL = "sequential"  # %s, %d, {} -> sırası değişirse yanlış argüman / çökme
    ESCAPE = "escape"  # \n, \t, gerçek satır sonu, \u00e9
    LITERAL = "literal"  # %%, &amp;, §a -> korunur, sırası serbest


_MOVABLE = frozenset({TokenKind.PLACEHOLDER, TokenKind.SEQUENTIAL})

_PRINTF_FLAGS = r"[-+0#']*"
_PRINTF_WIDTH = r"(?:\d+|\*)?(?:\.(?:\d+|\*))?"
_PRINTF_LENGTH = r"(?:hh|h|ll|l|L|q|j|z|t|I64|I32)?"
_PRINTF_CONV = r"[diouxXeEfFgGaAcsSp@]"
_PRINTF_TAIL = _PRINTF_FLAGS + _PRINTF_WIDTH + _PRINTF_LENGTH

# (ad, regex, tür). Çakışan eşleşmelerde en erken başlayan, eşitse en uzun olan kazanır.
DEFAULT_PATTERNS: tuple[tuple[str, str, TokenKind], ...] = (
    # Unity/TextMeshPro/HTML/Unreal zengin metin: <color=#FF0000>, </b>, <sprite name="x"/>, </>
    ("angle_markup", r"</?[A-Za-z][^<>\n]{0,200}?>|</>", TokenKind.MARKUP),
    # BBCode / Godot: [color=red], [url=https://...], [font=Noto Sans]
    ("bracket_attr", r"\[/?[A-Za-z_][\w\-]*=[^\[\]\n]{1,120}\]", TokenKind.MARKUP),
    # Tuş atamaları ve kısa köşeli kodlar: [E], [SPACE], [LMB], [b], [/b], [Root.GetName]
    ("bracket_token", r"\[/?[^\[\]\s]{1,40}\]", TokenKind.MARKUP),
    # printf: %1$s (konumsal, yer değiştirebilir), %(name)s (Python), %s/%d (sıralı)
    ("printf_positional", r"%\d+\$" + _PRINTF_TAIL + _PRINTF_CONV, TokenKind.PLACEHOLDER),
    ("printf_named", r"%\([A-Za-z_][\w.]*\)" + _PRINTF_TAIL + r"[diouxXeEfFgGcrsa]", TokenKind.PLACEHOLDER),
    ("printf_sequential", r"%" + _PRINTF_TAIL + _PRINTF_CONV, TokenKind.SEQUENTIAL),
    ("percent_literal", r"%%", TokenKind.LITERAL),
    # %PLAYER_NAME% (en az iki karakter: "%d%%" gibi printf dizileriyle karışmasın)
    ("percent_named", r"%[A-Za-z_][A-Za-z0-9_]+%", TokenKind.PLACEHOLDER),
    # ${var} (JS/Twine) ve $VAR$ (Paradox)
    ("dollar_brace", r"\$\{[^{}\n]*\}", TokenKind.PLACEHOLDER),
    ("dollar_named", r"\$[A-Za-z_][\w.|:]*\$", TokenKind.PLACEHOLDER),
    # Metin olarak yazılmış kaçış dizileri (CSV / bazı motorlar): \n \t \" \u00e9
    ("escape_literal", r"\\(?:u[0-9A-Fa-f]{4}|x[0-9A-Fa-f]{2}|[nrtvbf0\"'\\])", TokenKind.ESCAPE),
    # JSON'dan çözülmüş gerçek kontrol karakterleri
    ("escape_control", r"\r\n|[\r\n\t\v\f]", TokenKind.ESCAPE),
    ("html_entity", r"&(?:[A-Za-z][A-Za-z0-9]{1,31}|#\d{1,7}|#[xX][0-9A-Fa-f]{1,6});", TokenKind.LITERAL),
    # Minecraft/Paradox renk kodları: §a, §Y, §!
    ("section_color", r"§[0-9A-Za-z!]", TokenKind.LITERAL),
)

# Qt/Godot tarzı "%1" konumsal yer tutucular. Varsayılan değildir, çünkü Türkçe
# "%50" yüzde yazımıyla çakışır; Qt dizeleri için extra_patterns ile açın.
QT_POSITIONAL_PATTERN: tuple[str, str, TokenKind] = (
    "qt_positional", r"(?<![\w%])%\d{1,2}(?![\d$])", TokenKind.PLACEHOLDER,
)

# Ren'Py metin etiketleri ve BBCode adları: süslü/köşeli parantezde bunlar biçim etiketidir.
_KNOWN_MARKUP_NAMES = frozenset({
    "b", "i", "u", "s", "a", "p", "w", "k", "nw", "fast", "plain", "font", "color", "size",
    "alpha", "cps", "space", "vspace", "image", "rt", "rb", "outlinecolor", "url", "img",
    "center", "left", "right", "quote", "code", "list", "sup", "sub", "wave", "shake",
    "rainbow", "tornado", "fade", "outline", "table", "cell", "indent", "lb",
})
_BRACKET_WORD_RE = re.compile(r"^\[[a-z][a-z'\-]*\]$")
_BRACE_MARKUP_RE = re.compile(r"^\{(/?)([A-Za-z]+)(=[^{}]*)?\}$")
_ANGLE_NAME_RE = re.compile(r"^</?\s*([A-Za-z][\w.\-:]*)")
_BRACKET_NAME_RE = re.compile(r"^\[/?\s*([^\]=\s/]+)")


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    raw: str
    kind: TokenKind
    pattern: str
    markup_name: str | None = None
    closing: bool = False
    self_closing: bool = False


class RegexShield:
    """Kod parçalarını [TAG_n] ile maskeler, geri açar ve çeviriyi doğrular."""

    def __init__(
        self,
        *,
        protect_bracket_words: bool = True,
        extra_patterns: Iterable[tuple[str, str, TokenKind]] = (),
    ) -> None:
        """
        protect_bracket_words: True ise "[name]", "[sighs]" gibi köşeli parantezli düz
            kelimeler de korunur. Ren'Py "[player]" değişkenleri çevrilirse oyun çöker;
            yalnızca sahne yönergelerini çevirtmek istiyorsanız False yapın.
        extra_patterns: motorunuza özel (ad, regex, TokenKind) desenleri.
        """
        self.protect_bracket_words = protect_bracket_words
        self._patterns = tuple(
            (name, re.compile(regex), kind)
            for name, regex, kind in (*DEFAULT_PATTERNS, *extra_patterns)
        )

    # ------------------------------------------------------------------ public API

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Kodları [TAG_0], [TAG_1]... ile değiştirir. Bitişik kodlar tek etikette birleşir."""
        if not text:
            return text, {}
        tokens = self.tokenize(text)
        if not tokens:
            return text, {}

        parts: list[str] = []
        tag_map: dict[str, str] = {}
        cursor = 0
        for index, group in enumerate(self._group_adjacent(tokens)):
            start, end = group[0].start, group[-1].end
            tag = f"[TAG_{index}]"
            parts.append(text[cursor:start])
            parts.append(tag)
            tag_map[tag] = text[start:end]
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts), tag_map

    def unmask(self, masked_text: str, tag_map: Mapping[str, str]) -> str:
        """[TAG_n] etiketlerini orijinal kodlarla tek geçişte değiştirir."""
        return TAG_RE.sub(lambda m: tag_map.get(m.group(0), m.group(0)), masked_text)

    def validate(self, masked_source: str, translated_text: str, tag_map: Mapping[str, str]) -> bool:
        """Çeviri oyunu çökertmeden geri açılabiliyorsa True."""
        return not self.diagnose(masked_source, translated_text, tag_map)

    def diagnose(self, masked_source: str, translated_text: str, tag_map: Mapping[str, str]) -> list[str]:
        """Çevirideki tüm sorunları döndürür (boş liste = geçerli). Mesajlar modele geri beslenir."""
        if not isinstance(translated_text, str):
            return ["translation is not a string"]

        problems: list[str] = []
        source_tags = {m.group(0) for m in TAG_RE.finditer(masked_source)}
        if source_tags != set(tag_map):
            problems.append("internal error: masked source and tag map disagree")

        found = [m.group(0) for m in TAG_RE.finditer(translated_text)]
        counts = Counter(found)
        for tag in self._ordered_tags(tag_map):
            seen = counts.get(tag, 0)
            if seen == 0:
                problems.append(f"missing {tag}")
            elif seen > 1:
                problems.append(f"{tag} appears {seen} times; it must appear exactly once")
        unknown = sorted(set(counts) - set(tag_map), key=_tag_index)
        if unknown:
            problems.append(f"invented tag(s) {', '.join(unknown)} that are not in the source")

        stripped = TAG_RE.sub(_SENTINEL, translated_text)
        malformed = _LOOSE_TAG_RE.findall(stripped) + _BARE_TAG_RE.findall(stripped)
        if malformed:
            problems.append(f"malformed tag(s) {', '.join(malformed)}; copy tags exactly like [TAG_0]")

        problems.extend(self._leaked_code_problems(masked_source, stripped, tag_map))

        # Sıra ve iç içe geçme kontrolleri, her etiket tam bir kez varsa anlamlıdır.
        if not problems:
            problems.extend(self._order_problems(found, tag_map))
        return problems

    def tokenize(self, text: str) -> list[Token]:
        """Metindeki korunacak kodları çakışmasız ve soldan sağa sıralı döndürür."""
        candidates = [self._brace_token(text, start, end) for start, end in _brace_spans(text)]
        for name, regex, kind in self._patterns:
            for match in regex.finditer(text):
                if match.end() == match.start():
                    continue
                if (
                    name == "bracket_token"
                    and not self.protect_bracket_words
                    and _BRACKET_WORD_RE.match(match.group(0))
                    and match.group(0)[1:-1] not in _KNOWN_MARKUP_NAMES
                ):
                    continue
                candidates.append(self._make_token(match.start(), match.end(), match.group(0), kind, name))

        candidates.sort(key=lambda t: (t.start, -(t.end - t.start)))
        tokens: list[Token] = []
        last_end = 0
        for token in candidates:
            if token.start >= last_end:
                tokens.append(token)
                last_end = token.end
        return tokens

    def signature(self, text: str) -> Counter[str]:
        """Metindeki kodların çoklu kümesi. Son güvenlik kapısı: kaynak == çeviri olmalı."""
        return Counter(token.raw for token in self.tokenize(text))

    @staticmethod
    def has_translatable_text(masked_text: str) -> bool:
        """Etiketler çıkarıldığında geriye çevrilecek bir harf kalıyor mu?"""
        return any(ch.isalpha() for ch in TAG_RE.sub("", masked_text))

    # ------------------------------------------------------------------ internals

    def _leaked_code_problems(self, masked_source: str, stripped_translation: str,
                              tag_map: Mapping[str, str]) -> list[str]:
        problems: list[str] = []
        stripped_source = TAG_RE.sub(_SENTINEL, masked_source)
        leaked = Counter(t.raw for t in self.tokenize(stripped_translation))
        leaked -= Counter(t.raw for t in self.tokenize(stripped_source))
        if leaked:
            shown = ", ".join(repr(raw.replace(_SENTINEL, "[TAG]")) for raw in sorted(leaked))
            problems.append(f"code-like text {shown} outside the tags; only [TAG_n] tokens may carry code")

        for ch in "{}":
            if stripped_translation.count(ch) > stripped_source.count(ch):
                problems.append(f"unexpected '{ch}' (crashes string formatting); remove it")

        # printf ile biçimlenen dizede serbest '%' tanımsız davranıştır ("%50 c" -> "%50c").
        if any("%" in raw for raw in tag_map.values()):
            if stripped_translation.count("%") > stripped_source.count("%"):
                problems.append("unexpected '%' in a printf-formatted string; write the number without '%'")
        return problems

    def _order_problems(self, found_order: list[str], tag_map: Mapping[str, str]) -> list[str]:
        components = self._components(tag_map)
        source_sequence = [comp for tag in self._ordered_tags(tag_map) for comp in components[tag]]
        translated_sequence = [comp for tag in found_order for comp in components[tag]]
        problems: list[str] = []

        sequential = [cid for cid, token in translated_sequence if token.kind is TokenKind.SEQUENTIAL]
        if sequential != sorted(sequential):
            owners = [tag for tag in found_order if any(t.kind is TokenKind.SEQUENTIAL for _, t in components[tag])]
            problems.append(
                "tags holding sequential format specifiers (%s, %d, {}) were reordered: "
                f"keep {', '.join(sorted(owners, key=_tag_index))} in their original order"
            )

        closer_to_opener = _pair_markup(source_sequence)
        openers = set(closer_to_opener.values())
        tag_of = {cid: tag for tag in tag_map for cid, _ in components[tag]}
        stack: list[int] = []
        for cid, _token in translated_sequence:
            if cid in openers:
                stack.append(cid)
            elif cid in closer_to_opener:
                expected_opener = closer_to_opener[cid]
                if not stack or stack[-1] != expected_opener:
                    problems.append(
                        f"formatting tags {tag_of[expected_opener]} and {tag_of[cid]} must stay a pair: "
                        "the opening tag first, then the words they wrap, then the closing tag, without crossing other pairs"
                    )
                    break
                stack.pop()
        return problems

    def _components(self, tag_map: Mapping[str, str]) -> dict[str, list[tuple[int, Token]]]:
        """Her etiketi (birleşmiş olabilir) bileşen kodlarına ayırır; kaynak sırasına göre numaralar."""
        result: dict[str, list[tuple[int, Token]]] = {}
        next_id = 0
        for tag in self._ordered_tags(tag_map):
            raw = tag_map[tag]
            tokens = self.tokenize(raw) or [self._make_token(0, len(raw), raw, TokenKind.LITERAL, "opaque")]
            result[tag] = []
            for token in tokens:
                result[tag].append((next_id, token))
                next_id += 1
        return result

    @staticmethod
    def _ordered_tags(tag_map: Mapping[str, str]) -> list[str]:
        return sorted(tag_map, key=_tag_index)

    @staticmethod
    def _group_adjacent(tokens: list[Token]) -> list[list[Token]]:
        """Arada metin olmayan kodları tek etikette toplar.

        İstisna: satır sonu ile yer tutucu birleşmez. "{0}\\n" ayrı kalır ki model
        ismi Türkçe söz dizimine göre taşıyabilsin, satır sonu yerinde kalsın.
        """
        groups: list[list[Token]] = []
        for token in tokens:
            if groups and groups[-1][-1].end == token.start:
                kinds = {t.kind for t in groups[-1]}
                conflict = (
                    (token.kind is TokenKind.ESCAPE and kinds & _MOVABLE)
                    or (token.kind in _MOVABLE and TokenKind.ESCAPE in kinds)
                )
                if not conflict:
                    groups[-1].append(token)
                    continue
            groups.append([token])
        return groups

    def _brace_token(self, text: str, start: int, end: int) -> Token:
        raw = text[start:end]
        markup = _BRACE_MARKUP_RE.match(raw)
        if markup and (markup.group(1) or markup.group(3) or markup.group(2).lower() in _KNOWN_MARKUP_NAMES):
            return Token(start, end, raw, TokenKind.MARKUP, "brace_markup",
                         markup_name=markup.group(2).lower(), closing=bool(markup.group(1)))
        inner = raw[1:-1]
        if not raw.startswith("{{") and (inner == "" or inner.startswith(":")):
            kind = TokenKind.SEQUENTIAL  # "{}" / "{:.2f}" otomatik numaralı alanlar
        else:
            kind = TokenKind.PLACEHOLDER
        return Token(start, end, raw, kind, "brace")

    @staticmethod
    def _make_token(start: int, end: int, raw: str, kind: TokenKind, pattern: str) -> Token:
        if kind is not TokenKind.MARKUP:
            return Token(start, end, raw, kind, pattern)
        if raw == "</>":  # Unreal: herhangi bir açık etiketi kapatır
            return Token(start, end, raw, kind, pattern, markup_name="", closing=True)
        name_match = (_ANGLE_NAME_RE if raw.startswith("<") else _BRACKET_NAME_RE).match(raw)
        name = name_match.group(1).lower() if name_match else raw.lower()
        closing = raw.startswith("</") or raw.startswith("[/")
        return Token(start, end, raw, kind, pattern, markup_name=name, closing=closing,
                     self_closing=raw.endswith("/>"))


def _tag_index(tag: str) -> int:
    match = TAG_RE.fullmatch(tag)
    return int(match.group(1)) if match else 1 << 30


def _brace_spans(text: str) -> list[tuple[int, int]]:
    """Dengeli süslü parantez gruplarını bulur: {0}, {{var}}, ICU {n, plural, one {#} other {#}}.

    Kapanan her çift aday olarak eklenir; çakışma çözümünde en dıştaki kazanır.
    Kapanmamış bir '{' sonrasındaki geçerli {0} yine de korunur. Satır sonunu aşan
    parantez kod sayılmaz.
    """
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    for index, ch in enumerate(text):
        if ch == "{":
            stack.append(index)
        elif ch == "}" and stack:
            spans.append((stack.pop(), index + 1))
        elif ch in "\r\n":
            stack.clear()
    return spans


def _pair_markup(sequence: list[tuple[int, Token]]) -> dict[int, int]:
    """Kaynaktaki açılış/kapanış etiketlerini eşler: {kapanış_id: açılış_id}.

    Kapanmayan etiketler (Unity'de <color=red> kapanmadan kalabilir) ve tuş
    atamaları ([E]) eşsiz kalır; onlara sıra kısıtı uygulanmaz.
    """
    stack: list[tuple[str | None, int]] = []
    closer_to_opener: dict[int, int] = {}
    for cid, token in sequence:
        if token.kind is not TokenKind.MARKUP or token.self_closing:
            continue
        if not token.closing:
            stack.append((token.markup_name, cid))
            continue
        for position in range(len(stack) - 1, -1, -1):
            name, opener_id = stack[position]
            if token.markup_name == "" or name == token.markup_name:
                closer_to_opener[cid] = opener_id
                del stack[position:]
                break
    return closer_to_opener
