"""
Sağlayıcıdan bağımsız çevirmen altyapısı.

Her sağlayıcı yalnızca `_generate(system_prompt, user_message) -> (ham_json, usage)`
metodunu ve kendi SDK hatalarının sınıflandırmasını yazar. Ortak olanlar burada:

* Master çeviri sistem promptu (karakter tonu, deyim yasağı, sözlük, [TAG_n] kuralları)
* Katı JSON giriş/çıkış sözleşmesi ve hoşgörülü ayrıştırıcı
* Üç kademeli hata politikası:
    RETRY       -> geçici hata (429, 5xx, ağ, bozuk JSON): üstel bekleme ile tekrar dene
    FAIL_BATCH  -> bu blok bu haliyle çevrilemez (güvenlik reddi, çıktı kesildi):
                   üst katman bloğu ikiye bölüp yeniden dener
    FATAL       -> yapılandırma hatası (geçersiz anahtar, olmayan model): tüm çalışma durur
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from string import Template
from typing import Any, Callable, ClassVar, Mapping, Sequence

from core.glossary import GlossaryTerm

logger = logging.getLogger(__name__)

MAX_HINTED_WAIT = 120.0  # sağlayıcının "şu kadar sn sonra dene" önerisine uyulacak en uzun süre


# --------------------------------------------------------------------------- hatalar


class ProviderError(Exception):
    """Tüm sağlayıcı hatalarının tabanı."""


class FatalProviderError(ProviderError):
    """Yapılandırma/kimlik doğrulama hatası. Her blokta tekrarlanacağı için çalışma durdurulur."""


class TransientProviderError(ProviderError):
    """Tekrar denemeye değer geçici hata."""


class ResponseFormatError(TransientProviderError):
    """Model geçerli JSON döndürmedi."""


class ContentBlockedError(ProviderError):
    """Model ya da güvenlik filtresi bu bloğu reddetti."""


class OutputTruncatedError(ProviderError):
    """Çıktı token sınırında kesildi; blok küçültülmeli."""


class TranslationFailedError(ProviderError):
    """Blok, tüm denemelere rağmen çevrilemedi."""


class ErrorAction(Enum):
    RETRY = "retry"
    FAIL_BATCH = "fail_batch"
    FATAL = "fatal"


# --------------------------------------------------------------------------- veri tipleri


@dataclass(frozen=True)
class SourceItem:
    id: str
    text: str
    key: str | None = None

    def as_dict(self) -> dict[str, str]:
        entry = {"id": self.id}
        if self.key:
            entry["key"] = self.key
        entry["text"] = self.text
        return entry


@dataclass(frozen=True)
class ContextItem:
    text: str
    translation: str | None = None

    def as_dict(self) -> dict[str, str]:
        entry = {"text": self.text}
        if self.translation:
            entry["translation"] = self.translation
        return entry


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens


# --------------------------------------------------------------------------- prompt


_SYSTEM_PROMPT = Template("""\
You are the lead localizer of a video game, translating its $source_lang text into $target_lang. \
Your translations must read as if a native $target_lang writer who plays games had written the game \
in $target_lang from day one: natural, idiomatic and in character. A literal, word-for-word \
translation is a failure even when it is grammatically correct.

## Input
The user message contains one JSON object:
- "items": the strings to translate. Each has an "id", the "text", and usually a "key": the string's \
path in the game files. Use the key to tell what kind of string it is (menu button, item name, \
tooltip, notification, dialogue line) and who is speaking.
- "context": the lines that come right before these items, some with their accepted translation. \
Use them only for speaker, tone and continuity. Never translate them and never return them.
- "glossary": required terminology for this batch (see Glossary).
- "feedback": present only when your previous answer for some ids was rejected. It lists what was \
wrong; fix exactly that.

## Output
Return only this JSON object, with nothing before or after it:
{"translations": [{"id": "<id>", "text": "<translation>"}]}
- Exactly one entry for every item, with the same ids, in the same order. Never skip, merge, split \
or invent ids.
- "text" holds the translation alone: no notes, no alternatives, no surrounding quotes.

## Frozen tokens: a mistake here crashes the game
[TAG_0], [TAG_1], ... stand for code: variables such as a player name or a number, formatting tags, \
line breaks, button icons.
- Copy every token of the source into the translation exactly once and character for character: \
square brackets, uppercase TAG, underscore, the same number.
- Never translate, renumber, drop, repeat, merge or split a token, and never invent one. Never write \
any other code-like text yourself: no {...}, <...>, %s, \\n, or [X] of your own.
- Move a token only as far as $target_lang word order requires. Tokens that open and close \
formatting must still wrap the translation of the words they wrapped, opening token first.
- A token at the very start or end of the text, or one that stands for a line break, keeps that \
structural position.
- A token may hide a name or number you cannot see. Do not glue a grammatical suffix to a token when \
the suffix's form depends on that hidden value; rephrase so the token stands alone. For Turkish, \
"[TAG_0]'s sword" must not become "[TAG_0]'ın kılıcı" (wrong for most names); write \
"[TAG_0] adlı oyuncunun kılıcı" or restructure the sentence.

## Style
- Translate meaning and intent, never word by word. Idioms, slang, military radio calls and gamer \
jargon become what a $target_lang speaker would actually say in that situation. Examples into \
Turkish: "Fire in the hole!" -> "Siper alın!" (never "Delikteki ateş!"); "Cover me!" -> \
"Beni koruyun!" (never "Beni ört!"); "Piece of cake." -> "Çocuk oyuncağı."; "Roger that." -> \
"Anlaşıldı."
- Keep every character's voice: a gruff soldier stays terse and gruff, a noble stays formal, a child \
sounds young, a villain stays menacing. Keep the intensity of insults, profanity and violence; do \
not soften or censor.
- Fit the register to the string type: buttons and menu items are short commands; tooltips and \
descriptions are clear complete sentences; dialogue is natural spoken language.
- UI strings must fit the same space as the source: keep them about as short and pick the shorter \
natural wording.
- Keep proper nouns, brand names and invented names (characters, places, factions, spells) as they \
are unless the glossary says otherwise.
- Keep the source's capitalization style (ALL CAPS stays ALL CAPS), its sentence-final punctuation \
and its ellipses. Do not add a final period the source does not have.
- If a string needs no translation (an interjection, a name, a number), return it unchanged or \
adapted naturally.

## Glossary
Each glossary entry has "source", "target" and sometimes a "note". Whenever the source term appears, \
use the target term, inflected as the sentence requires, and follow the note. When the target equals \
the source, keep the term exactly as written.
$style_section""")


def build_system_prompt(source_lang: str, target_lang: str, style_rules: Sequence[str] = ()) -> str:
    """Sabit sistem promptu. Çalışma boyunca değişmez, böylece prompt önbelleğine alınabilir."""
    style_section = ""
    if style_rules:
        bullets = "\n".join(f"- {rule}" for rule in style_rules)
        style_section = f"\n## {target_lang} rules for this game\n{bullets}\n"
    return _SYSTEM_PROMPT.substitute(
        source_lang=source_lang, target_lang=target_lang, style_section=style_section
    )


def build_user_message(
    items: Sequence[SourceItem],
    context: Sequence[ContextItem] = (),
    glossary: Sequence[GlossaryTerm] = (),
    feedback: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Modele giden kullanıcı mesajı. Maliyet tahmini de birebir bu metni ölçer."""
    payload: dict[str, Any] = {}
    if glossary:
        payload["glossary"] = [term.as_prompt_dict() for term in glossary]
    if context:
        payload["context"] = [line.as_dict() for line in context]
    if feedback:
        item_ids = {item.id for item in items}
        rejected = [{"id": item_id, "problems": list(problems)}
                    for item_id, problems in feedback.items() if item_id in item_ids]
        if rejected:
            payload["feedback"] = rejected
    payload["items"] = [item.as_dict() for item in items]
    return (
        'Translate every entry in "items" and reply with the JSON object only.\n\n'
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def output_schema(*, strict: bool = True) -> dict[str, Any]:
    """Yapılandırılmış çıktı şeması: {"translations": [{"id", "text"}]}."""
    item: dict[str, Any] = {
        "type": "object",
        "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
        "required": ["id", "text"],
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"translations": {"type": "array", "items": item}},
        "required": ["translations"],
    }
    if strict:
        item["additionalProperties"] = False
        schema["additionalProperties"] = False
    return schema


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_translation_payload(raw: str) -> dict[str, str]:
    """Model yanıtını {id: çeviri} sözlüğüne çevirir.

    Markdown çitlerini ve JSON öncesi/sonrası gevezeliği tolere eder. Hatalı tek
    elemanları atlar (onlar eksik sayılıp yeniden istenir); aynı ID iki kez gelirse
    hangisinin doğru olduğu bilinemeyeceği için ikisini de atar.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ResponseFormatError("boş yanıt")
    text = _FENCE_RE.sub("", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _extract_json(text)

    entries = data.get("translations") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ResponseFormatError("'translations' listesi bulunamadı")

    result: dict[str, str] = {}
    duplicates: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item_id, translation = entry.get("id"), entry.get("text")
        if not isinstance(item_id, (str, int)) or isinstance(item_id, bool) or not isinstance(translation, str):
            continue
        item_id = str(item_id)
        if item_id in result:
            duplicates.add(item_id)
        result[item_id] = translation
    for item_id in duplicates:
        del result[item_id]
    if entries and not result and not duplicates:
        raise ResponseFormatError("yanıtta geçerli {id, text} öğesi yok")
    return result


def _extract_json(text: str) -> Any:
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ResponseFormatError("yanıtta JSON bulunamadı")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ResponseFormatError(f"geçersiz JSON: {exc.msg}") from exc


# --------------------------------------------------------------------------- temel sınıf


class BaseTranslator(ABC):
    provider_name: ClassVar[str] = "base"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        *,
        max_retries: int = 3,
        request_timeout: float = 120.0,
        backoff_base: float = 2.0,
        backoff_max: float = 30.0,
        min_interval: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """
        min_interval: iki API isteği arasında bırakılacak en az süre (saniye). Ücretsiz
            katmanların dakikalık istek sınırına (429) takılmamak için istekleri baştan yavaşlatır.
            Örn. dakikada 10 istek sınırı için 6-7 sn. 0 = sınırsız.
        """
        if max_retries < 1:
            raise ValueError("max_retries en az 1 olmalı")
        self.model = model
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.min_interval = max(0.0, min_interval)
        self._sleep = sleep
        self._last_request_at = float("-inf")
        self.usage = TokenUsage()
        self.request_count = 0

    def translate(
        self,
        items: Sequence[SourceItem],
        context: Sequence[ContextItem] = (),
        glossary: Sequence[GlossaryTerm] = (),
        feedback: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, str]:
        """Bir bloğu çevirir: {item_id: çeviri}. Eksik/hatalı satırların denetimi üst katmandadır."""
        if not items:
            return {}
        user_message = build_user_message(items, context, glossary, feedback)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self._throttle()
                self.request_count += 1
                raw, usage = self._generate(self.system_prompt, user_message)
                self.usage.add(usage)
                return parse_translation_payload(raw)
            except Exception as exc:  # SDK'ya özgü hatalar _classify_error ile ayrıştırılır
                action = self._classify_error(exc)
                if action is ErrorAction.FATAL:
                    if isinstance(exc, FatalProviderError):
                        raise
                    raise FatalProviderError(f"{self.provider_name}: {type(exc).__name__}: {exc}") from exc
                if action is ErrorAction.FAIL_BATCH:
                    raise TranslationFailedError(f"{self.provider_name}: {exc}") from exc
                last_error = exc
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt, exc)
                    logger.warning("%s: deneme %d/%d başarısız (%s: %s); %.1f sn sonra tekrar denenecek",
                                   self.provider_name, attempt, self.max_retries,
                                   type(exc).__name__, exc, delay)
                    self._sleep(delay)

        if last_error is not None and self._is_rate_limit(last_error):
            # Kota hâlâ doluysa bloğu bölmek daha çok istek demektir; anlaşılır şekilde dur.
            raise FatalProviderError(
                f"{self.provider_name}: istek kotası doldu ({self.max_retries} deneme 429 aldı). "
                "Birkaç dakika sonra tekrar deneyin ya da istek aralığını artırın "
                "(L10N_GEMINI_MIN_INTERVAL, örn. 10). Günlük kota bittiyse ertesi güne kadar bekleyin."
            ) from last_error
        raise TranslationFailedError(
            f"{self.provider_name}: {self.max_retries} denemenin hepsi başarısız oldu: {last_error}"
        ) from last_error

    def _throttle(self) -> None:
        """Bir önceki istekten bu yana min_interval geçmediyse aradaki farkı bekler."""
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (time.monotonic() - self._last_request_at)
        if wait > 0:
            # INFO: GUI'nin canlı logunda görünür (arayüz donmuş sanılmasın); CLI'da yalnızca -v ile.
            logger.info("%s: istek hızı sınırı için %.0f sn bekleniyor", self.provider_name, wait)
            self._sleep(wait)
        self._last_request_at = time.monotonic()

    def _is_rate_limit(self, exc: Exception) -> bool:
        """Hata bir istek/kota sınırı (HTTP 429) mı? Sağlayıcılar kendi SDK hatalarına göre ezer."""
        return False

    @abstractmethod
    def _generate(self, system_prompt: str, user_message: str) -> tuple[str, TokenUsage]:
        """Tek bir API çağrısı. Ham JSON metnini ve token kullanımını döndürür."""

    def _classify_error(self, exc: Exception) -> ErrorAction:
        if isinstance(exc, FatalProviderError):
            return ErrorAction.FATAL
        if isinstance(exc, TransientProviderError):
            return ErrorAction.RETRY
        if isinstance(exc, (ContentBlockedError, OutputTruncatedError)):
            return ErrorAction.FAIL_BATCH
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return ErrorAction.RETRY
        return ErrorAction.FATAL  # bilinmeyen hata: sessizce yutmak yerine görünür şekilde dur

    def _retry_after_seconds(self, exc: Exception) -> float | None:
        """Sağlayıcı 'retry-after' bildirdiyse saniye cinsinden döndürür."""
        return None

    def _retry_delay(self, attempt: int, exc: Exception) -> float:
        hinted = self._retry_after_seconds(exc)
        if hinted is not None:
            # Sağlayıcının önerdiği süreye uyulur; günlük kota gibi saatlik beklemeler 120 sn ile sınırlanır
            # (o durumda denemeler tükenir ve çalışma anlaşılır bir mesajla durur).
            return min(max(hinted, 0.0), MAX_HINTED_WAIT)
        return min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max) + random.uniform(0, 1)
