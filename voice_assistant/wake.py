"""Wake word on top of speech recognition.

GigaAM transcribes a phrase in ~0.08 s, so instead of a separate keyword model
every phrase is transcribed and checked for the name. Recognition varies
("Нора", "Норо", "Эйнора", "Нуора", "Ора"), hence the one-letter tolerance; to keep
"я видел нору лисы" from waking it, fuzzy matches are limited to the edges.
The exact name also works inside a sentence, even without ASR punctuation.
"""

from __future__ import annotations

import re

FILLERS = ("эй", "слушай", "привет", "ну", "окей", "ок", "так", "алло", "и")
JARVIS_NAMES = {"джарвиз", "джарвис", "jarvis"}


def is_exact_name(word: str, name: str) -> bool:
    return word == name or (name in JARVIS_NAMES and word in JARVIS_NAMES)


def _distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _is_name(word: str, name: str) -> bool:
    if is_exact_name(word, name):
        return True
    if len(word) >= 3 and word[0] == name[0] and _distance(word, name) <= 1:
        return True
    if word == name[1:]:  # the first sound clipped: "Ора" (but not "Пора": another first letter)
        return True
    # A filler glued to the name: "эйнора", "инора".
    return any(word.startswith(f) and _is_name(word[len(f) :], name) for f in FILLERS if len(word) > len(f) + 2)


def find_wake_word(text: str, name: str) -> str | None:
    """The request with the name removed ("" if only the name was said), or None if not addressed."""
    name = name.lower().replace("ё", "е")
    words = [(m.group().replace("ё", "е"), m.start(), m.end()) for m in re.finditer(r"[a-zа-яё]+", text.lower())]
    for word, _, end in words:
        if _is_name(word, name):
            return text[end:].lstrip(" ,.!?;:—-").strip()
        if word not in FILLERS:
            break
    if words:
        word, start, _ = words[-1]
        # At the end the name is a vocative, and GigaAM puts a comma before it: "Сколько времени, Нра?".
        # Without the comma only the exact name counts, so "видел нору" stays a sentence.
        vocative = text[:start].rstrip().endswith(",")
        if is_exact_name(word, name) or (vocative and _is_name(word, name)):
            return text[:start].rstrip(" ,.!?;:—-").strip()
    for word, start, end in words:
        if is_exact_name(word, name):
            # Keep the request on both sides: «поставь, Нора, таймер на минуту».
            return (text[:start].rstrip(" ,.!?;:—-") + " " + text[end:].lstrip(" ,.!?;:—-")).strip()
    return None


STOP_WORDS = ("стоп", "хватит", "подожди", "погоди", "замолчи", "замолчите", "помолчи", "тихо", "стой", "остановись", "достаточно", "перестань")
ECHO_SHARE = 0.6  # this share of words heard in her own current speech means it is just her echo


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zа-я]+", text.lower().replace("ё", "е"))


def find_interruption(text: str, name: str, echo: str) -> str | None:
    """While Nora talks the microphone also hears her. Returns the new request ("" for a bare
    «стоп») if the user cut in, or None if the phrase is her own voice or not addressed to her.

    Laptop speakers are loud, so a phrase is often her voice and the user's mixed: only words
    she is not saying right now count, and the name or a stop word among them interrupts.
    """
    name = name.lower().replace("ё", "е")
    spoken = set(_words(echo))
    words = _words(text)
    if not words:
        return None
    foreign = [w for w in words if w not in spoken]
    if len(foreign) < len(words) * (1 - ECHO_SHARE) and not any(w in STOP_WORDS for w in foreign):
        return None  # mostly her own words
    for match in re.finditer(r"[a-zа-яё]+", text.lower()):
        word = match.group().replace("ё", "е")
        if word in spoken:
            continue
        if word in STOP_WORDS:
            if word in ("помолчи", "замолчи"):
                return text[match.start():].strip()
            return ""
        if _is_name(word, name):
            rest = text[match.end() :].lstrip(" ,.!?;:—-").strip()
            first = _words(rest)[:1]
            return "" if first and first[0] in STOP_WORDS and first[0] not in ("помолчи", "замолчи") else rest
    return None
