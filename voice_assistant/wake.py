"""Wake word on top of speech recognition.

GigaAM transcribes a phrase in ~0.08 s, so instead of a separate keyword model
every phrase is transcribed and checked for the name. Recognition varies
("Нора", "Норо", "Эйнора", "Нуора", "Ора"), hence the one-letter tolerance; to keep
"я видел нору лисы" from waking it, the name must open the phrase (after an
optional "эй" / "слушай") or close it after a comma ("сколько времени, Нора?").
"""

from __future__ import annotations

import re

FILLERS = ("эй", "слушай", "привет", "ну", "окей", "ок", "так", "алло", "и")


def _distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _is_name(word: str, name: str) -> bool:
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
        if word == name or (vocative and _is_name(word, name)):
            return text[:start].rstrip(" ,.!?;:—-").strip()
    return None
