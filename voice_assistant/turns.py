"""Collect speech segments before committing a user's turn to any action."""

from __future__ import annotations

import re
import time

_INCOMPLETE = re.compile(
    r"(?:\b(?:и|а|но|или|чтобы|потому\s+что|если|когда|который|которая|которое|для|на|в|с|по|про)|"
    r"\b(?:я\s+хочу|мне\s+нужно|я\s+бы\s+хотел|можешь|поставь\s+таймер))\W*$", re.I,
)


class TurnCollector:
    def __init__(self, pause_s: float = 2.4, incomplete_pause_s: float = 4.5) -> None:
        self.pause_s = pause_s
        self.incomplete_pause_s = max(pause_s, incomplete_pause_s)
        self._parts: list[str] = []
        self._last_speech = 0.0
        self.heard_at = 0.0
        self.stt_s = 0.0

    @property
    def collecting(self) -> bool:
        return bool(self._parts)

    def clear(self) -> None:
        self._parts.clear()
        self.stt_s = 0.0

    def add(self, text: str, heard_at: float, stt_s: float = 0.0, silence_s: float = 0.0) -> None:
        if not self._parts:
            self.heard_at = heard_at - silence_s
            self.stt_s = 0.0
        self._parts.append(text.strip())
        self._last_speech = heard_at - silence_s
        self.stt_s += stt_s

    def take(self, speaking: bool, now: float | None = None, force: bool = False) -> str | None:
        if not self._parts or speaking:
            return None
        text = " ".join(self._parts)
        delay = self.incomplete_pause_s if _INCOMPLETE.search(text) else self.pause_s
        if not force and (time.monotonic() if now is None else now) - self._last_speech < delay:
            return None
        self._parts.clear()
        return text
