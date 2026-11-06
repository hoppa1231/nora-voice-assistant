"""Conversational quiet mode: resume on an explicit call or after a duration."""

from __future__ import annotations

import re
import time

from .tools.alarms import parse_duration, say_duration
from .tools import ToolError

_QUIET = re.compile(
    r"^(?:пожалуйста\s+)?(?:помолчи|замолчи|молчи|не\s+отвечай|тихий\s+режим)"
    r"(?:\s+(.*))?$", re.I,
)


class QuietMode:
    def __init__(self) -> None:
        self.until = 0.0

    @property
    def active(self) -> bool:
        return time.monotonic() < self.until

    def wake(self) -> None:
        self.until = 0.0

    def request(self, text: str) -> str | None:
        match = _QUIET.fullmatch(text.strip().rstrip(".!?…"))
        if not match:
            return None
        rest = (match.group(1) or "").strip()
        rest = re.sub(r"^(?:пожалуйста\s+)?(?:на\s+)?", "", rest, flags=re.I)
        if not rest or re.fullmatch(r"(?:до\s+(?:следующего\s+)?(?:вызова|обращения)|пока(?:\s+я)?\s+не\s+(?:позову|вызову)(?:\s+тебя)?|немного|некоторое\s+время|пожалуйста)", rest, re.I):
            self.until = float("inf")
            return "Хорошо, помолчу, пока не позовёшь по имени."
        try:
            seconds = parse_duration(rest)
        except ToolError:
            return "На сколько помолчать? Скажи, например: помолчи пять минут или помолчи до вызова."
        self.until = time.monotonic() + seconds
        return f"Хорошо, помолчу {say_duration(seconds)}. Если понадоблюсь раньше, позови по имени."
