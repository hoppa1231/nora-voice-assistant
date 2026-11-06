"""Exact everyday commands executed without a language-model round trip."""

from __future__ import annotations

import json
import re
import time

from .tools import ToolRegistry
from .tools.alarms import _WORD_NUMBERS

_TIMER = re.compile(r"^(?:(?:поставь|установи|запусти)\s+)?таймер(?:\s+(.*))?$|^засеки\s+(.+)$")
_DURATION_WORDS = set(_WORD_NUMBERS) | {"на", "и", "полтора", "полторы", "полчаса", "полминуты", "пол", "четверть"}


def duration_only(text: str) -> bool:
    words = re.findall(r"[a-zа-яё]+", text.lower())
    return bool(re.search(r"\d", text) or words) and all(
        w in _DURATION_WORDS or re.fullmatch(r"час(?:а|ов)?|минут(?:а|у|ы)?|секунд(?:а|у|ы)?|[hmsчмс]", w)
        for w in words
    ) and not re.search(r"-\s*\d|\bминус\b", text)


class QuickCommands:
    def __init__(self, tools: ToolRegistry) -> None:
        self.tools = tools
        self.timer_until = 0.0
        self.used_tool = False

    async def handle(self, text: str) -> str | None:
        self.used_tool = False
        t = text.lower().replace("ё", "е").strip().rstrip(".!?…")
        t = re.sub(r"^пожалуйста\s+|[,\s]+пожалуйста$", "", t).strip()
        args: dict = {}
        tool = None
        awaiting = time.monotonic() < self.timer_until
        self.timer_until = 0.0
        match = _TIMER.fullmatch(t)
        if t in ("стоп", "хватит", "подожди", "погоди", "стой", "остановись", "достаточно", "перестань"):
            return ""  # interruption has already stopped playback; do not start a new LLM reply
        if match or (awaiting and duration_only(t)):
            duration = next((v for v in match.groups() if v), "") if match else t
            duration = re.sub(r"^на\s+", "", duration).strip()
            if not duration:
                self.timer_until = time.monotonic() + 90
                return "На сколько поставить таймер?"
            parts = re.split(r"\s+для\s+", duration, maxsplit=1)
            if len(parts) > 1 and re.search(r"\b(?:и|а|затем|потом)\s+(?:открой|поставь|включи|выключи|отмени|напомни|скажи)\b", parts[1]):
                return None
            if not duration_only(parts[0]):
                return None  # ambiguous/composite requests stay with the LLM
            tool, args = "set_timer", {"duration": parts[0]}
            if len(parts) > 1:
                args["label"] = parts[1]
        elif re.fullmatch(r"(?:какие|покажи|перечисли)\s+(?:у\s+меня\s+)?(?:активные\s+)?таймеры|сколько\s+осталось(?:\s+(?:на|до)\s+таймер[ае])?|список\s+таймеров", t):
            tool = "list_alarms"
        elif t in ("отмени таймер", "удали таймер", "выключи таймер"):
            tool = "cancel_timer"
        elif t in ("отмени все таймеры", "удали все таймеры"):
            tool, args = "cancel_timer", {"all_timers": True}
        elif t in ("что ты умеешь", "какие у тебя инструменты", "список навыков", "покажи навыки"):
            tool = "list_skills"
        else:
            media = {"пауза": "pause", "поставь на паузу": "pause", "следующий трек": "next", "предыдущий трек": "previous", "продолжи музыку": "play"}
            volume = {"громче": "up", "погромче": "up", "сделай громче": "up", "сделай погромче": "up", "тише": "down", "потише": "down", "сделай тише": "down", "сделай потише": "down", "какая громкость": "get"}
            if t in media:
                tool, args = "media_control", {"action": media[t]}
            elif t in volume:
                tool, args = "volume", {"action": volume[t]}
        if tool is None or tool not in self.tools.names:
            return None
        self.used_tool = True
        return (await self.tools.call(tool, json.dumps(args, ensure_ascii=False))).text
