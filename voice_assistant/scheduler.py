"""Fires timers and reminders stored in memory, including ones missed while the app was off."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from .audio import chime
from .memory import Alarm, Memory
from .speech import SilentSpeaker, Speaker

log = logging.getLogger("va.alarms")

MISSED_AFTER = timedelta(minutes=1)


def announcement(alarm: Alarm, now: datetime, tz: ZoneInfo) -> str:
    missed = now - alarm.due_at > MISSED_AFTER
    if alarm.kind == "timer":
        text = f"Таймер: {alarm.text}." if not alarm.text.startswith("таймер") else f"{alarm.text.capitalize()} вышел."
    else:
        text = f"Напоминаю: {alarm.text}."
    if missed:
        text = f"Пропущено, было в {alarm.due_at.astimezone(tz):%H:%M}. {text}"
    return text


class AlarmScheduler:
    def __init__(self, memory: Memory, speaker: Speaker | SilentSpeaker, timezone: ZoneInfo) -> None:
        self._memory = memory
        self._speaker = speaker
        self._tz = timezone

    async def run(self) -> None:
        while True:
            now = datetime.now(UTC)
            for alarm in self._memory.due_alarms(now):
                if not self._memory.finish_alarm(alarm.id):
                    continue
                text = announcement(alarm, now, self._tz)
                print(f"\n⏰ {text}", flush=True)
                self._speaker.play_sound(chime(), 24000)
                self._speaker.say(text)
            await asyncio.sleep(1)
