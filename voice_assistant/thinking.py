"""Sparse waiting cues, independent of LLM output and conversation history."""
import asyncio
import contextlib
import random
import logging

log = logging.getLogger("va.thinking")


PHRASES = (
    "Секунду.", "Так, момент.", "Сейчас, дай подумать.",
    "Ладно, дай подумать.", "Ещё немного.", "Думаю над ответом.", "Момент.",
)


class ThinkingCues:
    def __init__(self, speaker, quiet, *, enabled=True, delay=4.0, interval=14.0, maximum=3):
        self.speaker, self.quiet = speaker, quiet
        self.enabled, self.delay, self.interval, self.maximum = enabled, delay, interval, maximum
        self._bag = []
        self._last = None

    def phrase(self):
        if not self._bag:
            self._bag = list(PHRASES)
            random.shuffle(self._bag)
            if self._bag[-1] == self._last:
                self._bag[0], self._bag[-1] = self._bag[-1], self._bag[0]
        self._last = self._bag.pop()
        return self._last

    async def _run(self):
        await asyncio.sleep(self.delay)
        count = 0
        while count < self.maximum:
            if not self.quiet.active and not self.speaker.speaking:
                try:
                    if self.speaker.say_waiting(self.phrase()):
                        count += 1
                except Exception:
                    log.exception("waiting cue failed; continuing without cues")
                    return
            await asyncio.sleep(self.interval)

    async def wait(self, operation):
        if not self.enabled:
            return await operation
        cues = asyncio.create_task(self._run())
        try:
            return await operation
        finally:
            cues.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cues
            self.speaker.cancel_waiting()
