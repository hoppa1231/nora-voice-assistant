"""Turns streamed LLM text into speech sentence by sentence.

Each finished sentence starts synthesizing at once, while the LLM keeps writing
the next one; playback follows in order. That is what makes the first word come
out roughly one sentence after the model starts answering, not after it ends.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable

import numpy as np

from .audio import Player
from .tts import TTS, Audio

log = logging.getLogger("va.speech")

_SENTENCE_END = re.compile(r"[.!?…]+[»\"')]*\s+|\n+")
_MARKDOWN = re.compile(r"[*_#`>|~]+")
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
_SPEAKABLE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]")


def clean_for_speech(text: str) -> str:
    text = _URL.sub("", text)
    text = _MARKDOWN.sub(" ", text)
    text = _EMOJI.sub("", text)
    text = re.sub(r"^\s*[-•]\s+", "", text)
    text = " ".join(text.split())
    return text if _SPEAKABLE.search(text) else ""


class SentenceSplitter:
    """Cuts streamed text into speakable pieces.

    The first piece of an answer is cut early, at the first comma after
    `first_min` characters: the time to first sound is what the user feels,
    and once audio plays the LLM (~60 chars/s) stays ahead of speech (~15 chars/s).
    """

    def __init__(self, soft_limit: int = 180, first_min: int = 40) -> None:
        self._buffer = ""
        self._soft_limit = soft_limit
        self._first_min = first_min
        self._first = True

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        sentences = []
        while True:
            match = _SENTENCE_END.search(self._buffer)
            if match:
                sentences.append(self._buffer[: match.end()])
                self._buffer = self._buffer[match.end() :]
                continue
            if self._first and len(self._buffer) > self._first_min:
                cut = self._buffer.find(", ", self._first_min)
                if cut < 0 and len(self._buffer) > 2 * self._first_min:
                    cut = self._buffer.rfind(" ")  # no comma coming soon: any word boundary will do
                if cut > 0:
                    sentences.append(self._buffer[: cut + 1])
                    self._buffer = self._buffer[cut + 1 :]
                    continue
            if len(self._buffer) > self._soft_limit:
                # A long run-on sentence: cut at the last comma (or space) to keep latency low.
                cut = max(self._buffer.rfind(", "), self._buffer.rfind("; "))
                if cut < self._soft_limit // 3:
                    cut = self._buffer.rfind(" ")
                if cut > 0:
                    sentences.append(self._buffer[: cut + 1])
                    self._buffer = self._buffer[cut + 1 :]
                    continue
            break
        pieces = [s.strip() for s in sentences if s.strip()]
        if pieces:
            self._first = False
        return pieces

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer.strip(), ""
        self._first = True
        return [rest] if rest else []


class Speaker:
    """Queue of things to say; synthesis runs ahead of playback, up to `parallel` items."""

    def __init__(
        self,
        tts: TTS,
        player: Player,
        on_speaking: Callable[[bool], None] | None = None,
        parallel: int = 2,
    ) -> None:
        self._tts = tts
        self._player = player
        self._on_speaking = on_speaking
        self._splitter = SentenceSplitter()
        self._queue: asyncio.Queue[asyncio.Task[Audio | None]] = asyncio.Queue()
        self._slots = asyncio.Semaphore(parallel)
        self._idle = asyncio.Event()
        self._idle.set()
        self._worker: asyncio.Task | None = None
        self.first_audio_at: float | None = None

    @property
    def speaking(self) -> bool:
        return not self._idle.is_set()

    def start(self) -> None:
        self._worker = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._worker:
            self._worker.cancel()

    def feed(self, text: str) -> None:
        for sentence in self._splitter.feed(text):
            self._enqueue_text(sentence)

    def end_of_text(self) -> None:
        for sentence in self._splitter.flush():
            self._enqueue_text(sentence)

    def say(self, text: str) -> None:
        self.feed(text)
        self.end_of_text()

    def play_sound(self, audio: np.ndarray, sample_rate: int) -> None:
        future: asyncio.Future[Audio | None] = asyncio.get_running_loop().create_future()
        future.set_result((audio, sample_rate))
        self._enqueue(future)

    async def wait_done(self) -> None:
        await self._idle.wait()

    def _enqueue_text(self, sentence: str) -> None:
        sentence = clean_for_speech(sentence)
        if sentence:
            self._enqueue(asyncio.create_task(self._synthesize(sentence)))

    def _enqueue(self, item: asyncio.Future[Audio | None]) -> None:
        if self._idle.is_set():
            self._idle.clear()
            if self._on_speaking:
                self._on_speaking(True)
        self._queue.put_nowait(item)

    async def _synthesize(self, sentence: str) -> Audio | None:
        async with self._slots:
            started = time.perf_counter()
            try:
                audio = await self._tts.synthesize(sentence)
            except Exception:
                log.exception("TTS failed for %r", sentence)
                return None
            log.debug("tts %.2fs for %r", time.perf_counter() - started, sentence)
            return audio

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                audio = await item
                if audio is not None:
                    if self.first_audio_at is None:
                        self.first_audio_at = time.monotonic()
                    await self._player.play(*audio)
            except Exception:
                log.exception("playback failed")
            finally:
                if self._queue.empty():
                    self._idle.set()
                    if self._on_speaking:
                        self._on_speaking(False)


class SilentSpeaker:
    """Stand-in for Speaker when voice output is off (`--mute`)."""

    speaking = False
    first_audio_at: float | None = None

    def start(self) -> None: ...
    async def close(self) -> None: ...
    def feed(self, text: str) -> None: ...
    def end_of_text(self) -> None: ...
    def say(self, text: str) -> None: ...
    def play_sound(self, audio: np.ndarray, sample_rate: int) -> None: ...
    async def wait_done(self) -> None: ...
