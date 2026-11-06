"""Text-to-speech backends: Edge (online) with automatic fallback to local Silero."""

from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from .config import TtsConfig

log = logging.getLogger("va.tts")

Audio = tuple[np.ndarray, int]


class TTS(Protocol):
    name: str

    async def synthesize(self, text: str) -> Audio: ...


class TtsError(RuntimeError):
    pass


class EdgeTTS:
    """Microsoft Edge read-aloud voices. Free and natural, but needs internet."""

    name = "edge"

    def __init__(self, voice: str, rate: str, first_chunk_timeout_s: float) -> None:
        self._voice = voice
        self._rate = rate
        self._first_chunk_timeout_s = first_chunk_timeout_s

    async def synthesize(self, text: str) -> Audio:
        import edge_tts

        mp3 = bytearray()
        loop = asyncio.get_running_loop()
        # Short deadline for the first chunk (slow or offline -> fall back fast), then
        # one for the rest of the sentence: the service sometimes stalls mid-stream.
        async with asyncio.timeout(self._first_chunk_timeout_s) as deadline:
            async for chunk in edge_tts.Communicate(text, self._voice, rate=self._rate).stream():
                if chunk["type"] == "audio":
                    if not mp3:
                        deadline.reschedule(loop.time() + 3 + len(text) * 0.03)
                    mp3 += chunk["data"]
        if not mp3:
            raise TtsError("edge returned no audio")
        audio, sample_rate = sf.read(io.BytesIO(bytes(mp3)), dtype="float32")
        return audio, sample_rate


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_UNITS = (
    (r"°\s*[CС]|°", ("градус", "градуса", "градусов")),
    (r"%", ("процент", "процента", "процентов")),
    (r"м/с", ("метр в секунду", "метра в секунду", "метров в секунду")),
    (r"км/ч", ("километр в час", "километра в час", "километров в час")),
)


def _plural_form(number: str, forms: tuple[str, str, str]) -> str:
    if "." in number or "," in number:
        return forms[1]  # "2,5 градуса"
    n = int(number)
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def normalize_for_silero(text: str) -> str:
    """Silero reads only Cyrillic words, so spell out numbers and common units."""
    from num2words import num2words

    for pattern, forms in _UNITS:
        text = re.sub(
            r"(\d+(?:[.,]\d+)?)\s*(?:" + pattern + ")",
            lambda m, forms=forms: f"{m.group(1)} {_plural_form(m.group(1), forms)}",
            text,
        )
    text = re.sub(r"\+(?=\d)", "плюс ", text).replace("−", "минус ")  # a "+" before a vowel marks stress
    text = re.sub(r"\bНор(?=[аеиуыо]\b)", "Н+ор", text)  # the assistant's name: НОра
    text = re.sub(r"\b[Дд]жарви[зс]\b", lambda m: m[0].replace("а", "+а", 1), text)
    text = re.sub(r"(\d):(\d\d)", r"\1 \2", text)

    def spell(match: re.Match[str]) -> str:
        number = match.group().replace(",", ".")
        try:
            return num2words(float(number) if "." in number else int(number), lang="ru")
        except (ValueError, OverflowError, NotImplementedError):
            return number

    return _NUMBER.sub(spell, text)


class SileroTTS:
    """Local Silero model loaded straight from its torch.package file."""

    name = "silero"
    MODEL_URL = "https://models.silero.ai/models/tts/ru/{name}"

    def __init__(self, model_path: Path, speaker: str, sample_rate: int) -> None:
        self._model_path = model_path
        self._speaker = speaker
        self._sample_rate = sample_rate
        self._model = None
        self._lock = threading.Lock()  # one synthesis at a time; the model is not thread-safe

    def load(self) -> None:
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        import torch

        if not self._model_path.exists():
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            url = self.MODEL_URL.format(name=self._model_path.name)
            log.info("downloading Silero model %s", url)
            torch.hub.download_url_to_file(url, str(self._model_path))
        started = time.perf_counter()
        importer = torch.package.PackageImporter(str(self._model_path))
        model = importer.load_pickle("tts_models", "model")
        model.to(torch.device("cpu"))
        with torch.inference_mode():  # the first synthesis is ~1 s slower; pay it now
            model.apply_tts(text="Привет.", speaker=self._speaker, sample_rate=self._sample_rate)
        self._model = model
        log.info("Silero %s loaded in %.1fs", self._model_path.stem, time.perf_counter() - started)

    async def synthesize(self, text: str) -> Audio:
        return await asyncio.to_thread(self._synthesize_blocking, text)

    def _synthesize_blocking(self, text: str) -> Audio:
        import torch

        with self._lock:
            self._load_locked()
            with torch.inference_mode():
                audio = self._model.apply_tts(
                    text=normalize_for_silero(text),
                    speaker=self._speaker,
                    sample_rate=self._sample_rate,
                    put_accent=True,
                    put_yo=True,
                )
        return audio.numpy().astype(np.float32), self._sample_rate


class FallbackTTS:
    """Uses the primary backend until it fails or is too slow, then the fallback.

    After a failure the primary is skipped for `retry_after_s`, so an outage
    costs one slow sentence instead of one per sentence.
    """

    def __init__(self, primary: TTS, fallback: TTS, retry_after_s: float) -> None:
        self.name = f"{primary.name}->{fallback.name}"
        self._primary = primary
        self._fallback = fallback
        self._retry_after_s = retry_after_s
        self._primary_blocked_until = 0.0

    async def synthesize(self, text: str) -> Audio:
        if time.monotonic() >= self._primary_blocked_until:
            try:
                return await self._primary.synthesize(text)
            except Exception as error:
                reason = "timeout" if isinstance(error, TimeoutError) else f"{type(error).__name__}: {error}"
                log.warning("%s failed (%s); using %s for %.0fs", self._primary.name, reason, self._fallback.name, self._retry_after_s)
                self._primary_blocked_until = time.monotonic() + self._retry_after_s
        return await self._fallback.synthesize(text)


def build_tts(config: TtsConfig) -> tuple[TTS, SileroTTS]:
    """Returns the backend to speak with and the Silero instance (to warm it up early)."""
    silero = SileroTTS(config.silero_model_path, config.silero_speaker, config.silero_sample_rate)
    if config.backend == "silero":
        return silero, silero
    if config.backend != "edge":
        raise ValueError(f"unknown TTS_BACKEND: {config.backend!r} (expected edge or silero)")
    edge = EdgeTTS(config.edge_voice, config.edge_rate, config.edge_first_chunk_timeout_s)
    return FallbackTTS(edge, silero, config.edge_retry_after_s), silero
