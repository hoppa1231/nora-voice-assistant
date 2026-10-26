"""Microphone input, Silero VAD utterance segmentation and audio playback."""

from __future__ import annotations

import asyncio
import collections
import logging

import numpy as np
import sounddevice as sd

from .config import AUDIO_SAMPLE_RATE, AudioConfig

log = logging.getLogger("va.audio")

FRAME_SAMPLES = 512  # 32 ms at 16 kHz, the window size Silero VAD is trained on
FRAME_MS = FRAME_SAMPLES * 1000 // AUDIO_SAMPLE_RATE


class Microphone:
    """Moves PortAudio frames from its callback thread into an asyncio queue.

    While muted (the assistant is thinking or talking) frames are dropped, so it
    never transcribes its own voice.
    """

    def __init__(self, device: int | str | None) -> None:
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=400)
        self._device = device
        self._muted = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if muted:
            while not self.queue.empty():
                self.queue.get_nowait()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("input status: %s", status)
        if self._muted or self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._put, indata[:, 0].copy())

    def _put(self, frame: np.ndarray) -> None:
        if self._muted:
            return
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(frame)


class WavMicrophone(Microphone):
    """Plays a recording into the pipeline in real time, as if spoken (`--wav`, for testing).

    Unlike the real microphone it pauses instead of dropping audio while muted,
    so a file with several requests is answered one by one.
    """

    def __init__(self, path: str) -> None:
        super().__init__(None)
        self._path = path
        self.finished = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._feed())

    def close(self) -> None:
        if self._task:
            self._task.cancel()

    async def _feed(self) -> None:
        import soundfile as sf

        audio, rate = sf.read(self._path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != AUDIO_SAMPLE_RATE:
            positions = np.arange(0, len(audio), rate / AUDIO_SAMPLE_RATE)
            audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
        audio = np.concatenate([audio, np.zeros(AUDIO_SAMPLE_RATE * 2, dtype=np.float32)])
        for start in range(0, len(audio) - FRAME_SAMPLES, FRAME_SAMPLES):
            while self._muted:
                await asyncio.sleep(0.05)
            self.queue.put_nowait(audio[start : start + FRAME_SAMPLES])
            await asyncio.sleep(FRAME_SAMPLES / AUDIO_SAMPLE_RATE)
        self.finished.set()

    def set_muted(self, muted: bool) -> None:
        self._muted = muted  # keep queued frames: nothing here is the assistant's own voice


class SpeechSegmenter:
    """Cuts a stream of 32 ms frames into utterances using Silero VAD."""

    def __init__(self, config: AudioConfig) -> None:
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad()
        self._cfg = config
        self._preroll: collections.deque[np.ndarray] = collections.deque(maxlen=max(1, config.speech_pad_ms // FRAME_MS))
        self._frames: list[np.ndarray] = []
        self._active = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._dc: float | None = None

    @property
    def active(self) -> bool:
        """Someone is speaking right now (an utterance is being collected)."""
        return self._active

    def reset(self) -> None:
        self._model.reset_states()
        self._preroll.clear()
        self._frames = []
        self._active = False
        self._speech_ms = self._silence_ms = 0

    def _clean(self, frame: np.ndarray) -> np.ndarray:
        """Remove the constant offset some built-in mics add (the 8845HS digital mic sits at +0.25:
        VAD then wakes up late and clips the first syllable — «Нора» became «Ора»), then apply gain."""
        mean = float(frame.mean())
        self._dc = mean if self._dc is None else 0.995 * self._dc + 0.005 * mean  # ~6 s time constant
        frame = frame - self._dc
        if self._cfg.mic_gain != 1.0:
            frame = np.clip(frame * self._cfg.mic_gain, -1.0, 1.0)
        return frame.astype(np.float32)

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed one frame; returns the whole utterance once the speaker pauses."""
        frame = self._clean(frame)
        with self._torch.inference_mode():
            prob = self._model(self._torch.from_numpy(frame), AUDIO_SAMPLE_RATE).item()

        if not self._active:
            self._preroll.append(frame)
            if prob >= self._cfg.vad_threshold:
                self._active = True
                self._frames = list(self._preroll)
                self._speech_ms = FRAME_MS
                self._silence_ms = 0
            return None

        self._frames.append(frame)
        # Hysteresis: once speech started, a slightly lower probability still counts as speech.
        if prob >= self._cfg.vad_threshold - 0.15:
            self._speech_ms += FRAME_MS
            self._silence_ms = 0
        else:
            self._silence_ms += FRAME_MS

        too_long = len(self._frames) * FRAME_MS >= self._cfg.max_utterance_s * 1000
        if self._silence_ms < self._cfg.min_silence_ms and not too_long:
            return None

        frames, speech_ms = self._frames, self._speech_ms
        self.reset()
        if speech_ms < self._cfg.min_speech_ms:
            return None
        return np.concatenate(frames)


class Player:
    def __init__(self, device: int | str | None) -> None:
        self._device = device

    async def play(self, audio: np.ndarray, sample_rate: int) -> None:
        await asyncio.to_thread(self._play_blocking, audio, sample_rate)

    def _play_blocking(self, audio: np.ndarray, sample_rate: int) -> None:
        sd.play(audio, sample_rate, device=self._device)
        sd.wait()

    def stop(self) -> None:
        sd.stop()


def chime(sample_rate: int = 24000) -> np.ndarray:
    """Two short rising tones played before an alarm announcement."""
    parts = []
    for freq in (880.0, 1320.0):
        t = np.arange(int(sample_rate * 0.16)) / sample_rate
        tone = 0.3 * np.sin(2 * np.pi * freq * t)
        fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)
        parts += [tone * fade, np.zeros(int(sample_rate * 0.04))]
    return np.concatenate(parts).astype(np.float32)
