"""Microphone input, Silero VAD utterance segmentation and audio playback."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import threading

import numpy as np
import sounddevice as sd

from .config import AUDIO_SAMPLE_RATE, AudioConfig

log = logging.getLogger("va.audio")

FRAME_SAMPLES = 512  # 32 ms at 16 kHz, the window size Silero VAD is trained on
FRAME_MS = FRAME_SAMPLES * 1000 // AUDIO_SAMPLE_RATE


class AudioInputError(RuntimeError):
    """An actionable audio startup failure, not a model/LLM failure."""


class StreamingResampler:
    """Continuous mono resampling with an anti-alias FIR and preserved sample phase."""

    def __init__(self, input_rate: int, output_rate: int = AUDIO_SAMPLE_RATE) -> None:
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.ratio = input_rate / output_rate
        self.bypass = input_rate == output_rate
        taps = 127
        cutoff = 0.45 * min(1.0, output_rate / input_rate)
        offsets = np.arange(taps) - (taps - 1) / 2
        kernel = 2 * cutoff * np.sinc(2 * cutoff * offsets) * np.hamming(taps)
        self.kernel = (kernel / kernel.sum()).astype(np.float32)
        self.reset()

    def reset(self) -> None:
        self._tail = np.zeros(len(self.kernel) - 1, dtype=np.float32)
        self._previous = 0.0
        self._inputs = self._outputs = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        if self.bypass or not len(samples):
            return np.asarray(samples, dtype=np.float32)
        combined = np.concatenate([self._tail, samples])
        filtered = np.convolve(combined, self.kernel, mode="valid")
        self._tail = combined[-len(self._tail):].copy()
        last_index = self._inputs + len(samples) - 1
        count = max(0, int(np.floor(last_index / self.ratio)) + 1 - self._outputs)
        positions = (self._outputs + np.arange(count)) * self.ratio
        result = np.interp(
            positions,
            np.arange(self._inputs - 1, last_index + 1),
            np.concatenate([[self._previous], filtered]),
        ).astype(np.float32)
        self._previous = float(filtered[-1])
        self._inputs += len(samples)
        self._outputs += count
        return result


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
        self._resampler = StreamingResampler(AUDIO_SAMPLE_RATE)
        self._pending = np.empty(0, dtype=np.float32)
        self._reset_capture = False

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        rates = [AUDIO_SAMPLE_RATE]
        try:
            for rate in rates:
                self._resampler = StreamingResampler(rate)
                try:
                    sd.check_input_settings(device=self._device, channels=1, dtype="float32", samplerate=rate)
                    self._stream = sd.InputStream(
                        samplerate=rate, channels=1, dtype="float32",
                        blocksize=round(FRAME_SAMPLES * rate / AUDIO_SAMPLE_RATE),
                        device=self._device, callback=self._callback,
                    )
                    self._stream.start()
                    if rate != AUDIO_SAMPLE_RATE:
                        log.info("Microphone uses native %d Hz; converting continuously to %d Hz", rate, AUDIO_SAMPLE_RATE)
                    return
                except sd.PortAudioError as error:
                    if self._stream is not None:
                        self._stream.close()
                        self._stream = None
                    if -9997 in error.args and len(rates) == 1:
                        native = round(sd.query_devices(self._device, "input")["default_samplerate"])
                        if native != AUDIO_SAMPLE_RATE:
                            rates.append(native)
                            continue
                    raise
        except sd.PortAudioError as error:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            raise AudioInputError(
                f"Не удалось открыть микрофон: {error}. Проверь PipeWire: "
                "systemctl --user status pipewire pipewire-pulse wireplumber. "
                "Если user bus недоступна, сначала восстанови пользовательский менеджер systemd "
                "или выйди и заново войди в сеанс."
            ) from error

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if muted:
            self._reset_capture = True
            while not self.queue.empty():
                self.queue.get_nowait()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("input status: %s", status)
        if self._muted or self._loop is None:
            return
        if self._reset_capture:
            self._resampler.reset()
            self._pending = np.empty(0, dtype=np.float32)
            self._reset_capture = False
        samples = self._resampler.process(indata[:, 0])
        self._pending = np.concatenate([self._pending, samples])
        while len(self._pending) >= FRAME_SAMPLES:
            frame = self._pending[:FRAME_SAMPLES].copy()
            self._pending = self._pending[FRAME_SAMPLES:]
            self._loop.call_soon_threadsafe(self._put, frame)

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


class PulseMicrophone(Microphone):
    """Capture a named Pulse/PipeWire source without changing global defaults."""

    def __init__(self, source: str) -> None:
        super().__init__(None)
        self.source = source
        self.failure: Exception | None = None
        self._task: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._capture())

    async def _capture(self) -> None:
        try:
            self._proc = await asyncio.create_subprocess_exec(
                "parec", f"--device={self.source}", "--raw", "--format=float32le",
                "--rate=16000", "--channels=1", "--latency-msec=32",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            while True:
                data = await self._proc.stdout.readexactly(FRAME_SAMPLES * 4)
                self._put(np.frombuffer(data, dtype="<f4").copy())
        except (OSError, asyncio.IncompleteReadError) as error:
            self.failure = RuntimeError(f"AEC microphone stopped: {error}")

    def close(self) -> None:
        if self._task:
            self._task.cancel()
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()

    async def aclose(self) -> None:
        self.close()
        if self._task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._proc:
            await self._proc.wait()


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
        self.max_utterance_s: float | None = None  # override, e.g. short chunks while Nora talks
        self.last_silence_s = 0.0

    @property
    def active(self) -> bool:
        """Someone is speaking right now (an utterance is being collected)."""
        return self._active

    @property
    def speaking(self) -> bool:
        """Actual speech, excluding the silence being collected at its end."""
        return self._active and self._silence_ms < 160

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

        too_long = len(self._frames) * FRAME_MS >= (self.max_utterance_s or self._cfg.max_utterance_s) * 1000
        if self._silence_ms < self._cfg.min_silence_ms and not too_long:
            return None

        frames, speech_ms = self._frames, self._speech_ms
        self.last_silence_s = self._silence_ms / 1000
        self.reset()
        if speech_ms < self._cfg.min_speech_ms:
            return None
        return np.concatenate(frames)


class Player:
    """Plays through its own output stream, written in short blocks, so stop() from another
    thread is safe and takes effect within a block. (sd.play/sd.stop share global state:
    stopping while a play starts crashed or hung the process — she «talked» in silence.)"""

    BLOCK_S = 0.05

    def __init__(self, device: int | str | None) -> None:
        self._device = device
        self._stop = threading.Event()

    async def play(self, audio: np.ndarray, sample_rate: int) -> None:
        self._stop.clear()
        # The thread cannot be killed, but a stuck device must not freeze the speech queue.
        timeout = len(audio) / sample_rate + 5
        try:
            await asyncio.wait_for(asyncio.to_thread(self._play_blocking, audio, sample_rate), timeout)
        except TimeoutError:
            self._stop.set()
            log.error("playback stuck for %.0fs, skipped", timeout)

    def _play_blocking(self, audio: np.ndarray, sample_rate: int) -> None:
        block = max(1, int(sample_rate * self.BLOCK_S))
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1, 1)
        with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32", device=self._device) as stream:
            for start in range(0, len(audio), block):
                if self._stop.is_set():
                    stream.abort()
                    return
                stream.write(audio[start : start + block])

    def stop(self) -> None:
        self._stop.set()


class PulsePlayer:
    """Play through the AEC reference sink; that same signal drives the speakers."""

    def __init__(self, sink: str) -> None:
        self.sink = sink
        self._proc: asyncio.subprocess.Process | None = None
        self._generation = 0

    async def play(self, audio: np.ndarray, sample_rate: int) -> None:
        generation = self._generation
        proc = await asyncio.create_subprocess_exec(
            "pacat", "--playback", f"--device={self.sink}", "--raw", "--format=float32le",
            f"--rate={sample_rate}", "--channels=1", "--latency-msec=50",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._proc = proc
        try:
            if generation != self._generation:
                proc.terminate()
                return
            await asyncio.wait_for(proc.communicate(np.asarray(audio, dtype="<f4").tobytes()), len(audio) / sample_rate + 5)
            if proc.returncode and generation == self._generation:
                raise RuntimeError(f"AEC playback failed: {proc.returncode}")
        except (BrokenPipeError, ConnectionResetError):
            if generation == self._generation:
                raise
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
            self._proc = None

    def stop(self) -> None:
        self._generation += 1
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()


def chime(sample_rate: int = 24000) -> np.ndarray:
    """Two short rising tones played before an alarm announcement."""
    parts = []
    for freq in (880.0, 1320.0):
        t = np.arange(int(sample_rate * 0.16)) / sample_rate
        tone = 0.3 * np.sin(2 * np.pi * freq * t)
        fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)
        parts += [tone * fade, np.zeros(int(sample_rate * 0.04))]
    return np.concatenate(parts).astype(np.float32)
