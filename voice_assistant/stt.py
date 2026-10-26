"""Speech-to-text: GigaAM v3 via sherpa-onnx (default) or faster-whisper.

On this CPU GigaAM transcribes a 3 s phrase in ~0.08 s with better Russian
accuracy; Whisper small needs ~2.5 s.
"""

from __future__ import annotations

import logging
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import AUDIO_SAMPLE_RATE, SttConfig

log = logging.getLogger("va.stt")

# Phrases Whisper invents on silence or noise, learned from subtitled YouTube videos.
HALLUCINATIONS = (
    "субтитры",
    "продолжение следует",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "dimatorzok",
    "редактор субтитров",
)

GIGAAM_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{name}.tar.bz2"


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class GigaAmTranscriber:
    def __init__(self, model_dir: Path, threads: int) -> None:
        import sherpa_onnx

        if not (model_dir / "model.int8.onnx").exists():
            _download_model(model_dir)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=threads,
            provider="cpu",
        )

    def transcribe(self, audio: np.ndarray) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(AUDIO_SAMPLE_RATE, audio)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()


class WhisperTranscriber:
    def __init__(self, config: SttConfig) -> None:
        from faster_whisper import WhisperModel

        self._language = config.language
        self._model = WhisperModel(
            config.whisper_model,
            device=config.device,
            compute_type=config.compute_type,
            cpu_threads=config.cpu_threads,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,
            temperature=0,
            vad_filter=False,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if any(phrase in text.lower() for phrase in HALLUCINATIONS):
            return ""
        return text


def build_transcriber(config: SttConfig) -> Transcriber:
    if config.engine == "gigaam":
        return GigaAmTranscriber(config.gigaam_model_dir, config.cpu_threads)
    if config.engine == "whisper":
        return WhisperTranscriber(config)
    raise ValueError(f"unknown STT_ENGINE: {config.engine!r} (expected gigaam or whisper)")


def _download_model(model_dir: Path) -> None:
    url = GIGAAM_RELEASE.format(name=model_dir.name)
    log.info("downloading %s (~160 MB)", url)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.bz2", dir=model_dir.parent) as archive:
        urllib.request.urlretrieve(url, archive.name)
        with tarfile.open(archive.name) as tar:
            tar.extractall(model_dir.parent, filter="data")
