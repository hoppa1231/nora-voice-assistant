"""Optional streaming interruption detector; GigaAM remains the final STT."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .wake import STOP_WORDS, _words, find_interruption, is_exact_name

log = logging.getLogger("va.barge")


class StreamingInterruption:
    def __init__(self, model_dir: Path, name: str) -> None:
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        if not model_dir.is_dir():
            raise RuntimeError(f"Vosk model not found: {model_dir}")
        self._recognizer = KaldiRecognizer(Model(str(model_dir)), 16000)
        self.name = name
        self._frames: list[np.ndarray] = []
        self._last_partial = ""

    def reset(self) -> None:
        self._recognizer.Reset()
        self._frames.clear()
        self._last_partial = ""

    def feed(self, frame: np.ndarray, echo: str = "") -> bool:
        self._frames.append(frame - float(frame.mean()))
        if len(self._frames) < 3:  # 96 ms; avoid running a decoder on every 32 ms frame
            return False
        pcm = np.clip(np.concatenate(self._frames) * 32767, -32768, 32767).astype("<i2").tobytes()
        self._frames.clear()
        final = self._recognizer.AcceptWaveform(pcm)
        data = json.loads(self._recognizer.Result() if final else self._recognizer.PartialResult())
        text = data.get("text" if final else "partial", "")
        if not text or text == self._last_partial:
            if final:
                self._last_partial = ""
            return False
        self._last_partial = "" if final else text
        words = _words(text)
        if not any(is_exact_name(w, self.name) or w in STOP_WORDS for w in words):
            return False  # do not fuzzy-match «нору», «пора» in partial background speech
        return find_interruption(text, self.name, echo) is not None


def build_detector(engine: str, model_dir: Path, name: str) -> StreamingInterruption | None:
    if engine in ("off", "gigaam"):
        return None
    try:
        detector = StreamingInterruption(model_dir, name)
    except (ImportError, RuntimeError) as error:
        if engine == "vosk":
            raise RuntimeError(f"Не удалось включить Vosk: {error}. Установи uv sync --extra barge-in и русскую модель.") from error
        log.info("Streaming interruption unavailable; using GigaAM: %s", error)
        return None
    log.info("Streaming Vosk interruption enabled")
    return detector


def main() -> None:
    import argparse
    import tempfile
    import urllib.request
    import zipfile
    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-model", action="store_true", required=True)
    parser.parse_args()
    target = load_config().vosk_model_dir
    if (target / "am" / "final.mdl").exists():
        print(f"Модель уже установлена: {target}")
        return
    url = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
    with tempfile.TemporaryDirectory(prefix="nora-vosk-") as temporary:
        archive = Path(temporary) / "model.zip"
        print("Скачиваю небольшую русскую модель Vosk (~45 МБ)…", flush=True)
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                path = Path(info.filename)
                if path.is_absolute() or ".." in path.parts or path.parts[0] != "vosk-model-small-ru-0.22":
                    raise RuntimeError("Небезопасный путь в архиве модели")
            bundle.extractall(temporary)
        downloaded = Path(temporary) / "vosk-model-small-ru-0.22"
        if target.exists():
            raise RuntimeError(f"Папка {target} уже существует, но модель неполная; существующие файлы не перезаписаны")
        target.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.move(str(downloaded), str(target))
    print(f"Установлено: {target}")


if __name__ == "__main__":
    main()
