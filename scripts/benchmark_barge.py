"""Reproducible synthetic Vosk/GigaAM interruption comparison (no microphone)."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from voice_assistant.audio import FRAME_SAMPLES, SpeechSegmenter
from voice_assistant.barge import StreamingInterruption
from voice_assistant.config import load_config
from voice_assistant.stt import build_transcriber
from voice_assistant.tts import SileroTTS
from voice_assistant.wake import find_interruption

CASES = [
    ("Нора стоп.", True, ""),
    ("Стоп.", True, ""),
    ("Нора поставь таймер.", True, ""),
    ("Сегодня в Петербурге солнечно.", False, ""),
    ("Пора спать.", False, ""),
    ("В лесу я увидел нору лисы.", False, ""),
    ("Меня зовут Нора.", False, "Меня зовут Нора"),
    ("Скажи стоп если надо.", False, "Скажи стоп если надо"),
]


async def main():
    cfg = load_config()
    torch.set_num_threads(4)
    stt = build_transcriber(cfg.stt)
    detector = StreamingInterruption(cfg.vosk_model_dir, cfg.wake_word or "нора")
    rows = []
    for voice in ("baya", "kseniya"):
        tts = SileroTTS(cfg.tts.silero_model_path, voice, 24000)
        for text, positive, echo in CASES:
            audio, rate = await tts.synthesize(text)
            pcm = np.interp(np.arange(0, len(audio), rate / 16000), np.arange(len(audio)), audio).astype(np.float32)
            pcm = np.concatenate([pcm, np.zeros(48000, dtype=np.float32)])
            segmenter = SpeechSegmenter(replace(cfg.audio, mic_gain=1.0))
            segmenter.max_utterance_s = 1.2
            detector.reset()
            found = {"vosk": None, "gigaam": None}
            compute = {"vosk": 0.0, "gigaam": 0.0}
            for start in range(0, len(pcm) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
                frame = pcm[start:start + FRAME_SAMPLES]
                audio_s = (start + FRAME_SAMPLES) / 16000
                if found["vosk"] is None:
                    began = time.perf_counter()
                    hit = detector.feed(frame, echo)
                    compute["vosk"] += time.perf_counter() - began
                    if hit:
                        found["vosk"] = round(audio_s, 3)
                if found["gigaam"] is None:
                    began = time.perf_counter()
                    utterance = segmenter.process(frame)
                    hit = utterance is not None and find_interruption(stt.transcribe(utterance), cfg.wake_word or "нора", echo) is not None
                    compute["gigaam"] += time.perf_counter() - began
                    if hit:
                        found["gigaam"] = round(audio_s, 3)
            row = {"voice": voice, "text": text, "expected_interrupt": positive, "detected_audio_s": found, "compute_s": {k: round(v, 4) for k, v in compute.items()}}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/nora-barge-benchmark.json")
    output.write_text(json.dumps({"note": "Synthetic voices, not live microphone or acoustic AEC quality. Times are positions within the input clip.", "rows": rows}, ensure_ascii=False, indent=2))
    print(f"Saved {output}", flush=True)


asyncio.run(main())
