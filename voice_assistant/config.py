"""Settings from environment variables, with `.env` in the project root as defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from .personas import NORA, Persona, get_persona

PROJECT_DIR = Path(__file__).resolve().parent.parent
SILERO_CACHE_DIR = Path.home() / ".cache/torch/hub/snakers4_silero-models_master/src/silero/model"

AUDIO_SAMPLE_RATE = 16000  # Silero VAD and Whisper both expect 16 kHz mono


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    extra_body: dict | None
    max_tool_steps: int


@dataclass(frozen=True)
class SttConfig:
    engine: str  # gigaam | whisper
    gigaam_model_dir: Path
    whisper_model: str
    language: str
    device: str
    compute_type: str
    cpu_threads: int
    min_chars: int


@dataclass(frozen=True)
class AudioConfig:
    input_device: int | str | None
    output_device: int | str | None
    vad_threshold: float
    min_speech_ms: int
    min_silence_ms: int
    speech_pad_ms: int
    max_utterance_s: float
    mic_gain: float


@dataclass(frozen=True)
class TtsConfig:
    backend: str  # edge | silero | none
    edge_voice: str
    edge_rate: str
    edge_first_chunk_timeout_s: float
    edge_retry_after_s: float
    silero_model_path: Path
    silero_speaker: str
    silero_sample_rate: int
    torch_threads: int


@dataclass(frozen=True)
class Config:
    llm: LlmConfig
    stt: SttConfig
    audio: AudioConfig
    tts: TtsConfig
    db_path: Path
    timezone: str
    default_city: str
    history_messages: int
    wake_word: str  # empty: answer every phrase
    wake_followup_s: float  # after an answer, the next phrase needs no name for this long
    indicator: str  # on-screen status pill: bottom | top | off
    barge_in: bool  # keep listening while answering, so «Нора, стоп» interrupts
    turn_pause_s: float = 2.4
    incomplete_pause_s: float = 4.5
    aec_mode: str = "auto"
    aec_source: str = ""
    aec_sink: str = ""
    music_duck: bool = True
    music_duck_factor: float = 0.35
    barge_engine: str = "auto"
    vosk_model_dir: Path = PROJECT_DIR / "models" / "vosk-model-small-ru-0.22"
    tool_groups: str = "all"
    disabled_tools: str = ""
    persona: Persona = NORA
    thinking_cues: bool = True
    thinking_delay_s: float = 4.0
    thinking_interval_s: float = 14.0


def _str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def _device(name: str) -> int | str | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return int(value) if value.isdigit() else value


def _json(name: str) -> dict | None:
    value = os.getenv(name, "").strip()
    return json.loads(value) if value else None


def _system_timezone() -> str:
    try:
        return Path("/etc/localtime").resolve().as_posix().split("zoneinfo/", 1)[1]
    except (OSError, IndexError):
        return "UTC"


def load_config(profile: str | None = None) -> Config:
    load_dotenv(PROJECT_DIR / ".env")
    persona = get_persona(profile or _str("ASSISTANT_PROFILE", "nora"))
    jarvis = persona.key == "jarvis"
    db_path = Path(_str("JARVIS_DB_PATH", str(PROJECT_DIR / "data" / "jarvis" / "assistant.db")) if jarvis else _str("DB_PATH", str(PROJECT_DIR / "data" / "assistant.db"))).expanduser()
    silero_model = _str("SILERO_MODEL", "v5_5_ru")
    config = Config(
        llm=LlmConfig(
            base_url=_str("LM_BASE_URL", "http://localhost:1234/v1"),
            api_key=_str("LM_API_KEY", "lm-studio"),
            model=_str("LM_MODEL", "qwen3-4b-instruct-2507"),
            temperature=_float("LM_TEMPERATURE", 0.3),
            max_tokens=_int("LM_MAX_TOKENS", 600),
            extra_body=_json("LM_EXTRA_BODY"),
            max_tool_steps=_int("LM_MAX_TOOL_STEPS", 6),
        ),
        stt=SttConfig(
            engine=_str("STT_ENGINE", "gigaam").lower(),
            gigaam_model_dir=Path(_str(
                "GIGAAM_MODEL_DIR",
                str(PROJECT_DIR / "models" / "sherpa-onnx-nemo-ctc-punct-giga-am-v3-russian-2025-12-16"),
            )).expanduser(),
            whisper_model=_str("WHISPER_MODEL", "small"),
            language=_str("STT_LANGUAGE", "ru"),
            device=_str("STT_DEVICE", "cpu"),
            compute_type=_str("STT_COMPUTE_TYPE", "int8"),
            cpu_threads=_int("STT_CPU_THREADS", 4),
            min_chars=_int("STT_MIN_CHARS", 3),
        ),
        audio=AudioConfig(
            input_device=_device("AUDIO_INPUT_DEVICE"),
            output_device=_device("AUDIO_OUTPUT_DEVICE"),
            vad_threshold=_float("VAD_THRESHOLD", 0.5),
            min_speech_ms=_int("VAD_MIN_SPEECH_MS", 250),
            min_silence_ms=_int("VAD_MIN_SILENCE_MS", 1000),
            speech_pad_ms=_int("VAD_SPEECH_PAD_MS", 300),
            max_utterance_s=_float("VAD_MAX_UTTERANCE_S", 60.0),
            mic_gain=_float("MIC_GAIN", 2.0),
        ),
        tts=TtsConfig(
            backend=_str("TTS_BACKEND", "silero").lower(),
            edge_voice=_str("JARVIS_EDGE_TTS_VOICE", "ru-RU-DmitryNeural") if jarvis else _str("EDGE_TTS_VOICE", "ru-RU-SvetlanaNeural"),
            edge_rate=_str("EDGE_TTS_RATE", "+0%"),
            edge_first_chunk_timeout_s=_float("EDGE_TTS_FIRST_CHUNK_TIMEOUT_S", 2.5),
            edge_retry_after_s=_float("EDGE_TTS_RETRY_AFTER_S", 60.0),
            silero_model_path=Path(_str("SILERO_MODEL_PATH", str(SILERO_CACHE_DIR / f"{silero_model}.pt"))),
            silero_speaker=_str("JARVIS_SILERO_SPEAKER", "eugene") if jarvis else _str("SILERO_SPEAKER", "baya"),
            silero_sample_rate=_int("SILERO_SAMPLE_RATE", 24000),
            torch_threads=_int("TORCH_THREADS", 4),
        ),
        db_path=db_path,
        timezone=_str("TIMEZONE", "") or _system_timezone(),
        default_city=_str("DEFAULT_CITY", ""),
        history_messages=_int("HISTORY_MESSAGES", 10),
        wake_word=(_str("JARVIS_WAKE_WORD", persona.wake_word) if jarvis else _str("WAKE_WORD", persona.wake_word)).lower(),
        wake_followup_s=_float("WAKE_FOLLOWUP_S", 15.0),
        indicator=_str("INDICATOR", "bottom").lower(),
        barge_in=_str("BARGE_IN", "1") not in ("0", "false", "no", "off"),
        turn_pause_s=_float("TURN_PAUSE_S", 2.4),
        incomplete_pause_s=_float("TURN_INCOMPLETE_PAUSE_S", 4.5),
        aec_mode=_str("AEC_MODE", "auto").lower(),
        aec_source=_str("AEC_SOURCE", ""),
        aec_sink=_str("AEC_SINK", ""),
        music_duck=_str("MUSIC_DUCK", "1").lower() not in ("0", "false", "off"),
        music_duck_factor=_float("MUSIC_DUCK_FACTOR", 0.35),
        barge_engine=_str("BARGE_ENGINE", "auto").lower(),
        vosk_model_dir=Path(_str("VOSK_MODEL_DIR", str(PROJECT_DIR / "models" / "vosk-model-small-ru-0.22"))).expanduser(),
        tool_groups=_str("TOOL_GROUPS", "all"),
        disabled_tools=_str("DISABLED_TOOLS", ""),
        persona=persona,
        thinking_cues=_str("THINKING_CUES", "1").lower() not in ("0", "false", "no", "off"),
        thinking_delay_s=_float("THINKING_CUE_DELAY_S", 4.0),
        thinking_interval_s=_float("THINKING_CUE_INTERVAL_S", 14.0),
    )
    if config.aec_mode not in ("auto", "on", "off"):
        raise ValueError("AEC_MODE должен быть auto, on или off")
    if not (config.thinking_delay_s >= 1 and config.thinking_interval_s >= 5):
        raise ValueError("THINKING_CUE_DELAY_S должен быть >= 1, THINKING_CUE_INTERVAL_S >= 5")
    if config.barge_engine not in ("auto", "vosk", "gigaam", "off"):
        raise ValueError("BARGE_ENGINE должен быть auto, vosk, gigaam или off")
    return config
