"""Entry point: `uv run python -m voice_assistant` (voice) or `--text` (keyboard)."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import logging
import os
import sys
import time
from zoneinfo import ZoneInfo

import httpx

from .assistant import Assistant, TurnStats
from .audio import Microphone, Player, SpeechSegmenter, WavMicrophone, chime
from .config import Config, load_config
from .indicator import Indicator
from .llm import ContextOverflow, LlmClient
from .memory import Memory
from .scheduler import AlarmScheduler
from .speech import SilentSpeaker, Speaker
from .tools import ToolContext, build_tools
from .tts import build_tts
from .wake import find_wake_word

log = logging.getLogger("va")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="voice_assistant", description=__doc__)
    parser.add_argument("--text", action="store_true", help="type instead of speaking into the microphone")
    parser.add_argument("--once", metavar="TEXT", help="answer one request and exit")
    parser.add_argument("--mute", action="store_true", help="no voice output, text only")
    parser.add_argument("--wav", metavar="FILE", help="use a recording instead of the microphone (testing)")
    parser.add_argument("--tts", choices=["edge", "silero"], help="override TTS_BACKEND")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args()


def print_turn(answer: str, stats: TurnStats, stt_s: float | None, first_audio_s: float | None) -> None:
    print(f"AI: {answer}")
    timings = []
    if stt_s is not None:
        timings.append(f"stt {stt_s:.2f}s")
    if stats.llm_first_text_s is not None:
        timings.append(f"llm first text {stats.llm_first_text_s:.2f}s")
    if first_audio_s is not None:
        timings.append(f"first audio {first_audio_s:.2f}s")
    timings.append(f"tools {stats.tool_calls}")
    timings.append(f"llm total {stats.total_s:.2f}s")
    print("   (" + ", ".join(timings) + ")\n", flush=True)


async def run(cfg: Config, args: argparse.Namespace) -> None:
    tz = ZoneInfo(cfg.timezone)
    memory = Memory(cfg.db_path)
    http = httpx.AsyncClient(timeout=15, follow_redirects=True)
    mic: Microphone | WavMicrophone | None = None
    segmenter: SpeechSegmenter | None = None
    busy = False  # transcribing or answering: the microphone is off
    thinking = False  # answering
    hearing = False  # someone speaks, or a phrase is being transcribed
    mic_off = False  # muted in the system mixer
    listen_until = 0.0  # follow-up window: phrases before this need no wake word
    indicator = Indicator(cfg.indicator)

    def refresh_indicator() -> None:
        if mic_off:
            state = "muted"
        elif speaker.speaking:
            state = "speaking"
        elif thinking:
            state = "thinking"
        elif time.monotonic() < listen_until or (mic is not None and not cfg.wake_word):
            state = "attentive"
        elif hearing:
            state = "hearing"
        else:
            state = "idle"
        indicator.set(state, listen_until if state == "attentive" and cfg.wake_word else 0.0, " · ".join(sorted(ctx.activity)))

    async def watch_indicator() -> None:
        nonlocal mic_off
        checked = 0.0
        while True:
            if isinstance(mic, Microphone) and not isinstance(mic, WavMicrophone) and time.monotonic() - checked > 3:
                checked = time.monotonic()
                mic_off = await source_muted()
            refresh_indicator()
            await asyncio.sleep(0.2)

    def update_mic(_: bool = False) -> None:
        refresh_indicator()
        if mic is None:
            return
        muted = busy or speaker.speaking
        if mic.muted and not muted and segmenter is not None:
            segmenter.reset()
        mic.set_muted(muted)

    if args.mute:
        speaker: Speaker | SilentSpeaker = SilentSpeaker()
    else:
        import torch

        torch.set_num_threads(cfg.tts.torch_threads)
        tts, silero = build_tts(cfg.tts)
        speaker = Speaker(tts, Player(cfg.audio.output_device), on_speaking=update_mic)
        # Warm up Silero in the background so a fallback never waits for model loading.
        asyncio.create_task(asyncio.to_thread(silero.load))
        log.info("TTS: %s", tts.name)
    speaker.start()

    def announce(text: str) -> None:
        nonlocal listen_until
        print(f"\n🔔 {text}", flush=True)
        assistant.add_announcement(text)
        speaker.play_sound(chime(), 24000)
        speaker.say(text)
        listen_until = time.monotonic() + cfg.wake_followup_s + len(text) * 0.07  # roughly until it finished talking

    ctx = ToolContext(memory, http, tz, cfg.default_city, announce=announce)
    tools = build_tools(ctx)
    assistant = Assistant(
        LlmClient(cfg.llm), tools, memory, speaker, tz,
        default_city=cfg.default_city,
        history_messages=cfg.history_messages,
        max_tool_steps=cfg.llm.max_tool_steps,
        pending=ctx.pending,
    )
    alarms = asyncio.create_task(AlarmScheduler(memory, speaker, tz).run())
    if not args.once:
        asyncio.create_task(assistant.warm_up())
    log.info("LLM: %s at %s; tools: %s", cfg.llm.model, cfg.llm.base_url, ", ".join(tools.names))

    if not args.once and cfg.indicator != "off":
        indicator.start()
        asyncio.create_task(watch_indicator())

    async def respond(text: str, heard_at: float, stt_s: float | None) -> None:
        nonlocal thinking
        speaker.first_audio_at = None
        thinking = True
        refresh_indicator()
        try:
            answer, stats = await assistant.handle(text)
        except Exception as error:
            import openai

            log.error("LLM error: %s: %s", type(error).__name__, error)
            if isinstance(error, openai.APIConnectionError):
                speaker.say("Не могу связаться с языковой моделью. Проверь, что LM Studio запущена.")
            elif isinstance(error, ContextOverflow):
                speaker.say("Запрос не влез в память модели, попробуй сформулировать короче.")
            else:
                speaker.say("Модель ответила ошибкой, подробности в журнале.")
            await speaker.wait_done()
            return
        finally:
            thinking = False
        await speaker.wait_done()
        first_audio = speaker.first_audio_at - heard_at if speaker.first_audio_at else None
        print_turn(answer, stats, stt_s, first_audio)

    try:
        if args.once:
            print(f"Ты: {args.once}")
            await respond(args.once, time.monotonic(), None)
        elif args.text:
            print("Текстовый режим. Пустая строка или Ctrl+D — выход, /new — новый разговор.\n")
            while True:
                try:
                    line = (await asyncio.to_thread(input, "Ты: ")).strip()
                except EOFError:
                    break
                if not line:
                    break
                if line == "/new":
                    assistant.reset_history()
                    continue
                await respond(line, time.monotonic(), None)
        else:
            from .stt import build_transcriber

            print("Загружаю распознавание речи…", flush=True)
            transcriber = build_transcriber(cfg.stt)
            segmenter = SpeechSegmenter(cfg.audio)
            mic = WavMicrophone(args.wav) if args.wav else Microphone(cfg.audio.input_device)
            mic.start()
            if cfg.wake_word:
                print(f"Слушаю. Обращайся по имени: «{cfg.wake_word.capitalize()}, …»; Ctrl+C — выход.\n", flush=True)
            else:
                print("Слушаю. Говори; Ctrl+C — выход.\n", flush=True)
            while True:
                try:
                    frame = await asyncio.wait_for(mic.queue.get(), timeout=0.5)
                except TimeoutError:
                    if isinstance(mic, WavMicrophone) and mic.finished.is_set():
                        break
                    continue
                utterance = segmenter.process(frame)
                if hearing != segmenter.active:
                    hearing = segmenter.active
                    refresh_indicator()
                if utterance is None:
                    continue
                heard_at = time.monotonic()
                busy = hearing = True
                update_mic()
                try:
                    text = await asyncio.to_thread(transcriber.transcribe, utterance)
                    stt_s = time.monotonic() - heard_at
                    if len(text) < cfg.stt.min_chars:
                        log.debug("ignored %r (%.1fs of audio)", text, len(utterance) / 16000)
                        continue
                    request = text
                    if cfg.wake_word and heard_at > listen_until:
                        request = find_wake_word(text, cfg.wake_word)
                        if request is None:
                            log.debug("not addressed: %r", text)
                            continue
                        if len(request) < cfg.stt.min_chars:  # just the name: chime and wait for the request
                            print(f"Ты: {text}", flush=True)
                            listen_until = time.monotonic() + cfg.wake_followup_s
                            speaker.play_sound(chime(), 24000)
                            await speaker.wait_done()
                            listen_until = time.monotonic() + cfg.wake_followup_s
                            continue
                    elif cfg.wake_word:
                        request = find_wake_word(text, cfg.wake_word) or text  # the name is fine here too
                    print(f"Ты: {text}", flush=True)
                    hearing = False
                    await respond(request, heard_at, stt_s)
                    listen_until = time.monotonic() + cfg.wake_followup_s
                finally:
                    busy = hearing = False
                    update_mic()
        await assistant.drain()
    finally:
        alarms.cancel()
        indicator.close()
        await speaker.close()
        if mic:
            mic.close()
        await http.aclose()
        memory.close()


async def source_muted() -> bool:
    """The default microphone is muted in PipeWire (the app would hear nothing)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        out, _ = await proc.communicate()
    except OSError:
        return False
    return b"MUTED" in out


_lock_file = None


def single_instance(path) -> bool:
    """Two copies would both answer, hear each other and share one LLM context window."""
    global _lock_file
    path.parent.mkdir(parents=True, exist_ok=True)
    _lock_file = open(path, "a+")
    try:
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _lock_file.seek(0)
        print(f"Нора уже запущена (pid {_lock_file.read().strip() or '?'}). Закрой ту копию или останови её: kill <pid>.")
        return False
    _lock_file.truncate(0)
    _lock_file.write(str(os.getpid()))
    _lock_file.flush()
    return True


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="   [%(name)s] %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config()
    if not args.once and not single_instance(cfg.db_path.parent / "nora.lock"):
        sys.exit(1)
    if args.tts:
        from dataclasses import replace

        cfg = replace(cfg, tts=replace(cfg.tts, backend=args.tts))
    try:
        asyncio.run(run(cfg, args))
    except KeyboardInterrupt:
        print("\nПока!")


if __name__ == "__main__":
    main()
