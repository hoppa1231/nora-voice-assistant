"""Entry point: `uv run python -m voice_assistant` (voice) or `--text` (keyboard)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import logging
import os
import signal
import sys
import time
from zoneinfo import ZoneInfo

import httpx

from .assistant import Assistant, TurnStats
from .audio import AudioInputError, Microphone, Player, PulseMicrophone, PulsePlayer, SpeechSegmenter, WavMicrophone, chime
from .audio_system import EchoCancellation, MusicDucker
from .config import PROJECT_DIR, Config, load_config
from .indicator import Indicator
from .listening import parse_command
from .llm import ContextOverflow, LlmClient
from .memory import Memory
from .scheduler import AlarmScheduler
from .speech import SilentSpeaker, Speaker
from .tools import ToolContext, build_tools
from .tts import build_tts
from .turns import TurnCollector
from .wake import find_interruption, find_wake_word

log = logging.getLogger("va")

BARGE_CHUNK_S = 1.2  # cap detection delay while she talks; ordinary turns use full segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="voice_assistant", description=__doc__)
    parser.add_argument("--text", action="store_true", help="type instead of speaking into the microphone")
    parser.add_argument("--once", metavar="TEXT", help="answer one request and exit")
    parser.add_argument("--mute", action="store_true", help="no voice output, text only")
    parser.add_argument("--wav", metavar="FILE", help="use a recording instead of the microphone (testing)")
    parser.add_argument("--tts", choices=["edge", "silero"], help="override TTS_BACKEND")
    parser.add_argument("--profile", choices=["nora", "jarvis"], help="assistant identity and voice (default: nora)")
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
    collected_turn = TurnCollector(cfg.turn_pause_s, cfg.incomplete_pause_s)
    indicator = Indicator(cfg.indicator, cfg.db_path.parent / "indicator-position.json", cfg.persona.name)
    aec = EchoCancellation()
    ducker = MusicDucker(cfg.music_duck_factor)
    audio_watch: asyncio.Task | None = None
    service_tasks: list[asyncio.Task] = []

    async def watch_audio() -> None:
        while True:
            active = speaker.speaking or thinking or (not assistant.quiet.active and (
                collected_turn.collecting or time.monotonic() < listen_until or listening.collecting
            ))
            await ducker.update(active)
            await asyncio.sleep(0.2)

    def refresh_indicator() -> None:
        ctx.pending.expire()
        if mic_off:
            state = "muted"
        elif speaker.speaking:
            state = "speaking"
        elif thinking:
            state = "thinking"
        elif not assistant.quiet.active and (collected_turn.collecting or time.monotonic() < listen_until or (mic is not None and not cfg.wake_word)):
            state = "attentive"
        elif hearing:
            state = "hearing"
        else:
            state = "idle"
        indicator.set(state, listen_until if state == "attentive" and cfg.wake_word else 0.0, " · ".join(sorted(ctx.activity)), tasks=ctx.tasks.snapshot())

    async def watch_indicator() -> None:
        nonlocal mic_off
        checked = 0.0
        while True:
            if isinstance(mic, Microphone) and not isinstance(mic, WavMicrophone) and time.monotonic() - checked > 3:
                checked = time.monotonic()
                mic_off = await source_muted()
            refresh_indicator()
            await asyncio.sleep(0.2)

    barge = False  # barge-in: the microphone stays on while answering (real microphone only)
    turn: asyncio.Task | None = None  # the answer being given, cancelled by an interruption

    def responding() -> bool:
        return (turn is not None and not turn.done()) or speaker.speaking

    def on_speaking(active: bool) -> None:
        if barge and not active and segmenter is not None and not segmenter.speaking:
            segmenter.reset()  # drop the tail of her own voice
        update_mic()

    async def interrupt() -> None:
        nonlocal listen_until
        # Stop playback before awaiting cancellation of a slow network/tool task.
        speaker.interrupt()
        if turn is not None and not turn.done():
            turn.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await turn
        listen_until = time.monotonic() + cfg.wake_followup_s
        refresh_indicator()

    def update_mic(_: bool = False) -> None:
        refresh_indicator()
        if mic is None:
            return
        if barge:
            if segmenter is not None:
                # Short chunks while she talks: «Нора, стоп» must not wait for her to pause.
                segmenter.max_utterance_s = BARGE_CHUNK_S if responding() else None
            mic.set_muted(False)
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
        player = Player(cfg.audio.output_device)
        speaker = Speaker(tts, player, on_speaking=on_speaking)
        # Warm up Silero in the background so a fallback never waits for model loading.
        service_tasks.append(asyncio.create_task(asyncio.to_thread(silero.load)))
        log.info("TTS: %s", tts.name)
    speaker.start()

    def announce(text: str) -> None:
        nonlocal listen_until
        print(f"\n🔔 {text}", flush=True)
        assistant.add_announcement(text)
        if assistant.quiet.active:
            return  # background agent results remain in history; timers still ring
        speaker.play_sound(chime(), 24000)
        speaker.say(text)
        listen_until = time.monotonic() + cfg.wake_followup_s + len(text) * 0.07  # roughly until it finished talking

    ctx = ToolContext(memory, http, tz, cfg.default_city, announce=announce, enabled_groups=cfg.tool_groups, disabled_tools=cfg.disabled_tools, persona=cfg.persona)
    tools = build_tools(ctx)
    assistant = Assistant(
        LlmClient(cfg.llm), tools, memory, speaker, tz,
        default_city=cfg.default_city,
        history_messages=cfg.history_messages,
        max_tool_steps=cfg.llm.max_tool_steps,
        pending=ctx.pending,
        listening=ctx.listening,
        persona=cfg.persona,
        wake_word=cfg.wake_word,
    )
    listening = ctx.listening
    alarms: asyncio.Task | None = None
    if not args.once:
        service_tasks.append(asyncio.create_task(assistant.warm_up()))
    log.info("LLM: %s at %s; tools: %s", cfg.llm.model, cfg.llm.base_url, ", ".join(tools.names))

    if not args.once:
        # Hotkey: `pkill -USR1 -f "python -m voice_assistant"` makes her stop talking.
        asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, lambda: asyncio.ensure_future(interrupt()))

    if not args.once and cfg.indicator != "off":
        indicator.start()
        service_tasks.append(asyncio.create_task(watch_indicator()))
    if cfg.music_duck and not (args.once or args.text or args.wav):
        audio_watch = asyncio.create_task(watch_audio())

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
        if not (args.once or args.text or args.wav) and cfg.aec_mode != "off":
            if cfg.audio.input_device is not None or cfg.audio.output_device is not None:
                if cfg.aec_mode == "on":
                    raise RuntimeError("Для AEC используй AEC_SOURCE/AEC_SINK вместо AUDIO_INPUT_DEVICE/AUDIO_OUTPUT_DEVICE")
                log.warning("Explicit PortAudio devices selected: AEC auto-routing skipped; use AEC_SOURCE/AEC_SINK instead")
            else:
                try:
                    await aec.start(cfg.aec_source, cfg.aec_sink)
                except (OSError, RuntimeError, TimeoutError) as error:
                    if cfg.aec_mode == "on":
                        raise RuntimeError(f"Не удалось включить AEC: {error}") from error
                    log.warning("AEC unavailable; using raw microphone: %s", error)
                if aec.active and isinstance(speaker, Speaker):
                    speaker.set_player(PulsePlayer(aec.sink))
        alarms = asyncio.create_task(AlarmScheduler(memory, speaker, tz).run())
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
            mic = WavMicrophone(args.wav) if args.wav else PulseMicrophone(aec.source) if aec.active else Microphone(cfg.audio.input_device)
            mic.start()
            if cfg.wake_word:
                print(f"Слушаю. Обращайся по имени: «{cfg.wake_word.capitalize()}, …»; Ctrl+C — выход.\n", flush=True)
            else:
                print("Слушаю. Говори; Ctrl+C — выход.\n", flush=True)

            barge = cfg.barge_in and not isinstance(mic, WavMicrophone)
            name = cfg.wake_word or cfg.persona.wake_word
            from .barge import build_detector
            detector = build_detector(cfg.barge_engine, cfg.vosk_model_dir, name) if barge else None
            detector_active = False

            async def answer(request: str, heard_at: float, stt_s: float | None) -> None:
                nonlocal busy, hearing, listen_until
                busy, hearing = True, False
                update_mic()
                try:
                    await respond(request, heard_at, stt_s)
                    listen_until = 0.0 if assistant.quiet.active else max(time.monotonic() + cfg.wake_followup_s, listening.until)
                finally:
                    busy = False
                    update_mic()

            async def dispatch(request: str, heard_at: float, stt_s: float | None) -> None:
                """Answer in the background while listening on (barge-in), or right here."""
                nonlocal turn
                if not barge:
                    await answer(request, heard_at, stt_s)
                    return
                turn = asyncio.create_task(answer(request, heard_at, stt_s))
                update_mic()

            while True:
                try:
                    frame = await asyncio.wait_for(mic.queue.get(), timeout=0.5)
                except TimeoutError:
                    frame = None
                    if getattr(mic, "failure", None):
                        raise mic.failure
                    if isinstance(mic, WavMicrophone) and mic.finished.is_set():
                        if collected := collected_turn.take(False, force=True):
                            await answer(collected, collected_turn.heard_at, collected_turn.stt_s)
                        if collected := listening.take(False, now=float("inf")):
                            await answer(collected, time.monotonic(), None)
                        break
                if detector is not None:
                    if responding():
                        detector_active = True
                        if frame is not None and await asyncio.to_thread(detector.feed, frame, speaker.recent_text):
                            log.info("streaming interruption detected")
                            await interrupt()
                            detector.reset()
                    elif detector_active:
                        detector.reset()
                        detector_active = False
                utterance = segmenter.process(frame) if frame is not None else None
                if hearing != segmenter.active and not responding():
                    hearing = segmenter.active
                    refresh_indicator()
                if utterance is None:
                    # Drain audio queued during STT before deciding the user stopped.
                    if not responding() and mic.queue.empty() and (collected := collected_turn.take(segmenter.active)):
                        await dispatch(collected, collected_turn.heard_at, collected_turn.stt_s)
                    # Long listening: the phrases so far are answered only after a long silence.
                    if not responding() and (collected := listening.take(segmenter.active)):
                        await dispatch(collected, time.monotonic(), None)
                    continue
                heard_at = time.monotonic()
                if barge and responding():
                    # She is talking or thinking: only an interruption matters, the rest is her own echo.
                    text = await asyncio.to_thread(transcriber.transcribe, utterance)
                    request = find_interruption(text, name, speaker.recent_text)
                    if request is None:
                        log.debug("while answering, ignored %r", text)
                        continue
                    print(f"Ты (перебил): {text}", flush=True)
                    await interrupt()
                    if len(request) >= cfg.stt.min_chars:
                        print(f"Ты: {request}", flush=True)
                        collected_turn.add(request, heard_at, silence_s=segmenter.last_silence_s)
                    continue
                patient = listening.active or listening.collecting  # keep the microphone on: the next phrase may follow at once
                busy = False  # keep capturing a continuation while STT runs
                hearing = True
                update_mic()
                try:
                    text = await asyncio.to_thread(transcriber.transcribe, utterance)
                    stt_s = time.monotonic() - heard_at
                    if len(text) < cfg.stt.min_chars:
                        log.debug("ignored %r (%.1fs of audio)", text, len(utterance) / 16000)
                        continue
                    request = text
                    if assistant.quiet.active:
                        request = find_wake_word(text, name)
                        if request is None:
                            continue
                        assistant.quiet.wake()
                        collected_turn.clear()
                        listen_until = time.monotonic() + cfg.wake_followup_s
                        if len(request) < cfg.stt.min_chars:
                            speaker.play_sound(chime(), 24000)
                            continue
                    if patient:
                        if cfg.wake_word:
                            request = find_wake_word(text, cfg.wake_word)
                            request = text if request is None else request
                        print(f"Ты: {text}", flush=True)
                        if len(request) < cfg.stt.min_chars:  # just the name
                            continue
                        listening.add(request)
                        if parse_command(request) == "off":  # «обычный режим» is answered at once
                            await dispatch(listening.take(False, now=float("inf")) or request, heard_at, stt_s)
                        continue
                    if cfg.wake_word and heard_at > listen_until and not collected_turn.collecting:
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
                    collected_turn.add(request, heard_at, stt_s, segmenter.last_silence_s)
                finally:
                    busy = hearing = False
                    update_mic()
            if turn is not None:
                await asyncio.wait([turn])
        await assistant.drain()
    finally:
        if audio_watch:
            audio_watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await audio_watch
        await ducker.restore()
        if turn is not None:
            turn.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await turn
        if alarms:
            alarms.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await alarms
        for task in service_tasks:
            task.cancel()
        await asyncio.gather(*service_tasks, return_exceptions=True)
        indicator.close()
        await speaker.close()
        if mic:
            if isinstance(mic, PulseMicrophone):
                await mic.aclose()
            else:
                mic.close()
        await aec.close()
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


def single_instance(path, name: str = "Нора", male: bool = False) -> bool:
    """Two copies would both answer, hear each other and share one LLM context window."""
    global _lock_file
    path.parent.mkdir(parents=True, exist_ok=True)
    _lock_file = open(path, "a+")
    try:
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _lock_file.seek(0)
        print(f"{name} уже {'запущен' if male else 'запущена'} (pid {_lock_file.read().strip() or '?'}). Закрой текущую копию: kill <pid>.")
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
    cfg = load_config(args.profile)
    # Both profiles share the microphone and must not answer one another.
    if not args.once and not single_instance(PROJECT_DIR / "data" / "nora.lock", cfg.persona.name, cfg.persona.male):
        sys.exit(1)
    if args.tts:
        from dataclasses import replace

        cfg = replace(cfg, tts=replace(cfg.tts, backend=args.tts))
    try:
        asyncio.run(run(cfg, args))
    except KeyboardInterrupt:
        print("\nПока!")
    except AudioInputError as error:
        log.error("Аудиовход недоступен. %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
