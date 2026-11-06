import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from voice_assistant.assistant import Assistant
from voice_assistant.commands import QuickCommands
from voice_assistant.memory import Memory
from voice_assistant.quiet import QuietMode
from voice_assistant.scheduler import AlarmScheduler
from voice_assistant.speech import SilentSpeaker
from voice_assistant.tools import ToolContext, build_tools
from voice_assistant.tools.alarms import parse_duration
from voice_assistant.tools.coder import command
from voice_assistant.turns import TurnCollector
from voice_assistant.wake import find_interruption, find_wake_word

MSK = ZoneInfo("Europe/Moscow")


class NoLlm:
    async def complete(self, *args, **kwargs):
        raise AssertionError("a direct command must not call the model")


def assistant_with_tools():
    memory = Memory(":memory:")
    ctx = ToolContext(memory, None, MSK, "")
    return Assistant(NoLlm(), build_tools(ctx), memory, SilentSpeaker(), MSK), memory


@pytest.mark.parametrize("text,expected", [
    ("А можешь, Нора, поставить таймер?", "А можешь поставить таймер?"),
    ("поставь нора таймер на пять минут", "поставь таймер на пять минут"),
    ("Нормально все", None),
    ("видел нору лисы", None),
])
def test_name_in_middle(text, expected):
    assert find_wake_word(text, "нора") == expected


def test_turn_waits_and_joins_continuation():
    turn = TurnCollector()
    turn.add("поставь таймер", 10, silence_s=1)
    assert turn.take(False, now=12) is None  # incomplete: longer pause
    turn.add("на пять минут", 12.5, silence_s=1)
    assert turn.take(True, now=20) is None
    assert turn.take(False, now=13) is None
    assert turn.take(False, now=14) == "поставь таймер на пять минут"
    assert turn.take(False, now=15) is None


def test_turn_keeps_request_across_max_length_segments():
    turn = TurnCollector()
    turn.add("я хочу", 60)
    assert turn.take(False, now=60) is None
    turn.add("чтобы ты проверила код", 80)
    assert turn.take(False, now=82.5) == "я хочу чтобы ты проверила код"


async def test_timer_without_llm_and_clarification():
    assistant, memory = assistant_with_tools()
    assert (await assistant.handle("поставь таймер"))[0] == "На сколько поставить таймер?"
    answer, _ = await assistant.handle("на пять минут")
    assert "5 минут" in answer
    assert len(memory.pending_alarms()) == 1
    answer, _ = await assistant.handle("сколько осталось до таймера")
    assert "осталось" in answer
    assert (await assistant.handle("отмени таймер"))[0] == "Отменила таймер."
    assert memory.pending_alarms() == []


async def test_stop_does_not_start_a_new_llm_answer():
    assistant, _ = assistant_with_tools()
    assert (await assistant.handle("стоп"))[0] == ""


async def test_timer_label_and_multiple_cancellation():
    assistant, memory = assistant_with_tools()
    assert "чая" in (await assistant.handle("поставь таймер на полторы минуты для чая"))[0]
    await assistant.handle("таймер на две минуты")
    reminder = memory.add_alarm("reminder", datetime.now(UTC) + timedelta(hours=1), "звонок")
    assert "несколько" in (await assistant.handle("отмени таймер"))[0]
    await assistant.handle("отмени все таймеры")
    assert [a.id for a in memory.pending_alarms()] == [reminder]


async def test_composite_or_negated_command_does_not_execute():
    assistant, memory = assistant_with_tools()
    quick = QuickCommands(assistant._tools)
    for text in ("не ставь таймер на пять минут", "поставь таймер на пять минут и открой браузер", "поставь таймер на минус пять минут", "поставь таймер на пять минут для чая и открой браузер"):
        assert await quick.handle(text) is None
    assert memory.pending_alarms() == []


async def test_quiet_until_explicit_wake():
    assistant, memory = assistant_with_tools()
    assert "помолчу" in (await assistant.handle("помолчи до вызова"))[0]
    assert assistant.quiet.active
    assert (await assistant.handle("таймер на минуту"))[0] == ""
    assert memory.pending_alarms() == []
    assert "Поставила" in (await assistant.handle("Нора таймер на минуту"))[0]
    assert not assistant.quiet.active


def test_quiet_duration_and_expiry(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("voice_assistant.quiet.time.monotonic", lambda: now[0])
    quiet = QuietMode()
    assert "5 минут" in quiet.request("помолчи на пять минут")
    assert quiet.active
    now[0] = 401
    assert not quiet.active


def test_quiet_survives_barge_in_parsing():
    assert find_interruption("Нора помолчи пять минут", "нора", "Рассказываю про погоду") == "помолчи пять минут"
    assert find_interruption("помолчи до вызова", "нора", "Рассказываю про погоду") == "помолчи до вызова"
    assert "помолчу" in QuietMode().request("помолчи пока я не позову")


async def test_due_timer_fires_once_without_llm():
    memory = Memory(":memory:")
    memory.add_alarm("timer", datetime.now(UTC) - timedelta(seconds=1), "чай")

    class Recorder(SilentSpeaker):
        texts = []
        sounds = []

        def say(self, text):
            self.texts.append(text)

        def play_sound(self, *args):
            self.sounds.append(args)

    speaker = Recorder()
    scheduler = asyncio.create_task(AlarmScheduler(memory, speaker, MSK).run())
    await asyncio.sleep(0)
    scheduler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await scheduler
    assert len(speaker.texts) == len(speaker.sounds) == 1
    assert "чай" in speaker.texts[0]
    assert memory.pending_alarms() == []


def test_codex_economy_and_sandbox(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    monkeypatch.delenv("CODEX_REASONING_EFFORT", raising=False)
    cmd = command("codex", "задача", tmp_path, "question", "brief", tmp_path / "answer")
    assert cmd[cmd.index("--model") + 1] == "gpt-6-luna"
    assert 'model_reasoning_effort="low"' in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    monkeypatch.setenv("CODEX_MODEL", "custom-model")
    assert "custom-model" in command("codex", "x", tmp_path, "change", "brief", tmp_path / "a")


@pytest.mark.parametrize("text", ["-5 минут", "минус пять минут"])
def test_timer_rejects_negative_duration(text):
    from voice_assistant.tools import ToolError
    with pytest.raises(ToolError):
        parse_duration(text)


async def test_interrupt_cancels_current_synthesis_without_waiting():
    from voice_assistant.speech import Speaker

    class StuckTts:
        async def synthesize(self, text):
            await asyncio.Event().wait()

    class Player:
        def stop(self):
            pass

    speaker = Speaker(StuckTts(), Player())
    speaker.start()
    speaker.say("Долгий ответ.")
    await asyncio.sleep(0)
    speaker.interrupt()
    await asyncio.wait_for(speaker.wait_done(), 0.2)
    await speaker.close()


async def test_microphone_loop_joins_timer_and_honours_quiet(tmp_path, monkeypatch):
    """Exercise the actual voice dispatch loop without devices or a model server."""
    import argparse
    import sqlite3
    from dataclasses import replace
    from voice_assistant import __main__ as main
    from voice_assistant.config import load_config

    transcripts = [
        ["Нора поставь таймер", "на пять минут"],
        ["помолчи до вызова"],
        ["таймер на десять минут"],  # ignored even inside the follow-up window
        ["поставь Нора таймер на две минуты"],
    ]

    class Recording:
        def __init__(self, path):
            self.queue = asyncio.Queue()
            self.finished = asyncio.Event()
            self.muted = False

        def start(self):
            self.task = asyncio.create_task(self.feed())

        async def feed(self):
            for group in transcripts:
                for text in group:
                    self.queue.put_nowait(text)
                    await asyncio.sleep(0.002)
                for _ in range(15):
                    self.queue.put_nowait(None)
                    await asyncio.sleep(0.005)
            self.finished.set()

        def set_muted(self, muted):
            self.muted = muted

        def close(self):
            self.task.cancel()

    class Segmenter:
        active = False
        last_silence_s = 0.0

        def __init__(self, cfg):
            pass

        def process(self, frame):
            return frame

        def reset(self):
            pass

    class Transcriber:
        def transcribe(self, audio):
            return audio

    class Llm(NoLlm):
        async def warm_up(self, *args):
            pass

    monkeypatch.setattr(main, "WavMicrophone", Recording)
    monkeypatch.setattr(main, "SpeechSegmenter", Segmenter)
    monkeypatch.setattr(main, "LlmClient", lambda cfg: Llm())
    monkeypatch.setattr("voice_assistant.stt.build_transcriber", lambda cfg: Transcriber())
    cfg = replace(load_config(), db_path=tmp_path / "assistant.db", indicator="off", turn_pause_s=0.015, incomplete_pause_s=0.04)
    args = argparse.Namespace(once=None, text=False, mute=True, wav="fake.wav")
    await asyncio.wait_for(main.run(cfg, args), 3)
    with sqlite3.connect(cfg.db_path) as db:
        texts = [r[0] for r in db.execute("SELECT text FROM alarms")]
    assert texts == ["таймер на 5 минут", "таймер на 2 минуты"]
