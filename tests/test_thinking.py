import asyncio
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from voice_assistant.assistant import Assistant
from voice_assistant.llm import Reply
from voice_assistant.memory import Memory
from voice_assistant.speech import SilentSpeaker, Speaker
from voice_assistant.thinking import PHRASES, ThinkingCues
from voice_assistant.tools import ToolRegistry


class Recorder(SilentSpeaker):
    def __init__(self):
        self.cues = []
        self.ready = asyncio.Event()
        self.cancelled = 0

    def say_waiting(self, text):
        self.cues.append(text)
        self.ready.set()
        return True

    def cancel_waiting(self):
        self.cancelled += 1


def controller(speaker, **kwargs):
    return ThinkingCues(speaker, SimpleNamespace(active=False), delay=0.005, interval=0.005, **kwargs)


async def test_fast_reply_has_no_cues():
    speaker = Recorder()

    async def immediate():
        return "Ответ"

    assert await controller(speaker).wait(immediate()) == "Ответ"
    assert speaker.cues == []


async def test_slow_reply_has_limited_non_repeating_cues():
    speaker = Recorder()
    cues = controller(speaker)
    task = asyncio.create_task(cues.wait(asyncio.Event().wait()))
    await asyncio.wait_for(speaker.ready.wait(), 1)
    await asyncio.sleep(0.06)
    assert len(speaker.cues) == 3
    assert all(a != b for a, b in zip(speaker.cues, speaker.cues[1:]))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert speaker.cancelled == 1
    count = len(speaker.cues)
    await asyncio.sleep(0.02)
    assert len(speaker.cues) == count


@pytest.mark.parametrize("mode", ["quiet", "speaking", "disabled"])
async def test_cues_respect_quiet_other_speech_and_setting(mode):
    speaker = Recorder()
    cues = controller(speaker, enabled=mode != "disabled")
    cues.quiet.active = mode == "quiet"
    speaker.speaking = mode == "speaking"
    await cues.wait(asyncio.sleep(0.025))
    assert speaker.cues == []


def test_phrase_shuffle_never_repeats_at_bag_boundary():
    cues = controller(Recorder())
    phrases = [cues.phrase() for _ in range(len(PHRASES) * 4)]
    assert all(a != b for a, b in zip(phrases, phrases[1:]))


async def test_cues_are_not_part_of_answer_or_conversation():
    speaker = Recorder()

    class Model:
        async def complete(self, *args, **kwargs):
            await speaker.ready.wait()
            return Reply("Вот ответ на вопрос.")

    assistant = Assistant(Model(), ToolRegistry(), Memory(":memory:"), speaker, ZoneInfo("Europe/Moscow"), thinking_delay_s=0.005)
    answer, _ = await asyncio.wait_for(assistant.handle("Расскажи о квантовой физике"), 1)
    assert speaker.cues
    assert answer == "Вот ответ на вопрос."
    assert not any(phrase in str(assistant._history) for phrase in speaker.cues)


async def test_cancelled_turn_stops_controller():
    speaker = Recorder()

    class Model:
        async def complete(self, *args, **kwargs):
            await asyncio.Event().wait()

    assistant = Assistant(Model(), ToolRegistry(), Memory(":memory:"), speaker, ZoneInfo("Europe/Moscow"), thinking_delay_s=0.005, thinking_interval_s=0.005)
    task = asyncio.create_task(assistant.handle("Долгий вопрос"))
    await asyncio.wait_for(speaker.ready.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    count = len(speaker.cues)
    await asyncio.sleep(0.02)
    assert len(speaker.cues) == count and speaker.cancelled == 1


class Player:
    def __init__(self):
        self.played = []

    async def play(self, audio, sample_rate):
        self.played.append(float(audio[0]))

    def stop(self):
        pass


async def test_pending_cue_does_not_delay_final_answer():
    started = asyncio.Event()

    class Tts:
        async def synthesize(self, text):
            if text == "Момент.":
                started.set()
                await asyncio.Event().wait()
            return np.ones(10), 16000

    player = Player()
    speaker = Speaker(Tts(), player)
    speaker.start()
    try:
        assert speaker.say_waiting("Момент.")
        await asyncio.wait_for(started.wait(), 1)
        speaker.say("Готов ответ.")
        await asyncio.wait_for(speaker.wait_done(), 1)
        assert player.played == [1.0]
        assert speaker.first_audio_at is not None
        assert not speaker._waiting
    finally:
        await speaker.close()


async def test_waiting_audio_is_cached_and_not_counted_as_answer_latency():
    class Tts:
        count = 0

        async def synthesize(self, text):
            self.count += 1
            return np.ones(10), 16000

    tts, player = Tts(), Player()
    speaker = Speaker(tts, player)
    speaker.start()
    try:
        for _ in range(2):
            assert speaker.say_waiting("Секунду.")
            await asyncio.wait_for(speaker.wait_done(), 1)
        assert tts.count == 1 and len(player.played) == 2
        assert speaker.first_audio_at is None
    finally:
        await speaker.close()


async def test_playing_waiting_phrase_finishes_then_real_answer():
    playing, release = asyncio.Event(), asyncio.Event()

    class Tts:
        async def synthesize(self, text):
            return np.array([1.0 if text == "Момент." else 2.0]), 16000

    class SlowPlayer(Player):
        async def play(self, audio, sample_rate):
            self.played.append(float(audio[0]))
            if audio[0] == 1:
                playing.set()
                await release.wait()

    player = SlowPlayer()
    speaker = Speaker(Tts(), player)
    speaker.start()
    try:
        speaker.say_waiting("Момент.")
        await asyncio.wait_for(playing.wait(), 1)
        speaker.say("Ответ.")
        release.set()
        await asyncio.wait_for(speaker.wait_done(), 1)
        assert player.played == [1.0, 2.0]
    finally:
        await speaker.close()


async def test_interrupt_drops_waiting_synthesis_and_tracking():
    started = asyncio.Event()

    class Tts:
        async def synthesize(self, text):
            started.set()
            await asyncio.Event().wait()

    player = Player()
    speaker = Speaker(Tts(), player)
    speaker.start()
    try:
        speaker.say_waiting("Секунду.")
        assert speaker.recent_text == "Секунду."
        await asyncio.wait_for(started.wait(), 1)
        speaker.interrupt()
        await asyncio.wait_for(speaker.wait_done(), 1)
        assert player.played == [] and not speaker._waiting
    finally:
        await speaker.close()


def test_thinking_settings_are_shared_by_profiles_and_validate(monkeypatch):
    from voice_assistant.config import load_config

    monkeypatch.setattr("voice_assistant.config.load_dotenv", lambda _: None)
    monkeypatch.setenv("THINKING_CUES", "0")
    monkeypatch.setenv("THINKING_CUE_DELAY_S", "6")
    monkeypatch.setenv("THINKING_CUE_INTERVAL_S", "18")
    for profile in ("nora", "jarvis"):
        cfg = load_config(profile)
        assert not cfg.thinking_cues
        assert cfg.thinking_delay_s == 6 and cfg.thinking_interval_s == 18
    monkeypatch.setenv("THINKING_CUE_INTERVAL_S", "0")
    with pytest.raises(ValueError, match="THINKING_CUE"):
        load_config()
