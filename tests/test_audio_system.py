import asyncio
import json

import numpy as np
import pytest

from voice_assistant.audio_system import EchoCancellation, MusicDucker
from voice_assistant.barge import StreamingInterruption, build_detector
from voice_assistant.memory import Memory
from voice_assistant.tools import ToolContext, ToolRegistry, build_tools
from zoneinfo import ZoneInfo


async def test_duck_restores_volume_without_resuming_paused_music(monkeypatch):
    volumes = {"playing": 0.8, "paused": 0.4}
    calls = []

    async def fake(*cmd, **kwargs):
        calls.append(cmd)
        if cmd == ("playerctl", "-l"):
            return "playing\npaused"
        player, action = cmd[2:4]
        if action == "status":
            return "Playing" if player == "playing" else "Paused"
        assert action == "volume"
        if len(cmd) == 5:
            volumes[player] = float(cmd[4])
        return str(volumes[player])

    monkeypatch.setattr("voice_assistant.audio_system.run", fake)
    ducker = MusicDucker(0.5)
    await ducker.update(True)
    assert volumes == {"playing": 0.4, "paused": 0.4}
    count = len(calls)
    await ducker.update(True)
    assert len(calls) == count
    await ducker.update(False)
    assert volumes == {"playing": 0.8, "paused": 0.4}
    assert not any("play" in cmd or "wpctl" in cmd for cmd in calls)


async def test_duck_preserves_manual_volume_change(monkeypatch):
    level = [1.0]

    async def fake(*cmd, **kwargs):
        if cmd == ("playerctl", "-l"):
            return "music"
        if cmd[-1] == "status":
            return "Playing"
        if len(cmd) == 5:
            level[0] = float(cmd[-1])
        return str(level[0])

    monkeypatch.setattr("voice_assistant.audio_system.run", fake)
    ducker = MusicDucker(0.35)
    await ducker.update(True)
    assert level[0] == 0.35
    level[0] = 0.6
    await ducker.restore()
    assert level[0] == 0.6


async def test_aec_owns_only_its_module_and_never_sets_defaults(monkeypatch):
    commands = []

    async def fake(*cmd, **kwargs):
        commands.append(cmd)
        return {"get-default-source": "physical_mic", "get-default-sink": "physical_speaker", "load-module": "123"}.get(cmd[1], "")

    monkeypatch.setattr("voice_assistant.audio_system.run", fake)
    monkeypatch.setattr("voice_assistant.audio_system.shutil.which", lambda _: "/bin/tool")
    aec = EchoCancellation()
    await aec.start()
    assert aec.active
    assert any("source_master=physical_mic" in cmd and "sink_master=physical_speaker" in cmd for cmd in commands)
    await aec.close()
    await aec.close()
    assert commands.count(("pactl", "unload-module", "123")) == 1
    assert not any("set-default-source" in cmd or "set-default-sink" in cmd for cmd in commands)


def test_streaming_detector_partial_and_echo():
    class Recognizer:
        text = "стоп"

        def AcceptWaveform(self, pcm):
            assert len(pcm) == 512 * 3 * 2
            return False

        def PartialResult(self):
            return json.dumps({"partial": self.text})

        def Reset(self):
            pass

    detector = StreamingInterruption.__new__(StreamingInterruption)
    detector._recognizer = Recognizer()
    detector.name = "нора"
    detector._frames = []
    detector._last_partial = ""
    frame = np.zeros(512, dtype=np.float32)
    assert not detector.feed(frame)
    assert not detector.feed(frame)
    assert detector.feed(frame)
    detector.reset()
    for _ in range(3):
        echo_trigger = detector.feed(frame, "скажи стоп")
    assert not echo_trigger


def test_optional_detector_falls_back_without_model(tmp_path):
    assert build_detector("off", tmp_path / "absent", "нора") is None
    assert build_detector("auto", tmp_path / "absent", "нора") is None
    with pytest.raises(RuntimeError):
        build_detector("vosk", tmp_path / "absent", "нора")


async def test_tool_timeout_and_cancellation():
    reg = ToolRegistry()
    cancelled = asyncio.Event()

    @reg.tool("stuck", timeout_s=0.01)
    async def stuck():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    result = await reg.call("stuck", "{}")
    assert not result.ok and "не ответил" in result.text
    assert cancelled.is_set()


def test_tool_group_selection_and_disabled_tools():
    ctx = ToolContext(Memory(":memory:"), None, ZoneInfo("Europe/Moscow"), "")
    reg = build_tools(ctx)
    reg.select("какая погода завтра")
    names = {s["function"]["name"] for s in reg.schemas(selected=True)}
    assert "get_weather" in names and "ask_coder" not in names
    assert "enable_skill" in names
    reg.select("объясни это подробнее")
    assert len(reg.schemas(selected=True)) == len(reg.schemas())
    ctx.enabled_groups = "alarms,media"
    ctx.disabled_tools = "volume"
    restricted = build_tools(ctx)
    assert "set_timer" in restricted.names
    assert "ask_coder" not in restricted.names and "volume" not in restricted.names


async def test_tool_selection_can_expand_for_followup():
    ctx = ToolContext(Memory(":memory:"), None, ZoneInfo("Europe/Moscow"), "")
    reg = build_tools(ctx)
    reg.select("погода")
    assert (await reg.call("enable_skill", '{"group":"web"}')).ok
    assert "web_fetch" in {s["function"]["name"] for s in reg.schemas(selected=True)}
    assert not (await reg.call("enable_skill", '{"group":"not_a_plugin"}')).ok


async def test_tool_rejects_string_boolean_before_cancelling_timers():
    ctx = ToolContext(Memory(":memory:"), None, ZoneInfo("Europe/Moscow"), "")
    reg = build_tools(ctx)
    await reg.call("set_timer", '{"duration":"пять минут"}')
    result = await reg.call("cancel_timer", '{"all_timers":"false"}')
    assert not result.ok and "boolean" in result.text
    assert len(ctx.memory.pending_alarms()) == 1


async def test_duck_cancellation_keeps_restore_state(monkeypatch):
    volume = [1.0]
    restoring = [False]

    async def fake(*cmd, **kwargs):
        if cmd == ("playerctl", "-l"):
            return "music"
        if cmd[-1] == "status":
            return "Playing"
        if restoring[0]:
            raise asyncio.CancelledError
        if len(cmd) == 5:
            volume[0] = float(cmd[-1])
        return str(volume[0])

    monkeypatch.setattr("voice_assistant.audio_system.run", fake)
    ducker = MusicDucker(0.5)
    await ducker.update(True)
    restoring[0] = True
    with pytest.raises(asyncio.CancelledError):
        await ducker.restore()
    assert "music" in ducker._saved
    restoring[0] = False
    await ducker.restore()
    assert volume[0] == 1.0
