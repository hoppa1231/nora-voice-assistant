import sys
from zoneinfo import ZoneInfo

import pytest

from voice_assistant.assistant import Assistant
from voice_assistant.config import load_config
from voice_assistant.memory import Memory
from voice_assistant.personas import JARVIS, NORA
from voice_assistant.prompts import build_system_prompt
from voice_assistant.speech import SilentSpeaker
from voice_assistant.tools import ToolContext, ToolRegistry, build_tools
from voice_assistant.tools.coder import BRIEF, SELF_NOTE
from voice_assistant.tts import normalize_for_silero
from voice_assistant.wake import find_interruption, find_wake_word, is_exact_name


class NoLlm:
    async def complete(self, *args, **kwargs):
        raise AssertionError("direct profile commands must not call the model")


def jarvis():
    memory = Memory(":memory:")
    ctx = ToolContext(memory, None, ZoneInfo("Europe/Moscow"), "", persona=JARVIS)
    tools = build_tools(ctx)
    return Assistant(NoLlm(), tools, memory, SilentSpeaker(), ctx.timezone, persona=JARVIS), tools


def test_profile_overrides_do_not_inherit_nora_voice_or_memory(monkeypatch, tmp_path):
    monkeypatch.setattr("voice_assistant.config.load_dotenv", lambda _: None)
    monkeypatch.setenv("WAKE_WORD", "нора")
    monkeypatch.setenv("SILERO_SPEAKER", "baya")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "nora.db"))
    for name in ("JARVIS_WAKE_WORD", "JARVIS_SILERO_SPEAKER", "JARVIS_DB_PATH", "JARVIS_EDGE_TTS_VOICE"):
        monkeypatch.delenv(name, raising=False)
    original = load_config("nora")
    male = load_config("jarvis")
    assert original.persona == NORA and original.wake_word == "нора"
    assert original.tts.silero_speaker == "baya"
    assert male.persona == JARVIS and male.wake_word == "джарвиз"
    assert male.tts.silero_speaker == "eugene"
    assert male.tts.edge_voice == "ru-RU-DmitryNeural"
    assert male.db_path != original.db_path and male.db_path.parent.name == "jarvis"


def test_jarvis_has_dedicated_environment_overrides(monkeypatch):
    monkeypatch.setenv("JARVIS_WAKE_WORD", "альфред")
    monkeypatch.setenv("JARVIS_SILERO_SPEAKER", "aidar")
    cfg = load_config("jarvis")
    assert cfg.wake_word == "альфред" and cfg.tts.silero_speaker == "aidar"


@pytest.mark.parametrize("name", ["Джарвиз", "Джарвис", "Jarvis"])
def test_jarvis_names_work_at_edges_and_in_middle(name):
    assert find_wake_word(f"{name}, поставь таймер", "джарвиз") == "поставь таймер"
    assert find_wake_word(f"поставь {name} таймер на минуту", "джарвиз") == "поставь таймер на минуту"
    assert find_wake_word(f"сколько времени, {name}", "джарвиз") == "сколько времени"
    assert is_exact_name(name.lower(), "джарвиз")
    assert find_interruption(f"{name} стоп", "джарвиз", "Сегодня солнечно") == ""


async def test_jarvis_timer_memory_and_quiet_have_male_responses():
    assistant, tools = jarvis()
    assert (await assistant.handle("Джарвис таймер на минуту"))[0] == "Поставил таймер на 1 минуту."
    assert (await tools.call("remember", '{"fact":"люблю кофе"}')).text == "Запомнил."
    assert (await assistant.handle("отмени таймер"))[0] == "Отменил таймер."
    await assistant.handle("помолчи до вызова")
    assert assistant.quiet.active
    assert (await assistant.handle("Нора таймер на минуту"))[0] == ""
    assert (await assistant.handle("Джарвиз таймер на две минуты"))[0] == "Поставил таймер на 2 минуты."
    assert not assistant.quiet.active
    assert (await assistant.handle("обычный режим"))[0] == "Вернулся в обычный режим."


async def test_profile_inflection_never_changes_user_label():
    _, tools = jarvis()
    result = await tools.call("set_timer", '{"duration":"минуту","label":"Поставила чай"}')
    assert result.text == "Поставил таймер «Поставила чай» на 1 минуту."


async def test_profile_never_inflects_retrieved_source_text():
    registry = ToolRegistry()
    registry.persona = JARVIS

    @registry.tool("страница")
    def web_fetch():
        return "Поставила рекорд: статья об участнице соревнований."

    assert (await registry.call("web_fetch", "{}")).text == "Поставила рекорд: статья об участнице соревнований."


def test_jarvis_prompt_and_coder_identity():
    prompt = build_system_prompt([], [], "", JARVIS)
    assert "Ты — Джарвиз" in prompt and "в мужском роде" in prompt
    assert "Ты — Нора" not in prompt and "Дерзкая" not in prompt
    assert "Ты — Нора" in build_system_prompt([], [], "", NORA)
    brief = BRIEF.format(assistant_name=JARVIS.name, project="/project", self_note=SELF_NOTE.format(assistant_name=JARVIS.name), mode_note="только чтение")
    assert "Джарвиз" in brief and "Нора" not in brief and "{assistant_name}" not in brief


def test_profile_cli_and_name_pronunciation(monkeypatch):
    from voice_assistant.__main__ import parse_args
    monkeypatch.setattr(sys, "argv", ["voice_assistant", "--profile", "jarvis", "--text"])
    assert parse_args().profile == "jarvis"
    assert normalize_for_silero("Джарвиз и Джарвис") == "Дж+арвиз и Дж+арвис"
