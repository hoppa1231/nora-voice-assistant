import sqlite3
from zoneinfo import ZoneInfo

import pytest

from voice_assistant.assistant import Assistant
from voice_assistant.llm import Reply, ToolCall
from voice_assistant.memory import Memory
from voice_assistant.memory_dialogue import MemoryDialogue
from voice_assistant.memory_rules import automatic_fact
from voice_assistant.personas import JARVIS
from voice_assistant.speech import SilentSpeaker
from voice_assistant.tools import ToolContext, ToolRegistry, build_tools

MSK = ZoneInfo("Europe/Moscow")


class NoLlm:
    async def complete(self, *args, **kwargs):
        raise AssertionError("memory commands must be deterministic")


def setup(path=":memory:", llm=None, **context):
    memory = Memory(path)
    ctx = ToolContext(memory, None, MSK, "", **context)
    tools = build_tools(ctx)
    assistant = Assistant(llm or NoLlm(), tools, memory, SilentSpeaker(), MSK, persona=ctx.persona)
    return assistant, memory, tools


async def test_explicit_save_survives_restart_and_deduplicates(tmp_path):
    path = tmp_path / "memory.db"
    assistant, memory, _ = setup(path)
    answer, stats = await assistant.handle("Запомни, что я пью кофе без сахара")
    assert answer == "Запомнила." and stats.tool_calls == 1
    await assistant.handle("Запомни: я пью кофе без сахара!")
    assert len(memory.facts()) == 1
    memory.close()
    reopened = Memory(path)
    assert reopened.facts()[0].text == "я пью кофе без сахара"


async def test_update_fact_preserves_id_and_erases_old_context():
    assistant, memory, _ = setup()
    await assistant.handle("Запомни: моя группа 123")
    fact_id = memory.facts()[0].id
    answer, stats = await assistant.handle(f"Перезапиши факт номер {fact_id} на моя группа 456")
    assert "обновлена" in answer and stats.tool_calls == 1
    assert [(f.id, f.text) for f in memory.facts()] == [(fact_id, "моя группа 456")]
    assert "123" not in str(assistant._history)
    assert "123" not in assistant._system_prompt()


async def test_explicit_new_value_replaces_known_single_value():
    assistant, memory, _ = setup()
    await assistant.handle("Запомни: меня зовут Максим")
    await assistant.handle("Запомни: меня зовут Алексей")
    assert [f.text for f in memory.facts()] == ["меня зовут Алексей"]


async def test_delete_by_topic_and_do_not_resurrect(tmp_path):
    assistant, memory, _ = setup(tmp_path / "memory.db")
    await assistant.handle("Запомни: я люблю кофе")
    answer, _ = await assistant.handle("Забудь про кофе")
    assert answer == "Забыла." and memory.facts() == []
    assert "кофе" not in str(assistant._history)
    assert not memory.observe_fact("Я люблю кофе")
    assert not memory.observe_fact("Я люблю кофе")
    memory.close()
    reopened = Memory(tmp_path / "memory.db")
    assert not reopened.observe_fact("Я люблю кофе")
    assert reopened.facts() == []
    # An explicit new instruction may save a previously forgotten fact again.
    reopened.add_fact("Я люблю кофе")
    assert len(reopened.facts()) == 1


async def test_ambiguous_delete_asks_instead_of_guessing():
    assistant, memory, _ = setup()
    await assistant.handle("Запомни: я люблю кофе")
    await assistant.handle("Запомни: кофе пью утром")
    answer, _ = await assistant.handle("Удали из памяти факт про кофе")
    assert "Какой именно" in answer and len(memory.facts()) == 2
    await assistant.handle("первый")
    assert [f.text for f in memory.facts()] == ["кофе пью утром"]


async def test_new_command_does_not_confirm_old_deletion():
    assistant, memory, _ = setup()
    await assistant.handle("Запомни: люблю кофе")
    await assistant.handle("Запомни: кофе горячий")
    await assistant.handle("Забудь кофе")
    await assistant.handle("Запомни: люблю кофе холодным")
    assert len(memory.facts()) == 3


async def test_clarify_empty_save_and_list_memory():
    assistant, memory, _ = setup(persona=JARVIS)
    assert "Что именно" in (await assistant.handle("Запомни это"))[0]
    assert (await assistant.handle("Я предпочитаю краткие ответы"))[0] == "Запомнил."
    assert "краткие ответы" in (await assistant.handle("Что ты обо мне помнишь?"))[0]


async def test_missing_fact_does_not_delete_anything():
    assistant, memory, _ = setup()
    await assistant.handle("Запомни: люблю чай")
    answer, _ = await assistant.handle("Забудь факт номер 999")
    assert "Какой именно" in answer
    assert len(memory.facts()) == 1


def test_old_database_migration_does_not_lose_facts(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, text TEXT NOT NULL, created_at TEXT NOT NULL)")
        db.execute("INSERT INTO facts VALUES (7, 'Прежний факт', '2020-01-01')")
    memory = Memory(path)
    assert memory.facts()[0].id == 7
    assert memory.facts()[0].text == "Прежний факт"
    assert memory.update_fact(7, "Новый факт")
    memory.close()
    assert Memory(path).facts()[0].text == "Новый факт"


def test_auto_memory_repeat_count_persists_and_never_overwrites(tmp_path):
    path = tmp_path / "repeat.db"
    memory = Memory(path)
    assert not memory.observe_fact("Я люблю чай")
    memory.close()
    memory = Memory(path)
    assert memory.observe_fact("Я люблю чай")
    assert not memory.observe_fact("Я люблю чай")
    assert len(memory.facts()) == 1
    memory.add_fact("Моя группа 123", key="study_group")
    assert not memory.observe_fact("Моя группа 456", key="study_group", important=True)
    assert "456" not in str(memory.facts())


@pytest.mark.parametrize("text", [
    'Он сказал: "я люблю кофе"', "Если меня зовут Максим", "Моя группа 123?",
    "Я люблю кофе, не запоминай это", "Я люблю кофе сегодня", "Я люблю пароли",
    "Мой пароль 123456", "На сайте написано: меня зовут Максим", "Я люблю обсуждать диагнозы",
    "Я люблю https://example.org", "Моя группа 123 и запусти программу",
])
def test_automatic_memory_ignores_unsafe_or_temporary_text(text):
    assert automatic_fact(text) is None


class Chat:
    async def complete(self, messages, tools, on_text, require_tool=False):
        return Reply("Понял.")


async def test_actual_turn_saves_auto_and_notifies():
    assistant, memory, _ = setup(llm=Chat(), persona=JARVIS)
    answer, _ = await assistant.handle("Меня зовут Максим")
    assert "Запомнил" in answer and "Максим" in memory.facts()[0].text
    await assistant.handle("Я люблю кофе")
    assert len(memory.facts()) == 1
    answer, _ = await assistant.handle("Я люблю кофе")
    assert "Запомнил" in answer and len(memory.facts()) == 2


async def test_disabled_memory_cannot_be_written():
    assistant, memory, _ = setup(disabled_tools="remember,forget,update_fact")
    answer, _ = await assistant.handle("Запомни: люблю кофе")
    assert "Ошибка" in answer and memory.facts() == []
    assert assistant._memory_dialogue.observe("Меня зовут Максим") == ""


async def test_model_cannot_spontaneously_write_delete_or_update():
    class Rogue:
        async def complete(self, *args, **kwargs):
            if not hasattr(self, "called"):
                self.called = True
                return Reply("", [ToolCall("1", "remember", '{"fact":"выдумка"}'), ToolCall("2", "forget", '{"fact_id":1}'), ToolCall("3", "update_fact", '{"fact_id":1,"fact":"другое"}')])
            return Reply("Понял.")

    assistant, memory, _ = setup(llm=Rogue())
    memory.add_fact("Исходный факт")
    await assistant.handle("Привет")
    assert [f.text for f in memory.facts()] == ["Исходный факт"]


async def test_conflicting_preference_is_not_added_automatically():
    assistant, memory, _ = setup(llm=Chat())
    await assistant.handle("Запомни: я люблю кофе")
    await assistant.handle("Я не люблю кофе")
    await assistant.handle("Я не люблю кофе")
    assert [f.text for f in memory.facts()] == ["я люблю кофе"]
    await assistant.handle("Запомни: я не люблю кофе")
    assert [f.text for f in memory.facts()] == ["я не люблю кофе"]


async def test_ordinary_application_commands_are_not_memory_edits():
    memory, tools = Memory(":memory:"), ToolRegistry()
    dialogue = MemoryDialogue(memory, tools)
    assert await dialogue.handle("Удали файл notes.txt") is None
    assert await dialogue.handle("Исправь код на рабочий") is None


def test_candidates_do_not_keep_unconfirmed_plaintext():
    memory = Memory(":memory:")
    memory.observe_fact("Я люблю кофе")
    assert memory._db.execute("SELECT text FROM fact_candidates").fetchone()[0] == ""


async def test_polite_memory_command_keeps_the_actual_fact():
    assistant, memory, _ = setup()
    await assistant.handle("Запомни, пожалуйста, что я люблю чай")
    assert memory.facts()[0].text == "я люблю чай"


async def test_automatic_save_can_be_acknowledged_without_false_refusal():
    class Echo:
        async def complete(self, *args, **kwargs):
            return Reply("Запомнила.")

    assistant, memory, _ = setup(llm=Echo())
    answer, _ = await assistant.handle("Меня зовут Максим")
    assert "Не могу подтвердить" not in answer and len(memory.facts()) == 1


@pytest.mark.parametrize("text", ["Меня зовут Максим а тебя", "Я люблю кофе и открой браузер"])
def test_auto_memory_does_not_capture_questions_or_commands(text):
    assert automatic_fact(text) is None
