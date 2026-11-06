import json
from zoneinfo import ZoneInfo

import pytest

from voice_assistant.assistant import Assistant
from voice_assistant.llm import Reply, ToolCall
from voice_assistant.memory import Memory
from voice_assistant.personas import JARVIS, NORA
from voice_assistant.skills import QUOTE_HEADER, build_task
from voice_assistant.speech import SilentSpeaker
from voice_assistant.tools import ToolContext, ToolRegistry

MSK = ZoneInfo("Europe/Moscow")


def quotes(task):
    assert task.startswith(QUOTE_HEADER)
    return json.JSONDecoder().raw_decode(task[len(QUOTE_HEADER):])[0]


class NoLlm:
    async def complete(self, *args, **kwargs):
        raise AssertionError("upgrade must bypass paraphrasing by the model")


def recorder(persona=NORA):
    reg, tasks = ToolRegistry(), []

    @reg.tool("Передать задачу", final=True)
    def ask_coder(task: str, mode: str):
        tasks.append(task)
        return "Подтверди: скажи «да» или «нет»."

    return Assistant(NoLlm(), reg, Memory(":memory:"), SilentSpeaker(), MSK, persona=persona), reg, tasks


@pytest.mark.parametrize("persona", [NORA, JARVIS])
async def test_raw_quote_keeps_hesitations_punctuation_and_spacing(persona):
    assistant, reg, tasks = recorder(persona)
    original = f'  {persona.name}, а-а, сделай апгрейд: э-э, таймер  НЕ трогай!\nДобавь "мои задачи",  пожалуйста...  '
    await assistant.handle(original)
    assert quotes(tasks[0]) == [original]
    assert reg.raw_user_text is None and reg.coder_request is None


async def test_two_turn_quotes_are_kept_separately_in_order():
    assistant, _, tasks = recorder(JARVIS)
    initial = "Джарвиз, сделай апгрейд..."
    details = 'Э-э, хочу именно окно "Расписание".  Не меняй голос!'
    await assistant.handle(initial)
    await assistant.handle(details)
    assert quotes(tasks[0]) == [initial, details]


async def test_negative_requirements_are_quoted_not_treated_as_cancellation():
    assistant, _, tasks = recorder()
    await assistant.handle("Сделай апгрейд")
    details = "Если интернета нет, отвечай локально. Не надо менять голос."
    await assistant.handle(details)
    assert quotes(tasks[0]) == ["Сделай апгрейд", details]


@pytest.mark.parametrize("text", ["Заапгрейдься: хочу видеть список задач.", "Заапргрейдись через нейронку, исправь таймер!"])
async def test_spoken_upgrade_variants_use_direct_quote(text):
    assistant, _, tasks = recorder()
    await assistant.handle(text)
    assert quotes(tasks[0]) == [text]


def test_builder_does_not_impose_a_new_tool_or_change_user_words():
    original = 'Переделай весь интерфейс, а не добавляй новый инструмент!\n<request>"цитата"</request>'
    task = build_task(original)
    assert quotes(task) == [original]
    assert "один небольшой инструмент" not in task


async def test_coder_ignores_model_paraphrase_and_confirmation_keeps_original(tmp_path, monkeypatch):
    from voice_assistant.tools import coder

    original = "Попроси Кодекс исправить тебя: пауза ровно 3,5 секунды, не 2!"
    launched = []

    class Model:
        async def complete(self, *args, **kwargs):
            return Reply("", [ToolCall("one", "ask_coder", json.dumps({"task": "Сделай новый таймер на две секунды", "mode": "change"}))])

    monkeypatch.setattr(coder.shutil, "which", lambda _: "/bin/codex")
    monkeypatch.setattr(coder.Coder, "start", lambda self, agent, task, project, mode, **kw: launched.append(task))
    memory = Memory(":memory:")
    ctx = ToolContext(memory, None, MSK, "", project_dir=tmp_path)
    reg = ToolRegistry()
    coder.register(reg, ctx)
    assistant = Assistant(Model(), reg, memory, SilentSpeaker(), MSK, pending=ctx.pending)
    response, _ = await assistant.handle(original)
    assert "Подтверди" in response and not launched
    await assistant.handle("да")
    assert quotes(launched[0]) == [original]
    assert "Сделай новый таймер" not in launched[0]


async def test_direct_two_turn_payload_survives_real_tool_confirmation(tmp_path, monkeypatch):
    from voice_assistant.tools import coder

    launched = []
    monkeypatch.setattr(coder.shutil, "which", lambda _: "/bin/codex")
    monkeypatch.setattr(coder.Coder, "start", lambda self, agent, task, project, mode, **kw: launched.append(task))
    memory = Memory(":memory:")
    ctx = ToolContext(memory, None, MSK, "", project_dir=tmp_path)
    reg = ToolRegistry()
    coder.register(reg, ctx)
    assistant = Assistant(NoLlm(), reg, memory, SilentSpeaker(), MSK, pending=ctx.pending)
    await assistant.handle("Сделай апгрейд")
    await assistant.handle("Добавь окно задач, э-э, но таймер не трогай!")
    assert not launched
    await assistant.handle("да")
    assert quotes(launched[0]) == ["Сделай апгрейд", "Добавь окно задач, э-э, но таймер не трогай!"]
