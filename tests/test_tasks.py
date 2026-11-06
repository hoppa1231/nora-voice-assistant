import asyncio
from zoneinfo import ZoneInfo

import pytest

from voice_assistant.assistant import Assistant
from voice_assistant.claims import needs_execution, unsupported_claim
from voice_assistant.llm import Reply, ToolCall
from voice_assistant.memory import Memory
from voice_assistant.overlay_model import clamp_position, load_position, save_position
from voice_assistant.speech import SilentSpeaker
from voice_assistant.tasks import TaskBoard
from voice_assistant.tools import PendingActions, ToolContext, ToolError, ToolRegistry
from voice_assistant.tools.coder import Coder

MSK = ZoneInfo("Europe/Moscow")


class Recorder(SilentSpeaker):
    def __init__(self):
        self.texts = []

    def feed(self, text):
        self.texts.append(text)


class Replies:
    def __init__(self, *replies):
        self.replies = iter(replies)

    async def complete(self, messages, tools, on_text, require_tool=False):
        reply = next(self.replies)
        on_text(reply.text)
        return reply


async def test_quoted_request_reaches_coder_without_model():
    reg, tasks = ToolRegistry(), []

    @reg.tool("Делегирование", final=True)
    def ask_coder(task: str, mode: str) -> str:
        tasks.append(task)
        return "Поручить Кодексу? Подтверди: скажи «да» или «нет»."

    assistant = Assistant(Replies(), reg, Memory(":memory:"), SilentSpeaker(), MSK)
    answer, stats = await assistant.handle("А-а, добавь Э-э, через Кодекс сейчас себе функцию, чтобы ты мог просматривать нормально расписание")
    assert len(tasks) == 1 and "расписание" in tasks[0]
    assert "Подтверди" in answer and stats.tool_calls == 1


@pytest.mark.parametrize("utterance", ["Попроси агента проверить расписание", "Ты меня понял?"])
async def test_unverified_claim_never_reaches_speech(utterance):
    lie = "Поручу Кодексу создать функцию. Сейчас работаю над этим."
    speaker = Recorder()
    assistant = Assistant(Replies(Reply(lie)), ToolRegistry(), Memory(":memory:"), speaker, MSK)
    answer, stats = await assistant.handle(utterance)
    assert "Не могу подтвердить" in answer and stats.tool_calls == 0
    assert all(lie not in text for text in speaker.texts)
    assert "Поручу" not in "".join(speaker.texts)


async def test_tool_preamble_cannot_hide_confirmation():
    reg = ToolRegistry()

    @reg.tool("Делегирование", final=True)
    def ask_coder(task: str, mode: str) -> str:
        return "Поручить Кодексу? Подтверди: скажи «да» или «нет»."

    speaker = Recorder()
    reply = Reply("Кодекс уже работает!", [ToolCall("1", "ask_coder", '{"task":"расписание", "mode":"change"}')])
    assistant = Assistant(Replies(reply), reg, Memory(":memory:"), speaker, MSK)
    answer, stats = await assistant.handle("Попроси агента проверить расписание")
    assert "Подтверди" in answer and stats.tool_calls == 1
    assert "уже работает" not in "".join(speaker.texts)


async def test_failed_tool_is_not_success():
    reg = ToolRegistry()

    @reg.tool("Делегирование")
    def ask_coder(task: str) -> str:
        raise ToolError("не установлен codex")

    llm = Replies(Reply("", [ToolCall("1", "ask_coder", '{"task":"расписание"}')]), Reply("Поручу Кодексу, работаю над этим."))
    speaker = Recorder()
    assistant = Assistant(llm, reg, Memory(":memory:"), speaker, MSK)
    answer, _ = await assistant.handle("Попроси агента проверить расписание")
    assert "Не могу подтвердить" in answer and "Поручу" not in "".join(speaker.texts)


async def test_chat_and_text_composition_still_work():
    assistant = Assistant(Replies(Reply("Краткое резюме.")), ToolRegistry(), Memory(":memory:"), SilentSpeaker(), MSK)
    assert (await assistant.handle("Сделай краткое резюме"))[0] == "Краткое резюме."
    assert not needs_execution("Как поставить таймер?")
    assert unsupported_claim("Таймер запущен", set())
    assert not unsupported_claim("Таймер запущен", {"set_timer"})


@pytest.mark.parametrize("response", ["Какую программу открыть?", "Не могу кликать по интерфейсу: нет инструмента."])
async def test_honest_limits_and_clarification_are_preserved(response):
    assistant = Assistant(Replies(Reply(response)), ToolRegistry(), Memory(":memory:"), SilentSpeaker(), MSK)
    assert (await assistant.handle("Открой приложение"))[0] == response


def test_pending_cancellation_and_expiry_update_board(monkeypatch):
    pending, board = PendingActions(), TaskBoard()
    task = board.create("Расписание", "Кодекс")
    pending.request("Разрешить?", lambda: None, lambda: board.update(task, "cancelled"))
    assert board.snapshot()[0]["state"] == "awaiting_confirmation"
    monkeypatch.setattr(PendingActions, "TTL_S", -1)
    pending.expire()
    assert pending.take() is None
    assert board.snapshot()[0]["state"] == "cancelled"


@pytest.mark.parametrize("failure", [False, True])
async def test_coder_lifecycle_uses_actual_process(tmp_path, monkeypatch, failure):
    from voice_assistant.tools import coder as module

    ctx = ToolContext(Memory(":memory:"), None, MSK, "", project_dir=tmp_path)
    coder = Coder(ctx)
    proceed = asyncio.Event()

    class Process:
        pid = 123
        returncode = None

        async def communicate(self):
            await proceed.wait()
            self.returncode = 1 if failure else 0
            return b"Done", b"failed" if failure else b""

    async def spawn(*args, **kwargs):
        assert ctx.tasks.snapshot()[0]["state"] == "preparing"
        return Process()

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    coder.start("codex", "Проверить расписание", tmp_path, "question")
    assert ctx.tasks.snapshot()[0]["state"] == "preparing"
    await asyncio.sleep(0)
    assert ctx.tasks.snapshot()[0]["state"] == "running"
    proceed.set()
    await coder._task
    assert ctx.tasks.snapshot()[0]["state"] == ("failed" if failure else "done")
    assert not coder.busy and not ctx.activity


def test_positions_are_saved_and_clamped(tmp_path):
    path = str(tmp_path / "jarvis" / "indicator-position.json")
    assert load_position(path) is None
    save_position(path, 77, 101)
    assert load_position(path) == (77, 101)
    assert clamp_position(-40, 9999, 356, 84, 2048, 1280) == (0, 1196)
    assert clamp_position(50, 50, 356, 84, 100, 50) == (0, 0)


def test_board_hides_old_results_but_keeps_active_jobs(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("voice_assistant.tasks.time.monotonic", lambda: now[0])
    board = TaskBoard()
    first = board.create("Первая", "Кодекс")
    second = board.create("Вторая", "Кодекс", "running")
    board.update(first, "done")
    now[0] += 91
    assert [t["id"] for t in board.snapshot()] == [second]


@pytest.mark.parametrize("tests_pass", [False, True])
async def test_self_change_is_done_only_after_tests(tmp_path, monkeypatch, tests_pass):
    from voice_assistant.tools import coder as module

    ctx = ToolContext(Memory(":memory:"), None, MSK, "", project_dir=tmp_path)
    coder = Coder(ctx)
    commits = []

    async def commit(*args):
        commits.append(args)
        return True

    class Process:
        pid = 456
        returncode = 0

        async def communicate(self):
            return b"Done", b""

    async def spawn(*args, **kwargs):
        return Process()

    async def check(project):
        assert ctx.tasks.snapshot()[0]["state"] == "checking"
        return tests_pass, "test report"

    monkeypatch.setattr(module, "commit_all", commit)
    monkeypatch.setattr(module, "run_tests", check)
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    coder.start("codex", "Новое умение", tmp_path, "change")
    await coder._task
    assert ctx.tasks.snapshot()[0]["state"] == ("done" if tests_pass else "failed")
    assert len(commits) == (2 if tests_pass else 1)
    assert ("перезапустись" in coder.job.summary) is tests_pass


async def test_coder_cancellation_stops_process(tmp_path, monkeypatch):
    from voice_assistant.tools import coder as module

    ctx = ToolContext(Memory(":memory:"), None, MSK, "", project_dir=tmp_path)
    coder = Coder(ctx)

    class Process:
        pid = 789
        returncode = None
        killed = False

        async def communicate(self):
            await asyncio.Event().wait()

        def kill(self):
            self.killed = True

        async def wait(self):
            self.returncode = -9

    process = Process()

    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    coder.start("codex", "Проверка", tmp_path, "question")
    await asyncio.sleep(0)
    coder._task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await coder._task
    assert process.killed and not ctx.activity
    assert ctx.tasks.snapshot()[0]["state"] == "cancelled"


async def test_pending_job_status_is_not_running(tmp_path, monkeypatch):
    from voice_assistant.tools import coder as module

    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/codex")
    ctx = ToolContext(Memory(":memory:"), None, MSK, "", project_dir=tmp_path)
    reg = ToolRegistry()
    module.register(reg, ctx)
    result = await reg.call("ask_coder", '{"task":"Читать расписание","mode":"change"}')
    assert "Подтверди" in result.text
    assert ctx.tasks.snapshot()[0]["state"] == "awaiting_confirmation"
    assert "ещё не запущена" in (await reg.call("coder_status", "{}")).text
    ctx.pending.clear()
    assert ctx.tasks.snapshot()[0]["state"] == "cancelled"
