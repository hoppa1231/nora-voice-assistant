from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from voice_assistant.assistant import Assistant
from voice_assistant.memory import Memory
from voice_assistant.speech import SilentSpeaker
from voice_assistant.tools import PendingActions, ToolContext, ToolRegistry, build_tools
from voice_assistant.tools.coder import command, summarize
from voice_assistant.tools.computer import DesktopApp, find_app, find_window, load_apps

MSK = ZoneInfo("Europe/Moscow")


def write_desktop(directory: Path, app_id: str, body: str) -> None:
    (directory / f"{app_id}.desktop").write_text("[Desktop Entry]\nType=Application\n" + body)


def test_load_and_find_apps(tmp_path):
    write_desktop(tmp_path, "org.telegram.desktop", "Name=Telegram\nExec=telegram-desktop\n")
    write_desktop(tmp_path, "firefox", "Name=Firefox\nGenericName=Web Browser\nGenericName[ru]=Веб-браузер\n")
    write_desktop(tmp_path, "org.kde.dolphin", "Name=Dolphin\nGenericName=File Manager\nName[ru]=Dolphin\n")
    write_desktop(tmp_path, "kitty", "Name=kitty\nGenericName=Terminal emulator\n")
    write_desktop(tmp_path, "hidden", "Name=Hidden\nNoDisplay=true\n")
    apps = load_apps((tmp_path,))
    assert "hidden" not in {a.id for a in apps}
    found = {q: find_app(q, apps).id for q in ("телеграм", "браузер", "проводник", "терминал", "Firefox", "веб-браузер")}
    assert found == {
        "телеграм": "org.telegram.desktop",
        "браузер": "firefox",
        "проводник": "org.kde.dolphin",
        "терминал": "kitty",
        "Firefox": "firefox",
        "веб-браузер": "firefox",
    }
    assert find_app("фотошоп", apps) is None


def test_find_app_prefers_name_over_keyword():
    apps = [
        DesktopApp("thunar-settings", "Настройки Thunar", "thunar settings настройки thunar"),
        DesktopApp("steam", "Steam", "steam game"),
    ]
    assert find_app("стим", apps).id == "steam"


def test_find_window_most_recent_first():
    clients = [
        {"address": "0x1", "class": "firefox", "title": "Новости", "focusHistoryID": 3},
        {"address": "0x2", "class": "firefox", "title": "Почта", "focusHistoryID": 1},
        {"address": "0x3", "class": "org.telegram.desktop", "title": "Чат", "focusHistoryID": 2},
    ]
    assert find_window("браузер", clients)["address"] == "0x2"
    assert find_window("телеграм", clients)["address"] == "0x3"
    assert find_window("новости", clients)["address"] == "0x1"
    assert find_window("стим", clients) is None


def test_summarize_takes_itog_line():
    answer = "Сделал модуль.\n\n```py\ncode\n```\nИТОГ: Добавила **умение** читать RSS, тесты проходят."
    assert summarize(answer) == "Добавила умение читать RSS, тесты проходят."
    assert summarize("Просто ответ без итога.") == "Просто ответ без итога."


def test_coder_commands_limit_permissions(tmp_path):
    ask = command("claude", "что это", tmp_path, "question", "brief", tmp_path / "o")
    assert "Bash" in ask[ask.index("--disallowedTools") :] and "--permission-mode" not in ask
    change = command("claude", "сделай", tmp_path, "change", "brief", tmp_path / "o")
    assert change[change.index("--permission-mode") + 1] == "acceptEdits"
    assert not any(t in ("Bash", "Bash(*)") for t in change)
    codex = command("codex", "сделай", tmp_path, "question", "brief", tmp_path / "o")
    assert codex[codex.index("--sandbox") + 1] == "read-only"


class NoLlm:
    async def complete(self, *args, **kwargs):
        raise AssertionError("confirmation must not reach the LLM")


async def test_confirmation_is_decided_in_code():
    pending = PendingActions()
    assistant = Assistant(NoLlm(), ToolRegistry(), Memory(":memory:"), SilentSpeaker(), MSK, pending=pending)
    calls = []

    async def run() -> str:
        calls.append(1)
        return "Сделала."

    assert pending.request("Закрыть окно?", run) == "Закрыть окно? Подтверди: скажи «да» или «нет»."
    assert (await assistant.handle("Да, давай"))[0] == "Сделала."
    pending.request("Закрыть окно?", run)
    assert (await assistant.handle("нет, не надо"))[0] == "Ладно, не буду."
    pending.request("Закрыть окно?", run)
    assert (await assistant.handle("да нет, не надо"))[0] == "Ладно, не буду."
    assert calls == [1]


async def test_coder_change_waits_for_confirmation(tmp_path):
    (tmp_path / "blog").mkdir()
    async with httpx.AsyncClient() as http:
        ctx = ToolContext(Memory(":memory:"), http, MSK, "", project_dir=tmp_path / "app", projects_root=tmp_path)
        reg = build_tools(ctx)
    assert {"open_app", "close_window", "ask_coder", "restart_assistant"} <= set(reg.names)
    result = await reg.call("ask_coder", '{"task": "добавь тёмную тему", "mode": "change", "project": "blog"}')
    assert result.text.endswith("Подтверди: скажи «да» или «нет».") and "blog" in result.text
    missing = await reg.call("ask_coder", '{"task": "x", "mode": "change", "project": "нету"}')
    assert missing.text.startswith("Ошибка: нет проекта") and "blog" in missing.text


@pytest.mark.parametrize("text", ["Поставь таймер", "да"])
def test_pending_expires(text, monkeypatch):
    pending = PendingActions()

    async def run() -> str:
        return ""

    pending.request("x", run)
    monkeypatch.setattr(PendingActions, "TTL_S", -1)
    assert pending.take() is None


def test_spoken_window_names():
    from voice_assistant.tools.coder import short
    from voice_assistant.tools.computer import app_name

    assert app_name({"class": "org.telegram.desktop"}) == "telegram"
    assert app_name({"class": "firefox"}) == "firefox"
    assert short("Создать новое умение, которое позволит читать новости из RSS-каналов по команде", 40) == "Создать новое умение, которое позволит…"


async def test_context_overflow_drops_history():
    from voice_assistant.llm import ContextOverflow, Reply

    class SmallLlm:
        async def complete(self, messages, tools, on_text, require_tool=False):
            if len(messages) > 2:
                raise ContextOverflow("Context size has been exceeded.")
            on_text("Ок.")
            return Reply("Ок.")

    assistant = Assistant(SmallLlm(), ToolRegistry(), Memory(":memory:"), SilentSpeaker(), MSK)
    assistant._turns = [[{"role": "user", "content": "старое"}, {"role": "assistant", "content": "старое"}]]
    assert (await assistant.handle("привет"))[0] == "Ок."
    assert len(assistant._turns) == 1


@pytest.mark.parametrize(
    ("text", "request_text"),
    [
        ("Допиши функцию", ""),
        ("Нора, сделай так", ""),
        ("Сделай так, пожалуйста.", ""),
        ("Добавь себе новую функцию", ""),
        ("Допиши функцию, чтобы ты говорила курс доллара", "Допиши функцию, чтобы ты говорила курс доллара"),
        ("Сделай так, чтобы ты умела читать новости", "Сделай так, чтобы ты умела читать новости"),
        ("Сделай так же", None),
        ("Сделай так, чтобы было погромче", None),
        ("Сделай погромче", None),
        ("Какая функция у печени?", None),
    ],
)
def test_new_function_requests(text, request_text):
    from voice_assistant.skills import parse_request

    assert parse_request(text) == request_text


def fake_coder() -> tuple[ToolRegistry, list[str]]:
    reg, tasks = ToolRegistry(), []

    @reg.tool("x", final=True)
    def ask_coder(task: str, mode: str) -> str:
        assert mode == "change"
        tasks.append(task)
        return "Поручить Клоду? Подтверди: скажи «да» или «нет»."

    return reg, tasks


async def test_new_function_asks_details_first():
    import json
    from voice_assistant.skills import ASK_DETAILS, QUOTE_HEADER

    def quotes(task):
        return json.JSONDecoder().raw_decode(task.removeprefix(QUOTE_HEADER))[0]

    reg, tasks = fake_coder()
    assistant = Assistant(NoLlm(), reg, Memory(":memory:"), SilentSpeaker(), MSK)
    assert (await assistant.handle("Допиши функцию"))[0] == ASK_DETAILS
    answer = (await assistant.handle("Пусть по утрам говорит, сколько дней до выходных"))[0]
    assert answer.endswith("Подтверди: скажи «да» или «нет».")
    assert quotes(tasks[0]) == ["Допиши функцию", "Пусть по утрам говорит, сколько дней до выходных"]
    assert "простая и понятная" not in tasks[0]

    assert (await assistant.handle("Сделай так"))[0] == ASK_DETAILS
    assert (await assistant.handle("Нет, не надо"))[0] == "Ладно, не буду."
    await assistant.handle("Допиши функцию, чтобы ты считала калории")
    assert len(tasks) == 2 and quotes(tasks[1]) == ["Допиши функцию, чтобы ты считала калории"]


async def test_new_function_question_expires(monkeypatch):
    from voice_assistant import skills

    class OkLlm:
        async def complete(self, messages, tools, on_text, require_tool=False):
            from voice_assistant.llm import Reply

            on_text("Ок.")
            return Reply("Ок.")

    reg, tasks = fake_coder()
    assistant = Assistant(OkLlm(), reg, Memory(":memory:"), SilentSpeaker(), MSK)
    await assistant.handle("Допиши функцию")
    monkeypatch.setattr(skills, "TTL_S", -1)
    assert (await assistant.handle("Какая погода?"))[0] == "Ок." and not tasks


async def test_new_function_reports_coder_errors():
    assistant = Assistant(NoLlm(), ToolRegistry(), Memory(":memory:"), SilentSpeaker(), MSK)
    answer = (await assistant.handle("Допиши функцию, чтобы ты считала калории"))[0]
    assert answer == "Не получилось: нет инструмента ask_coder."


@pytest.mark.parametrize(
    ("text", "command", "only"),
    [
        ("Нора, я хочу говорить подольше", "on", True),
        ("Не перебивай меня, пожалуйста", "on", True),
        ("Давай не будем перебрасываться, дай договорить", "on", True),
        ("Включи долгое слушание", "on", False),
        ("Не перебивай, я расскажу про свой день на работе", "on", False),
        ("Хватит меня слушать", "off", True),
        ("Вернись в обычный режим", "off", False),
        ("Можешь снова перебивать", "off", True),
        ("Какая погода завтра?", None, False),
        ("У нас был долгий разговор с мамой", None, False),
    ],
)
def test_long_listening_phrases(text, command, only):
    from voice_assistant.listening import only_command, parse_command

    assert parse_command(text) == command
    if command:
        assert only_command(text) == only


def test_long_listening_waits_for_a_long_pause(monkeypatch):
    import time

    from voice_assistant import listening

    mode = listening.LongListening()
    assert not mode.active
    mode.start()
    assert mode.active and mode.until - time.monotonic() > 590
    mode.add("Сначала я хочу рассказать,", now=100.0)
    assert mode.take(False, now=101.0) is None  # a usual pause is not the end
    mode.add("что было утром.", now=102.0)
    assert mode.take(True, now=110.0) is None  # still speaking
    assert mode.take(False, now=102.0 + listening.PAUSE_S) == "Сначала я хочу рассказать, что было утром."
    assert mode.take(False, now=1000.0) is None
    mode.stop()
    assert not mode.active


async def test_long_listening_switch_is_decided_in_code():
    from voice_assistant.listening import STARTED, STOPPED, LongListening

    mode = LongListening()
    assistant = Assistant(NoLlm(), ToolRegistry(), Memory(":memory:"), SilentSpeaker(), MSK, listening=mode)
    assert (await assistant.handle("Нора, я хочу говорить подольше"))[0] == STARTED
    assert mode.active
    assert (await assistant.handle("Хватит меня слушать"))[0] == STOPPED
    assert not mode.active


async def test_long_listening_inside_a_request_still_asks_the_model():
    from voice_assistant.listening import LongListening
    from voice_assistant.llm import Reply

    class OkLlm:
        async def complete(self, messages, tools, on_text, require_tool=False):
            on_text("Слушаю.")
            return Reply("Слушаю.")

    mode = LongListening()
    assistant = Assistant(OkLlm(), ToolRegistry(), Memory(":memory:"), SilentSpeaker(), MSK, listening=mode)
    assert (await assistant.handle("Не перебивай, я расскажу про свой день на работе"))[0] == "Слушаю."
    assert mode.active


async def test_long_listening_tool():
    async with httpx.AsyncClient() as http:
        ctx = ToolContext(Memory(":memory:"), http, MSK, "")
        reg = build_tools(ctx)
    result = await reg.call("long_listening", '{"on": true}')
    assert result.final and ctx.listening.active
    await reg.call("long_listening", '{"on": false}')
    assert not ctx.listening.active


def test_main_module_imports():
    import voice_assistant.__main__ as main

    assert main.parse_command("не перебивай") == "on"


async def test_speaker_interrupt_drops_queue():
    import asyncio

    import numpy as np

    from voice_assistant.speech import Speaker

    played: list[float] = []

    class SlowTts:
        async def synthesize(self, sentence):
            await asyncio.sleep(0.01)
            return np.zeros(10, dtype=np.float32), 24000

    class FakePlayer:
        def __init__(self):
            self.stopped = asyncio.Event()

        async def play(self, audio, rate):
            played.append(rate)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.stopped.wait(), 1)  # «plays» until stopped

        def stop(self):
            self.stopped.set()

    import contextlib

    speaker = Speaker(SlowTts(), FakePlayer())
    speaker.start()
    speaker.say("Первое предложение довольно длинное, чтобы его разрезать. Второе. Третье. Четвёртое.")
    await asyncio.sleep(0.05)
    assert speaker.speaking and "Второе" in speaker.recent_text
    speaker.interrupt()
    await asyncio.wait_for(speaker.wait_done(), 1)
    assert len(played) == 1 and not speaker.speaking
    speaker.say("Снова.")  # still works after an interruption
    await asyncio.sleep(0.05)
    assert len(played) == 2
    await speaker.close()


@pytest.mark.parametrize(
    ("text", "details"),
    [
        ("Сделай апгрейд", ""),
        ("Нора, сделай апгрейд", ""),
        ("Апгрейд: научись говорить курс доллара утром", "Апгрейд: научись говорить курс доллара утром"),
        ("Давай апгрейд", ""),
        ("Улучши себя", ""),
        ("Какой апгрейд у видеокарты лучше", None),
    ],
)
def test_upgrade_phrase(text, details):
    from voice_assistant.skills import parse_request

    assert parse_request(text) == details


async def test_upgrade_dialogue_listens_patiently():
    from voice_assistant.listening import LongListening

    listening = LongListening()
    assistant = Assistant(NoLlm(), ToolRegistry(), Memory(":memory:"), SilentSpeaker(), MSK, listening=listening)
    answer, _ = await assistant.handle("Сделай апгрейд")
    assert answer.startswith("Давай, излагай задачу") and listening.active
