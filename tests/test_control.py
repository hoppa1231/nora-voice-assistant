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
    assert (await assistant.handle("нет, не надо"))[0] == "Хорошо, не буду."
    pending.request("Закрыть окно?", run)
    assert (await assistant.handle("да нет, не надо"))[0] == "Хорошо, не буду."
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
