"""Desktop control on Hyprland: launch apps from .desktop files, windows, workspaces, browser, typing."""

from __future__ import annotations

import asyncio
import configparser
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from . import ToolContext, ToolError, ToolRegistry

APP_DIRS = (
    Path.home() / ".local/share/applications",
    Path("/usr/local/share/applications"),
    Path("/usr/share/applications"),
    Path("/var/lib/flatpak/exports/share/applications"),
    Path.home() / ".local/share/flatpak/exports/share/applications",
)

# Spoken name -> words found in .desktop files.
SYNONYMS = {
    "браузер": "firefox web browser",
    "файрфокс": "firefox",
    "фаерфокс": "firefox",
    "хром": "chrome chromium",
    "телеграм": "telegram",
    "телеграмм": "telegram",
    "дискорд": "discord",
    "стим": "steam",
    "проводник": "files file manager nautilus dolphin thunar",
    "файлы": "files file manager",
    "терминал": "terminal kitty alacritty foot konsole",
    "консоль": "terminal",
    "настройки": "settings",
    "калькулятор": "calculator",
    "музыка": "music spotify",
    "спотифай": "spotify",
    "обсидиан": "obsidian",
    "вскод": "code visual studio",
    "код": "code visual studio",
    "редактор": "editor text",
    "лм студио": "lm studio",
}

TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ы": "y", "э": "e", "ю": "yu",
    "я": "ya", "ь": "", "ъ": "",
})


@dataclass
class DesktopApp:
    id: str  # file name without .desktop, what gtk-launch takes
    name: str
    words: str  # lowercased Name, Name[ru], GenericName, Keywords, id


def load_apps(dirs: tuple[Path, ...] = APP_DIRS) -> list[DesktopApp]:
    apps: dict[str, DesktopApp] = {}
    for directory in dirs:
        for path in sorted(directory.glob("*.desktop")) if directory.is_dir() else ():
            app_id = path.stem
            if app_id in apps:  # ~/.local overrides system entries
                continue
            parser = configparser.ConfigParser(interpolation=None, strict=False)
            parser.optionxform = str
            try:
                parser.read(path, encoding="utf-8")
                entry = parser["Desktop Entry"]
            except (configparser.Error, KeyError, UnicodeDecodeError):
                continue
            if entry.get("Type", "Application") != "Application" or entry.get("NoDisplay") == "true" or entry.get("Hidden") == "true":
                continue
            name = entry.get("Name[ru]") or entry.get("Name") or app_id
            fields = [entry.get(k, "") for k in ("Name", "Name[ru]", "GenericName", "GenericName[ru]", "Keywords", "Keywords[ru]")]
            apps[app_id] = DesktopApp(app_id, name, " ".join([*fields, app_id.replace(".", " ").replace("-", " ")]).lower())
    return list(apps.values())


def find_app(query: str, apps: list[DesktopApp]) -> DesktopApp | None:
    query = query.lower().replace("ё", "е").strip()
    candidates = [query, query.translate(TRANSLIT)]
    if query in SYNONYMS:
        candidates.insert(0, SYNONYMS[query])
    best, best_score = None, 0.0
    for candidate in candidates:
        terms = [t for t in re.split(r"\W+", candidate) if len(t) >= 2]
        for app in apps:
            name = app.name.lower()
            if name == candidate or app.id.lower() == candidate:
                return app
            words = set(re.split(r"[\W_]+", app.words))
            score = 0.0
            for i, term in enumerate(terms):
                weight = 1 / (1 + i)  # synonyms list the most likely word first
                if term in words:
                    score += 2 * weight
                elif any(w.startswith(term) for w in words if len(term) >= 3):
                    score += weight
            if score and name.startswith(terms[0] if terms else "\0"):
                score += 1
            if score > best_score:
                best, best_score = app, score
    return best


async def run(*cmd: str, timeout: float = 5) -> str:
    if shutil.which(cmd[0]) is None:
        raise ToolError(f"не установлен {cmd[0]}")
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    if proc.returncode != 0:
        raise ToolError((err or out).decode().strip() or f"{cmd[0]} завершился с кодом {proc.returncode}")
    return out.decode().strip()


async def launch(*cmd: str) -> None:
    """Start a program detached from the assistant (it survives the assistant's exit)."""
    if os.getenv("HYPRLAND_INSTANCE_SIGNATURE") and shutil.which("hyprctl"):
        result = await run("hyprctl", "dispatch", "exec", " ".join(_shell_quote(c) for c in cmd))
        if result.strip() != "ok":
            raise ToolError(result)
        return
    if shutil.which(cmd[0]) is None:
        raise ToolError(f"не установлен {cmd[0]}")
    await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
    )


def _shell_quote(arg: str) -> str:
    return arg if re.fullmatch(r"[\w@%+=:,./-]+", arg) else "'" + arg.replace("'", "'\\''") + "'"


async def windows() -> list[dict]:
    return json.loads(await run("hyprctl", "clients", "-j"))


def find_window(query: str, clients: list[dict]) -> dict | None:
    query = query.lower().strip()
    variants = {query, query.translate(TRANSLIT), *SYNONYMS.get(query, "").split()} - {""}
    for client in sorted(clients, key=lambda c: c.get("focusHistoryID", 99)):  # most recently used first
        haystack = f"{client.get('class', '')} {client.get('initialClass', '')} {client.get('title', '')}".lower()
        if any(v in haystack for v in variants):
            return client
    return None


def describe_window(client: dict) -> str:
    title = client.get("title") or client.get("class") or "без названия"
    return f"{client.get('class', '?')}: «{_clip(title, 60)}» (стол {client.get('workspace', {}).get('name', '?')})"


def app_name(client: dict) -> str:
    """«org.telegram.desktop» -> «telegram»: something that can be said aloud."""
    parts = [p for p in re.split(r"[.\-_]", client.get("class") or "") if p.lower() not in ("org", "com", "io", "desktop", "kde", "gnome", "app")]
    return max(parts, key=len) if parts else "окно"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def register(reg: ToolRegistry, ctx: ToolContext) -> None:
    @reg.tool(
        "Запустить программу на компьютере по названию (браузер, телеграм, терминал, проводник, стим…).",
        {"name": {"type": "string", "description": "Название программы, как сказал пользователь."}},
        required=("name",),
        final=True,
    )
    async def open_app(name: str) -> str:
        app = await asyncio.to_thread(lambda: find_app(name, load_apps()))
        if app is None:
            raise ToolError(f"не нашёл программу «{name}» среди установленных")
        await launch("gtk-launch", app.id)
        return f"Открываю {app.name}."

    @reg.tool("Список открытых окон.")
    async def list_windows() -> str:
        clients = [c for c in await windows() if c.get("mapped", True)]
        if not clients:
            return "Открытых окон нет."
        return "Окна: " + "; ".join(describe_window(c) for c in clients)

    @reg.tool(
        "Переключиться на открытое окно программы.",
        {"name": {"type": "string", "description": "Программа или часть заголовка окна."}},
        required=("name",),
        final=True,
    )
    async def focus_window(name: str) -> str:
        client = find_window(name, await windows())
        if client is None:
            raise ToolError(f"нет открытого окна «{name}»")
        await run("hyprctl", "dispatch", "focuswindow", f"address:{client['address']}")
        return "Переключила."

    @reg.tool(
        "Закрыть окно программы (спросит подтверждение).",
        {"name": {"type": "string", "description": "Программа или часть заголовка окна."}},
        required=("name",),
        final=True,
    )
    async def close_window(name: str) -> str:
        client = find_window(name, await windows())
        if client is None:
            raise ToolError(f"нет открытого окна «{name}»")
        address = client["address"]

        async def close() -> str:
            await run("hyprctl", "dispatch", "closewindow", f"address:{address}")
            return "Закрыла."

        return ctx.pending.request(f"Закрыть {app_name(client)}?", close)

    @reg.tool(
        "Перейти на рабочий стол с номером.",
        {"number": {"type": "integer"}},
        required=("number",),
        final=True,
    )
    async def switch_workspace(number: int) -> str:
        await run("hyprctl", "dispatch", "workspace", str(int(number)))
        return f"Рабочий стол {int(number)}."

    @reg.tool(
        "Открыть сайт в браузере пользователя (показать на экране, а не прочитать).",
        {"url": {"type": "string"}},
        required=("url",),
        final=True,
    )
    async def open_url(url: str) -> str:
        url = url.strip()
        if not re.match(r"^https?://", url):
            if " " in url or "." not in url:
                raise ToolError("это не адрес сайта; для поиска есть browser_search")
            url = "https://" + url
        await launch("xdg-open", url)
        return "Открываю в браузере."

    @reg.tool(
        "Открыть поиск в браузере на экране (когда пользователь хочет сам посмотреть результаты).",
        {"query": {"type": "string"}},
        required=("query",),
        final=True,
    )
    async def browser_search(query: str) -> str:
        await launch("xdg-open", f"https://duckduckgo.com/?q={quote_plus(query)}")
        return "Открыла поиск в браузере."

    @reg.tool(
        "Напечатать текст в активное окно, как с клавиатуры (диктовка).",
        {"text": {"type": "string"}},
        required=("text",),
        final=True,
    )
    async def type_text(text: str) -> str:
        await run("wtype", "--", text, timeout=30)
        return "Напечатала."
