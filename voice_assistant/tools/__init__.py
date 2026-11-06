"""Tools the LLM can call. Each module registers its tools into a ToolRegistry."""

from __future__ import annotations

import inspect
import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from ..listening import LongListening
from ..memory import Memory
from ..personas import NORA, Persona
from ..tasks import TaskBoard

log = logging.getLogger("va.tools")

Handler = Callable[..., Awaitable[str] | str]


class ToolError(Exception):
    """A failure the model should see and explain to the user."""


@dataclass
class PendingAction:
    description: str
    run: Callable[[], Awaitable[str]]
    created: float
    on_cancel: Callable[[], None] | None = None


class PendingActions:
    """An action waiting for a spoken «да»/«нет». The answer is parsed by code, not the
    model, so a misheard or hallucinated tool call cannot confirm itself."""

    TTL_S = 90

    def __init__(self) -> None:
        self._action: PendingAction | None = None

    def request(self, description: str, run: Callable[[], Awaitable[str]], on_cancel: Callable[[], None] | None = None) -> str:
        self.clear()
        self._action = PendingAction(description, run, time.monotonic(), on_cancel)
        return f"{description} Подтверди: скажи «да» или «нет»."

    def take(self) -> PendingAction | None:
        action, self._action = self._action, None
        if action and time.monotonic() - action.created > self.TTL_S:
            if action.on_cancel:
                action.on_cancel()
            return None
        return action

    def clear(self) -> None:
        action, self._action = self._action, None
        if action and action.on_cancel:
            action.on_cancel()

    def expire(self) -> None:
        if self._action and time.monotonic() - self._action.created > self.TTL_S:
            self.clear()


def _announce_nowhere(text: str) -> None:
    log.info("announce: %s", text)


@dataclass
class ToolContext:
    memory: Memory
    http: httpx.AsyncClient
    timezone: ZoneInfo
    default_city: str
    project_dir: Path = Path(__file__).resolve().parents[2]
    projects_root: Path = Path.home() / "Projects"
    pending: PendingActions = field(default_factory=PendingActions)
    announce: Callable[[str], None] = _announce_nowhere  # speak later, e.g. when a background task ends
    activity: set[str] = field(default_factory=set)  # running background jobs, shown on screen
    listening: LongListening = field(default_factory=LongListening)  # «не перебивай»: up to 10 min without the name
    enabled_groups: str = "all"
    disabled_tools: str = ""
    persona: Persona = NORA
    tasks: TaskBoard = field(default_factory=TaskBoard)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Handler
    # The result is already the full spoken answer, so the turn can end without
    # asking the LLM to rephrase it (saves one LLM round trip, ~1.5 s).
    final: bool = False
    group: str = "general"
    timeout_s: float = 30.0


@dataclass
class ToolResult:
    text: str
    ok: bool
    final: bool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._group = "general"
        self._disabled: set[str] = set()
        self._selected: set[str] | None = None
        self.persona = NORA
        self.user_text: str | None = None
        self.raw_user_text: str | None = None
        self.coder_request: str | None = None
        self.memory_edit_allowed = False
        self.memory_save_allowed = False

    def tool(self, description: str, properties: dict | None = None, required: tuple[str, ...] = (), final: bool = False, timeout_s: float = 30.0):
        def register(handler: Handler) -> Handler:
            if handler.__name__ in self._disabled:
                return handler
            if handler.__name__ in self._tools:
                raise ValueError(f"duplicate tool: {handler.__name__}")
            parameters = {"type": "object", "properties": properties or {}, "required": list(required)}
            self._tools[handler.__name__] = Tool(handler.__name__, description, parameters, handler, final, self._group, timeout_s)
            return handler

        return register

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, selected: bool = False) -> list[dict]:
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self._tools.values() if not selected or self._selected is None or t.group in self._selected
        ]

    def select(self, text: str, previous_tools: list[str] = ()) -> None:
        """Conservative routing: unknown requests retain the complete enabled catalog."""
        hints = {
            "alarms": r"таймер|напом|будильник|засеки",
            "media": r"музык|громк|громче|тише|плеер|трек|пауз|звук",
            "weather": r"погод|температур|дожд|ветер|прогноз",
            "web": r"найди|поищи|поиск|сайт|страниц|расписани|новост|курс|пары|заняти",
            "computer": r"открой|закрой|окн|рабочий\s+стол|браузер|напечатай|программ",
            "coder": r"кодекс|кодексу|клод|агент|код\b|функци|навык|умени|апгрейд|перезапуст|откати|проект",
        }
        groups = {group for group, pattern in hints.items() if re.search(pattern, text, re.I)}
        if not groups:
            self._selected = None
            return
        groups.update(self._tools[n].group for n in previous_tools if n in self._tools)
        self._selected = groups | {"memory", "listening", "general", "catalog"}

    def catalog(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for tool in self._tools.values():
            groups.setdefault(tool.group, []).append(tool.name)
        return groups

    async def call(self, name: str, arguments: str) -> ToolResult:
        tool = self._tools.get(name)
        text = await self._run(tool, name, arguments)
        ok = not text.startswith("Ошибка")
        return ToolResult(text, ok, ok and tool is not None and tool.final)

    async def _run(self, tool: Tool | None, name: str, arguments: str) -> str:
        if tool is None:
            return f"Ошибка: нет инструмента {name}."
        try:
            args = json.loads(arguments) if arguments.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except ValueError as error:
            return f"Ошибка: аргументы не в формате JSON ({error})."
        known = inspect.signature(tool.handler).parameters
        args = {k: v for k, v in args.items() if k in known}
        try:
            for key in tool.parameters.get("required", []):
                if key not in args:
                    raise ToolError(f"не указан аргумент {key}")
            for key, value in args.items():
                schema = tool.parameters.get("properties", {}).get(key, {})
                if value is None and known[key].default is None:
                    continue
                expected = schema.get("type")
                valid = {
                    "string": isinstance(value, str), "integer": type(value) is int,
                    "number": type(value) in (int, float), "boolean": type(value) is bool,
                    "object": isinstance(value, dict), "array": isinstance(value, list),
                }
                if expected in valid and not valid[expected]:
                    raise ToolError(f"аргумент {key}: ожидается {expected}")
                if "enum" in schema and value not in schema["enum"]:
                    raise ToolError(f"недопустимое значение аргумента {key}")
            result = tool.handler(**args)
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, tool.timeout_s)
            # Only our fixed confirmations are inflected; retrieved text is evidence.
            if name in {"set_timer", "cancel_alarm", "cancel_timer", "remember", "forget", "ask_coder", "undo_coder_change"}:
                return self.persona.stock(str(result))
            return str(result)
        except TimeoutError:
            return f"Ошибка: инструмент {name} не ответил за {tool.timeout_s:g} секунд."
        except ToolError as error:
            return f"Ошибка: {error}"
        except TypeError as error:
            return f"Ошибка: неверные аргументы ({error})."
        except Exception as error:
            log.exception("tool %s crashed", name)
            return f"Ошибка: {type(error).__name__}: {error}"


def build_tools(ctx: ToolContext) -> ToolRegistry:
    from . import alarms, coder, computer, listening, media, memory_tools, weather, web

    modules = {"alarms": alarms, "memory": memory_tools, "web": web, "weather": weather, "media": media, "computer": computer, "coder": coder, "listening": listening}
    enabled = set(modules) if ctx.enabled_groups == "all" else {g.strip() for g in ctx.enabled_groups.split(",") if g.strip()}
    unknown = enabled - modules.keys()
    if unknown:
        raise ValueError(f"Unknown TOOL_GROUPS: {', '.join(sorted(unknown))}")
    registry = ToolRegistry()
    registry.persona = ctx.persona
    registry._disabled = {n.strip() for n in ctx.disabled_tools.split(",") if n.strip()}
    for name, module in modules.items():
        if name not in enabled:
            continue
        registry._group = name
        module.register(registry, ctx)
    registry._group = "catalog"

    @registry.tool("Список включённых навыков и доступных инструментов.", final=True)
    def list_skills() -> str:
        descriptions = {
            "alarms": "Таймеры и напоминания", "memory": "Память и сохранённые сценарии",
            "web": "Поиск и чтение сайтов", "weather": "Погода", "media": "Музыка и громкость",
            "computer": "Программы, окна и рабочие столы", "coder": "Задачи Кодексу и обновление моих умений",
            "listening": "Режим долгого слушания",
        }
        return "Умею: " + "; ".join(descriptions.get(g, g) for g in registry.catalog() if g != "catalog") + "."

    @registry.tool("Подключить к текущему запросу инструменты другого включённого навыка. Список — list_skills.", {"group": {"type": "string"}}, required=("group",))
    def enable_skill(group: str) -> str:
        if group not in registry.catalog():
            raise ToolError(f"навык {group} не включён в настройках")
        if registry._selected is not None:
            registry._selected.add(group)
        return f"Доступны: {', '.join(registry.catalog()[group])}."

    registry._group = "general"
    return registry
