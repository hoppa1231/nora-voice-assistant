"""Tools the LLM can call. Each module registers its tools into a ToolRegistry."""

from __future__ import annotations

import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from ..listening import LongListening
from ..memory import Memory

log = logging.getLogger("va.tools")

Handler = Callable[..., Awaitable[str] | str]


class ToolError(Exception):
    """A failure the model should see and explain to the user."""


@dataclass
class PendingAction:
    description: str
    run: Callable[[], Awaitable[str]]
    created: float


class PendingActions:
    """An action waiting for a spoken «да»/«нет». The answer is parsed by code, not the
    model, so a misheard or hallucinated tool call cannot confirm itself."""

    TTL_S = 90

    def __init__(self) -> None:
        self._action: PendingAction | None = None

    def request(self, description: str, run: Callable[[], Awaitable[str]]) -> str:
        self._action = PendingAction(description, run, time.monotonic())
        return f"{description} Подтверди: скажи «да» или «нет»."

    def take(self) -> PendingAction | None:
        action, self._action = self._action, None
        if action and time.monotonic() - action.created > self.TTL_S:
            return None
        return action


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


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Handler
    # The result is already the full spoken answer, so the turn can end without
    # asking the LLM to rephrase it (saves one LLM round trip, ~1.5 s).
    final: bool = False


@dataclass
class ToolResult:
    text: str
    ok: bool
    final: bool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, description: str, properties: dict | None = None, required: tuple[str, ...] = (), final: bool = False):
        def register(handler: Handler) -> Handler:
            parameters = {"type": "object", "properties": properties or {}, "required": list(required)}
            self._tools[handler.__name__] = Tool(handler.__name__, description, parameters, handler, final)
            return handler

        return register

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self._tools.values()
        ]

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
            result = tool.handler(**args)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except ToolError as error:
            return f"Ошибка: {error}"
        except TypeError as error:
            return f"Ошибка: неверные аргументы ({error})."
        except Exception as error:
            log.exception("tool %s crashed", name)
            return f"Ошибка: {type(error).__name__}: {error}"


def build_tools(ctx: ToolContext) -> ToolRegistry:
    from . import alarms, coder, computer, listening, media, memory_tools, weather, web

    registry = ToolRegistry()
    for module in (alarms, memory_tools, web, weather, media, computer, coder, listening):
        module.register(registry, ctx)
    return registry
