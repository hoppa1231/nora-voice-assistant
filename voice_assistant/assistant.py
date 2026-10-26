"""One conversation turn: prompt -> streamed answer -> tool calls -> spoken reply."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .llm import ContextOverflow, LlmClient
from .memory import Memory
from .prompts import build_system_prompt, build_turn_context
from .routines import learn_routine, match_routine, routine_hint
from .speech import SilentSpeaker, Speaker
from .tools import PendingActions, ToolRegistry

log = logging.getLogger("va.assistant")

HISTORY_TOOL_RESULT_CHARS = 300
OVERFLOW_TOOL_RESULT_CHARS = 1500
SLOW_TOOLS = {"web_fetch", "web_search"}

YES = re.compile(r"^\W*(?:да|давай|запускай|подтверждаю|ага|конечно|выполняй|делай|можно|угу)\b", re.IGNORECASE)
NO = re.compile(r"\b(?:нет|не\s+надо|не\s+нужно|отмена|отмени|стоп|не\s+делай)\b", re.IGNORECASE)

# Commands that always need a tool. A small model sometimes just answers "done"
# to these, so the first request forces a tool call (LM Studio: tool_choice=required).
ACTION_COMMAND = re.compile(
    r"^\W*(?:пожалуйста\W+)?(?:запомни|запиши|забудь|напомни|засеки|поставь\s+(?:таймер|будильник|напоминание)|"
    r"таймер\b|отмени|сохрани|включи|выключи|поставь\s+на\s+паузу|сделай\s+(?:громче|тише|погромче|потише))",
    re.IGNORECASE,
)


@dataclass
class TurnStats:
    llm_first_text_s: float | None = None
    tool_calls: int = 0
    total_s: float = 0.0


class Assistant:
    def __init__(
        self,
        llm: LlmClient,
        tools: ToolRegistry,
        memory: Memory,
        speaker: Speaker | SilentSpeaker,
        timezone: ZoneInfo,
        default_city: str = "",
        history_messages: int = 10,
        max_tool_steps: int = 6,
        pending: PendingActions | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._memory = memory
        self._speaker = speaker
        self._tz = timezone
        self._default_city = default_city
        self._turns: list[list[dict]] = []
        self._history_turns = max(2, history_messages // 2)
        self._max_tool_steps = max_tool_steps
        self._cached_system: str | None = None
        self._background: set[asyncio.Task] = set()
        self._pending = pending or PendingActions()

    @property
    def _history(self) -> list[dict]:
        return [message for turn in self._turns for message in turn]

    def reset_history(self) -> None:
        self._turns.clear()

    def add_announcement(self, text: str) -> None:
        """Something said outside a turn (agent finished): keep it so «а подробнее?» works."""
        self._remember_turn([{"role": "user", "content": "[Служебно: фоновая задача завершилась]"}, {"role": "assistant", "content": text}])

    def _remember_turn(self, turn: list[dict]) -> None:
        self._turns.append(turn)
        if len(self._turns) > self._history_turns:
            # Drop half at once: every trim invalidates the server's prompt cache.
            self._turns = self._turns[-(self._history_turns // 2) :]

    async def _confirm(self, user_text: str) -> str | None:
        """Answer to a pending «Подтверди» question, decided by code; None if it is not one."""
        action = self._pending.take()
        if action is None:
            return None
        if NO.search(user_text):  # checked first: «да нет, не надо» is a no
            return "Хорошо, не буду."
        if YES.match(user_text):
            try:
                return await action.run()
            except Exception as error:  # ToolError or a crash: either way say why
                log.warning("confirmed action failed: %s", error)
                return f"Не получилось: {error}"
        log.info("pending action dropped: %s", action.description)
        return None  # any other request cancels the question and is handled normally

    def _system_prompt(self) -> str:
        return build_system_prompt(self._memory.facts(), self._memory.routines(), self._default_city)

    async def warm_up(self) -> None:
        """Pre-compute the prompt prefix after it changed (startup, new fact or routine).

        Otherwise the next request pays ~7 s on the iGPU to reprocess ~2k tokens.
        """
        system = self._system_prompt()
        if system == self._cached_system:
            return
        self._cached_system = system
        messages = [{"role": "system", "content": system}, *self._history, {"role": "user", "content": "."}]
        started = time.monotonic()
        try:
            await self._llm.warm_up(messages, self._tools.schemas())
        except Exception as error:
            log.warning("LLM warm-up failed: %s", error)
            self._cached_system = None
            return
        log.debug("prompt cache warmed in %.1fs", time.monotonic() - started)

    async def handle(self, user_text: str) -> tuple[str, TurnStats]:
        started = time.monotonic()
        stats = TurnStats()
        confirmed = await self._confirm(user_text)
        if confirmed is not None:
            self._speaker.feed(confirmed)
            self._speaker.end_of_text()
            self._remember_turn([{"role": "user", "content": user_text}, {"role": "assistant", "content": confirmed}])
            stats.total_s = time.monotonic() - started
            return confirmed, stats
        system = self._system_prompt()
        self._cached_system = system
        context = build_turn_context(datetime.now(self._tz), self._memory.pending_alarms())
        routine = match_routine(user_text, self._memory.routines())
        if routine:
            log.info("routine «%s» matches", routine.name)
            context += "\n" + routine_hint(routine)
        # The turn is recorded exactly as sent, tool calls included: the server's prompt
        # cache needs an unchanged prefix, and a history of "done!" replies without
        # tool calls teaches the model to claim actions it never performed.
        turn: list[dict] = [{"role": "user", "content": f"{context}\n{user_text}"}]
        messages = [{"role": "system", "content": system}, *self._history, *turn]
        spoken: list[str] = []

        def on_text(delta: str) -> None:
            if stats.llm_first_text_s is None:
                stats.llm_first_text_s = time.monotonic() - started
            spoken.append(delta)
            self._speaker.feed(delta)

        if routine and routine.url:
            # The routine knows the page: open it in code with the question as the query,
            # instead of spending an LLM round trip (and risking a wrong query) on the call.
            on_text("Сейчас посмотрю. ")
            await self._prefetch(routine.url, user_text, messages, turn, stats)
        require_tool = (routine is not None and not routine.url) or bool(ACTION_COMMAND.match(user_text))
        for step in range(self._max_tool_steps):
            reply = await self._complete(messages, on_text, require_tool and step == 0)
            if not reply.tool_calls:
                break
            if reply.text and not reply.text.endswith((" ", "\n")):
                on_text(" ")
            call_message = {
                "role": "assistant",
                "content": reply.text or None,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c in reply.tool_calls
                ],
            }
            messages.append(call_message)
            turn.append(call_message)
            if not spoken and any(c.name in SLOW_TOOLS for c in reply.tool_calls):
                on_text("Сейчас посмотрю. ")  # web pages take a while; don't sit in silence
            results = []
            for call in reply.tool_calls:
                stats.tool_calls += 1
                t0 = time.monotonic()
                result = await self._tools.call(call.name, call.arguments)
                results.append(result)
                log.info("→ %s(%s) [%.1fs]: %s", call.name, _short_args(call.arguments), time.monotonic() - t0, _preview(result.text))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result.text})
                turn.append({"role": "tool", "tool_call_id": call.id, "content": _clip(result.text, HISTORY_TOOL_RESULT_CHARS)})
            if all(r.final for r in results):
                if not reply.text.strip():  # the model said nothing yet: its result is the answer
                    on_text(" ".join(r.text for r in results))
                break
        else:
            on_text(" Не получилось уложиться в несколько шагов, попробуй сформулировать иначе.")

        self._speaker.end_of_text()
        answer = "".join(spoken).strip()
        turn.append({"role": "assistant", "content": answer})
        self._remember_turn(turn)
        stats.total_s = time.monotonic() - started
        names = [c["function"]["name"] for m in turn for c in m.get("tool_calls") or []]
        if "web_fetch" in names and "save_routine" not in names and routine is None:
            self._spawn(self._learn_routine(turn))
        elif self._system_prompt() != system:
            self._spawn(self.warm_up())
        return answer, stats

    async def _complete(self, messages: list[dict], on_text, require_tool: bool):
        """One LLM call; if the prompt overflows the context, shed history, then shorten tool results."""
        for _ in range(3):
            try:
                return await self._llm.complete(messages, self._tools.schemas(), on_text, require_tool=require_tool)
            except ContextOverflow:
                history = len(self._history)
                if history:
                    log.warning("context overflow: dropping %d history messages", history)
                    del messages[1 : 1 + history]
                    self._turns.clear()
                    continue
                clipped = False
                for message in messages:
                    if message["role"] == "tool" and len(message["content"]) > OVERFLOW_TOOL_RESULT_CHARS:
                        message["content"] = _clip(message["content"], OVERFLOW_TOOL_RESULT_CHARS)
                        clipped = True
                if not clipped:
                    raise
                log.warning("context overflow: tool results shortened")
        return await self._llm.complete(messages, self._tools.schemas(), on_text, require_tool=require_tool)

    async def _prefetch(self, url: str, query: str, messages: list[dict], turn: list[dict], stats: TurnStats) -> None:
        arguments = json.dumps({"url": url, "query": query}, ensure_ascii=False)
        call = {"id": "routine_fetch", "type": "function", "function": {"name": "web_fetch", "arguments": arguments}}
        t0 = time.monotonic()
        result = await self._tools.call("web_fetch", arguments)
        stats.tool_calls += 1
        log.info("→ web_fetch(%s) [%.1fs, routine]: %s", _short_args(arguments), time.monotonic() - t0, _preview(result.text))
        for target in (messages, turn):
            target.append({"role": "assistant", "content": None, "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.text})
        turn.append({"role": "tool", "tool_call_id": call["id"], "content": _clip(result.text, HISTORY_TOOL_RESULT_CHARS)})

    async def drain(self, timeout: float = 20) -> None:
        """Let background work (routine learning, cache warm-up) finish before exit."""
        if self._background:
            await asyncio.wait(self._background, timeout=timeout)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _learn_routine(self, turn: list[dict]) -> None:
        try:
            await learn_routine(self._llm, self._memory, turn)
        except Exception as error:
            log.warning("routine learning failed: %s", error)
        await self.warm_up()


def _short_args(arguments: str) -> str:
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return arguments[:80]
    return ", ".join(f"{k}={v!r}" for k, v in args.items())[:160]


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _preview(result: str) -> str:
    return _clip(" ".join(result.split()), 160)
