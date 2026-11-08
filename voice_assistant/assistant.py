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

from .listening import STARTED, STOPPED, LongListening, only_command, parse_command
from .llm import ContextOverflow, LlmClient
from .commands import QuickCommands
from .claims import explains_nonexecution, needs_execution, unsupported_claim
from .quiet import QuietMode
from .memory import Memory
from .memory_dialogue import MemoryDialogue
from .prompts import build_system_prompt, build_turn_context
from .personas import NORA, Persona
from .routines import learn_routine, match_routine, routine_hint
from .skills import DETAILS_WINDOW_S, SkillDialogue, build_task, clean_request, parse_request
from .speech import SilentSpeaker, Speaker
from .tools import PendingActions, ToolRegistry
from .thinking import ThinkingCues

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
        listening: LongListening | None = None,
        quiet: QuietMode | None = None,
        persona: Persona = NORA,
        wake_word: str | None = None,
        thinking_cues: bool = True,
        thinking_delay_s: float = 4.0,
        thinking_interval_s: float = 14.0,
    ) -> None:
        self._llm = llm
        self.persona = persona
        self._wake_word = persona.wake_word if wake_word is None else wake_word or persona.wake_word
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
        self._skill = SkillDialogue()
        self._listening = listening or LongListening()
        self.quiet = quiet or QuietMode()
        self._thinking_cues = ThinkingCues(speaker, self.quiet, enabled=thinking_cues, delay=thinking_delay_s, interval=thinking_interval_s)
        self._quick = QuickCommands(tools)
        self._memory_dialogue = MemoryDialogue(memory, tools)

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
            if action.on_cancel:
                action.on_cancel()
            return "Ладно, не буду."
        if YES.match(user_text):
            try:
                return await action.run()
            except Exception as error:  # ToolError or a crash: either way say why
                log.warning("confirmed action failed: %s", error)
                return f"Не получилось: {error}"
        log.info("pending action dropped: %s", action.description)
        if action.on_cancel:
            action.on_cancel()
        return None  # any other request cancels the question and is handled normally

    async def _new_function(self, user_text: str, raw_text: str | None = None) -> str | None:
        """«Допиши функцию» / «сделай так»: ask for details, then hand the task to ask_coder.
        None if the phrase is not part of this dialogue."""
        raw_text = user_text if raw_text is None else raw_text
        request = self._skill.take()
        if request is not None:
            self._listening.stop()  # the details have been said
            if re.fullmatch(r"\W*(?:нет(?:[,\s]+не\s+надо)?|не\s+надо|не\s+нужно|отмена|отмени|стоп|не\s+делай)\W*", user_text, re.I):
                return "Ладно, не буду."
            task = build_task(request, raw_text, persona=self.persona)
        else:
            request = parse_request(user_text)
            if request is None:
                return None
            if not request:
                # The details are usually a long thought: listen patiently (no name, answer after a long pause).
                self._listening.start(DETAILS_WINDOW_S)
                return self._skill.ask(raw_text)
            task = build_task(raw_text, persona=self.persona)
        self._tools.coder_request = task
        result = await self._tools.call("ask_coder", json.dumps({"task": task, "mode": "change"}, ensure_ascii=False))
        self._direct_tool_calls += 1
        if not result.ok:
            return "Не получилось: " + result.text.removeprefix("Ошибка: ")
        return result.text

    def _switch_listening(self, user_text: str) -> str | None:
        """«Не перебивай» / «обычный режим». The mode is switched even inside a longer request;
        only a bare switch is answered here, anything else still goes to the model."""
        command = parse_command(user_text)
        if command is None:
            return None
        if command == "on":
            self._listening.start()
        else:
            self._listening.stop()
        log.info("long listening %s", command)
        if not only_command(user_text):
            return None
        return STARTED if command == "on" else self.persona.stock(STOPPED)

    def _system_prompt(self) -> str:
        prompt = build_system_prompt(self._memory.facts(), self._memory.routines(), self._default_city, self.persona)
        return prompt + "\nРеально включённые инструменты: " + ", ".join(self._tools.names) + ". Остальные действия недоступны."

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
        try:
            return await self._thinking_cues.wait(self._handle(user_text))
        finally:
            self._tools.user_text = None
            self._tools.raw_user_text = None
            self._tools.coder_request = None
            self._tools.memory_edit_allowed = False
            self._tools.memory_save_allowed = False

    async def _handle(self, user_text: str) -> tuple[str, TurnStats]:
        raw_text = user_text
        self._tools.raw_user_text = raw_text
        started = time.monotonic()
        stats = TurnStats()
        self._direct_tool_calls = 0
        self._memory_dialogue.changed = False
        self._memory_dialogue.used_tool = False
        user_text = clean_request(user_text)
        previous = [c["function"]["name"] for m in self._history for c in m.get("tool_calls") or []]
        self._tools.select(user_text, previous)
        # Keyboard callers use the same explicit wake-up rule as microphone input.
        if self.quiet.active:
            from .wake import find_wake_word
            addressed = find_wake_word(user_text, self._wake_word)
            if addressed is None:
                return "", stats
            self.quiet.wake()
            user_text = addressed or "Ты здесь?"
        if self.persona.male:
            from .wake import find_wake_word
            addressed = find_wake_word(user_text, self._wake_word)
            if addressed is not None:
                user_text = addressed or "Ты здесь?"
        self._tools.user_text = user_text
        self._tools.memory_edit_allowed = bool(
            re.match(r"^\W*(?:пожалуйста\W+)?(?:забудь|удали|перезапиши|обнови|исправь|измени|замени)\b", user_text, re.I)
            and (re.search(r"\b(?:факт|факты|память|памяти|информацию|запись)\b", user_text, re.I) or re.match(r"^\W*забудь\b", user_text, re.I))
        )
        confirmed = await self._confirm(user_text)
        if confirmed is None:
            confirmed = self.quiet.request(user_text)
            if confirmed is not None:
                self._listening.stop()
                self._listening.clear()
                self._pending.clear()
        if confirmed is None:
            confirmed = await self._memory_dialogue.handle(user_text)
            stats.tool_calls += int(self._memory_dialogue.used_tool)
            if self._memory_dialogue.changed:
                # Old facts must not survive a correction/deletion in dialogue history.
                self.reset_history()
        if confirmed is None:
            confirmed = await self._quick.handle(user_text)
            stats.tool_calls += int(self._quick.used_tool)
        if confirmed is None:
            confirmed = await self._new_function(user_text, raw_text)
        if confirmed is None:
            confirmed = self._switch_listening(user_text)
        if confirmed is not None:
            stats.tool_calls += self._direct_tool_calls
            confirmed = self.persona.stock(confirmed)
            self._speaker.feed(confirmed)
            self._speaker.end_of_text()
            recorded_text = "[Команда изменения памяти обработана]" if self._memory_dialogue.changed else user_text
            self._remember_turn([{"role": "user", "content": recorded_text}, {"role": "assistant", "content": confirmed}])
            stats.total_s = time.monotonic() - started
            return confirmed, stats
        memory_notice = self._memory_dialogue.observe(user_text)
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
        # The automatic notice is emitted only after a real SQLite insertion.
        successful_tools: set[str] = {"remember"} if memory_notice else set()
        memory_changed = False

        def on_text(delta: str) -> None:
            if stats.llm_first_text_s is None:
                stats.llm_first_text_s = time.monotonic() - started

        def emit(text: str) -> None:
            if text:
                spoken.append(text)
                self._speaker.feed(text)

        emit(memory_notice)

        if routine and routine.url:
            # The routine knows the page: open it in code with the question as the query,
            # instead of spending an LLM round trip (and risking a wrong query) on the call.
            emit("Проверяю. " if self.persona.male else "Ща гляну. ")
            await self._prefetch(routine.url, user_text, messages, turn, stats)
        require_tool = (routine is not None and not routine.url) or bool(ACTION_COMMAND.match(user_text)) or needs_execution(user_text)
        for step in range(self._max_tool_steps):
            reply = await self._complete(messages, on_text, require_tool and step == 0)
            if not reply.tool_calls:
                if (require_tool and stats.tool_calls == 0 and not explains_nonexecution(reply.text)) or unsupported_claim(reply.text, successful_tools):
                    emit("Не могу подтвердить выполнение: нет успешного вызова нужного инструмента. Попробуй сформулировать просьбу точнее.")
                else:
                    emit(reply.text)
                break
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
                emit("Проверяю. " if self.persona.male else "Ща гляну. ")
            results = []
            for call in reply.tool_calls:
                stats.tool_calls += 1
                t0 = time.monotonic()
                result = await self._tools.call(call.name, call.arguments)
                if result.ok:
                    successful_tools.add(call.name)
                    if call.name in {"remember", "update_fact", "forget"}:
                        memory_changed = True
                results.append(result)
                log.info("→ %s(%s) [%.1fs]: %s", call.name, _short_args(call.arguments), time.monotonic() - t0, _preview(result.text))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result.text})
                turn.append({"role": "tool", "tool_call_id": call.id, "content": _clip(result.text, HISTORY_TOOL_RESULT_CHARS)})
            if all(r.final for r in results):
                emit(" ".join(r.text for r in results))
                break
        else:
            emit(" Не получилось уложиться в несколько шагов, попробуй сформулировать иначе.")

        self._speaker.end_of_text()
        answer = "".join(spoken).strip()
        turn.append({"role": "assistant", "content": answer})
        if memory_changed:
            self.reset_history()
            self._remember_turn([{"role": "user", "content": "[Память изменена; актуальные факты в системном контексте]"}, {"role": "assistant", "content": "Запись памяти обработана."}])
        else:
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
                return await self._llm.complete(messages, self._tools.schemas(selected=True), on_text, require_tool=require_tool)
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
        return await self._llm.complete(messages, self._tools.schemas(selected=True), on_text, require_tool=require_tool)

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
