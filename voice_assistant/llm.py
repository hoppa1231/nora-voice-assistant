"""Streaming chat client for any OpenAI-compatible server (LM Studio by default)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import APIError, AsyncOpenAI

from .config import LlmConfig


class ContextOverflow(Exception):
    """The prompt does not fit the model's context window (LM Studio: «Context size has been exceeded»)."""


def _is_overflow(error: Exception) -> bool:
    text = str(error).lower()
    return "context" in text and ("exceed" in text or "overflow" in text or "too long" in text)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LlmClient:
    def __init__(self, config: LlmConfig) -> None:
        self._cfg = config
        self._client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key, timeout=120)
        # One request at a time: LM Studio splits one context window between parallel
        # requests, so a background warm-up next to a real turn could overflow it.
        self._lock = asyncio.Lock()

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        on_text: Callable[[str], None],
        require_tool: bool = False,
    ) -> Reply:
        """Streams the answer: text deltas go to `on_text` as they arrive, tool calls are collected."""
        async with self._lock:
            try:
                return await self._complete(messages, tools, on_text, require_tool)
            except APIError as error:
                if _is_overflow(error):
                    raise ContextOverflow(str(error)) from error
                raise

    async def _complete(self, messages: list[dict], tools: list[dict], on_text: Callable[[str], None], require_tool: bool) -> Reply:
        stream = await self._client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,
            tools=tools or None,
            tool_choice=("required" if require_tool else "auto") if tools else None,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            stream=True,
            extra_body=self._cfg.extra_body,
        )
        text: list[str] = []
        calls: dict[int, ToolCall] = {}
        try:
            await self._read(stream, text, calls, on_text)
        finally:
            await stream.close()  # interrupted: tell the server to stop generating
        tool_calls = [calls[i] for i in sorted(calls)]
        for i, call in enumerate(tool_calls):
            call.id = call.id or f"call_{i}"
        return Reply("".join(text), tool_calls)

    @staticmethod
    async def _read(stream, text: list[str], calls: dict[int, ToolCall], on_text: Callable[[str], None]) -> None:
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text.append(delta.content)
                on_text(delta.content)
            for part in delta.tool_calls or []:
                call = calls.setdefault(part.index, ToolCall(id="", name="", arguments=""))
                if part.id:
                    call.id = part.id
                if part.function and part.function.name:
                    call.name += part.function.name
                if part.function and part.function.arguments:
                    call.arguments += part.function.arguments

    async def warm_up(self, messages: list[dict], tools: list[dict]) -> None:
        """Have the server process (and cache) this prompt prefix before the real request needs it."""
        async with self._lock:
            await self._client.chat.completions.create(
                model=self._cfg.model,
                messages=messages,
                tools=tools or None,
                max_tokens=1,
                temperature=0,
                extra_body=self._cfg.extra_body,
            )
