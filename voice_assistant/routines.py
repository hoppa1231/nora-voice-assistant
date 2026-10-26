"""Saved routines: recognising a request that matches one, and learning new ones.

A 4B model reliably follows a routine handed to it in the current message, but
not one buried in the system prompt, and it often forgets to save one while busy
navigating a site. So both steps are done by code around the model.
"""

from __future__ import annotations

import json
import logging
import re

from .llm import LlmClient
from .memory import Memory, Routine

log = logging.getLogger("va.routines")


def _stem(word: str) -> str:
    # Russian endings vary: "пары", "пар", "парах". Keep the first 3-5 letters.
    return word[: max(3, min(5, len(word) - 1))]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е"))


def match_routine(text: str, routines: list[Routine]) -> Routine | None:
    words = _words(text)
    for routine in routines:
        for trigger in routine.triggers or (routine.name,):
            stems = [_stem(w) for w in _words(trigger) if len(w) >= 3]
            if stems and all(any(w.startswith(s) for w in words) for s in stems):
                return routine
    return None


def routine_hint(routine: Routine) -> str:
    if routine.url:
        return f"[Подходит сохранённый сценарий «{routine.name}»: {routine.instructions} Страница уже открыта ниже; ответь по ней, ничего не переспрашивая.]"
    return f"[Подходит сохранённый сценарий «{routine.name}»: {routine.instructions} Выполни его сейчас, ничего не переспрашивая.]"


LEARN_PROMPT = """Ты помогаешь голосовому ассистенту запоминать, как выполнять повторяющиеся просьбы.
Ниже просьба пользователя, страницы, которые ассистент открыл (по порядку), и его ответ.

Если пользователь объяснил, где искать то, что он будет спрашивать снова (например, своё расписание: сайт, раздел, группа),
вызови learn_routine. Если сохранённый сценарий уже это покрывает, просьба разовая или данные не нашлись — вызови skip."""

LEARN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "learn_routine",
            "description": "Сохранить сценарий.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Короткое общее имя без дат: «расписание»."},
                    "triggers": {"type": "string", "description": "Слова через запятую, по которым понятно, что просьба про это: «расписание, пары, занятия»."},
                    "url": {"type": "string", "description": "Прямой адрес страницы, где нашлись данные (из открытых)."},
                    "note": {"type": "string", "description": "Что ещё важно: группа, подгруппа. Можно пусто."},
                },
                "required": ["name", "triggers", "url"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "skip", "description": "Ничего не сохранять.", "parameters": {"type": "object", "properties": {}}},
    },
]


def _fetched_pages(turn: list[dict]) -> list[tuple[str, str]]:
    """(url, first line of the result) for every successful web_fetch in the turn."""
    results = {m["tool_call_id"]: m["content"] for m in turn if m["role"] == "tool"}
    pages = []
    for message in turn:
        for call in message.get("tool_calls") or []:
            if call["function"]["name"] != "web_fetch":
                continue
            result = results.get(call["id"], "")
            match = re.match(r"Адрес: (\S+)", result)
            if match:
                pages.append((match.group(1), " ".join(result.split())[:200]))
    return pages


async def learn_routine(llm: LlmClient, memory: Memory, turn: list[dict]) -> Routine | None:
    pages = _fetched_pages(turn)
    if not pages:
        return None
    existing = "\n".join(f"«{r.name}»: {r.instructions}" for r in memory.routines()) or "нет"
    user_text = turn[0]["content"].split("\n", 1)[-1]
    report = (
        f"Просьба пользователя: {user_text}\n\nОткрытые страницы:\n"
        + "\n".join(f"- {head}" for _, head in pages)
        + f"\n\nОтвет ассистента: {turn[-1]['content']}\n\nУже сохранённые сценарии: {existing}"
    )
    reply = await llm.complete(
        [{"role": "system", "content": LEARN_PROMPT}, {"role": "user", "content": report}],
        LEARN_TOOLS,
        on_text=lambda _: None,
        require_tool=True,
    )
    call = next((c for c in reply.tool_calls if c.name == "learn_routine"), None)
    if call is None:
        return None
    try:
        args = json.loads(call.arguments)
        name = args["name"].strip()
    except (ValueError, KeyError, AttributeError):
        return None
    urls = [url for url, _ in pages]
    url = args.get("url", "").strip()
    if url not in urls:
        url = urls[-1]  # never trust a URL the model did not actually open
    instructions = f"Данные на странице {url}. Ответь на вопрос по ней."
    if str(args.get("note", "")).strip():
        instructions += f" {str(args['note']).strip()}"
    triggers = list(dict.fromkeys(t.strip().lower() for t in [name, *str(args.get("triggers", "")).split(",")] if t.strip()))
    memory.save_routine(name, instructions, triggers, url)
    log.info("learned routine «%s» (%s): %s", name, ", ".join(triggers), url)
    return Routine(name.lower(), instructions, tuple(triggers), url)
