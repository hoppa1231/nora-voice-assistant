"""«Сделай апгрейд» / «допиши функцию» / «сделай так»: a short dialogue, decided by code, that turns a vague
request into a task for the coding agent.

A bare request gets one clarifying question; the next phrase is taken as the details.
A request that already has details goes straight to ask_coder, which still asks «да или нет».
"""

from __future__ import annotations

import re
import json
import time
from .personas import NORA, Persona

ASK_DETAILS = (
    "Давай, излагай задачу для Кодекса. Говори спокойно, я не перебью; "
    "когда закончишь, помолчи несколько секунд."
)
DETAILS_WINDOW_S = 180.0  # patient listening while the task is dictated
TTL_S = 90  # the details must follow soon, otherwise the question is forgotten
MIN_DETAIL_WORDS = 2


def clean_request(text: str) -> str:
    """Remove spoken hesitations only; keep the actual task and its parameters."""
    return re.sub(r"(?i)(?<![а-яёa-z])(?:а-а+|э-э+|э+|эм+|м-м+)(?![а-яёa-z])[,\s]*", " ", text).strip(" ,")

_PREFIX = r"^\W*(?:(?:нора|джарви[зс]|jarvis)\W+)?(?:пожалуйста\W+)?"
_NEW_FUNCTION = re.compile(
    _PREFIX + r"(?:допиши|добавь|создай|сделай)\s+(?:(?:мне|себе|сейчас|пожалуйста|новую|новое|новый|ещё|еще|одну|одно|такую|через\s+(?:кодекс|кодекса|клода))[,\s]+)*"
    r"(?:функци[юя]|умение|навык)(?:\s+(?:мне|себе))?\b\W*(?P<details>.*)$",
    re.IGNORECASE | re.DOTALL,
)
# «Сделай апгрейд», «апгрейд: научись…», «давай обновим тебя» — any change to Nora herself.
_UPGRADE = re.compile(
    _PREFIX + r"(?:(?:сделай|сделаем|давай|запусти|нужен|нужно)\s+(?:себе\s+)?)?"
    r"(?:апгрейд\w*|апгрэйд\w*|upgrade|заапгрейд\w*|заапргрейд\w*|обнови\s+себя|обновим\s+тебя|улучши\s+себя|прокачай\s+себя|доработай\s+себя)\b\W*(?P<details>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_DO_IT_LIKE = re.compile(_PREFIX + r"сделай\s+так(?:\W+пожалуйста)?\W*$", re.IGNORECASE)
_SO_THAT_YOU_CAN = re.compile(
    _PREFIX + r"сделай\s+так\W+чтобы\s+ты\s+(?:умел[аи]?|мог(?:ла)?|научил(?:ась|ся)|сам[а]?)\b\W*(?P<details>.*)$",
    re.IGNORECASE | re.DOTALL,
)

QUOTE_HEADER = "Дословные реплики пользователя (JSON, без пересказа):\n"


def parse_request(text: str) -> str | None:
    """None if the phrase is not a request for a new function; '' if it has no details yet;
    otherwise the phrase itself, to be passed to the agent."""
    text = clean_request(text)
    if _DO_IT_LIKE.match(text):
        return ""
    match = _UPGRADE.match(text) or _NEW_FUNCTION.match(text) or _SO_THAT_YOU_CAN.match(text)
    if match is None:
        return None
    return text.strip() if len(match.group("details").split()) >= MIN_DETAIL_WORDS else ""


def build_task(request: str, details: str = "", persona: Persona = NORA) -> str:
    quotes = [request, details] if details else [request]
    return (
        QUOTE_HEADER + json.dumps(quotes, ensure_ascii=False, indent=2)
        + f"\n\nСлужебный контекст (не слова пользователя): это запрос на изменение кода помощника «{persona.name}». "
        "Самостоятельно разбери намерение по исходным репликам, не подменяй их пересказом ассистента "
        "и не ограничивай задачу добавлением одного инструмента. Проверь результат тестами."
    )


class SkillDialogue:
    """Remembers that Nora has just asked for details."""

    def __init__(self) -> None:
        self._request: str | None = None
        self._asked = 0.0

    def ask(self, request: str) -> str:
        self._request, self._asked = request, time.monotonic()
        return ASK_DETAILS

    def take(self) -> str | None:
        """The original request, if a question about it is still open."""
        request, self._request = self._request, None
        if request is not None and time.monotonic() - self._asked > TTL_S:
            return None
        return request
