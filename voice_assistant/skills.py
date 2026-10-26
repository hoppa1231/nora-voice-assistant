"""«Допиши функцию» / «сделай так»: a short dialogue, decided by code, that turns a vague
request into a task for the coding agent.

A bare request gets one clarifying question; the next phrase is taken as the details.
A request that already has details goes straight to ask_coder, which still asks «да или нет».
"""

from __future__ import annotations

import re
import time

ASK_DETAILS = "Хорошо. Расскажи подробнее: что должна делать новая функция, когда ей срабатывать и что ответить?"
TTL_S = 90  # the details must follow soon, otherwise the question is forgotten
MIN_DETAIL_WORDS = 2

_PREFIX = r"^\W*(?:нора\W+)?(?:пожалуйста\W+)?"
_NEW_FUNCTION = re.compile(
    _PREFIX + r"(?:допиши|добавь|создай|сделай)\s+(?:(?:мне|себе|новую|новое|новый|ещё|еще|одну|одно|такую)\s+)*"
    r"(?:функци[юя]|умение|навык)(?:\s+(?:мне|себе))?\b\W*(?P<details>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_DO_IT_LIKE = re.compile(_PREFIX + r"сделай\s+так(?:\W+пожалуйста)?\W*$", re.IGNORECASE)
_SO_THAT_YOU_CAN = re.compile(
    _PREFIX + r"сделай\s+так\W+чтобы\s+ты\s+(?:умела|могла|научилась|сама)\b\W*(?P<details>.*)$",
    re.IGNORECASE | re.DOTALL,
)

GUIDE = """Это просьба добавить Норе новую функцию (инструмент для модели). Требования:
- простая и понятная: один небольшой инструмент или доработка существующего, без лишних настроек;
- полезная в повседневных делах; если похожее уже есть — доработай его, а не дублируй;
- ясное описание инструмента для модели, короткий ответ голосом (если ответ готов сразу — final=True), понятные ошибки через ToolError;
- без новых платных сервисов и ключей, если можно обойтись;
- добавь тест и строку в таблицу «Что умеет» в README.md.
Если просьба неоднозначна, выбери самый простой разумный вариант и скажи о нём в ИТОГ."""


def parse_request(text: str) -> str | None:
    """None if the phrase is not a request for a new function; '' if it has no details yet;
    otherwise the phrase itself, to be passed to the agent."""
    if _DO_IT_LIKE.match(text):
        return ""
    match = _NEW_FUNCTION.match(text) or _SO_THAT_YOU_CAN.match(text)
    if match is None:
        return None
    return text.strip() if len(match.group("details").split()) >= MIN_DETAIL_WORDS else ""


def build_task(request: str, details: str = "") -> str:
    request = request.strip().rstrip(".!?…")
    text = f"{request}: {details.strip()}" if details.strip() else request
    return f"{text.rstrip('.')}.\n\n{GUIDE}"


class SkillDialogue:
    """Remembers that Nora has just asked for details."""

    def __init__(self) -> None:
        self._request: str | None = None
        self._asked = 0.0

    def ask(self, request: str) -> str:
        self._request, self._asked = request.strip(), time.monotonic()
        return ASK_DETAILS

    def take(self) -> str | None:
        """The original request, if a question about it is still open."""
        request, self._request = self._request, None
        if request is not None and time.monotonic() - self._asked > TTL_S:
            return None
        return request
