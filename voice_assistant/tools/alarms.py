"""Timers and reminders. Durations and times are parsed here, not by the model:
small models get "полтора часа" or time zones wrong."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from . import ToolContext, ToolError, ToolRegistry

_WORD_NUMBERS = {
    "ноль": 0, "один": 1, "одну": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
    "восемнадцать": 18, "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40,
    "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90, "сто": 100,
}
_UNITS = (
    (re.compile(r"^(час|ч$|h$|hour)"), 3600),
    (re.compile(r"^(мин|м$|m$)"), 60),
    (re.compile(r"^(сек|с$|s$)"), 1),
)
_ISO = re.compile(r"^P?T?(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?$", re.I)


def parse_duration(text: str) -> int:
    """'5 минут', 'полтора часа', '1 час 20 минут', 'PT90S' -> seconds."""
    raw = text.strip()
    if re.search(r"-\s*\d|\bминус\b", raw, re.I):
        raise ToolError("длительность должна быть положительной")
    iso = _ISO.match(raw.replace(" ", ""))
    if iso and any(iso.groups()):
        h, m, s = (float(g or 0) for g in iso.groups())
        return round(h * 3600 + m * 60 + s)

    t = raw.lower().replace("ё", "е").replace(",", ".")
    t = re.sub(r"\bполтора\b|\bполторы\b", "1.5", t)
    t = re.sub(r"\bпол\s*(часа|минуты)", r"0.5 \1", t)
    t = re.sub(r"\bчетверть\s+часа", "15 минут", t)
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-zа-я]+", t)

    total = 0.0
    number: float | None = None
    last_unit: int | None = None
    for token in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            number = (number or 0) + float(token)
            continue
        if token in _WORD_NUMBERS:
            number = (number or 0) + _WORD_NUMBERS[token]
            continue
        for pattern, seconds in _UNITS:
            if pattern.match(token):
                total += (1 if number is None else number) * seconds
                number = None
                last_unit = seconds
                break
    if number is not None:
        # "час двадцать" -> the trailing number is in the next smaller unit;
        # a bare number for a timer most often means minutes.
        total += number * (60 if last_unit in (None, 3600) else 1)
    if total <= 0:
        raise ToolError(f"не понял длительность «{text}»")
    return round(total)


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """plural(5, ("минута", "минуты", "минут")) -> "5 минут"."""
    if n % 10 == 1 and n % 100 != 11:
        form = forms[0]
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        form = forms[1]
    else:
        form = forms[2]
    return f"{n} {form}"


def say_duration(seconds: int) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(plural(hours, ("час", "часа", "часов")))
    if minutes:
        parts.append(plural(minutes, ("минуту", "минуты", "минут")))
    if secs:
        parts.append(plural(secs, ("секунду", "секунды", "секунд")))
    return " ".join(parts)


def say_when(due: datetime, now: datetime) -> str:
    days = (due.date() - now.date()).days
    day = {0: "сегодня", 1: "завтра", 2: "послезавтра"}.get(days, f"{due:%d.%m}")
    return f"{day} в {due:%H:%M}"


def parse_local_time(text: str, ctx: ToolContext) -> datetime:
    """ISO date/time in the user's zone. A zone suffix (often a bogus 'Z') is ignored on purpose."""
    cleaned = re.sub(r"(Z|[+-]\d\d:?\d\d)$", "", text.strip().replace("T", " "))
    try:
        naive = datetime.fromisoformat(cleaned)
    except ValueError as error:
        raise ToolError(f"не понял время «{text}», нужен формат ГГГГ-ММ-ДД ЧЧ:ММ") from error
    return naive.replace(tzinfo=ctx.timezone)


def register(reg: ToolRegistry, ctx: ToolContext) -> None:
    @reg.tool(
        "Поставить таймер.",
        {
            "duration": {"type": "string", "description": "Длительность словами, как сказал пользователь: «5 минут», «полтора часа», «1 час 20 минут»."},
            "label": {"type": "string", "description": "Только если пользователь сказал, для чего таймер («чай», «пельмени»). Иначе не передавай."},
        },
        required=("duration",),
        final=True,
    )
    def set_timer(duration: str, label: str = "") -> str:
        seconds = parse_duration(duration)
        if label.strip().lower() in ("таймер", "timer", duration.strip().lower()):
            label = ""  # the model sometimes fills the label with nothing new
        due = datetime.now(UTC) + timedelta(seconds=seconds)
        ctx.memory.add_alarm("timer", due, label or f"таймер на {say_duration(seconds)}")
        return f"Поставила таймер «{label}» на {say_duration(seconds)}." if label else f"Поставила таймер на {say_duration(seconds)}."

    @reg.tool(
        "Поставить напоминание на дату и время.",
        {
            "when": {"type": "string", "description": "Местное время пользователя, формат ГГГГ-ММ-ДД ЧЧ:ММ, без часового пояса."},
            "text": {"type": "string", "description": "О чём напомнить."},
        },
        required=("when", "text"),
        final=True,
    )
    def add_reminder(when: str, text: str) -> str:
        due = parse_local_time(when, ctx)
        now = datetime.now(ctx.timezone)
        if due <= now:
            raise ToolError(f"время {due:%Y-%m-%d %H:%M} уже прошло")
        ctx.memory.add_alarm("reminder", due, text)
        return f"Напомню {say_when(due, now)}: {text}."

    @reg.tool("Список активных таймеров и напоминаний.")
    def list_alarms() -> str:
        alarms = ctx.memory.pending_alarms()
        if not alarms:
            return "Активных таймеров и напоминаний нет."
        now = datetime.now(UTC)
        lines = []
        for a in alarms:
            when = say_when(a.due_at.astimezone(ctx.timezone), now.astimezone(ctx.timezone))
            left = say_duration(max(1, round((a.due_at - now).total_seconds())))
            kind = "таймер" if a.kind == "timer" else "напоминание"
            lines.append(f"#{a.id} {kind} «{a.text}» {when}, осталось {left}")
        return "\n".join(lines)

    @reg.tool(
        "Отменить таймер или напоминание по номеру.",
        {"alarm_id": {"type": "integer"}},
        required=("alarm_id",),
        final=True,
    )
    def cancel_alarm(alarm_id: int) -> str:
        if not ctx.memory.finish_alarm(int(alarm_id)):
            raise ToolError(f"нет активного таймера или напоминания #{alarm_id}")
        return "Отменила."

    @reg.tool(
        "Отменить единственный таймер или все таймеры. Напоминания не затрагивает.",
        {"all_timers": {"type": "boolean"}}, final=True,
    )
    def cancel_timer(all_timers: bool = False) -> str:
        timers = [a for a in ctx.memory.pending_alarms() if a.kind == "timer"]
        if not timers:
            return "Активных таймеров нет."
        if len(timers) > 1 and not all_timers:
            return "Таймеров несколько. Уточни номер или скажи: отмени все таймеры."
        for timer in timers:
            ctx.memory.finish_alarm(timer.id)
        return "Отменила все таймеры." if all_timers else "Отменила таймер."
