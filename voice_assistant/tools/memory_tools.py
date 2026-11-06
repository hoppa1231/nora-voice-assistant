"""Long-term memory tools: facts about the user and routines ("how to do X")."""

from __future__ import annotations

from . import ToolContext, ToolError, ToolRegistry
from ..memory_rules import fact_key, explicit_memory_request


def register(reg: ToolRegistry, ctx: ToolContext) -> None:
    @reg.tool(
        "Запомнить навсегда факт о пользователе, его предпочтение или постоянную инструкцию.",
        {"fact": {"type": "string", "description": "Короткая формулировка от третьего лица, например «Пьёт кофе без сахара»."}},
        required=("fact",),
        final=True,
    )
    def remember(fact: str) -> str:
        if reg.user_text is not None and not reg.memory_save_allowed and not explicit_memory_request(reg.user_text):
            raise ToolError("самостоятельная память обрабатывается отдельными правилами; явной просьбы запомнить не было")
        key = fact_key(fact)
        old = [f for f in ctx.memory.facts(limit=-1) if key and fact_key(f.text) == key]
        if len(old) == 1 and old[0].text != fact.strip():
            ctx.memory.update_fact(old[0].id, fact, key=key)
            return "Запись в памяти обновлена."
        ctx.memory.add_fact(fact, key=key)
        return "Запомнила."

    @reg.tool("Перезаписать существующий факт, сохранив его номер. Только по просьбе пользователя; при неоднозначности уточни номер.",
              {"fact_id": {"type": "integer"}, "fact": {"type": "string"}}, required=("fact_id", "fact"), final=True)
    def update_fact(fact_id: int, fact: str) -> str:
        if reg.user_text is not None and not reg.memory_edit_allowed:
            raise ToolError("перезапись факта разрешена только по просьбе пользователя")
        if not ctx.memory.update_fact(fact_id, fact, key=fact_key(fact)):
            raise ToolError(f"нет факта #{fact_id}")
        return "Запись в памяти обновлена."

    @reg.tool("Показать сохранённые факты с номерами для просмотра, исправления или удаления.", final=True)
    def list_facts() -> str:
        facts = ctx.memory.facts(limit=-1)
        return "В памяти пока нет фактов." if not facts else "Помню: " + "; ".join(f"#{f.id}: {f.text}" for f in facts)

    @reg.tool(
        "Забыть факт по его номеру из списка «Что ты знаешь о пользователе».",
        {"fact_id": {"type": "integer"}},
        required=("fact_id",),
        final=True,
    )
    def forget(fact_id: int) -> str:
        if reg.user_text is not None and not reg.memory_edit_allowed:
            raise ToolError("удаление факта разрешено только по просьбе пользователя")
        if not ctx.memory.delete_fact(int(fact_id)):
            raise ToolError(f"нет факта #{fact_id}")
        return "Забыла."

    @reg.tool(
        "Сохранить или обновить сценарий: как выполнять повторяющуюся просьбу (какой сайт открыть, "
        "какую группу или раздел выбрать, что искать). Сохраняй прямые ссылки, как только их нашёл.",
        {
            "name": {"type": "string", "description": "Короткое общее имя без дат, например «расписание»."},
            "instructions": {"type": "string", "description": "Пошагово, с точными адресами страниц и параметрами."},
            "triggers": {"type": "string", "description": "Слова через запятую, по которым понятно, что просьба про это: «расписание, пары, занятия»."},
            "url": {"type": "string", "description": "Прямой адрес страницы с нужными данными, если он уже известен."},
        },
        required=("name", "instructions", "triggers"),
    )
    def save_routine(name: str, instructions: str, triggers: str = "", url: str = "") -> str:
        ctx.memory.save_routine(name, instructions, [name, *triggers.split(",")], url)
        return f"Сценарий «{name}» сохранён."

    @reg.tool("Удалить сохранённый сценарий.", {"name": {"type": "string"}}, required=("name",))
    def delete_routine(name: str) -> str:
        if not ctx.memory.delete_routine(name):
            raise ToolError(f"нет сценария «{name}»")
        return f"Сценарий «{name}» удалён."
