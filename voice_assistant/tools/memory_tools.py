"""Long-term memory tools: facts about the user and routines ("how to do X")."""

from __future__ import annotations

from . import ToolContext, ToolError, ToolRegistry


def register(reg: ToolRegistry, ctx: ToolContext) -> None:
    @reg.tool(
        "Запомнить навсегда факт о пользователе, его предпочтение или постоянную инструкцию.",
        {"fact": {"type": "string", "description": "Короткая формулировка от третьего лица, например «Пьёт кофе без сахара»."}},
        required=("fact",),
        final=True,
    )
    def remember(fact: str) -> str:
        ctx.memory.add_fact(fact)
        return "Запомнила."

    @reg.tool(
        "Забыть факт по его номеру из списка «Что ты знаешь о пользователе».",
        {"fact_id": {"type": "integer"}},
        required=("fact_id",),
        final=True,
    )
    def forget(fact_id: int) -> str:
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
