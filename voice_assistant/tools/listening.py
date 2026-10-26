"""Long listening for the model: the usual phrases are caught in code (assistant.py),
this covers unusual wording."""

from __future__ import annotations

from ..listening import STARTED, STOPPED
from . import ToolContext, ToolRegistry


def register(reg: ToolRegistry, ctx: ToolContext) -> None:
    @reg.tool(
        "Долгое слушание: пользователь хочет говорить долго, не перебрасываться короткими фразами, просит не перебивать. "
        "on=true — до 10 минут слушать без имени и отвечать только после долгой паузы; on=false — обычный режим.",
        {"on": {"type": "boolean"}},
        final=True,
    )
    def long_listening(on: bool = True) -> str:
        if on:
            ctx.listening.start()
            return STARTED
        ctx.listening.stop()
        return STOPPED
