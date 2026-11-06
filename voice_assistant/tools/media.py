"""Music and volume control through playerctl (MPRIS) and wpctl (PipeWire)."""

from __future__ import annotations

import asyncio
import re

from . import ToolContext, ToolError, ToolRegistry

SINK = "@DEFAULT_AUDIO_SINK@"
PLAYERCTL = {"play": "play", "pause": "pause", "toggle": "play-pause", "next": "next", "previous": "previous"}


async def run(*cmd: str) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError as error:
        raise ToolError(f"не установлен {cmd[0]}") from error
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=5)
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        raise
    if proc.returncode != 0:
        message = (err or out).decode().strip()
        if "No players found" in message:
            raise ToolError("нет запущенного плеера")
        raise ToolError(message or f"{cmd[0]} завершился с кодом {proc.returncode}")
    return out.decode().strip()


async def current_volume() -> str:
    raw = await run("wpctl", "get-volume", SINK)  # "Volume: 0.40 [MUTED]"
    match = re.search(r"([\d.]+)", raw)
    level = round(float(match.group(1)) * 100) if match else None
    return f"{level}%" + (", звук выключен" if "MUTED" in raw else "")


def register(reg: ToolRegistry, ctx: ToolContext) -> None:
    @reg.tool(
        "Управление музыкой/видео в плеере.",
        {"action": {"type": "string", "enum": list(PLAYERCTL), "description": "toggle = пауза/продолжить"}},
        required=("action",),
        final=True,
    )
    async def media_control(action: str) -> str:
        if action not in PLAYERCTL:
            raise ToolError(f"неизвестное действие {action}")
        await run("playerctl", PLAYERCTL[action])
        if action in ("next", "previous", "play"):
            try:
                title = await run("playerctl", "metadata", "--format", "{{artist}} — {{title}}")
                return f"Играет: {title}."
            except ToolError:
                pass
        return "Готово."

    @reg.tool(
        "Громкость системы.",
        {
            "action": {"type": "string", "enum": ["up", "down", "set", "mute", "unmute", "get"]},
            "percent": {"type": "integer", "description": "Для set — уровень 0–100; для up/down — шаг (по умолчанию 10)."},
        },
        required=("action",),
        final=True,
    )
    async def volume(action: str, percent: int | None = None) -> str:
        if action in ("up", "down"):
            step = int(percent or 10)
            await run("wpctl", "set-volume", "-l", "1.0", SINK, f"{step}%{'+' if action == 'up' else '-'}")
        elif action == "set":
            if percent is None:
                raise ToolError("для set нужен percent")
            await run("wpctl", "set-volume", "-l", "1.0", SINK, f"{max(0, min(100, int(percent)))}%")
        elif action in ("mute", "unmute"):
            await run("wpctl", "set-mute", SINK, "1" if action == "mute" else "0")
        elif action != "get":
            raise ToolError(f"неизвестное действие {action}")
        return f"Громкость {await current_volume()}."
