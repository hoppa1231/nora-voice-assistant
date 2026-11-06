"""Transient PipeWire AEC routing and reversible per-player music ducking."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import shutil

log = logging.getLogger("va.audio_system")


async def run(*cmd: str, timeout: float = 4) -> str:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        raise
    if proc.returncode:
        raise RuntimeError(err.decode(errors="replace").strip() or f"{cmd[0]}: {proc.returncode}")
    return out.decode().strip()


class EchoCancellation:
    def __init__(self) -> None:
        self.module_id: str | None = None
        self.source = f"nora_aec_source_{os.getpid()}"
        self.sink = f"nora_aec_sink_{os.getpid()}"

    @property
    def active(self) -> bool:
        return self.module_id is not None

    async def start(self, source_master: str = "", sink_master: str = "") -> None:
        for binary in ("pactl", "parec", "pacat"):
            if not shutil.which(binary):
                raise RuntimeError(f"не установлен {binary}")
        source_master = source_master or await run("pactl", "get-default-source")
        sink_master = sink_master or await run("pactl", "get-default-sink")
        result = await run(
            "pactl", "load-module", "module-echo-cancel",
            f"source_name={self.source}", f"sink_name={self.sink}",
            f"source_master={source_master}", f"sink_master={sink_master}",
            "aec_method=webrtc", "rate=48000", "channels=1", "channel_map=mono",
        )
        if not result.isdigit():
            raise RuntimeError(f"некорректный идентификатор AEC: {result}")
        self.module_id = result
        log.info("AEC enabled: %s -> %s", source_master, sink_master)

    async def close(self) -> None:
        if self.module_id is not None:
            module_id, self.module_id = self.module_id, None
            try:
                await run("pactl", "unload-module", module_id)
            except (OSError, RuntimeError, TimeoutError):
                log.exception("could not unload owned AEC module %s", module_id)


class MusicDucker:
    """Only attenuate playing MPRIS players; never change the system volume."""

    def __init__(self, factor: float = 0.35) -> None:
        self.factor = min(1.0, max(0.05, factor))
        self._saved: dict[str, tuple[float, float]] = {}
        self._active = False

    async def update(self, active: bool) -> None:
        if active == self._active:
            return
        self._active = active
        if not active:
            await self.restore()
            return
        try:
            players = (await run("playerctl", "-l")).splitlines()
        except (OSError, RuntimeError, TimeoutError):
            return
        for player in players:
            try:
                if await run("playerctl", "--player", player, "status") != "Playing":
                    continue
                original = float(await run("playerctl", "--player", player, "volume"))
                lowered = original * self.factor
                self._saved[player] = original, lowered
                await run("playerctl", "--player", player, "volume", str(lowered))
            except (OSError, RuntimeError, TimeoutError, ValueError):
                log.debug("player %s does not support ducking", player)

    async def restore(self) -> None:
        for player, (original, lowered) in list(self._saved.items()):
            try:
                current = float(await run("playerctl", "--player", player, "volume"))
                # Keep deliberate user changes made during the conversation.
                if math.isclose(current, lowered, abs_tol=0.002):
                    await run("playerctl", "--player", player, "volume", str(original))
            except (OSError, RuntimeError, TimeoutError, ValueError):
                log.debug("player %s disappeared during ducking", player)
            self._saved.pop(player, None)
        self._active = False
