"""Feeds the on-screen status pill (overlay.py, run by the system Python with GTK)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger("va.indicator")

OVERLAY = Path(__file__).with_name("overlay.py")
SYSTEM_PYTHON = "/usr/bin/python3"


class Indicator:
    def __init__(self, position: str = "bottom", settings_path: Path | None = None, name: str = "") -> None:
        self._position = position
        self._settings_path = settings_path
        self._name = name
        self._proc: subprocess.Popen | None = None
        self._last: tuple | None = None

    def start(self) -> None:
        if not os.getenv("WAYLAND_DISPLAY") or not shutil.which(SYSTEM_PYTHON):
            log.info("no Wayland session: on-screen indicator off")
            return
        try:
            self._proc = subprocess.Popen(
                [SYSTEM_PYTHON, "-I", str(OVERLAY), self._position, str(self._settings_path or ""), self._name],
                stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        except OSError as error:
            log.warning("indicator failed to start: %s", error)

    def set(self, state: str, until: float = 0.0, badge: str = "", tasks: list[dict] | None = None) -> None:
        """Send only changes; `until` (monotonic) is the end of the follow-up countdown."""
        tasks = tasks or []
        key = (state, until, badge, json.dumps(tasks, sort_keys=True))
        if self._proc is None or key == self._last:
            return
        if self._proc.poll() is not None:
            log.warning("indicator exited: %s", (self._proc.stderr.read() or "").strip()[-300:])
            self._proc = None
            return
        self._last = key
        try:
            self._proc.stdin.write(json.dumps({"state": state, "seconds": max(0.0, until - time.monotonic()) if until else 0, "badge": badge, "tasks": tasks, "name": self._name}, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._proc = None

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
