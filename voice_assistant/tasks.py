"""Actual background job state, shared by tools and the on-screen indicator."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass

ACTIVE = {"awaiting_confirmation", "preparing", "running", "checking"}


@dataclass
class BackgroundTask:
    id: str
    title: str
    agent: str
    state: str
    created: float
    updated: float
    detail: str = ""


class TaskBoard:
    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}

    def create(self, title: str, agent: str, state: str = "awaiting_confirmation") -> str:
        now = time.monotonic()
        id_ = uuid.uuid4().hex[:10]
        self._tasks[id_] = BackgroundTask(id_, " ".join(title.split())[:240], agent, state, now, now)
        # Keep a bounded session history; never discard active jobs.
        finished = sorted((t for t in self._tasks.values() if t.state not in ACTIVE), key=lambda t: t.updated)
        for task in finished[:-12]:
            self._tasks.pop(task.id, None)
        return id_

    def update(self, id_: str, state: str, detail: str = "") -> None:
        task = self._tasks[id_]
        task.state, task.detail, task.updated = state, detail, time.monotonic()

    def snapshot(self, limit: int = 3) -> list[dict]:
        now = time.monotonic()
        tasks = [t for t in self._tasks.values() if t.state in ACTIVE or now - t.updated < 90]
        tasks.sort(key=lambda t: (t.state not in ACTIVE, -t.created))
        return [asdict(t) for t in tasks[:limit]]


def task_summary(task: dict) -> str:
    labels = {"awaiting_confirmation": "Жду подтверждения", "preparing": "Запускаю", "running": "Выполняется", "checking": "Проверяю", "done": "Готово", "failed": "Ошибка", "cancelled": "Отменено"}
    return labels.get(task.get("state"), "Неизвестный статус")
