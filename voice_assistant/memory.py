"""Long-term memory in SQLite: facts about the user, saved routines and alarms."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routines (
    name TEXT PRIMARY KEY,
    instructions TEXT NOT NULL,
    triggers TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alarms (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('timer', 'reminder')),
    due_at TEXT NOT NULL,
    text TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS alarms_pending ON alarms (done, due_at);
"""


@dataclass(frozen=True)
class Fact:
    id: int
    text: str


@dataclass(frozen=True)
class Routine:
    name: str
    instructions: str
    triggers: tuple[str, ...] = ()  # words that mean "this request is for this routine"
    url: str = ""  # page to open right away when the routine matches


@dataclass(frozen=True)
class Alarm:
    id: int
    kind: str
    due_at: datetime  # aware, UTC
    text: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Memory:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    # facts

    def add_fact(self, text: str) -> int:
        with self._db:
            cur = self._db.execute("INSERT INTO facts (text, created_at) VALUES (?, ?)", (text.strip(), _now()))
        return cur.lastrowid

    def delete_fact(self, fact_id: int) -> bool:
        with self._db:
            return self._db.execute("DELETE FROM facts WHERE id = ?", (fact_id,)).rowcount > 0

    def facts(self, limit: int = 100) -> list[Fact]:
        rows = self._db.execute("SELECT id, text FROM facts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [Fact(*row) for row in reversed(rows)]

    # routines

    def save_routine(
        self, name: str, instructions: str, triggers: list[str] | tuple[str, ...] = (), url: str = ""
    ) -> None:
        cleaned = ",".join(dict.fromkeys(t.strip().lower() for t in triggers if t.strip()))
        with self._db:
            self._db.execute(
                "INSERT INTO routines (name, instructions, triggers, url, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET instructions = excluded.instructions, "
                "triggers = excluded.triggers, url = excluded.url, updated_at = excluded.updated_at",
                (name.strip().lower(), instructions.strip(), cleaned, url.strip(), _now()),
            )

    def delete_routine(self, name: str) -> bool:
        with self._db:
            return self._db.execute("DELETE FROM routines WHERE name = ?", (name.strip().lower(),)).rowcount > 0

    def routines(self) -> list[Routine]:
        rows = self._db.execute("SELECT name, instructions, triggers, url FROM routines ORDER BY name").fetchall()
        return [
            Routine(name, instructions, tuple(t for t in triggers.split(",") if t), url)
            for name, instructions, triggers, url in rows
        ]

    # alarms

    def add_alarm(self, kind: str, due_at: datetime, text: str) -> int:
        with self._db:
            cur = self._db.execute(
                "INSERT INTO alarms (kind, due_at, text) VALUES (?, ?, ?)",
                (kind, due_at.astimezone(UTC).isoformat(timespec="seconds"), text.strip()),
            )
        return cur.lastrowid

    def pending_alarms(self) -> list[Alarm]:
        rows = self._db.execute("SELECT id, kind, due_at, text FROM alarms WHERE done = 0 ORDER BY due_at").fetchall()
        return [Alarm(id_, kind, datetime.fromisoformat(due), text) for id_, kind, due, text in rows]

    def due_alarms(self, now: datetime) -> list[Alarm]:
        return [a for a in self.pending_alarms() if a.due_at <= now]

    def finish_alarm(self, alarm_id: int) -> bool:
        with self._db:
            return self._db.execute("UPDATE alarms SET done = 1 WHERE id = ? AND done = 0", (alarm_id,)).rowcount > 0
