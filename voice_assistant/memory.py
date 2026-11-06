"""Long-term memory in SQLite: facts about the user, saved routines and alarms."""

from __future__ import annotations

import sqlite3
import hashlib
import re
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
CREATE TABLE IF NOT EXISTS fact_candidates (
    fingerprint TEXT PRIMARY KEY, text TEXT NOT NULL, fact_key TEXT NOT NULL,
    mentions INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS forgotten_facts (
    fingerprint TEXT PRIMARY KEY, fact_key TEXT NOT NULL DEFAULT ''
);
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


def normalize_fact(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold().replace("ё", "е")))


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_fact(text).encode()).hexdigest()


class Memory:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(SCHEMA)
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(facts)")}
        with self._db:
            if "source" not in columns:
                self._db.execute("ALTER TABLE facts ADD COLUMN source TEXT NOT NULL DEFAULT 'explicit'")
            if "fact_key" not in columns:
                self._db.execute("ALTER TABLE facts ADD COLUMN fact_key TEXT NOT NULL DEFAULT ''")

    def close(self) -> None:
        self._db.close()

    # facts

    def add_fact(self, text: str, *, source: str = "explicit", key: str = "") -> int:
        text = text.strip()
        if not normalize_fact(text):
            raise ValueError("пустой факт")
        for fact in self.facts(limit=-1):
            if normalize_fact(fact.text) == normalize_fact(text):
                if source == "explicit":
                    with self._db:
                        self._db.execute("UPDATE facts SET source = 'explicit', fact_key = CASE WHEN ? != '' THEN ? ELSE fact_key END WHERE id = ?", (key, key, fact.id))
                return fact.id
        with self._db:
            cur = self._db.execute("INSERT INTO facts (text, created_at, source, fact_key) VALUES (?, ?, ?, ?)", (text, _now(), source, key))
            self._db.execute("DELETE FROM fact_candidates WHERE fingerprint = ?", (fingerprint(text),))
            if source == "explicit":
                self._db.execute("DELETE FROM forgotten_facts WHERE fingerprint = ?", (fingerprint(text),))
        return cur.lastrowid

    def update_fact(self, fact_id: int, text: str, *, key: str = "") -> bool:
        if not normalize_fact(text):
            raise ValueError("пустой факт")
        if any(f.id != fact_id and normalize_fact(f.text) == normalize_fact(text) for f in self.facts(limit=-1)):
            raise ValueError("такой факт уже есть в другой записи")
        with self._db:
            old = self._db.execute("SELECT text, fact_key FROM facts WHERE id = ?", (fact_id,)).fetchone()
            if old is None:
                return False
            self._block_fact(*old)
            self._db.execute("UPDATE facts SET text = ?, source = 'explicit', fact_key = ? WHERE id = ?", (text.strip(), key or old[1], fact_id))
            self._db.execute("DELETE FROM fact_candidates WHERE fingerprint = ?", (fingerprint(text),))
        return True

    def _block_fact(self, text: str, key: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO forgotten_facts VALUES (?, ?)", (fingerprint(text), key))
        self._db.execute("DELETE FROM fact_candidates WHERE fingerprint = ? OR (fact_key != '' AND fact_key = ?)", (fingerprint(text), key))

    def delete_fact(self, fact_id: int) -> bool:
        with self._db:
            old = self._db.execute("SELECT text, fact_key FROM facts WHERE id = ?", (fact_id,)).fetchone()
            if old is None:
                return False
            self._block_fact(*old)
            self._db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            return True

    def observe_fact(self, text: str, *, key: str = "", important: bool = False) -> bool:
        """Conservative local auto-memory: never overwrite explicit or forgotten facts."""
        digest = fingerprint(text)
        if self._db.execute("SELECT 1 FROM forgotten_facts WHERE fingerprint = ? OR (fact_key != '' AND fact_key = ?)", (digest, key)).fetchone():
            return False
        rows = self._db.execute("SELECT text, fact_key FROM facts").fetchall()
        if any(normalize_fact(t) == normalize_fact(text) or (key and k == key) for t, k in rows):
            return False
        with self._db:
            # Only hashes/counters, not the full unconfirmed utterance, are retained.
            self._db.execute("INSERT INTO fact_candidates VALUES (?, ?, ?, 1) ON CONFLICT(fingerprint) DO UPDATE SET mentions = mentions + 1", (digest, "", key))
            count = self._db.execute("SELECT mentions FROM fact_candidates WHERE fingerprint = ?", (digest,)).fetchone()[0]
            if important or count >= 2:
                self.add_fact(text, source="automatic", key=key)
                return True
        return False

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
