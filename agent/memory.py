"""Persistent local memory (SQLite) so the agent remembers across cycles and restarts.

Tables
  seen_entries  - forum entries already considered (never re-processed)
  actions       - every write the agent intends/makes, keyed by an idempotency key.
                  An action is saved as 'pending' BEFORE the POST, so a crash or lost
                  acknowledgement can be reconciled on the next run.
  cycles        - one row per scheduled run, with its outcome (evidence log)
  kv            - small counters/flags: consecutive_failures, halted, my_user_id
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_entries (
    entry_id      INTEGER PRIMARY KEY,
    parent_id     INTEGER,
    user_id       INTEGER,
    is_mine       INTEGER NOT NULL DEFAULT 0,
    first_seen_at REAL NOT NULL,
    cycle_id      TEXT
);
CREATE TABLE IF NOT EXISTS actions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    idem_key         TEXT NOT NULL UNIQUE,
    kind             TEXT NOT NULL,          -- reply | new_thread
    parent_id        INTEGER,
    body             TEXT NOT NULL,
    reason           TEXT,
    context_ids      TEXT NOT NULL,          -- JSON list of entry ids this responds to
    status           TEXT NOT NULL,          -- pending | verified | not_posted
    canvas_entry_id  INTEGER,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    cycle_id         TEXT
);
CREATE TABLE IF NOT EXISTS cycles (
    id           TEXT PRIMARY KEY,
    started_at   REAL NOT NULL,
    finished_at  REAL,
    outcome      TEXT,
    detail       TEXT
);
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def idempotency_key(kind: str, parent_id: int | None, body: str) -> str:
    raw = f"{kind}|{parent_id}|{' '.join(body.split())}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class Memory:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path), isolation_level=None)  # autocommit; explicit txns below
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")  # survive power loss / kill -9
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def txn(self) -> Iterator[sqlite3.Connection]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ kv

    def get(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: object) -> None:
        self.db.execute("INSERT INTO kv(key,value) VALUES(?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))

    @property
    def consecutive_failures(self) -> int:
        return int(self.get("consecutive_failures", "0") or 0)

    @property
    def halted(self) -> bool:
        return self.get("halted", "0") == "1"

    # ---------------------------------------------------------------- seen

    def seen_ids(self) -> set[int]:
        return {r[0] for r in self.db.execute("SELECT entry_id FROM seen_entries")}

    def mark_seen(self, entries: list[dict], my_user_id: int, cycle_id: str) -> None:
        now = time.time()
        with self.txn() as db:
            for e in entries:
                db.execute(
                    "INSERT OR IGNORE INTO seen_entries(entry_id,parent_id,user_id,is_mine,"
                    "first_seen_at,cycle_id) VALUES(?,?,?,?,?,?)",
                    (e["id"], e.get("parent_id"), e.get("user_id"),
                     int(e.get("user_id") == my_user_id), now, cycle_id))

    def mark_seen_ids(self, ids: list[int], cycle_id: str) -> None:
        now = time.time()
        with self.txn() as db:
            for i in ids:
                db.execute("INSERT OR IGNORE INTO seen_entries(entry_id,first_seen_at,cycle_id) "
                           "VALUES(?,?,?)", (int(i), now, cycle_id))

    # ------------------------------------------------------------- actions

    def get_action(self, idem_key: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM actions WHERE idem_key=?", (idem_key,)).fetchone()

    def record_pending(self, *, idem_key: str, kind: str, parent_id: int | None, body: str,
                       reason: str, context_ids: list[int], cycle_id: str) -> bool:
        """Durably record intent BEFORE posting. False if this key already exists."""
        now = time.time()
        try:
            self.db.execute(
                "INSERT INTO actions(idem_key,kind,parent_id,body,reason,context_ids,status,"
                "created_at,updated_at,cycle_id) VALUES(?,?,?,?,?,?, 'pending', ?,?,?)",
                (idem_key, kind, parent_id, body, reason, json.dumps(context_ids), now, now,
                 cycle_id))
            return True
        except sqlite3.IntegrityError:
            return False

    def retry_not_posted(self, idem_key: str, cycle_id: str) -> None:
        """Re-arm an action that was confirmed never to have reached Canvas."""
        self.db.execute("UPDATE actions SET status='pending', updated_at=?, cycle_id=? "
                        "WHERE idem_key=? AND status='not_posted'",
                        (time.time(), cycle_id, idem_key))

    def mark_action(self, idem_key: str, status: str, canvas_entry_id: int | None = None) -> None:
        self.db.execute(
            "UPDATE actions SET status=?, canvas_entry_id=COALESCE(?, canvas_entry_id), "
            "updated_at=? WHERE idem_key=?", (status, canvas_entry_id, time.time(), idem_key))

    def pending_actions(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM actions WHERE status='pending' ORDER BY id"))

    def posts_since(self, since: float) -> int:
        """Writes that did, or may have, reached Canvas since `since` (pending counts)."""
        row = self.db.execute("SELECT COUNT(*) FROM actions WHERE created_at>=? "
                              "AND status IN ('pending','verified')", (since,)).fetchone()
        return int(row[0])

    def my_posts(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(self.db.execute(
            "SELECT * FROM actions WHERE status IN ('pending','verified') "
            "ORDER BY id DESC LIMIT ?", (limit,)))

    def replied_parents(self) -> set[int]:
        return {r[0] for r in self.db.execute(
            "SELECT parent_id FROM actions WHERE kind='reply' "
            "AND status IN ('pending','verified') AND parent_id IS NOT NULL")}

    # -------------------------------------------------------------- cycles

    def start_cycle(self, cycle_id: str) -> None:
        self.db.execute("INSERT INTO cycles(id,started_at) VALUES(?,?)", (cycle_id, time.time()))

    def finish_cycle(self, cycle_id: str, outcome: str, detail: str, failed: bool) -> None:
        with self.txn() as db:
            db.execute("UPDATE cycles SET finished_at=?, outcome=?, detail=? WHERE id=?",
                       (time.time(), outcome, detail, cycle_id))
        if failed:
            self.set("consecutive_failures", self.consecutive_failures + 1)
        else:
            self.set("consecutive_failures", 0)

    def last_real_cycle_start(self) -> float | None:
        """Start time of the latest cycle that was not a dry run (None if never)."""
        row = self.db.execute(
            "SELECT MAX(started_at) FROM cycles WHERE outcome IS NULL "
            "OR outcome != 'dry_run_would_post'").fetchone()
        return row[0] if row and row[0] else None

    def recent_cycles(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM cycles ORDER BY started_at DESC LIMIT ?",
                                    (limit,)))
