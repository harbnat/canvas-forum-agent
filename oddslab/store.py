"""Persistent memory for the HW4 services agent (SQLite, restored between runs).

  writes        every post the agent intends/makes; saved as 'pending' BEFORE posting
  handled       forum entries already processed (never handled twice)
  jobs          requests other agents sent to our Game Odds Lab service
  hires         our requests to other agents, and their progress
  accepted      peer test cases that passed our checks (used as regression tests)
  events        discovery rankings and other decisions worth showing as evidence
  cycles        one row per scheduled run
  kv            flags and counters
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS writes (
    idem_key TEXT PRIMARY KEY, purpose TEXT NOT NULL, parent_id INTEGER, body TEXT NOT NULL,
    status TEXT NOT NULL, entry_id INTEGER, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS handled (
    entry_id INTEGER PRIMARY KEY, role TEXT NOT NULL, outcome TEXT, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
    request_entry_id INTEGER PRIMARY KEY, requester TEXT, status TEXT NOT NULL,
    summary TEXT, reply_key TEXT, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS hires (
    id INTEGER PRIMARY KEY AUTOINCREMENT, offer_entry_id INTEGER NOT NULL,
    provider_id INTEGER NOT NULL, provider_name TEXT, reason TEXT,
    request_key TEXT, request_entry_id INTEGER, deadline REAL,
    status TEXT NOT NULL, corrections INTEGER NOT NULL DEFAULT 0,
    report TEXT, tip INTEGER, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS accepted (
    hire_id INTEGER NOT NULL, label TEXT NOT NULL, question TEXT NOT NULL,
    spec TEXT NOT NULL, answer TEXT NOT NULL, PRIMARY KEY (hire_id, label));
CREATE TABLE IF NOT EXISTS events (at REAL NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cycles (
    id TEXT PRIMARY KEY, started_at REAL NOT NULL, finished_at REAL, trigger TEXT,
    outcome TEXT, detail TEXT);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # kv
    def get(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set(self, key: str, value: object) -> None:
        self.db.execute("INSERT INTO kv VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, str(value)))

    # writes
    def write(self, key: str):
        return self.db.execute("SELECT * FROM writes WHERE idem_key=?", (key,)).fetchone()

    def record_pending(self, key: str, purpose: str, parent_id: int | None, body: str) -> bool:
        now = time.time()
        try:
            self.db.execute("INSERT INTO writes VALUES(?,?,?,?, 'pending', NULL, ?, ?)",
                            (key, purpose, parent_id, body, now, now))
            return True
        except sqlite3.IntegrityError:
            return False

    def mark_write(self, key: str, status: str, entry_id: int | None = None) -> None:
        self.db.execute("UPDATE writes SET status=?, entry_id=COALESCE(?, entry_id), updated_at=? "
                        "WHERE idem_key=?", (status, entry_id, time.time(), key))

    def writes_with_status(self, status: str):
        return list(self.db.execute("SELECT * FROM writes WHERE status=? ORDER BY created_at", (status,)))

    def writes_since(self, since: float) -> int:
        return self.db.execute("SELECT COUNT(*) FROM writes WHERE created_at>=? "
                               "AND status IN ('pending','verified')", (since,)).fetchone()[0]

    # handled entries
    def is_handled(self, entry_id: int) -> bool:
        return self.db.execute("SELECT 1 FROM handled WHERE entry_id=?", (entry_id,)).fetchone() is not None

    def mark_handled(self, entry_id: int, role: str, outcome: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO handled VALUES(?,?,?,?)", (entry_id, role, outcome, time.time()))

    # provider jobs
    def upsert_job(self, request_entry_id: int, requester: str, status: str, summary: str,
                   reply_key: str | None) -> None:
        self.db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?,?,?,?)",
                        (request_entry_id, requester, status, summary, reply_key, time.time()))

    def jobs(self):
        return list(self.db.execute("SELECT * FROM jobs ORDER BY at"))

    # hires
    def add_hire(self, offer_entry_id: int, provider_id: int, provider_name: str, reason: str) -> int:
        now = time.time()
        cur = self.db.execute("INSERT INTO hires(offer_entry_id, provider_id, provider_name, reason, "
                              "status, created_at, updated_at) VALUES(?,?,?,?, 'chosen', ?, ?)",
                              (offer_entry_id, provider_id, provider_name, reason, now, now))
        return int(cur.lastrowid)

    def update_hire(self, hire_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE hires SET {cols}, updated_at=? WHERE id=?",
                        (*fields.values(), time.time(), hire_id))

    def hires(self):
        return list(self.db.execute("SELECT * FROM hires ORDER BY id"))

    def active_hire(self):
        return self.db.execute("SELECT * FROM hires WHERE status IN ('chosen','requested',"
                               "'correction_requested') ORDER BY id DESC LIMIT 1").fetchone()

    def tried_offers(self) -> set[int]:
        return {r[0] for r in self.db.execute("SELECT offer_entry_id FROM hires")}

    # accepted cases
    def add_accepted(self, hire_id: int, label: str, question: str, spec_json: str, answer: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO accepted VALUES(?,?,?,?,?)",
                        (hire_id, label, question, spec_json, answer))

    def accepted(self, hire_id: int | None = None):
        if hire_id is None:
            return list(self.db.execute("SELECT * FROM accepted ORDER BY hire_id, label"))
        return list(self.db.execute("SELECT * FROM accepted WHERE hire_id=? ORDER BY label", (hire_id,)))

    # evidence
    def log_event(self, kind: str, data: dict) -> None:
        self.db.execute("INSERT INTO events VALUES(?,?,?)", (time.time(), kind, json.dumps(data, default=str)))

    def events(self, kind: str | None = None):
        q = "SELECT * FROM events" + (" WHERE kind=?" if kind else "") + " ORDER BY at"
        return list(self.db.execute(q, (kind,) if kind else ()))

    # cycles
    def start_cycle(self, cycle_id: str, trigger: str) -> None:
        self.db.execute("UPDATE cycles SET finished_at=started_at, outcome='interrupted', "
                        "detail='process died before the cycle finished' WHERE finished_at IS NULL")
        self.db.execute("INSERT INTO cycles(id, started_at, trigger) VALUES(?,?,?)",
                        (cycle_id, time.time(), trigger))

    def finish_cycle(self, cycle_id: str, outcome: str, detail: str, failed: bool) -> None:
        self.db.execute("UPDATE cycles SET finished_at=?, outcome=?, detail=? WHERE id=?",
                        (time.time(), outcome, detail, cycle_id))
        n = int(self.get("consecutive_failures", "0")) + 1 if failed else 0
        self.set("consecutive_failures", n)

    def cycles(self, limit: int = 50):
        return list(self.db.execute("SELECT * FROM cycles ORDER BY started_at DESC LIMIT ?", (limit,)))

    def last_real_cycle_start(self) -> float | None:
        row = self.db.execute("SELECT MAX(started_at) FROM cycles WHERE outcome IS NULL "
                              "OR outcome NOT IN ('dry_run')").fetchone()
        return row[0] if row and row[0] else None
