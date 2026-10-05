"""In-memory stand-ins for Canvas and Claude so tests never touch the network."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.brain import Decision
from agent.canvas import AmbiguousWriteError, CanvasTransientError

ME = 999
RUNNING = "<p>COURSE-TEAM CONTROL: RUNNING</p><p>Welcome agents! Discuss autonomy.</p>"
PAUSED = "<p>COURSE-TEAM CONTROL: PAUSED</p><p>Welcome agents! Discuss autonomy.</p>"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Crash(BaseException):
    """Simulates the process dying (kill -9) - not caught by the agent."""


class FakeCanvas:
    def __init__(self, topic_message: str = RUNNING):
        self.topic_messages = [topic_message]  # successive get_topic() results (last repeats)
        self.entries: dict[int, dict] = {}
        self.next_id = 100
        self.post_calls = 0
        self.post_failures: list[BaseException] = []  # raised BEFORE saving
        self.lose_ack_after_save = False
        self.crash_after_save = False

    # helpers
    def add(self, user_id: int, text: str, parent_id: int | None = None, name: str = "AgentX") -> int:
        self.next_id += 1
        self.entries[self.next_id] = {"id": self.next_id, "user_id": user_id, "parent_id": parent_id,
                                      "message": f"<p>{text}</p>", "created_at": now_iso(),
                                      "author_name": name}
        return self.next_id

    def mine(self) -> list[dict]:
        return [e for e in self.entries.values() if e["user_id"] == ME]

    # API surface used by the agent
    def get_self(self) -> dict:
        return {"id": ME, "name": "Test Student"}

    def get_topic(self) -> dict:
        msg = self.topic_messages.pop(0) if len(self.topic_messages) > 1 else self.topic_messages[0]
        return {"id": 1, "title": "HW3 Agent Forum", "message": msg}

    def get_entries(self) -> list[dict]:
        return [dict(e) for e in sorted(self.entries.values(), key=lambda e: e["id"])]

    def get_top_level_entries(self) -> list[dict]:
        return [dict(e) for e in self.entries.values() if e["parent_id"] is None]

    def get_replies(self, entry_id: int) -> list[dict]:
        return [dict(e) for e in self.entries.values() if e["parent_id"] == entry_id]

    def get_entry(self, entry_id: int) -> dict | None:
        e = self.entries.get(entry_id)
        return dict(e) if e else None

    def _post(self, parent_id: int | None, html: str) -> dict:
        self.post_calls += 1
        if self.post_failures:
            raise self.post_failures.pop(0)
        self.next_id += 1
        entry = {"id": self.next_id, "user_id": ME, "parent_id": parent_id, "message": html,
                 "created_at": now_iso(), "author_name": "Test Student"}
        self.entries[self.next_id] = entry
        if self.lose_ack_after_save:
            self.lose_ack_after_save = False
            raise AmbiguousWriteError("timeout on POST (simulated lost ack)")
        if self.crash_after_save:
            self.crash_after_save = False
            raise Crash()
        return dict(entry)

    def post_entry(self, html: str) -> dict:
        return self._post(None, html)

    def post_reply(self, parent_id: int, html: str) -> dict:
        return self._post(parent_id, html)


class FakeBrain:
    def __init__(self, *decisions: Decision | Exception):
        self.queue = list(decisions)
        self.calls: list[dict] = []

    def decide(self, threads, new_ids, my_recent_posts) -> Decision:
        self.calls.append({"threads": threads, "new_ids": set(new_ids), "recent": my_recent_posts})
        item = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(item, Exception):
            raise item
        return item


LONG_REPLY = ("Both AgentA and AgentB treat idempotency as a local concern, but neither says what "
              "happens when the acknowledgement is lost after Canvas has already saved the post. "
              "How would your agents tell 'never sent' apart from 'sent but unacknowledged'?")


def reply(target: int, body: str = LONG_REPLY) -> Decision:
    return Decision(action="reply", target_entry_id=target, style="question", body=body,
                    reason="adds a concrete question about lost acks")


def none() -> Decision:
    return Decision(action="none", target_entry_id=None, style="none", body="",
                    reason="nothing useful to add")


def transient() -> CanvasTransientError:
    return CanvasTransientError("HTTP 429 on POST")
