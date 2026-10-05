"""Deliberate failure injection for the 'recover from one failure' requirement.

Enable with `python -m agent run --inject-fault <name>`. Each fault fires at
most once per process so the agent's recovery path is what gets exercised.
"""

from __future__ import annotations

FAULTS = {
    "http_500": "First Canvas GET returns a synthetic HTTP 500 (tests retry + backoff).",
    "malformed_response": "First Canvas GET returns non-JSON garbage (tests parsing + retry).",
    "lost_ack": "The POST really reaches Canvas, then the response is 'lost' as a timeout "
    "(tests reconcile: no duplicate post).",
    "crash_after_post": "The POST reaches Canvas, then the process dies before saving state "
    "(tests restart recovery: next run reconciles, no duplicate).",
    "duplicate_event": "The same decision is executed twice in one cycle (tests idempotency key).",
    "malformed_llm": "The model's decision comes back as unparseable output (tests fail-closed).",
}


class Faults:
    def __init__(self, name: str | None = None):
        if name is not None and name not in FAULTS:
            raise ValueError(f"unknown fault {name!r}; choose from {', '.join(FAULTS)}")
        self.name = name
        self._fired: set[str] = set()

    def fire(self, name: str) -> bool:
        """True exactly once if `name` is the active fault."""
        if self.name == name and name not in self._fired:
            self._fired.add(name)
            return True
        return False

    def armed(self, name: str) -> bool:
        return self.name == name
