"""One scheduled agent cycle: observe -> reconcile -> decide -> gate -> write -> verify.

Run by cron every few hours via `python -m agent run`. Every cycle ends with an
explicit outcome recorded in memory and in logs/agent.jsonl, including the
cycles where the agent deliberately did not post.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from . import safety
from .brain import Brain, Decision, MalformedDecision
from .canvas import AmbiguousWriteError, CanvasClient, CanvasError, CanvasTransientError, backoff_delay
from .config import Config
from .events import EventLog
from .faults import Faults
from .memory import Memory, idempotency_key

log = logging.getLogger(__name__)

POST_ATTEMPTS = 3
VERIFY_ATTEMPTS = 4
MAX_THREADS_IN_PROMPT = 6
MAX_ENTRIES_PER_THREAD = 15
MAX_CHARS_PER_ENTRY = 1200


def root_of(e: dict, by_id: dict[int, dict]) -> int:
    """Id of the top-level entry (thread start) that `e` belongs to."""
    visited = set()
    while e.get("parent_id") and e["parent_id"] in by_id and e["id"] not in visited:
        visited.add(e["id"])
        e = by_id[e["parent_id"]]
    return e["id"]


class Paused(Exception):
    pass


class PostFailed(Exception):
    pass


@dataclass
class CycleResult:
    outcome: str
    detail: str
    failed: bool = False
    posted_urls: list[str] = field(default_factory=list)


def parse_time(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class Agent:
    def __init__(self, cfg: Config, canvas: CanvasClient, brain: Brain, memory: Memory,
                 events: EventLog, faults: Faults | None = None, dry_run: bool = False,
                 sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.time):
        self.cfg = cfg
        self.canvas = canvas
        self.brain = brain
        self.memory = memory
        self.events = events
        self.faults = faults or Faults()
        self.dry_run = dry_run
        self.sleep = sleep
        self.now = now
        self.cycle_id = ""

    # ================================================================= entry

    def run_cycle(self) -> CycleResult:
        self.cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + secrets.token_hex(3)
        if self.memory.halted:
            result = CycleResult("halted", "stopped after repeated failures; run "
                                 "`python -m agent reset` after investigating")
            self.events.log(self.cycle_id, "cycle_end", outcome=result.outcome, detail=result.detail)
            return result

        trigger = os.environ.get("AGENT_TRIGGER", "local")
        self.memory.start_cycle(self.cycle_id, trigger)
        self.events.log(self.cycle_id, "cycle_start", trigger=trigger, dry_run=self.dry_run,
                        fault=self.faults.name)
        try:
            result = self._run()
        except Paused as e:
            result = CycleResult("paused", str(e))
        except (CanvasError, MalformedDecision, PostFailed) as e:
            result = CycleResult("error", f"{type(e).__name__}: {e}", failed=True)
        except Exception as e:  # anthropic errors, unexpected bugs: still count + stop rule
            log.exception("unexpected error in cycle")
            result = CycleResult("error", f"{type(e).__name__}: {e}", failed=True)

        self.memory.finish_cycle(self.cycle_id, result.outcome, result.detail, result.failed)
        failures = self.memory.consecutive_failures
        if result.failed and failures >= self.cfg.max_consecutive_failures:
            self.memory.set("halted", "1")
            self.events.log(self.cycle_id, "halted", consecutive_failures=failures)
            log.error("HALTED after %d consecutive failed cycles", failures)
        self.events.log(self.cycle_id, "cycle_end", outcome=result.outcome, detail=result.detail,
                        failed=result.failed, consecutive_failures=failures,
                        posted=result.posted_urls)
        return result

    # ============================================================ main logic

    def _run(self) -> CycleResult:
        me = self._whoami()
        topic = self.canvas.get_topic()
        control = safety.control_state(topic.get("message"))
        self.events.log(self.cycle_id, "control_line", state=control)

        entries = self._load_entries()
        by_id = {e["id"]: e for e in entries}

        notes = self._reconcile_pending(me)

        # Our own entries are remembered but never treated as input.
        mine = [e for e in entries if e.get("user_id") == me]
        self.memory.mark_seen(mine, me, self.cycle_id)

        if control != "RUNNING":
            raise Paused(f"control line says {control}; not posting" + notes)

        seen = self.memory.seen_ids()
        unseen = [e for e in entries if e["id"] not in seen and e.get("user_id") != me]
        # Old entries are context only: don't revive threads that went quiet days ago.
        cutoff = self.now() - self.cfg.max_entry_age_hours * 3600
        stale = [e for e in unseen if (parse_time(e.get("created_at")) or self.now()) < cutoff]
        if stale:
            self.memory.mark_seen(stale, me, self.cycle_id)
            self.events.log(self.cycle_id, "skipped_stale", count=len(stale),
                            older_than_hours=self.cfg.max_entry_age_hours)
        stale_ids = {e["id"] for e in stale}
        new = [e for e in unseen if e["id"] not in stale_ids]
        self.events.log(self.cycle_id, "observed", total_entries=len(entries), new=len(new),
                        new_ids=[e["id"] for e in new])
        if not new:
            return CycleResult("no_post_nothing_new",
                               "no new entries from other agents since last cycle" + notes)

        budget = self._post_budget(entries, me)
        if budget <= 0:
            return CycleResult("no_post_rate_limited",
                               f"hourly post limit ({self.cfg.max_posts_per_hour}) reached; "
                               "new entries left unseen for next cycle" + notes)

        threads, considered, thread_notes = self._build_threads(entries, by_id, new, me)
        recent = self._my_previous_bodies(entries, me)[:10]
        decision = self.brain.decide(threads, considered, recent, thread_notes)
        self.events.log(self.cycle_id, "decision", action=decision.action, style=decision.style,
                        target=decision.target_entry_id, reason=decision.reason,
                        considered=sorted(considered))

        result = self._act(decision, me, by_id, considered, CycleResult("none", ""))
        if self.faults.fire("duplicate_event"):
            self.events.log(self.cycle_id, "fault_injected", fault="duplicate_event",
                            note="re-executing the same decision")
            again = self._act(decision, me, by_id, considered, result)
            result.detail += f" | duplicate re-execution -> {again.outcome}"
        result.detail += notes
        return result

    # =========================================================== helpers

    def _whoami(self) -> int:
        cached = self.memory.get("my_user_id")
        if cached:
            return int(cached)
        me = self.canvas.get_self()
        self.memory.set("my_user_id", me["id"])
        return int(me["id"])

    def _load_entries(self) -> list[dict]:
        out = []
        for e in self.canvas.get_entries():
            if e.get("deleted"):
                continue
            e["text"] = safety.html_to_text(e.get("message"))
            if e["text"]:
                out.append(e)
        return out

    def _my_previous_bodies(self, entries: list[dict], me: int) -> list[str]:
        """My earlier posts, newest first: local memory plus what is live on Canvas.

        Including the live copies means a lost memory file cannot make the agent
        repeat itself.
        """
        local = [r["body"] for r in self.memory.my_posts(limit=50)]
        remote = [e["text"] for e in sorted(entries, key=lambda e: e["id"], reverse=True)
                  if e.get("user_id") == me]
        return local + [t for t in remote if t not in local]

    def _post_budget(self, entries: list[dict], me: int) -> int:
        hour_ago = self.now() - 3600
        local = self.memory.posts_since(hour_ago)
        remote = sum(1 for e in entries if e.get("user_id") == me
                     and (parse_time(e.get("created_at")) or 0) >= hour_ago)
        used = max(local, remote)
        return min(self.cfg.max_posts_per_cycle, self.cfg.max_posts_per_hour - used)

    def _build_threads(self, entries: list[dict], by_id: dict[int, dict],
                       new: list[dict], me: int) -> tuple[list[list[dict]], set[int], list[str]]:
        roots: dict[int, list[dict]] = {}
        for e in entries:
            roots.setdefault(root_of(e, by_id), []).append(e)
        new_roots: list[int] = []
        for e in sorted(new, key=lambda x: x["id"], reverse=True):
            r = root_of(e, by_id)
            if r not in new_roots:
                new_roots.append(r)
        chosen = new_roots[:MAX_THREADS_IN_PROMPT]  # most recently active threads first

        threads, considered, notes = [], set(), []
        new_ids = {e["id"] for e in new}
        for r in chosen:
            notes.append(self._cooldown_note(r, entries, by_id, me))
            members = sorted(roots[r], key=lambda x: x["id"])
            if len(members) > MAX_ENTRIES_PER_THREAD:
                members = [members[0]] + members[-(MAX_ENTRIES_PER_THREAD - 1):]
            thread = []
            for e in members:
                text = e["text"]
                if len(text) > MAX_CHARS_PER_ENTRY:
                    text = text[:MAX_CHARS_PER_ENTRY] + " [...truncated]"
                parent = by_id.get(e.get("parent_id") or -1)
                thread.append({**e, "text": text,
                               "replies_to_me": bool(parent and parent.get("user_id") == me)})
                if e["id"] in new_ids:
                    considered.add(e["id"])
            threads.append(thread)
        return threads, considered, notes

    def _last_post_in_thread(self, root: int, entries: list[dict], by_id: dict[int, dict],
                             me: int) -> float | None:
        times = [parse_time(e.get("created_at")) for e in entries
                 if e.get("user_id") == me and root_of(e, by_id) == root]
        times = [t for t in times if t]
        return max(times) if times else None

    def _cooldown_note(self, root: int, entries: list[dict], by_id: dict[int, dict],
                       me: int) -> str:
        last = self._last_post_in_thread(root, entries, by_id, me)
        if last is None or self.now() - last >= self.cfg.thread_cooldown_hours * 3600:
            return ""
        hours = (self.now() - last) / 3600
        return (f"COOLDOWN: you posted in this thread {hours:.1f}h ago. Do not post here again "
                "unless a NEW entry replies directly to one of your posts (marked replies_to_you).")

    # ================================================================ acting

    def _act(self, d: Decision, me: int, by_id: dict[int, dict], considered: set[int],
             prev: CycleResult) -> CycleResult:
        def done(outcome: str, detail: str, failed: bool = False) -> CycleResult:
            if not failed:
                self.memory.mark_seen_ids(sorted(considered), self.cycle_id)
            return CycleResult(outcome, detail, failed, prev.posted_urls)

        if d.action == "none":
            return done("no_post_by_choice", d.reason)

        kind = d.action
        parent_id = d.target_entry_id if kind == "reply" else None
        body = d.body.strip()

        if kind == "reply":
            target = by_id.get(parent_id or -1)
            if target is None:
                return done("blocked_by_gate", f"reply target {parent_id} is not in this forum")
            if target.get("user_id") == me:
                return done("blocked_by_gate", "model tried to reply to my own post")
            parent = by_id.get(target.get("parent_id") or -1)
            answers_me = bool(parent and parent.get("user_id") == me)
            last = self._last_post_in_thread(root_of(target, by_id), list(by_id.values()),
                                             by_id, me)
            cooldown = self.cfg.thread_cooldown_hours * 3600
            if last is not None and self.now() - last < cooldown and not answers_me:
                return done("blocked_by_gate",
                            f"thread cooldown: I posted in this thread "
                            f"{(self.now() - last) / 3600:.1f}h ago and nobody replied to me")

        key = idempotency_key(kind, parent_id, body)
        existing = self.memory.get_action(key)
        if existing is not None and existing["status"] in ("pending", "verified"):
            self.events.log(self.cycle_id, "duplicate_suppressed", idem_key=key,
                            status=existing["status"], canvas_entry_id=existing["canvas_entry_id"])
            return done("duplicate_suppressed",
                        f"identical action {key} already {existing['status']}; not posting again")

        if kind == "reply":
            already = parent_id in self.memory.replied_parents() or any(
                e.get("parent_id") == parent_id and e.get("user_id") == me for e in by_id.values())
            if already:
                return done("blocked_by_gate", f"I already replied to entry {parent_id}")

        own_previous = self._my_previous_bodies(list(by_id.values()), me)
        problems = safety.check_body(body, own_previous)
        if problems:
            # Body deliberately not logged: it may be what tripped a secret filter.
            self.events.log(self.cycle_id, "blocked_by_gate", problems=problems)
            return done("blocked_by_gate", "; ".join(problems))

        if self.dry_run:
            self.events.log(self.cycle_id, "dry_run", kind=kind, parent_id=parent_id, body=body)
            target = self.cfg.entry_url(parent_id) if parent_id else "a new thread"
            return CycleResult("dry_run_would_post",
                               f"{kind} to {target}\n----- draft -----\n{body}\n-----------------")

        if existing is not None:  # previously confirmed never posted; safe to try again
            self.memory.retry_not_posted(key, self.cycle_id)
        else:
            self.memory.record_pending(idem_key=key, kind=kind, parent_id=parent_id, body=body,
                                       reason=d.reason, context_ids=sorted(considered),
                                       cycle_id=self.cycle_id)
        self.events.log(self.cycle_id, "pending_recorded", idem_key=key, kind=kind,
                        parent_id=parent_id)

        html_msg = safety.text_to_html(body) + (
            f"<p><em>— {safety.html.escape(self.cfg.agent_name)} (autonomous course agent)</em></p>")
        started = self.now()
        try:
            entry = self._post_with_recovery(kind, parent_id, html_msg, body, me, started)
        except Paused:
            self.memory.mark_action(key, "not_posted")
            raise
        except (PostFailed, CanvasError) as e:
            found = self._find_my_post(kind, parent_id, body, me)
            if found is None:
                self.memory.mark_action(key, "not_posted")
                raise PostFailed(f"{e}; confirmed nothing was saved") from e
            entry = found

        entry_id = int(entry["id"])
        self.events.log(self.cycle_id, "posted", idem_key=key, canvas_entry_id=entry_id)
        if not self._verify(entry_id, body, me):
            # Leave it 'pending' so the next cycle reconciles it instead of reposting.
            raise PostFailed(f"posted entry {entry_id} could not be verified yet")

        self.memory.mark_action(key, "verified", entry_id)
        self.memory.mark_seen_ids([entry_id], self.cycle_id)
        url = self.cfg.entry_url(entry_id)
        self.events.log(self.cycle_id, "verified", canvas_entry_id=entry_id, url=url)
        out = done("posted", f"{kind} {url} ({d.style}): {d.reason}")
        out.posted_urls = prev.posted_urls + [url]
        return out

    def _ensure_running(self) -> None:
        topic = self.canvas.get_topic()
        state = safety.control_state(topic.get("message"))
        if state != "RUNNING":
            self.events.log(self.cycle_id, "control_line", state=state, at="pre_write")
            raise Paused(f"control line says {state} at write time; not posting")

    def _post_with_recovery(self, kind: str, parent_id: int | None, html_msg: str, body: str,
                            me: int, started: float) -> dict:
        last: Exception | None = None
        for attempt in range(POST_ATTEMPTS):
            self._ensure_running()  # read the control line before EVERY write attempt
            try:
                if kind == "reply":
                    assert parent_id is not None
                    return self.canvas.post_reply(parent_id, html_msg)
                return self.canvas.post_entry(html_msg)
            except AmbiguousWriteError as e:
                last = e
                self.events.log(self.cycle_id, "write_ambiguous", error=str(e), attempt=attempt + 1)
                self.sleep(backoff_delay(attempt))
                found = self._find_my_post(kind, parent_id, body, me)
                if found is not None:
                    self.events.log(self.cycle_id, "reconciled_after_ambiguous_write",
                                    canvas_entry_id=found["id"],
                                    note="post was saved; NOT re-posting")
                    return found
            except CanvasTransientError as e:
                last = e
                self.events.log(self.cycle_id, "write_retry", error=str(e), attempt=attempt + 1)
                self.sleep(backoff_delay(attempt, e.retry_after))
        raise PostFailed(f"gave up after {POST_ATTEMPTS} attempts: {last}")

    def _find_my_post(self, kind: str, parent_id: int | None, body: str, me: int) -> dict | None:
        try:
            candidates = (self.canvas.get_replies(parent_id) if kind == "reply" and parent_id
                          else self.canvas.get_top_level_entries())
        except CanvasError as e:
            log.warning("could not reconcile (%s)", e)
            return None
        for c in candidates:
            if c.get("user_id") == me and safety.contains_post(c.get("message"), body):
                return c
        return None

    def _verify(self, entry_id: int, body: str, me: int) -> bool:
        for attempt in range(VERIFY_ATTEMPTS):
            try:
                entry = self.canvas.get_entry(entry_id)
            except CanvasError:
                entry = None
            if entry and entry.get("user_id") == me and safety.contains_post(entry.get("message"), body):
                return True
            self.sleep(backoff_delay(attempt))
        return False

    def _reconcile_pending(self, me: int) -> str:
        """Resolve writes left 'pending' by a crash / lost ack in an earlier run."""
        notes = []
        for a in self.memory.pending_actions():
            found = self._find_my_post(a["kind"], a["parent_id"], a["body"], me)
            if found is not None:
                self.memory.mark_action(a["idem_key"], "verified", int(found["id"]))
                self.memory.mark_seen_ids(json.loads(a["context_ids"]) + [int(found["id"])],
                                          self.cycle_id)
                self.events.log(self.cycle_id, "reconciled_pending", idem_key=a["idem_key"],
                                canvas_entry_id=found["id"], result="was_saved_not_reposting")
                notes.append(f"recovered earlier post {found['id']} without reposting")
            else:
                self.memory.mark_action(a["idem_key"], "not_posted")
                self.events.log(self.cycle_id, "reconciled_pending", idem_key=a["idem_key"],
                                result="never_saved_marked_not_posted")
                notes.append(f"earlier attempt {a['idem_key'][:8]} never reached Canvas")
        return (" | " + "; ".join(notes)) if notes else ""
