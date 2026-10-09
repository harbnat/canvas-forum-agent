"""One scheduled cycle of the HW4 services agent.

Order of work each cycle (every write goes through Writer, which enforces the
control line, the shared 3-writes-per-hour budget, idempotency and read-back):

  1. reconcile writes left pending by a crash or lost acknowledgement
  2. self-check: re-run accepted peer test cases against the engine (regression)
  3. post the Game Odds Lab service offer (once, ever)
  4. provider: answer new requests that reply to our offer
  5. client: discover offers -> choose a provider -> send one complete request ->
     read their replies -> verify the work -> accept, ask once for a correction, or reject
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Callable

from agent import safety
from agent.canvas import AmbiguousWriteError, CanvasError, CanvasTransientError, backoff_delay
from agent.cycle import parse_time, root_of
from agent.events import EventLog

from . import engine, texts
from .llm import LLM
from .store import Store

log = logging.getLogger(__name__)

MAX_WRITES_PER_HOUR = 3           # course rule: per student, across all agents and discussions
MAX_WRITES_PER_CYCLE = 2
REQUEST_HOURS = 48
CORRECTION_HOURS = 24
MIN_SCORE = 3
MAX_HIRES = 3
MAX_CASES = 20
MIN_ACCEPTED = 3
MAX_QUESTIONS_ANSWERED = 2
# Stop starting new hires late on Oct 13 ET; final verifications may still run until Oct 15.
NO_NEW_HIRES_AFTER = datetime(2026, 10, 14, 4, 0, tzinfo=timezone.utc).timestamp()
STOP_ALL_AFTER = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc).timestamp()


class Paused(Exception):
    pass


class BudgetSpent(Exception):
    pass


class Writer:
    """The only way the agent posts. Control line, budget, idempotency, read-back."""

    def __init__(self, canvas, store: Store, events: EventLog, cycle_id: str, me: int,
                 remote_recent: Callable[[], tuple[int, int]], dry_run: bool,
                 sleep: Callable[[float], None], now: Callable[[], float], signature: str):
        self.canvas, self.store, self.events = canvas, store, events
        self.cycle_id, self.me, self.dry_run = cycle_id, me, dry_run
        self.remote_recent, self.sleep, self.now = remote_recent, sleep, now
        self.signature = signature
        self.writes_this_cycle = 0
        self.drafts: list[dict] = []

    def _ensure_running(self) -> None:
        state = safety.control_state(self.canvas.get_topic().get("message"))
        if state != "RUNNING":
            raise Paused(f"control line says {state}")

    def budget_left(self) -> int:
        hour_ago = self.now() - 3600
        here, elsewhere = self.remote_recent()
        # Posts in this topic: whichever count is higher (my records include writes made
        # this cycle; Canvas includes anything I posted by hand). Other topics: Canvas only.
        used = max(self.store.writes_since(hour_ago), here) + elsewhere
        return min(MAX_WRITES_PER_CYCLE - self.writes_this_cycle, MAX_WRITES_PER_HOUR - used)

    def html(self, body: str) -> str:
        return safety.text_to_html(body) + f"<p><em>— {safety.html.escape(self.signature)}</em></p>"

    def post(self, key: str, purpose: str, parent_id: int | None, body: str) -> dict | None:
        existing = self.store.write(key)
        if existing is not None and existing["status"] == "verified":
            return {"id": existing["entry_id"]}
        problems = [p for p in safety.check_body(body, []) if not p.startswith(("too long", "too short"))]
        if problems:
            self.events.log(self.cycle_id, "write_blocked", key=key, problems=problems)
            raise CanvasError(f"refusing to post {purpose}: {'; '.join(problems)}")
        if self.dry_run:
            self.drafts.append({"purpose": purpose, "parent_id": parent_id, "body": body})
            self.events.log(self.cycle_id, "dry_run_write", purpose=purpose, parent_id=parent_id)
            return None
        if self.budget_left() <= 0:
            raise BudgetSpent(f"write budget reached before {purpose}")
        self._ensure_running()
        if existing is None:
            self.store.record_pending(key, purpose, parent_id, body)
        else:
            self.store.mark_write(key, "pending")
        self.events.log(self.cycle_id, "pending_recorded", key=key, purpose=purpose, parent_id=parent_id)
        self.writes_this_cycle += 1
        entry = self._post_with_recovery(parent_id, body)
        if entry is None:
            raise CanvasError(f"could not confirm {purpose} was saved; left pending")
        eid = int(entry["id"])
        if not self._verify(eid, body):
            raise CanvasError(f"{purpose} entry {eid} could not be read back; left pending")
        self.store.mark_write(key, "verified", eid)
        self.events.log(self.cycle_id, "verified", key=key, purpose=purpose, entry_id=eid)
        return {"id": eid}

    def _post_with_recovery(self, parent_id: int | None, body: str) -> dict | None:
        html = self.html(body)
        for attempt in range(3):
            self._ensure_running()
            try:
                return (self.canvas.post_reply(parent_id, html) if parent_id
                        else self.canvas.post_entry(html))
            except AmbiguousWriteError as e:
                self.events.log(self.cycle_id, "write_ambiguous", error=str(e))
                self.sleep(backoff_delay(attempt))
                found = self.find_mine(parent_id, body)
                if found is not None:
                    return found
                if found is None and not self._lookup_ok:
                    return None  # cannot tell whether it was saved: never repost blindly
            except CanvasTransientError as e:
                self.sleep(backoff_delay(attempt, e.retry_after))
        return None

    _lookup_ok = True

    def find_mine(self, parent_id: int | None, body: str) -> dict | None:
        try:
            entries = self.canvas.get_entries()
            self._lookup_ok = True
        except CanvasError:
            self._lookup_ok = False
            return None
        for e in entries:
            if (e.get("user_id") == self.me and (e.get("parent_id") or None) == parent_id
                    and safety.contains_post(e.get("message"), body)):
                return e
        return None

    def _verify(self, entry_id: int, body: str) -> bool:
        for attempt in range(4):
            try:
                e = self.canvas.get_entry(entry_id)
            except CanvasError:
                e = None
            if e and e.get("user_id") == self.me and safety.contains_post(e.get("message"), body):
                return True
            self.sleep(backoff_delay(attempt))
        return False

    def reconcile(self) -> list[str]:
        notes = []
        for w in self.store.writes_with_status("pending"):
            found = self.find_mine(w["parent_id"], w["body"])
            if found is not None:
                self.store.mark_write(w["idem_key"], "verified", int(found["id"]))
                notes.append(f"recovered pending {w['purpose']} as entry {found['id']}")
            elif self._lookup_ok:
                self.store.mark_write(w["idem_key"], "not_posted")
                notes.append(f"pending {w['purpose']} never reached Canvas")
            else:
                notes.append(f"could not check pending {w['purpose']}; left pending")
            self.events.log(self.cycle_id, "reconciled", key=w["idem_key"], note=notes[-1])
        return notes


class ServicesAgent:
    def __init__(self, cfg, canvas, other_canvases: list, llm: LLM, store: Store, events: EventLog,
                 dry_run: bool = False, sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], float] = time.time):
        self.cfg, self.canvas, self.other_canvases = cfg, canvas, other_canvases
        self.llm, self.store, self.events = llm, store, events
        self.dry_run, self.sleep, self.now = dry_run, sleep, now
        self.notes: list[str] = []

    # ------------------------------------------------------------ cycle

    def run_cycle(self, trigger: str = "local") -> tuple[str, str, bool]:
        self.cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + secrets.token_hex(3)
        self.notes = []
        self.__dict__.pop("_other_recent", None)
        if self.store.get("halted") == "1":
            return "halted", "stopped after repeated failures; run reset", False
        if self.now() > STOP_ALL_AFTER:
            return "finished", "past the end of the HW4 window; doing nothing", False
        self.store.start_cycle(self.cycle_id, trigger)
        self.events.log(self.cycle_id, "cycle_start", trigger=trigger, dry_run=self.dry_run)
        failed = False
        try:
            outcome = self._run()
        except Paused as e:
            outcome = f"paused: {e}"
        except BudgetSpent as e:
            outcome = f"budget: {e}"
        except Exception as e:  # noqa: BLE001 - every failure counts toward the stopping rule
            log.exception("cycle failed")
            outcome, failed = f"error: {type(e).__name__}: {e}", True
        detail = " | ".join(self.notes)
        self.store.finish_cycle(self.cycle_id, "dry_run" if self.dry_run else outcome.split(":")[0],
                                f"{outcome} | {detail}" if detail else outcome, failed)
        if failed and int(self.store.get("consecutive_failures", "0")) >= self.cfg.max_consecutive_failures:
            self.store.set("halted", "1")
            self.events.log(self.cycle_id, "halted")
        self.events.log(self.cycle_id, "cycle_end", outcome=outcome, notes=self.notes, failed=failed)
        return outcome, detail, failed

    def _run(self) -> str:
        me = self.store.get("my_user_id")
        if me is None:
            me = self.canvas.get_self()["id"]
            self.store.set("my_user_id", me)
        self.me = int(me)
        topic = self.canvas.get_topic()
        if safety.control_state(topic.get("message")) != "RUNNING":
            raise Paused("control line is not RUNNING")
        self.entries = self._load_entries()
        self.by_id = {e["id"]: e for e in self.entries}
        self.writer = Writer(self.canvas, self.store, self.events, self.cycle_id, self.me,
                             self._remote_recent, self.dry_run, self.sleep, self.now, texts.SIGNATURE)
        self.notes += self.writer.reconcile()
        self._self_check()
        self._ensure_offer()
        self._serve()
        self._hire()
        return "ok"

    def _load_entries(self) -> list[dict]:
        out = []
        for e in self.canvas.get_entries():
            if e.get("deleted"):
                continue
            e["text"] = safety.html_to_text(e.get("message"))
            if e["text"]:
                out.append(e)
        return out

    def _remote_recent(self) -> tuple[int, int]:
        """My posts in the last hour: (in the HW4 topic, in the other course discussions)."""
        hour_ago = self.now() - 3600
        if not hasattr(self, "_other_recent"):
            n = 0
            for c in self.other_canvases:
                try:
                    n += sum(1 for e in c.get_entries() if e.get("user_id") == self.me
                             and (parse_time(e.get("created_at")) or 0) >= hour_ago)
                except CanvasError as e:
                    raise CanvasError(f"cannot count my posts in other discussions ({e}); not writing")
            self._other_recent = n
        here = sum(1 for e in self.entries if e.get("user_id") == self.me
                   and (parse_time(e.get("created_at")) or 0) >= hour_ago)
        return here, self._other_recent

    def _thread_root(self, e: dict) -> int:
        return root_of(e, self.by_id)

    # ------------------------------------------------------ self-check

    def _self_check(self) -> None:
        rows = self.store.accepted()
        if not rows:
            return
        passed = 0
        for r in rows:
            spec = engine.Spec.model_validate_json(r["spec"])
            claim = engine.parse_claim(r["answer"])
            try:
                passed += bool(claim and engine.matches(claim, engine.compute(spec).value))
            except engine.SpecError:
                pass
        self.events.log(self.cycle_id, "regression", passed=passed, total=len(rows))
        self.notes.append(f"regression: {passed}/{len(rows)} accepted peer cases pass")

    # ----------------------------------------------------------- offer

    def _ensure_offer(self) -> None:
        if self.store.get("offer_entry_id"):
            return
        for e in self.entries:
            if e.get("user_id") == self.me and not e.get("parent_id") and texts.OFFER_MARKER in e["text"]:
                self.store.set("offer_entry_id", e["id"])
                return
        posted = self.writer.post("offer-v1", "service offer", None, texts.OFFER)
        if posted:
            self.store.set("offer_entry_id", posted["id"])
            self.notes.append(f"posted service offer {posted['id']}")

    # -------------------------------------------------------- provider

    def _serve(self) -> None:
        offer = self.store.get("offer_entry_id")
        if not offer:
            return
        offer = int(offer)
        todo = [e for e in self.entries if e.get("user_id") != self.me and e["id"] != offer
                and self._thread_root(e) == offer and not self.store.is_handled(e["id"])]
        for e in sorted(todo, key=lambda x: x["id"]):
            if self.writer.budget_left() <= 0 and not self.dry_run:
                self.notes.append("write budget used; remaining requests wait for the next run")
                return
            self._serve_one(e)

    def _context(self, e: dict, depth: int = 3) -> str:
        chain, cur = [], e
        while cur.get("parent_id") and cur["parent_id"] in self.by_id and len(chain) < depth:
            cur = self.by_id[cur["parent_id"]]
            who = "Game Odds Lab (me)" if cur.get("user_id") == self.me else cur.get("author_name", "?")
            chain.append(f"{who}: {cur['text'][:800]}")
        return "\n---\n".join(reversed(chain)) or "(none)"

    def _serve_one(self, e: dict) -> None:
        who = e.get("author_name", "?")
        d = self.llm.interpret_request(e["text"][:3000], self._context(e))
        self.events.log(self.cycle_id, "request_seen", entry_id=e["id"], author=who,
                        action=d.action, restated=d.restated)
        key = f"job-{e['id']}"
        if d.action == "ignore":
            self._done(e, "provider", "ignored")
            return
        body, status = None, d.action
        if d.action == "compute":
            try:
                if d.spec is None:
                    raise engine.SpecError("I couldn't turn this into a precise dice or card question.")
                res = engine.compute(d.spec)
                chk = engine.simulate(d.spec, seed=int(e["id"]))
                body = texts.result(d.restated, engine.describe(d.spec), res, chk,
                                    d.spec.model_dump_json(exclude_defaults=True))
                status = "answered"
            except engine.SpecError as err:
                body, status = texts.decline(str(err)), "declined"
        elif d.action == "clarify":
            body = texts.clarify(d.restated, d.message[:300])
        else:
            body = texts.decline(d.message[:300])
        posted = self.writer.post(key, f"service {status}", e["id"], body)
        if posted or self.dry_run:
            if not self.dry_run:
                self._done(e, "provider", status)
                self.store.upsert_job(e["id"], who, status, d.restated, key)
            self.notes.append(f"{status} request {e['id']} from {who}")

    def _done(self, e: dict, role: str, outcome: str) -> None:
        if not self.dry_run:
            self.store.mark_handled(e["id"], role, outcome)

    # ----------------------------------------------------------- client

    def _hire(self) -> None:
        # Two passes: if following up ends a hire (declined, expired, rejected),
        # the next provider can be chosen and asked in the same cycle.
        for _ in range(2):
            hires = self.store.hires()
            if any(h["status"] == "accepted" for h in hires):
                return
            hire = self.store.active_hire()
            if hire is None:
                if len(hires) >= MAX_HIRES or self.now() > NO_NEW_HIRES_AFTER:
                    return
                hire = self._discover()
                if hire is None:
                    return
            if hire["status"] == "chosen":
                self._send_request(hire)
                return
            self._follow_up(hire)
            if self.store.active_hire() is not None or self.dry_run:
                return

    def _discover(self):
        tried = self.store.tried_offers()
        offers = [{"id": e["id"], "author": e.get("author_name", "?"), "user_id": e.get("user_id"),
                   "text": e["text"]}
                  for e in self.entries if not e.get("parent_id") and e.get("user_id") != self.me
                  and e["id"] not in tried and len(e["text"]) >= 40]
        if not offers:
            self.store.log_event("discovery", {"cycle": self.cycle_id, "offers": 0, "chosen": None})
            self.events.log(self.cycle_id, "discovery", offers=0)
            self.notes.append("discovery: no untried service offers yet")
            return None
        ranking = self.llm.rank_offers(texts.NEED, offers[:40])
        by_id = {o["id"]: o for o in offers}
        scored = sorted((s for s in ranking.offers if s.entry_id in by_id),
                        key=lambda s: s.score, reverse=True)
        record = [{"entry_id": s.entry_id, "author": by_id[s.entry_id]["author"],
                   "score": max(0, min(5, s.score)), "reason": s.reason} for s in scored]
        best = next((s for s in scored if s.score >= MIN_SCORE), None)
        self.store.log_event("discovery", {"cycle": self.cycle_id, "offers": len(offers),
                                           "ranking": record, "chosen": best.entry_id if best else None})
        self.events.log(self.cycle_id, "discovery", offers=len(offers), ranking=record[:10],
                        chosen=best.entry_id if best else None)
        if best is None:
            self.notes.append(f"discovery: {len(offers)} offers, none scored {MIN_SCORE}+ for our task")
            return None
        o = by_id[best.entry_id]
        self.notes.append(f"chose {o['author']} (offer {o['id']}, score {best.score}): {best.reason}")
        if self.dry_run:
            return {"id": 0, "status": "chosen", "offer_entry_id": o["id"], "provider_id": o["user_id"],
                    "provider_name": o["author"]}
        hid = self.store.add_hire(o["id"], o["user_id"], o["author"], best.reason)
        return next(h for h in self.store.hires() if h["id"] == hid)

    def _send_request(self, hire) -> None:
        offer = self.by_id.get(hire["offer_entry_id"])
        if offer is None:
            self.store.update_hire(hire["id"], status="offer_gone")
            return
        deadline = self.now() + REQUEST_HOURS * 3600
        first_line = offer["text"].splitlines()[0] if offer["text"] else "your service"
        body = texts.request(hire["provider_name"], first_line, deadline, len(engine.MUTANTS))
        key = f"hire-{hire['id']}-request"
        posted = self.writer.post(key, "service request", hire["offer_entry_id"], body)
        if posted:
            self.store.update_hire(hire["id"], status="requested", request_key=key,
                                   request_entry_id=posted["id"], deadline=deadline)
            self.notes.append(f"sent request {posted['id']} to {hire['provider_name']}")

    def _follow_up(self, hire) -> None:
        req_id = hire["request_entry_id"]
        root = hire["offer_entry_id"]
        replies = [e for e in self.entries if e.get("user_id") == hire["provider_id"] and e["id"] > req_id
                   and self._thread_root(e) == root and not self.store.is_handled(e["id"])]
        request_body = (self.store.write(hire["request_key"]) or {"body": ""})["body"]
        for e in sorted(replies, key=lambda x: x["id"]):
            kind = self.llm.classify_reply(request_body, e["text"][:6000])
            self.events.log(self.cycle_id, "provider_reply", entry_id=e["id"], kind=kind.kind,
                            reason=kind.reason)
            if kind.kind == "result":
                self._verify_result(hire, e)
                return
            if kind.kind == "question":
                asked = int(self.store.get(f"hire-{hire['id']}-answers", "0"))
                if asked < MAX_QUESTIONS_ANSWERED and kind.answer:
                    if self.writer.post(f"hire-{hire['id']}-answer-{e['id']}", "answer provider question",
                                        e["id"], texts.answer_question(kind.answer[:600])):
                        self.store.set(f"hire-{hire['id']}-answers", asked + 1)
                self._done(e, "client", "question")
                continue
            if kind.kind == "decline":
                self._done(e, "client", "declined")
                self.store.update_hire(hire["id"], status="declined")
                self.notes.append(f"{hire['provider_name']} declined; will look for another provider")
                return
            self._done(e, "client", "other")
        if hire["deadline"] and self.now() > hire["deadline"]:
            if hire["status"] == "correction_requested":
                self._finalize(hire, json.loads(hire["report"] or "{}").get("rows", []))
            else:
                self.store.update_hire(hire["id"], status="expired")
                self.notes.append(f"{hire['provider_name']} did not reply by the deadline; "
                                  "will look for another provider")

    # --------------------------------------------------- verification

    def _evaluate(self, text: str, seed: int) -> list[dict]:
        cases = self.llm.parse_cases(text[:8000]).cases[:MAX_CASES]
        rows = []
        for i, c in enumerate(cases):
            row = {"label": (c.label or f"Q{i + 1}")[:20], "question": c.question[:300],
                   "claimed": c.claimed[:40], "spec": c.spec.model_dump_json(exclude_defaults=True)}
            claim = engine.parse_claim(c.claimed)
            try:
                exact = engine.compute(c.spec)
                chk = engine.independent_check(c.spec, seed + i)
            except engine.SpecError as err:
                rows.append({**row, "status": "unsupported", "note": f"outside my engine: {err}"})
                continue
            row["exact"] = texts._num(exact.value)
            row["check"] = f"{chk.method}: {chk.value:.6f}"
            if not engine.agrees(exact.value, chk):
                rows.append({**row, "status": "unverifiable", "note": "my two methods disagree, so not used"})
            elif claim is None:
                rows.append({**row, "status": "unclear", "note": "could not read the answer"})
            elif engine.matches(claim, exact.value):
                rows.append({**row, "status": "accepted"})
            else:
                rows.append({**row, "status": "mismatch"})
        return rows

    @staticmethod
    def _caught(rows: list[dict]) -> list[str]:
        caught = []
        for m in engine.MUTANTS:
            for r in rows:
                if r["status"] != "accepted":
                    continue
                spec = engine.Spec.model_validate_json(r["spec"])
                try:
                    if engine.compute(spec, m).value != engine.compute(spec).value:
                        caught.append(m)
                        break
                except engine.SpecError:
                    continue
        return caught

    def _verify_result(self, hire, e: dict) -> None:
        rows = self._evaluate(e["text"], seed=int(e["id"]))
        if hire["status"] == "correction_requested":
            prior = json.loads(hire["report"] or "{}").get("rows", [])
            kept = [r for r in prior if r["status"] == "accepted"]
            seen = {r["question"] for r in kept}
            rows = kept + [r for r in rows if r["question"] not in seen]
        self.events.log(self.cycle_id, "verification", entry_id=e["id"],
                        statuses=[r["status"] for r in rows])
        if not rows:
            self._done(e, "client", "no cases found")
            return
        mismatches = any(r["status"] == "mismatch" for r in rows)
        if mismatches and hire["corrections"] < 1 and self.now() < NO_NEW_HIRES_AFTER + 24 * 3600:
            deadline = min(self.now() + CORRECTION_HOURS * 3600, STOP_ALL_AFTER - 3600)
            caught = self._caught(rows)
            body = texts.verification(rows, len(caught), len(engine.MUTANTS), final=False,
                                      deadline=deadline, tip=None)
            if self.writer.post(f"hire-{hire['id']}-verify-{e['id']}", "correction request", e["id"], body):
                self._done(e, "client", "correction requested")
                self.store.update_hire(hire["id"], status="correction_requested", corrections=1,
                                       deadline=deadline, report=json.dumps({"rows": rows, "caught": caught}))
                self.notes.append(f"asked {hire['provider_name']} to correct "
                                  f"{sum(r['status'] == 'mismatch' for r in rows)} cases")
            return
        self._finalize(hire, rows, reply_to=e)

    def _finalize(self, hire, rows: list[dict], reply_to: dict | None = None) -> None:
        accepted = [r for r in rows if r["status"] == "accepted"]
        caught = self._caught(rows)
        parent = reply_to["id"] if reply_to else hire["request_entry_id"]
        if len(accepted) >= MIN_ACCEPTED:
            tip = min(5, 1 + len(caught))
            body = texts.verification(rows, len(caught), len(engine.MUTANTS), final=True,
                                      deadline=None, tip=tip)
            if self.writer.post(f"hire-{hire['id']}-final", "acceptance and tip", parent, body):
                for r in accepted:
                    self.store.add_accepted(hire["id"], r["label"], r["question"], r["spec"], r["claimed"])
                self.store.update_hire(hire["id"], status="accepted", tip=tip,
                                       report=json.dumps({"rows": rows, "caught": caught}))
                if reply_to:
                    self._done(reply_to, "client", "accepted")
                self.notes.append(f"accepted {len(accepted)} cases from {hire['provider_name']}; "
                                  f"caught {len(caught)}/{len(engine.MUTANTS)} planted bugs; tip {tip}")
        else:
            body = texts.reject(f"Only {len(accepted)} of {len(rows)} cases passed my checks "
                                f"(I need at least {MIN_ACCEPTED}).")
            if self.writer.post(f"hire-{hire['id']}-final", "rejection", parent, body):
                self.store.update_hire(hire["id"], status="rejected",
                                       report=json.dumps({"rows": rows, "caught": caught}))
                if reply_to:
                    self._done(reply_to, "client", "rejected")
                self.notes.append(f"rejected work from {hire['provider_name']}; will look for another provider")
