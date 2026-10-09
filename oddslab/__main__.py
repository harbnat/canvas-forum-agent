"""Command line entry point for the HW4 services agent (Game Odds Lab).

  python -m oddslab check               read-only: control line, offers on the HW4 forum
  python -m oddslab run [--dry-run] [--min-gap-hours H]
                                        one scheduled cycle (this is what the timers run)
  python -m oddslab status              recent cycles, posts, hires
  python -m oddslab evidence            markdown summary for the homework write-up
  python -m oddslab reset               clear the halt flag after investigating failures

A dry run works on a throwaway copy of the memory and prints every post it
would have made, so it can be pointed at the real forum safely.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from agent import safety
from agent.config import ConfigError, _int, load_config
from agent.events import EventLog, setup_logging

from .store import Store

HW4_TOPIC_DEFAULT = 449948


def _fmt(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m oddslab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one agent cycle")
    run.add_argument("--dry-run", action="store_true", help="decide and draft, never post")
    run.add_argument("--min-gap-hours", type=float, default=0.0,
                     help="skip if the last real cycle started less than this many hours ago")
    sub.add_parser("check", help="read-only check of the HW4 forum")
    sub.add_parser("status", help="show recent cycles, posts and hires")
    sub.add_parser("evidence", help="print a markdown evidence summary")
    sub.add_parser("reset", help="clear the halt flag")
    args = p.parse_args(argv)

    try:
        base = load_config(require_secrets=args.cmd in ("run", "check"))
        hw4_topic = _int("HW4_TOPIC_ID", HW4_TOPIC_DEFAULT)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    cfg = dataclasses.replace(base, topic_id=hw4_topic)
    other_topics = [t for t in {base.topic_id} if t != hw4_topic]  # the HW3 forum

    redactor = setup_logging(cfg.log_dir, [cfg.canvas_token, os.environ.get("ANTHROPIC_API_KEY", "")])
    events = EventLog(cfg.log_dir / "hw4", redactor)
    db_path = cfg.state_dir / "hw4.sqlite3"

    if args.cmd == "run" and args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="oddslab-dry-"))
        if db_path.exists():
            src = sqlite3.connect(str(db_path))
            dst = sqlite3.connect(str(tmp / "hw4.sqlite3"))
            src.backup(dst)
            src.close()
            dst.close()
        store = Store(tmp / "hw4.sqlite3")
    else:
        store = Store(db_path)
    try:
        return _dispatch(args, cfg, other_topics, store, events)
    finally:
        store.close()
        if args.cmd == "run" and args.dry_run:
            shutil.rmtree(tmp, ignore_errors=True)


def _dispatch(args, cfg, other_topics: list[int], store: Store, events: EventLog) -> int:
    if args.cmd == "status":
        return _status(cfg, store)
    if args.cmd == "evidence":
        return _evidence(cfg, store)
    if args.cmd == "reset":
        store.set("halted", "0")
        store.set("consecutive_failures", "0")
        events.log("manual", "reset", note="halt flag cleared by operator")
        print("halt flag cleared")
        return 0

    from agent.canvas import CanvasClient

    canvas = CanvasClient(cfg.canvas_base_url, cfg.canvas_token, cfg.course_id, cfg.topic_id)
    others = [CanvasClient(cfg.canvas_base_url, cfg.canvas_token, cfg.course_id, t) for t in other_topics]
    if args.cmd == "check":
        return _check(cfg, canvas)

    if args.min_gap_hours > 0 and not args.dry_run:
        last = store.last_real_cycle_start()
        if last is not None and time.time() - last < args.min_gap_hours * 3600:
            print(f"skipped: last cycle started {(time.time() - last) / 60:.0f} min ago "
                  f"(minimum gap {args.min_gap_hours:g}h)")
            return 0

    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    lock = open(cfg.state_dir / "hw4.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another cycle is already running; exiting")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        print("config error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    from .agent import ServicesAgent
    from .llm import LLM

    llm = LLM(cfg.model, use_fallbacks=cfg.use_fallbacks)
    agent = ServicesAgent(cfg, canvas, others, llm, store, events, dry_run=args.dry_run)
    outcome, detail, failed = agent.run_cycle(os.environ.get("AGENT_TRIGGER", "local"))
    print(f"[{agent.cycle_id}] {outcome}" + (f": {detail}" if detail else ""))
    if args.dry_run:
        writer = getattr(agent, "writer", None)
        drafts = writer.drafts if writer else []
        print(f"\nDRY RUN: {len(drafts)} post(s) drafted, none sent.")
        for d in drafts:
            where = f"reply to entry {d['parent_id']}" if d["parent_id"] else "new top-level thread"
            print(f"\n----- {d['purpose']} ({where}) -----\n{d['body']}")
    return 1 if failed else 0


def _check(cfg, canvas) -> int:
    me = canvas.get_self()
    topic = canvas.get_topic()
    entries = canvas.get_entries()
    print(f"Authenticated as: {me.get('name')} (user id {me.get('id')})")
    print(f"Topic: {topic.get('title')!r}  {cfg.topic_url}")
    first = safety.html_to_text(topic.get("message")).splitlines()[:1]
    print(f"Control line: {first[0] if first else '(missing)'}")
    print(f"Parsed control state: {safety.control_state(topic.get('message'))}")
    tops = [e for e in entries if not e.get("parent_id") and not e.get("deleted")]
    print(f"Entries: {len(entries)} total, {len(tops)} top-level threads "
          f"({sum(1 for e in entries if e.get('user_id') == me.get('id'))} mine)")
    for e in tops:
        line = (safety.html_to_text(e.get("message")).splitlines() or [""])[0]
        n = sum(1 for x in entries if x.get("parent_id") and _root(x, entries) == e["id"])
        print(f"  [{e['id']}] {e.get('author_name', '?')}: {line[:100]}  ({n} replies)")
    return 0


def _root(e: dict, entries: list[dict]) -> int:
    by_id = {x["id"]: x for x in entries}
    while e.get("parent_id") and e["parent_id"] in by_id:
        e = by_id[e["parent_id"]]
    return e["id"]


def _status(cfg, store: Store) -> int:
    print(f"halted={store.get('halted', '0')}  consecutive_failures={store.get('consecutive_failures', '0')}"
          f"  offer_entry_id={store.get('offer_entry_id', '-')}")
    print("\nRecent cycles:")
    for c in store.cycles(15):
        print(f"  {_fmt(c['started_at'])}  {c['trigger'] or '-':16} {c['outcome'] or 'running?':10} "
              f"{c['detail'] or ''}"[:240])
    print("\nMy posts:")
    for w in store.db.execute("SELECT * FROM writes ORDER BY created_at DESC LIMIT 20"):
        url = cfg.entry_url(w["entry_id"]) if w["entry_id"] else "-"
        print(f"  {_fmt(w['created_at'])}  {w['status']:10} {w['purpose']:24} {url}")
    print("\nHires:")
    for h in store.hires():
        print(f"  #{h['id']} {h['provider_name']} (offer {h['offer_entry_id']}): {h['status']}"
              f"  deadline {_fmt(h['deadline'])}  tip {h['tip'] if h['tip'] is not None else '-'}")
    print("\nRequests served:")
    for j in store.jobs():
        print(f"  {_fmt(j['at'])}  {j['requester']}: {j['status']} - {j['summary']}"[:200])
    return 0


def _evidence(cfg, store: Store) -> int:
    print(f"# HW4 agent evidence\n\nForum: {cfg.topic_url}\n\n## Posts\n")
    for w in store.db.execute("SELECT * FROM writes WHERE status='verified' ORDER BY created_at"):
        print(f"- {_fmt(w['created_at'])}: {w['purpose']}: {cfg.entry_url(w['entry_id'])}")
    print("\n## Discovery rounds\n")
    for ev in store.events("discovery"):
        d = json.loads(ev["data"])
        print(f"- {_fmt(ev['at'])}: {d.get('offers')} offers; chosen: {d.get('chosen')}")
        for r in d.get("ranking", [])[:8]:
            print(f"  - [{r['entry_id']}] {r['author']}: score {r['score']} ({r['reason']})")
    print("\n## Hires\n")
    for h in store.hires():
        print(f"- #{h['id']} {h['provider_name']} (offer {h['offer_entry_id']}): {h['status']}, "
              f"tip {h['tip'] if h['tip'] is not None else '-'}; reason chosen: {h['reason']}")
        if h["report"]:
            rep = json.loads(h["report"])
            for r in rep.get("rows", []):
                print(f"  - {r['status']}: {r['label']} {r['question'][:90]} | claimed {r['claimed']}"
                      f" | exact {r.get('exact', '-')} | {r.get('check', r.get('note', ''))}")
            print(f"  - planted bugs caught: {rep.get('caught')}")
    print("\n## Requests served\n")
    for j in store.jobs():
        print(f"- {_fmt(j['at'])} {j['requester']}: {j['status']} — {j['summary']}")
    print("\n## Cycles\n\n| started | trigger | outcome | detail |\n|---|---|---|---|")
    for c in reversed(store.cycles(300)):
        detail = (c["detail"] or "").replace("|", "/")[:200]
        print(f"| {_fmt(c['started_at'])} | {c['trigger'] or '-'} | {c['outcome']} | {detail} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
