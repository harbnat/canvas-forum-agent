"""Command line entry point.

  python -m agent check                 read-only connectivity check (no writes, no LLM)
  python -m agent run [--dry-run] [--inject-fault NAME]
                                        one scheduled cycle (this is what cron runs)
  python -m agent status                recent cycles, posts, failure counter
  python -m agent evidence              markdown summary for the homework write-up
  python -m agent reset                 clear the halt flag after investigating failures
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from datetime import datetime, timezone

from . import safety
from .config import ConfigError, load_config
from .events import EventLog, setup_logging
from .faults import FAULTS, Faults
from .memory import Memory


def _fmt(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one agent cycle")
    run.add_argument("--dry-run", action="store_true", help="decide but never post")
    run.add_argument("--inject-fault", choices=sorted(FAULTS), help="exercise a failure path")
    sub.add_parser("check", help="read-only connectivity check")
    sub.add_parser("status", help="show recent cycles and posts")
    sub.add_parser("evidence", help="print a markdown evidence summary")
    sub.add_parser("reset", help="clear the halt flag")
    args = p.parse_args(argv)

    needs_secrets = args.cmd in ("run", "check")
    try:
        cfg = load_config(require_secrets=needs_secrets)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    secrets = [cfg.canvas_token, os.environ.get("ANTHROPIC_API_KEY", "")]
    redactor = setup_logging(cfg.log_dir, secrets)
    memory = Memory(cfg.state_dir / "memory.sqlite3")
    events = EventLog(cfg.log_dir, redactor)
    try:
        return _dispatch(args, cfg, memory, events)
    finally:
        memory.close()  # checkpoints the WAL so state/ is one complete file


def _dispatch(args, cfg, memory: Memory, events: EventLog) -> int:
    if args.cmd == "status":
        return _status(cfg, memory)
    if args.cmd == "evidence":
        return _evidence(cfg, memory, events)
    if args.cmd == "reset":
        memory.set("halted", "0")
        memory.set("consecutive_failures", "0")
        events.log("manual", "reset", note="halt flag cleared by operator")
        print("halt flag cleared")
        return 0

    from .canvas import CanvasClient

    faults = Faults(getattr(args, "inject_fault", None))
    canvas = CanvasClient(cfg.canvas_base_url, cfg.canvas_token, cfg.course_id, cfg.topic_id,
                          faults=faults)
    if args.cmd == "check":
        return _check(cfg, canvas)

    # One cycle at a time: if cron fires while a slow cycle is still running, skip.
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    lock = open(cfg.state_dir / "agent.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another cycle is already running; exiting")
        return 0

    from .brain import Brain
    from .cycle import Agent

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        print("config error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2
    brain = Brain(model=cfg.model, agent_name=cfg.agent_name, use_fallbacks=cfg.use_fallbacks,
                  faults=faults)
    agent = Agent(cfg, canvas, brain, memory, events, faults=faults, dry_run=args.dry_run)
    result = agent.run_cycle()
    print(f"[{agent.cycle_id}] {result.outcome}: {result.detail}")
    for url in result.posted_urls:
        print(f"  posted: {url}")
    return 1 if result.failed else 0


def _check(cfg, canvas) -> int:
    me = canvas.get_self()
    topic = canvas.get_topic()
    entries = canvas.get_entries()
    print(f"Authenticated as: {me.get('name')} (user id {me.get('id')})")
    print(f"Topic: {topic.get('title')!r}  {cfg.topic_url}")
    first = safety.html_to_text(topic.get("message")).splitlines()[:1]
    print(f"Control line: {first[0] if first else '(missing)'}")
    print(f"Parsed control state: {safety.control_state(topic.get('message'))}")
    print(f"Entries visible: {len(entries)} "
          f"({sum(1 for e in entries if e.get('user_id') == me.get('id'))} are mine)")
    return 0


def _status(cfg, memory: Memory) -> int:
    print(f"halted={memory.halted}  consecutive_failures={memory.consecutive_failures}")
    print("\nRecent cycles:")
    for c in memory.recent_cycles(15):
        print(f"  {_fmt(c['started_at'])}  {c['outcome'] or 'running?':24} {c['detail'] or ''}"[:200])
    print("\nMy posts:")
    for a in memory.my_posts(20):
        url = cfg.entry_url(a["canvas_entry_id"]) if a["canvas_entry_id"] else "-"
        print(f"  {_fmt(a['created_at'])}  {a['status']:9} {a['kind']:10} {url}")
    return 0


def _evidence(cfg, memory: Memory, events: EventLog) -> int:
    print(f"# Agent activity evidence\n\nForum: {cfg.topic_url}\n")
    print("## Posts made autonomously\n")
    for a in reversed(memory.my_posts(100)):
        if a["status"] == "verified":
            print(f"- {_fmt(a['created_at'])} — {a['kind']} — {cfg.entry_url(a['canvas_entry_id'])}"
                  f" — _{a['reason']}_")
    print("\n## Scheduled cycles\n\n| started | outcome | detail |\n|---|---|---|")
    for c in reversed(memory.recent_cycles(200)):
        detail = (c["detail"] or "").replace("|", "/")[:160]
        print(f"| {_fmt(c['started_at'])} | {c['outcome']} | {detail} |")
    fault_cycles = {e["cycle"] for e in events.read() if e.get("event") == "cycle_start" and e.get("fault")}
    if fault_cycles:
        print("\n## Failure-injection runs (from logs/agent.jsonl)\n\n```")
        for e in events.read():
            if e.get("cycle") in fault_cycles or e.get("event") == "reconciled_pending":
                e.pop("body", None)
                print(e)
        print("```")
    return 0


if __name__ == "__main__":
    sys.exit(main())
