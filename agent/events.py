"""Structured evidence log (logs/agent.jsonl) plus secret redaction for all logging."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path


class Redactor(logging.Filter):
    """Scrub configured secrets from every log record, whatever logger emitted it."""

    def __init__(self, secrets: list[str]):
        super().__init__()
        self.secrets = [s for s in secrets if s and len(s) >= 8]

    def scrub(self, text: str) -> str:
        for s in self.secrets:
            text = text.replace(s, "[REDACTED]")
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.scrub(record.getMessage())
        record.args = ()
        return True


class EventLog:
    def __init__(self, log_dir: Path, redactor: Redactor | None = None):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "agent.jsonl"
        self.redactor = redactor or Redactor([])

    def log(self, cycle_id: str, event: str, **fields: object) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "cycle": cycle_id, "event": event, **fields}
        line = self.redactor.scrub(json.dumps(record, default=str))
        with self.path.open("a") as f:
            f.write(line + "\n")
            f.flush()
        logging.getLogger("agent.events").info("%s %s", event,
                                               {k: v for k, v in fields.items() if k != "body"})

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


def setup_logging(log_dir: Path, secrets: list[str]) -> Redactor:
    log_dir.mkdir(parents=True, exist_ok=True)
    redactor = Redactor(secrets)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(log_dir / "agent.log")):
        h.setFormatter(fmt)
        h.addFilter(redactor)
        root.addHandler(h)
    for noisy in ("httpx", "httpx2", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return redactor
