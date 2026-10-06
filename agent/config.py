"""Configuration loaded from environment variables (and an optional .env file).

Secrets (CANVAS_TOKEN, ANTHROPIC_API_KEY) are only ever read from the
environment. They are never written to disk, logged, or sent anywhere except
their own service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

HARD_MAX_POSTS_PER_HOUR = 3  # assignment limit; config can lower it, never raise it


class ConfigError(Exception):
    pass


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, # comments. Existing env wins."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ConfigError(f"{name} must be an integer") from e


@dataclass(frozen=True)
class Config:
    canvas_base_url: str
    canvas_token: str = field(repr=False)
    course_id: int
    topic_id: int
    model: str
    agent_name: str
    state_dir: Path
    log_dir: Path
    max_posts_per_hour: int
    max_posts_per_cycle: int
    max_consecutive_failures: int
    use_fallbacks: bool
    max_entry_age_hours: int = 72
    thread_cooldown_hours: int = 12

    @property
    def topic_url(self) -> str:
        return f"{self.canvas_base_url}/courses/{self.course_id}/discussion_topics/{self.topic_id}"

    def entry_url(self, entry_id: int) -> str:
        return f"{self.topic_url}?entry_id={entry_id}"


def load_config(require_secrets: bool = True) -> Config:
    load_dotenv(Path.cwd() / ".env")

    token = os.environ.get("CANVAS_TOKEN", "").strip()
    if require_secrets and not token:
        raise ConfigError("CANVAS_TOKEN is not set (put it in .env or your environment)")

    base_url = os.environ.get("CANVAS_BASE_URL", "https://canvas.mit.edu").rstrip("/")
    if not base_url.startswith("https://"):
        raise ConfigError("CANVAS_BASE_URL must use https")

    per_hour = min(_int("AGENT_MAX_POSTS_PER_HOUR", 3), HARD_MAX_POSTS_PER_HOUR)

    return Config(
        canvas_base_url=base_url,
        canvas_token=token,
        course_id=_int("CANVAS_COURSE_ID", 40577),
        topic_id=_int("CANVAS_TOPIC_ID", 448963),
        model=os.environ.get("AGENT_MODEL", "claude-opus-5-5").strip(),
        agent_name=os.environ.get("AGENT_NAME", "Threadweaver").strip() or "Threadweaver",
        state_dir=Path(os.environ.get("AGENT_STATE_DIR", "state")),
        log_dir=Path(os.environ.get("AGENT_LOG_DIR", "logs")),
        max_posts_per_hour=max(per_hour, 0),
        max_posts_per_cycle=max(min(_int("AGENT_MAX_POSTS_PER_CYCLE", 1), per_hour), 0),
        max_consecutive_failures=max(_int("AGENT_MAX_CONSECUTIVE_FAILURES", 3), 1),
        use_fallbacks=os.environ.get("AGENT_USE_FALLBACKS", "1").strip() != "0",
        max_entry_age_hours=max(_int("AGENT_MAX_ENTRY_AGE_HOURS", 72), 1),
        thread_cooldown_hours=max(_int("AGENT_THREAD_COOLDOWN_HOURS", 12), 0),
    )
