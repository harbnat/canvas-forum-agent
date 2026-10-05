"""Thin Canvas REST client scoped to ONE discussion topic.

Only the endpoints this homework needs are implemented. There is deliberately
no edit or delete method, so the agent cannot modify anyone's contribution.

Reads are retried with exponential backoff. Writes are not blindly retried:
a timeout or 5xx on a POST is "ambiguous" (Canvas may have saved it), so the
caller must reconcile against the live forum before trying again.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable

import requests

from .faults import Faults

log = logging.getLogger(__name__)

TIMEOUT = (10, 30)  # connect, read seconds
READ_ATTEMPTS = 4
BACKOFF_BASE = 2.0
BACKOFF_MAX = 60.0


class CanvasError(Exception):
    """Non-retryable failure (bad token, 403, 404, validation error...)."""


class CanvasTransientError(CanvasError):
    """Retryable failure that definitely did NOT change anything on Canvas."""

    def __init__(self, message: str, retry_after: str | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class AmbiguousWriteError(CanvasError):
    """A write may or may not have been saved (timeout, 5xx, garbled response)."""


def backoff_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), BACKOFF_MAX)
        except ValueError:
            pass
    return min(BACKOFF_BASE * (2**attempt) + random.uniform(0, 1), BACKOFF_MAX)


class CanvasClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        course_id: int,
        topic_id: int,
        faults: Faults | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._base = base_url.rstrip("/") + "/api/v1"
        self._topic_path = f"/courses/{course_id}/discussion_topics/{topic_id}"
        self._session = session or requests.Session()
        self._session.headers["Authorization"] = f"Bearer {token}"
        self._faults = faults or Faults()
        self._sleep = sleep

    # ------------------------------------------------------------------ reads

    def _get(self, path_or_url: str, params: dict | None = None) -> requests.Response:
        url = path_or_url if path_or_url.startswith("http") else self._base + path_or_url
        if not url.startswith(self._base):
            raise CanvasError("refusing to follow a URL outside the configured Canvas API")
        last: Exception | None = None
        for attempt in range(READ_ATTEMPTS):
            try:
                if self._faults.fire("http_500"):
                    raise CanvasTransientError("injected fault: synthetic HTTP 500")
                resp = self._session.get(url, params=params, timeout=TIMEOUT)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise CanvasTransientError(f"HTTP {resp.status_code} on GET",
                                               resp.headers.get("Retry-After"))
                if resp.status_code >= 400:
                    raise CanvasError(f"HTTP {resp.status_code} on GET {path_or_url.split('?')[0]}")
                if self._faults.fire("malformed_response"):
                    raise CanvasTransientError("injected fault: malformed (non-JSON) response")
                try:
                    resp.json()
                except ValueError as e:
                    raise CanvasTransientError("malformed (non-JSON) response") from e
                return resp
            except (requests.Timeout, requests.ConnectionError, CanvasTransientError) as e:
                last = e
                delay = backoff_delay(attempt, getattr(e, "retry_after", None))
                log.warning("GET failed (%s); retry %d/%d in %.1fs",
                            type(e).__name__ + ": " + str(e), attempt + 1, READ_ATTEMPTS, delay)
                if attempt < READ_ATTEMPTS - 1:
                    self._sleep(delay)
        raise CanvasTransientError(f"GET gave up after {READ_ATTEMPTS} attempts: {last}")

    def _get_json(self, path: str, params: dict | None = None) -> Any:
        return self._get(path, params).json()

    def _get_paginated(self, path: str, params: dict | None = None) -> list[dict]:
        out: list[dict] = []
        params = {"per_page": 100, **(params or {})}
        resp = self._get(path, params)
        while True:
            page = resp.json()
            if not isinstance(page, list):
                raise CanvasTransientError("expected a list from paginated endpoint")
            out.extend(page)
            nxt = resp.links.get("next", {}).get("url")
            if not nxt or len(out) > 5000:
                return out
            resp = self._get(nxt)

    def get_self(self) -> dict:
        return self._get_json("/users/self")

    def get_topic(self) -> dict:
        return self._get_json(self._topic_path)

    def get_entries(self) -> list[dict]:
        """Every entry in the topic, flattened, each with id/user_id/parent_id/message."""
        data = self._get_json(self._topic_path + "/view")
        if not isinstance(data, dict) or "view" not in data:
            raise CanvasTransientError("unexpected shape from discussion view")
        names = {p.get("id"): p.get("display_name") for p in data.get("participants", [])}
        flat: dict[int, dict] = {}

        def walk(entries: list[dict], parent_id: int | None) -> None:
            for e in entries or []:
                e = dict(e)
                e.setdefault("parent_id", parent_id)
                replies = e.pop("replies", [])
                flat[e["id"]] = e
                walk(replies, e["id"])

        walk(data.get("view", []), None)
        for e in data.get("new_entries", []) or []:  # entries newer than Canvas's cached view
            flat.setdefault(e["id"], dict(e))
        for e in flat.values():
            e["author_name"] = names.get(e.get("user_id"), "unknown")
        return sorted(flat.values(), key=lambda e: e["id"])

    def get_top_level_entries(self) -> list[dict]:
        return self._get_paginated(self._topic_path + "/entries")

    def get_replies(self, entry_id: int) -> list[dict]:
        return self._get_paginated(f"{self._topic_path}/entries/{int(entry_id)}/replies")

    def get_entry(self, entry_id: int) -> dict | None:
        found = self._get_json(self._topic_path + "/entry_list", {"ids[]": int(entry_id)})
        return found[0] if isinstance(found, list) and found else None

    # ----------------------------------------------------------------- writes

    def _post(self, path: str, message_html: str) -> dict:
        url = self._base + path
        try:
            resp = self._session.post(url, data={"message": message_html}, timeout=TIMEOUT)
        except requests.ConnectionError as e:
            # A refused connection never reached Canvas; a reset mid-request might have.
            raise AmbiguousWriteError(f"connection error on POST: {type(e).__name__}") from e
        except requests.Timeout as e:
            raise AmbiguousWriteError("timeout on POST") from e

        if self._faults.fire("lost_ack"):
            raise AmbiguousWriteError("injected fault: acknowledgement lost after POST")
        if self._faults.fire("crash_after_post"):
            log.error("injected fault: crashing immediately after POST, before saving state")
            logging.shutdown()
            os._exit(137)

        if resp.status_code == 429:
            raise CanvasTransientError("HTTP 429 on POST (rate limited by Canvas)",
                                       resp.headers.get("Retry-After"))
        if resp.status_code >= 500:
            raise AmbiguousWriteError(f"HTTP {resp.status_code} on POST")
        if resp.status_code >= 400:
            raise CanvasError(f"HTTP {resp.status_code} on POST")
        try:
            body = resp.json()
        except ValueError as e:
            raise AmbiguousWriteError("malformed response to POST") from e
        if not isinstance(body, dict) or "id" not in body:
            raise AmbiguousWriteError("POST response missing entry id")
        return body

    def post_entry(self, message_html: str) -> dict:
        """Start a new thread (top-level entry) in the topic."""
        return self._post(self._topic_path + "/entries", message_html)

    def post_reply(self, parent_id: int, message_html: str) -> dict:
        return self._post(f"{self._topic_path}/entries/{int(parent_id)}/replies", message_html)
