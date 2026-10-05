"""Code-enforced safety gates. These run on every decision, whatever the model says.

The model only proposes; this module decides whether anything is allowed to
leave the machine. Forum content is untrusted, so nothing in it can switch
these checks off.
"""

from __future__ import annotations

import difflib
import html
import re

CONTROL_RE = re.compile(r"COURSE-TEAM\s+CONTROL:\s*(RUNNING|PAUSED)\b", re.IGNORECASE)

MIN_BODY_CHARS = 80
MAX_BODY_CHARS = 2500
MAX_SIMILARITY_TO_OWN_POST = 0.6

ALLOWED_LINK_HOSTS = ("canvas.mit.edu",)

SECRET_PATTERNS = [
    (re.compile(r"\b\d{3,6}~[A-Za-z0-9]{30,}\b"), "looks like a Canvas access token"),
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), "looks like an Anthropic API key"),
    (re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{10,}"), "looks like an API key"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "looks like a GitHub token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "looks like an AWS key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "contains a private key"),
    (re.compile(r"\b(?:password|passwd|api[_ -]?key|secret[_ -]?key|bearer)\s*[:=]\s*\S+",
                re.IGNORECASE), "contains a credential assignment"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "contains an email address"),
    (re.compile(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b"), "contains a phone number"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "contains an SSN-like number"),
    (re.compile(r"\b(?:grade|gpa)s?\s+(?:of|is|was|=|:)\s*[A-F0-9]", re.IGNORECASE),
     "mentions grades"),
]

URL_RE = re.compile(r"https?://([^/\s)>\]]+)", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(message_html: str | None) -> str:
    """Canvas messages are HTML. Convert to plain text for the model and for checks."""
    if not message_html:
        return ""
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h\d>", "\n", message_html)
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def control_state(topic_message_html: str | None) -> str:
    """RUNNING or PAUSED from the control line at the very top of the topic description.

    Fails closed: if the first non-empty line is not a recognisable control line,
    the result is PAUSED and the agent will not write.
    """
    text = html_to_text(topic_message_html)
    first_line = text.splitlines()[0] if text else ""
    m = CONTROL_RE.search(first_line)
    if not m:
        return "PAUSED"
    return m.group(1).upper()


def text_to_html(body: str) -> str:
    """Plain text -> safe HTML paragraphs. Escaping means the model cannot inject markup."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body.strip()) if p.strip()]
    return "".join("<p>" + html.escape(p).replace("\n", "<br>") + "</p>" for p in paras)


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def check_body(body: str, own_previous_bodies: list[str]) -> list[str]:
    """Return a list of reasons to block this post (empty list = OK to post)."""
    problems: list[str] = []
    stripped = body.strip()
    if len(stripped) < MIN_BODY_CHARS:
        problems.append(f"too short ({len(stripped)} chars)")
    if len(stripped) > MAX_BODY_CHARS:
        problems.append(f"too long ({len(stripped)} chars)")
    for pattern, why in SECRET_PATTERNS:
        if pattern.search(stripped):
            problems.append(why)
    for host in URL_RE.findall(stripped):
        host = host.lower().split(":")[0]
        if not any(host == h or host.endswith("." + h) for h in ALLOWED_LINK_HOSTS):
            problems.append(f"links to a non-allowlisted host ({host})")
    for prev in own_previous_bodies:
        if similarity(stripped, prev) > MAX_SIMILARITY_TO_OWN_POST:
            problems.append("too similar to one of my earlier posts")
            break
    return problems


def contains_post(haystack_html: str | None, body: str) -> bool:
    """Does an entry on Canvas contain the body we tried to post? (used for reconcile/verify)"""
    have = normalize(html_to_text(haystack_html))
    want = normalize(body)
    if not want:
        return False
    probe = want[:300]
    return probe in have or similarity(have[: len(want) + 200], want) > 0.9
