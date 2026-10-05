# Threadweaver: an autonomous Canvas forum agent

Homework 3, "Make Your Agent Autonomous". Threadweaver wakes up on a schedule,
reads the **Homework 3: Agent Discussion Forum** on Canvas, decides on its own
whether it has something useful to add, and if so posts one reply or thread.
Then it verifies the post was saved and goes back to sleep. Nobody prompts it
between runs.

**Persona:** synthesizer and Socratic questioner. It either connects what several
agents have said (who agrees, who disagrees, what's still open) or asks one
pointed question or counterexample that moves a thread forward. When neither
would add anything, it stays quiet and logs why.

---

## Setup

Requires Python 3.10+ and macOS or Linux.

```bash
git clone https://github.com/harbnat/canvas-forum-agent.git
cd canvas-forum-agent
python3 -m venv .venv
./.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env        # then edit .env (it is gitignored)
```

Fill in `.env`:

| Variable | What |
|---|---|
| `CANVAS_TOKEN` | Your own token: Canvas → Account → Settings → **+ New Access Token**, with an expiry shortly after the due date. |
| `ANTHROPIC_API_KEY` | Claude API key. |
| `CANVAS_COURSE_ID` / `CANVAS_TOPIC_ID` | Defaults point at the HW3 forum (`40577` / `448963`). |

Then check the setup, rehearse, and run the tests:

```bash
./.venv/bin/python -m pytest -q                 # 37 offline tests, no network
./.venv/bin/python -m agent check               # read-only: who am I, control line, entry count
./.venv/bin/python -m agent run --dry-run       # full cycle, but never posts
./.venv/bin/python -m agent run                 # one real cycle
```

### Start the schedule (once; after that it runs by itself)

**cron** (macOS or Linux): `crontab -e`, then add this line with your absolute path:

```
17 */3 * * * /ABSOLUTE/PATH/TO/canvas-forum-agent/scripts/run_cycle.sh
```

**launchd** (macOS alternative): see `scripts/com.threadweaver.agent.plist`.

On macOS, if cron can't read the folder, move the repo out of `~/Documents` and
`~/Desktop`, or give `cron` Full Disk Access. The machine must be awake for runs
to happen.

### Day-to-day commands

```bash
./.venv/bin/python -m agent status     # recent cycles, my posts, failure counter
./.venv/bin/python -m agent evidence   # markdown summary for the write-up
./.venv/bin/python -m agent reset      # clear the halt flag after investigating
crontab -l / crontab -e                # see or stop the schedule
```

---

## Architecture

```
 cron (every 3h) ──▶ scripts/run_cycle.sh ──▶ python -m agent run
                                                   │
   ┌───────────────────────────────────────────────┴──────────────────────────┐
   │ 0. lock file (one cycle at a time) · halted? → exit                        │
   │ 1. GET topic → parse control line (fail closed: missing = PAUSED)          │
   │ 2. GET all entries (view + new_entries) → strip HTML                       │
   │ 3. Reconcile any 'pending' writes from a crashed / interrupted earlier run │
   │ 4. PAUSED? → stop.  Nothing new from other agents? → stop (no LLM call)    │
   │ 5. Post budget: ≤1 per cycle, ≤3 per rolling hour (local AND remote count) │
   │ 6. Claude decides → structured JSON {action, target, style, body, reason}  │
   │ 7. Code gates: target valid · not me · one reply per entry · no secrets/PII│
   │    · links only to canvas.mit.edu · not similar to my past posts · length  │
   │ 8. Save 'pending' action with an idempotency key (SQLite, fsync)           │
   │ 9. Re-read control line → POST (backoff on 429; reconcile on ambiguity)    │
   │10. Verify: re-fetch the entry, check author + text → mark 'verified'       │
   └────────────────────────────────────────────────────────────────────────────┘
```

| Module | Role |
|---|---|
| `agent/canvas.py` | Canvas REST client scoped to one topic. Reads retry with exponential backoff and jitter (honours `Retry-After`). There are no edit or delete methods. A POST timeout or 5xx is treated as **ambiguous**, never blindly retried. |
| `agent/memory.py` | SQLite memory: `seen_entries`, `actions` (idempotency key and status), `cycles` (outcome of every run), and `kv` (failure counter, halt flag). |
| `agent/brain.py` | One Claude call (`claude-opus-5-5`, structured output, no tools). Forum text goes inside `<untrusted_forum_content>`. |
| `agent/safety.py` | Control-line parser, HTML escaping, and content gates (secrets, PII, links, near-duplicates). |
| `agent/cycle.py` | The cycle above: decision logic, rate limits, recovery, and the stopping rule. |
| `agent/events.py` | `logs/agent.jsonl` evidence log, with secrets redacted from every log line. |
| `agent/faults.py` | Opt-in failure injection used to show recovery. |

### Decision logic
- No new entries from other agents means no model call. Outcome: `no_post_nothing_new`.
- Otherwise Claude sees the threads containing new entries, plus its own recent posts.
  It returns `none` (the default when nothing is genuinely useful), `reply` (preferred),
  or `new_thread`, with a one-line reason that is logged either way.
- Code checks the result. A blocked decision becomes `blocked_by_gate` and nothing is posted.

### Persistent memory and idempotency
- Each entry it has considered is stored in `seen_entries`, so it's never re-processed.
  Its own entries are stored too and never treated as input.
- Each write is saved as `pending` **before** the POST. Its key is
  `sha256(kind | parent | body)`, so re-running the same decision can't post twice.
- After the POST, the agent re-fetches the entry and checks the author and text,
  then marks it `verified`.
- At the start of each cycle, any leftover `pending` action is reconciled against
  Canvas. If found, it's marked verified and not re-posted. If not found, it's
  marked `not_posted`.

### Rate limits and stopping rule
- At most 1 post per cycle and 3 per rolling hour. The hourly count takes the larger
  of the local and live-forum counts, so losing local state can't reset it. The
  config can lower the hourly cap but code clamps it to 3.
- Reads retry 4 times with backoff (2s, 4s, 8s plus jitter). POSTs get up to 3
  attempts, and each attempt re-reads the control line first.
- A cycle that ends in error increments `consecutive_failures`. After **3 in a row**,
  the agent sets `halted` and every later cron run exits at once until a person runs
  `python -m agent reset`. Any successful cycle resets the counter.

### Safety and blast radius
- **Credentials:** the Canvas token is read only from the environment or `.env`
  (gitignored). It is sent only in the Canvas `Authorization` header and redacted
  from every log line. The Claude key is handled the same way. Nothing secret is
  ever given to the model.
- **Capabilities:** the model has no tools, so it can't run commands, read files,
  browse, or call Canvas. Its only effect is a JSON proposal, which code checks.
  The Canvas client is limited to one topic, follows no URLs outside the Canvas API,
  and has no edit or delete method.
- **Prompt injection:** forum text is labelled as untrusted data. The system prompt
  says it can't change the rules. Even if the model were fooled, gates written in
  code block secrets, emails, phone numbers, grades, outside links, self-replies,
  repeats, and over-limit posting. Post text is HTML-escaped, so it can't inject markup.
- **Control line:** checked at the start of each cycle and again right before every
  POST. Anything other than an exact `COURSE-TEAM CONTROL: RUNNING` on the first
  line counts as PAUSED.

---

## Failure injection (requirement 6)

Each fault fires once, so you see the recovery path in action:

| Command | What it shows |
|---|---|
| `python -m agent run --inject-fault lost_ack` | The POST reaches Canvas, then the response is "lost". The agent finds its own post and does **not** post again. |
| `python -m agent run --inject-fault crash_after_post` | The process dies right after the POST, before saving. The next `python -m agent run` reconciles the pending action: `recovered earlier post … without reposting`. |
| `python -m agent run --inject-fault duplicate_event` | The same decision is executed twice. The second is `duplicate_suppressed`. |
| `python -m agent run --inject-fault http_500` | A synthetic 500 on a read, which is retried with backoff. |
| `python -m agent run --inject-fault malformed_response` | A garbled (non-JSON) Canvas response, which is retried. |
| `python -m agent run --inject-fault malformed_llm` | An unparseable model decision. The agent fails closed (no post) and the failure counts toward the stopping rule. |

`python -m agent evidence` prints these runs from `logs/agent.jsonl` for the write-up.

---

## What not to submit

`.env`, `state/` and `logs/` are gitignored. Before sharing a screenshot or log
excerpt, check that it has no token. Tokens are redacted in logs, but check anyway.
