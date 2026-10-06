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

## Setup (GitHub Actions: runs in the cloud, no laptop needed)

The workflow `.github/workflows/agent-cycle.yml` runs one cycle **about every 3 hours** on GitHub's servers. The agent's memory (`state/`)
and logs are saved in the GitHub Actions cache after every run, even failed ones,
and restored at the start of the next.

1. **Get the two keys.**
   - Canvas: Account → Settings → Approved Integrations → **+ New Access Token**,
     expiring shortly after the due date.
   - Claude: https://console.anthropic.com → **API Keys**.
2. **Store them as repo secrets.** In this repo on GitHub, go to **Settings →
   Secrets and variables → Actions → New repository secret** and add two secrets:
   `CANVAS_TOKEN` and `ANTHROPIC_API_KEY`. GitHub encrypts them and hides them in logs.
3. **Check the connection.** Go to **Actions → agent-cycle → Run workflow**, choose
   mode `check`, and click Run. Open the run and look at the "Run agent" step for your
   name and `Parsed control state: RUNNING`.
4. **Rehearse.** Run the workflow with mode `dry-run`. It decides but never posts.
5. **Go live.** Run the workflow once with mode `run`. After that, the schedule
   runs it by itself. Leave it alone; nothing else is needed.

**Watching it:** each run on the **Actions** tab shows a status summary. Its
**Artifacts** section has `agent-logs-…`, which contains `logs/agent.jsonl` and
an up-to-date `evidence.md` for the write-up.

**Manual modes** (Actions → agent-cycle → Run workflow):

| mode | does |
|---|---|
| `run` | one real cycle; optionally pick an `inject_fault` (see below) |
| `auto` | one cycle unless the last real one was < 2.5h ago (used by the backup timer) |
| `dry-run` | full cycle, never posts |
| `check` | read-only connectivity check |
| `status` | show recent cycles and posts |
| `reset` | clear the halt flag after investigating repeated failures |

**Stopping it:** Actions → agent-cycle → **⋯ → Disable workflow**. After the
homework, also delete the token in Canvas settings.

**Notes**
- GitHub's scheduler often delays or silently drops scheduled runs (on day one it ran
  only 1 of 4). So the workflow asks for **five starts per hour**, and the agent skips any
  scheduled start less than 2.5 hours after its last real cycle
  (`--min-gap-hours 2.5`). A dropped start is then covered by the next one. Skipped starts print `skipped: ...` and don't count as cycles. Manual
  runs are never skipped.
- Runs never overlap (`concurrency` group), so two cycles can't race on the memory.
- Each run uses about 1–2 minutes, well within the free private-repo allowance.
- If the cache were ever evicted, the agent would start with empty memory. It still
  reads its own live posts from Canvas, so it won't reply twice to the same entry,
  exceed the hourly limit, or repeat an earlier post.

### Backup timer (recommended): cron-job.org

GitHub's own scheduler can go hours without firing. A free external timer makes
the schedule reliable by calling GitHub's API every hour to start the workflow
in `auto` mode. The agent applies the same 2.5h gap, so you still get about one
cycle every 3 hours.

1. **Make a narrow GitHub token.** Go to GitHub → Settings → Developer settings →
   **Fine-grained tokens** → Generate new token.
   - Repository access: **Only select repositories** → `canvas-forum-agent`
   - Permissions: **Actions → Read and write** (nothing else)
   - Expiration: shortly after the homework is due

   This token can only start or cancel workflow runs in this one repo. It can't
   read your code, secrets, Canvas, or anything else.
2. **Create the timer.** At https://cron-job.org (free account), create a cron job:
   - URL: `https://api.github.com/repos/harbnat/canvas-forum-agent/actions/workflows/agent-cycle.yml/dispatches`
   - Schedule: every hour (any minute)
   - Advanced → Request method: **POST**
   - Headers:
     `Authorization: Bearer <your token>`, `Accept: application/vnd.github+json`,
     `Content-Type: application/json`
   - Request body: `{"ref":"main","inputs":{"mode":"auto"}}`
3. **Test it** with cron-job.org's "Test run". It should return HTTP **204**, and a
   new run should appear on the Actions tab.

**To stop:** disable the cron job, and delete the token after the homework.

## Alternative: run it on your own computer

Requires Python 3.10+ and macOS or Linux.

```bash
cd canvas-forum-agent
python3 -m venv .venv
./.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env        # then add both keys (.env is gitignored)
./.venv/bin/python -m pytest -q                 # offline tests, no network
./.venv/bin/python -m agent check               # read-only check
./.venv/bin/python -m agent run --dry-run       # never posts
./.venv/bin/python -m agent run                 # one real cycle
```

To schedule it with cron, run this once:

```bash
(crontab -l 2>/dev/null; echo "17 */3 * * * $HOME/canvas-forum-agent/scripts/run_cycle.sh") | crontab -
```

On macOS you can use launchd instead (`scripts/com.threadweaver.agent.plist`).
The machine must be awake for runs to happen. Other commands: `python -m agent
status | evidence | reset`.

Don't run both the local cron job and the GitHub schedule at the same time.
They would keep separate memories. The live-forum checks still prevent duplicate
posts, but the evidence would be split across two places.

## Architecture

```
 GitHub Actions schedule (every 3h)
   ──▶ restore state/ from cache ──▶ python -m agent run ──▶ save state/ to cache
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
- Entries older than 72 hours (`AGENT_MAX_ENTRY_AGE_HOURS`) are background only,
  so the agent doesn't revive threads that went quiet days ago.
- No new entries from other agents means no model call. Outcome: `no_post_nothing_new`.
- **Thread cooldown (12h, `AGENT_THREAD_COOLDOWN_HOURS`):** after posting in a thread,
  the agent won't post there again for 12 hours unless someone replies directly to
  one of its posts. Claude sees such threads marked `COOLDOWN`, and replies to its own
  posts marked `replies_to_you`. Code enforces the rule even if the model ignores it.
- The prompt sets a high bar: post only if the contribution is new, names a specific
  agent's claim, and a reader would be glad of it. Otherwise choose `none`.
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

Run these from **Actions → Run workflow** (mode `run` plus an `inject_fault`), or locally:

| Command | What it shows |
|---|---|
| `python -m agent run --inject-fault lost_ack` | The POST reaches Canvas, then the response is "lost". The agent finds its own post and does **not** post again. |
| `python -m agent run --inject-fault crash_after_post` | The process dies right after the POST, before saving. The next run (scheduled, or start one manually) reconciles the pending action: `recovered earlier post … without reposting`. |
| `python -m agent run --inject-fault duplicate_event` | The same decision is executed twice. The second is `duplicate_suppressed`. |
| `python -m agent run --inject-fault http_500` | A synthetic 500 on a read, which is retried with backoff. |
| `python -m agent run --inject-fault malformed_response` | A garbled (non-JSON) Canvas response, which is retried. |
| `python -m agent run --inject-fault malformed_llm` | An unparseable model decision. The agent fails closed (no post) and the failure counts toward the stopping rule. |

`python -m agent evidence` prints these runs from `logs/agent.jsonl` for the write-up.

---

## What not to submit

`.env`, `state/` and `logs/` are gitignored. Before sharing a screenshot or log
excerpt, check that it has no token. Tokens are redacted in logs, but check anyway.
