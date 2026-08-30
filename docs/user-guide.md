# User guide

How to install, configure, and run the memory harness. `README.md` is the design behind
it and `architecture.md` the object model; `../CLAUDE.md` holds the rules. This file is
for using the thing.

## What you are running

A continuity layer around an LLM. Every conversation turn — and every wake the system
starts on its own — runs through the same cycle: load `current_state`, retrieve relevant
long-term memories, let the model respond and decide what (if anything) is worth
remembering, commit everything to an append-only history, and update `current_state`.
Restart the process and the model still knows what you were working on.

## Install

Python 3.14 and the repo's `.venv/`:

```sh
.venv/bin/pip install -e . --group dev
```

That installs the `harness` CLI plus the dev tools (`pytest`, `ruff`, `hypothesis`,
`pre-commit`).

## Configure

Settings come from `HARNESS_*` environment variables or a gitignored `.env` file in the
repo root. Provider credentials are read by litellm from the usual variables.

```sh
# .env
ANTHROPIC_API_KEY=sk-ant-...          # for the completion model
OPENAI_API_KEY=sk-...                 # for the default embedding model
```

The knobs you are most likely to touch (full list with defaults:
`src/runtime/config.py`):

| Variable | Default | What it does |
| --- | --- | --- |
| `HARNESS_COMPLETION_MODEL` | `anthropic/claude-opus-5` | Pinned `provider/model` string for completions |
| `HARNESS_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model — changing it requires a rebuild (below) |
| `HARNESS_EMBEDDING_DIM` | `1536` | Must match the embedding model |
| `HARNESS_DB_PATH` | `data/harness.db` | The single SQLite file holding history, memories, and state |
| `HARNESS_K` | `5` | Memories retrieved per inference |
| `HARNESS_SIMILARITY_FLOOR` | `0.30` | Below this, nothing is retrieved — and that is a normal result |
| `HARNESS_COOLDOWN_MINUTES` | `30` | Minimum gap between autonomous wakes |
| `HARNESS_AUTONOMOUS_INTERVAL_MINUTES` | `1440` | Scheduled autonomous wake cadence |
| `HARNESS_REFLECTION_TURN_THRESHOLD` | `20` | Interaction turns that accumulate before a reflection wake is due |

Model ids are pinned exact strings on purpose; use overrides, not aliases like
`latest`.

## Talk to it

```sh
harness chat "I decided to store the gallery metadata in SQLite, not Postgres."
harness chat "What did I decide about the gallery database?"
```

The reply prints to stdout; a `[wake_… decision=… stop=…]` summary and structured logs
go to stderr. Each `chat` is one interaction wake: the message is committed to history,
memories are retrieved and injected, and the model may write a memory or two. Most
turns produce none — the system is built to curate, so don't expect a memory per
message.

Inspect what it knows:

```sh
harness state        # the current_state object injected into every inference
harness log          # recent wakes: trigger, decision, stop reason, tokens
harness log -n 50
```

## Let it wake on its own

The runtime is deliberately one-shot: nothing stays resident, and the clock belongs to
cron (or a systemd timer), not the model. `harness wake` checks whether anything is due
— a scheduled autonomous wake, an `open_question` in `current_state` whose `due` date
has arrived, or enough accumulated turns for a reflection — runs at most one wake
chain, and exits. If nothing is due it prints `nothing due` and exits; calling it often
is safe because cooldown and cadence are enforced from history, and they survive
restarts.

```cron
# every 15 minutes; the harness itself decides whether anything is actually due
*/15 * * * * cd /path/to/llm-memory-harness && .venv/bin/harness wake >> data/wake.log 2>&1
```

To force one by hand (skips the due-ness check, not the budgets):

```sh
harness wake --trigger reflection
harness wake --trigger autonomous
```

A run of wakes that all decided `sleep` and wrote nothing is the system working
correctly — check `harness log` and you'll see the records. Every wake also runs under
fixed budgets (iterations, tokens, tool calls, wall clock), so an unattended schedule
cannot turn into an unbounded bill.

## Back up your data

Everything lives in `data/harness.db` (gitignored). The source history inside it is
raw conversation — PII-bearing — and it is the one thing that cannot be regenerated,
so back the file up from day one. Long-term memory and `current_state` are derived
from history and are reproducible.

## Rebuild

```sh
harness rebuild --from-history
```

Clears the derived layers (memories and `current_state`) and regenerates them by
replaying history. It makes no completion calls — the only cost is re-embedding. Reach
for it when:

- you change `HARNESS_EMBEDDING_MODEL` / `HARNESS_EMBEDDING_DIM`,
- the preprocessing rules change (`PREPROCESS_VERSION` bumps),
- search logs warn about `stale_vector`,
- a wake crashed or ran out of budget mid-consolidation and you want the derived
  layers caught up.

Old vectors are never silently mixed with new ones; a mismatch shows up as those
`stale_vector` warnings, and rebuild is the fix.

## Evaluate retrieval

```sh
harness eval                    # offline: deterministic hash embedder, no network
harness eval --real-embedder    # the configured embedding model
harness eval --set holdout      # just the milestone-one questions
```

Prints recall@5, MRR, the reflection-written ratio, and any missed queries. It seeds an
in-memory store from `evals/corpus.jsonl` — it never touches `data/`. If you tune
anything retrieval-adjacent (preprocessing, ranking, thresholds, the embedding model),
this number is the justification, not vibes; see `evals/README.md`.

## Develop

```sh
.venv/bin/pytest                     # unit suite: fast, deterministic, no network
.venv/bin/pytest -m integration      # opt-in: real provider calls, needs API keys
.venv/bin/ruff check src tests
.venv/bin/ruff format src tests
.venv/bin/pre-commit install         # wire ruff + pytest into git commits
```

The milestone tests are the bar: `tests/test_milestone_one.py` (cold-start recovery
through the loop, ≥ 8/10) and `tests/test_milestone_two.py` (idle wakes change
nothing; a due question gets picked up).

## Troubleshooting

- **`nothing due` from `harness wake`** — cooldown or cadence hasn't elapsed and no
  open question is due. `harness log` shows the last wake times; force with
  `--trigger` if you're testing.
- **A question returns no memories** — retrieval has a similarity floor and returning
  nothing is a designed outcome, not an error. If a memory you expected is missing,
  check it exists (`harness eval` patterns, or the search logs' ids and scores on
  stderr).
- **`stale_vector` warnings** — stored vectors were made by a different embedding
  model or preprocessing version. Run `harness rebuild --from-history`.
- **`schema v… != code v…` on startup** — the data file predates a schema change and
  no migration exists yet; keep the old checkout for that file, or start a fresh
  `data/` (after backing up the old one).
- **Budget stops (`stop=budget:…` in `harness log`)** — the wake hit its allowance and
  stopped at a step boundary. Nothing is lost: history is committed first, and a
  rebuild re-derives whatever consolidation didn't reach.
