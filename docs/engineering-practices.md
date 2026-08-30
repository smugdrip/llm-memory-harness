# Engineering practices

How to build this repo without the failure modes common to retrieval systems, or to processes that run
themselves. `README.md` is the design; `../CLAUDE.md` holds the rules, stated once — this file gives
rationale and tooling, and does not restate them. Nothing here is settled by code yet; amend it as reality
disagrees.

## Toolchain

`pyproject.toml` as the single source of config. **uv** for resolution and locking, **ruff** for lint and
format (replacing black/isort/flake8), **pyright** or mypy `strict` over `src/`, **pytest** for tests,
wired into `pre-commit` so local and CI agree. None of these are installed yet.

Python 3.14 (the existing `.venv/`) is the target. Re-check wheel availability before adding any compiled
dependency rather than assuming it.

## Implementation choices

`architecture.md` has the object model — which components exist and what each promises. What belongs here
is which implementation to reach for first.

Start boring: SQLite with `sqlite-vec` behind `MemoryStore`, or brute-force cosine over a numpy array. At
MVP scale, exact search over a few thousand vectors is fast and has no index-tuning failure modes. Swap in
something heavier when measured volume justifies it, not before.

Avoid LangChain / LlamaIndex. Retrieval orchestration *is* the product, and a framework hides the exact
layer this project exists to design. litellm is not that kind of dependency — it normalizes one HTTP call
and orchestrates nothing.

Budget accounting is one object threaded through the cycle, decremented at every model and tool call and
checked at step boundaries rather than mid-write. Because history is committed before the derived writes,
"ran out of budget" and "crashed" have the same recovery: the record is there and rebuild re-derives the
rest.

Cooldown is persisted state, not a `sleep()`. Store the last wake time so a restarted process does not
immediately fire the autonomous trigger it was in the middle of cooling down from.

## Rebuilds

`rebuild --from-history` regenerates long-term memory and `current_state` from immutable history. Build it
early and keep it working: once it exists, changing the embedding model or the preprocessing rules stops
being a migration problem and becomes a re-run. Two consequences beyond the idempotency rule:

- The store needs a schema version and migrations, SQLite included.
- Rebuild cost scales with history, and the embedding backend sets the price of it. Batch the calls.
- Supersession has to survive a rebuild. Replaying history in order should re-derive the same chains, but
  that only holds if curation makes the same supersede call the second time. Worth an explicit round-trip
  test: a rebuild that quietly resurrects retired memories looks exactly like a rebuild that worked.

## Testing

- Inject a fake embedder that derives a deterministic vector from a hash. Curation, metadata, provenance,
  ranking, and `current_state` updates are all testable that way, with no network.
- Register the `integration` marker in `pyproject.toml` and exclude it from the default run via
  `addopts`, so the opt-in suite stays opt-in.
- Record/replay (vcrpy, or hand-rolled JSON fixtures) for the LLM calls in the curation step.
- Golden files for `canonical_text`. When a curation prompt changes you want to read the diff in
  canonicalization output, not guess at the effect.
- Property tests (hypothesis) for preprocessing against the invariant the design already states: names,
  dates, and project terms survive. That invariant is only real if something checks it.
- Drive the loop with an injected clock and a scripted trigger source. Two properties are worth asserting
  outright: an idle wake writes zero memory rows and leaves `current_state` byte-identical, and a wake that
  exhausts its budget stops at a step boundary with its history record intact.
- Fake the decide step the way the embedder is faked. Budgets, cooldown, transitions and the no-op path are
  all decision-independent, and testing them with a real model in the way makes them slow and flaky for no
  added coverage.

## Evaluation

Build the eval set before tuning retrieval. Hand-label 30–50 `(query, expected_memory_ids)` pairs over a
fixed corpus, commit them under `evals/`, and measure **recall@k** and MRR. Without this, every change to
preprocessing, curation, or the embedding model is judged on impression.

The first milestone is already an eval — make it an automated test over the frozen corpus with an explicit
pass threshold.

If LLM-as-judge is used for memory quality, pin the judge model id and treat changing it as a change to
the metric itself.

The drift check is the second eval and it needs the first one to exist: run N reflection cycles over the
frozen corpus, then re-measure recall@k. Reflection that consolidates leaves it flat or improves it;
reflection that editorializes pushes real answers out of the top `k`, and the number moves well before
anyone notices by reading. Track the ratio of reflection-written to interaction-written memories next to
it — a store trending toward its own commentary shows up there earlier than in recall.

## Configuration

Typed `Settings` via `pydantic-settings`, sourced from environment variables; `.env` stays gitignored. Do
not construct API clients at module import — it makes the code untestable.

Pin exact model ids (e.g. `claude-opus-5`, never a date-suffixed variant or a floating "latest" alias), so
a provider-side default change cannot silently alter retrieval behavior or output quality.

Have the curation step return structured output (`output_config.format`, or a `strict: true` tool) rather
than parsing JSON out of prose. Candidate memories are a schema; enforce it at the API boundary.

## Observability

Structured logging (structlog) with a single run id threaded through both write and search paths.

For every inference log the exact input the model received — system instructions, thread context, the
retrieved memories, `current_state`, and the trigger — along with **the retrieved memory ids and their
scores**, token counts from `response.usage`, latency, and cost. The ids and scores are enough to debug
retrieval; they are not enough to debug behavior, and when the answers change the first question is always
what the model actually saw.

Prompt caching is the awkward case here: the prefix is stable, but memories and `current_state` change
every turn. Cache lookup is prefix-matched over `tools` → `system` → `messages`, so putting the retrieved
set in the top-level `system` block invalidates the cache on every request. Append it to `messages` as a
`{"role": "system"}` entry instead — supported on Opus 5, Opus 4.8 and Fable 5, not Sonnet 5 — after the
user turn it was retrieved for, which is where the ordering rules put it anyway (never `messages[0]`; last
or followed by an assistant turn). Watch `usage.cache_read_input_tokens`: zero across repeated requests
means something upstream is still changing.

Once wakes are recorded, the history record and the log overlap heavily — both want the retrieved ids, the
token counts, the latency. Keep both, for different reasons: the history record is replayable ground truth
and has to stay stable, while the log is operational and free to change shape. Where they overlap, write
the record first and derive the log line from it, so the two cannot disagree about what happened.

For autonomous wakes also log the decision and whether the cycle was a no-op. A stretch of wakes that all
decided `sleep` is the system behaving correctly and should cost almost nothing to confirm; a stretch that
all decided `reflect` is a runaway, and that shows up on the bill before it shows up in the answers.

## Reliability

Batch embedding calls rather than embedding one record per request. Retries with jitter, explicit
timeouts, and a most-specific-first exception chain (`RateLimitError` before a broad `APIStatusError`) so
retryable and non-retryable failures stay distinguishable.

An autonomous wake has nobody watching it fail. Alert on the loop's own health — wakes that error, wakes
that hit the wall clock, cooldown violations — because the visible symptom of a broken reflection trigger
is nothing happening, which is indistinguishable from the system correctly deciding there was nothing to
do.

## Data handling

`data/` is gitignored. Source history is PII-bearing — raw conversation, plus the model's own output and
the state snapshots recorded beside it — and it is the one thing in the system that cannot be regenerated.
Back it up from day one and keep it append-only. Everything in the memory layer is
reproducible from it.

## Open decisions

Blocking, in build order:

1. **Embedding backend** — hosted API or local `sentence-transformers`. It fixes `embedding_dim`, the cost
   of every rebuild, and whether the integration suite needs network at all; local pulls in torch (2.13.0
   publishes cp314 wheels, checked 2026-08-30). Needed at build step 2.
2. **Curation model and effort level** — curation runs at the end of every non-idle cycle and is the cost
   driver, so it need not be the same model as the main loop. Needed at build step 6.
3. **Runtime shape** — the orchestrator settles the entry point; what remains is whether it ships as a CLI
   that owns the process or a library others embed, which decides who owns the trigger loop. Needed at
   build step 8.
4. **What drives the clock** — an in-process scheduler holding a daemon open, or a one-shot `wake` command
   invoked by cron or a systemd timer. The one-shot version is easier to reason about and makes invariant
   14 structural rather than a matter of discipline; a daemon costs less per wake because the process and
   its caches stay warm. Needed at build step 9.
5. **Reflection cadence** — elapsed time, turns accumulated, or a threshold on unconsolidated memories. It
   sets the cost floor of running the system with nobody talking to it. Needed at build step 9.
6. **Whether the caching strategy survives litellm** — the volatile block belongs in `messages[]`, and the
   Anthropic way to put it there is a `{"role": "system"}` entry, which litellm may hoist into the
   top-level `system` parameter and thereby invalidate the prefix on every request. Test it against a real
   provider and watch the cache-hit counter; the fallback is a user-role block, which is provider-neutral
   and gives up the operator channel. Needed at build step 8.
