# Engineering practices

How to build this repo without the failure modes common to retrieval systems. `README.md` is the design;
`../CLAUDE.md` holds the rules, stated once — this file gives rationale and tooling, and does not restate
them. Nothing here is settled by code yet; amend it as reality disagrees.

## Toolchain

`pyproject.toml` as the single source of config. **uv** for resolution and locking, **ruff** for lint and
format (replacing black/isort/flake8), **pyright** or mypy `strict` over `src/`, **pytest** for tests,
wired into `pre-commit` so local and CI agree. None of these are installed yet.

Python 3.14 (the existing `.venv/`) is the target. Re-check wheel availability before adding any compiled
dependency rather than assuming it.

## Module boundaries

The embedding backend and the vector store are the two dependencies guaranteed to change — and the
embedder is a different vendor from the LLM, since Anthropic exposes no first-party embeddings endpoint.
Define both as `Protocol`s, `Embedder` and `MemoryStore`, and keep provider SDK types out of the core
memory module.

Start with the boring implementation: SQLite with `sqlite-vec`, or brute-force cosine over a numpy array.
At MVP scale, exact search over a few thousand vectors is fast and has no index-tuning failure modes.
Swap in something heavier when measured volume justifies it, not before.

Avoid LangChain / LlamaIndex here. Retrieval orchestration *is* the product; a framework hides the exact
layer this project exists to design.

## Rebuilds

`rebuild --from-history` regenerates long-term memory and `current_state` from immutable history. Build it
early and keep it working: once it exists, changing the embedding model or the preprocessing rules stops
being a migration problem and becomes a re-run. Two consequences beyond the idempotency rule:

- The store needs a schema version and migrations, SQLite included.
- Rebuild cost scales with history, and the embedding backend sets the price of it. Batch the calls.

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

## Evaluation

Build the eval set before tuning retrieval. Hand-label 30–50 `(query, expected_memory_ids)` pairs over a
fixed corpus, commit them under `evals/`, and measure **recall@k** and MRR. Without this, every change to
preprocessing, curation, or the embedding model is judged on impression.

The first milestone is already an eval — make it an automated test over the frozen corpus with an explicit
pass threshold.

If LLM-as-judge is used for memory quality, pin the judge model id and treat changing it as a change to
the metric itself.

## Configuration

Typed `Settings` via `pydantic-settings`, sourced from environment variables; `.env` stays gitignored. Do
not construct API clients at module import — it makes the code untestable.

Pin exact model ids (e.g. `claude-opus-5`, never a date-suffixed variant or a floating "latest" alias), so
a provider-side default change cannot silently alter retrieval behavior or output quality.

Have the curation step return structured output (`output_config.format`, or a `strict: true` tool) rather
than parsing JSON out of prose. Candidate memories are a schema; enforce it at the API boundary.

## Observability

Structured logging (structlog) with a single run id threaded through both write and search paths.

For every inference log the query, **the retrieved memory ids and their scores**, token counts from
`response.usage`, latency, and cost. Retrieval quality is undiagnosable after the fact without the
retrieved set recorded.

Prompt caching is the awkward case here: the prefix is stable, but memories and `current_state` change
every turn. Cache lookup is prefix-matched over `tools` → `system` → `messages`, so putting the retrieved
set in the top-level `system` block invalidates the cache on every request. Append it to `messages` as a
`{"role": "system"}` entry instead — supported on Opus 5, Opus 4.8 and Fable 5, not Sonnet 5 — after the
user turn it was retrieved for, which is where the ordering rules put it anyway (never `messages[0]`; last
or followed by an assistant turn). Watch `usage.cache_read_input_tokens`: zero across repeated requests
means something upstream is still changing.

## Reliability

Batch embedding calls rather than embedding one record per request. Retries with jitter, explicit
timeouts, and a most-specific-first exception chain (`RateLimitError` before a broad `APIStatusError`) so
retryable and non-retryable failures stay distinguishable.

## Data handling

`data/` is gitignored. Raw conversation history is PII-bearing and is the one thing in the system that
cannot be regenerated — back it up from day one and keep it append-only. Everything in the memory layer is
reproducible from it.

## Open decisions

Blocking, in build order:

1. **Embedding backend** — hosted API or local `sentence-transformers`. It fixes `embedding_dim`, the cost
   of every rebuild, and whether the integration suite needs network at all; local pulls in torch (2.13.0
   publishes cp314 wheels, checked 2026-08-30). Needed at build step 2.
2. **Curation model and effort level** — curation runs after every turn and is the cost driver, so it need
   not be the same model as the main loop. Needed at build step 6.
3. **Runtime shape** — CLI chat loop, or a library others embed. Only blocks build step 8.
