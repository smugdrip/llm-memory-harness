# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Doc contract

This file is normative: rules and invariants live here, stated once, in checkable form. `docs/` explains
*why* and does not restate a rule. Keep it that way as the docs grow — a rule changes in one place.

- `docs/README.md` — the design: stores, inference loop, record shapes, build order.
- `docs/engineering-practices.md` — toolchain, testing, eval, observability, and the open decisions.

## Status: design-only

The repo holds `docs/`, a Python `.gitignore`, `LICENSE`, and an empty Python 3.14 `.venv/` (pip only).
There is no dependency manifest, build system, or test suite, so **there are no build/lint/test commands
to run yet.** Add them here as they land.

Python 3.14 is settled — the `.venv/` and `.gitignore` are the only stack signals and they agree. The
rest of the toolchain is the intended default, not yet installed.

## What is being built

A continuity layer that gives an LLM durable memory across sessions. Four stores, deliberately not
merged: **short-term** (the live thread), **long-term** (curated embedded memories behind
`memory.write()` / `memory.search()`), **`current_state`** (a small object injected on every inference
without a search), and **source history** (immutable raw conversations).

## Invariants

Design decisions that non-obviously constrain implementation, and the ones easiest to violate silently.

1. Source history is immutable ground truth. Long-term memory and `current_state` are **derived** layers
   and must be fully regenerable by `rebuild --from-history`.
2. `memory.write()` is idempotent on `source_id`, or a rebuild duplicates every memory.
3. Appending to history is the commit point and happens first; curation and embedding are downstream and
   retryable, because rebuild exists.
4. Memories are superseded — `superseded_by` set, row excluded from search — never deleted.
5. Every memory carries provenance (`raw_text` + `source_id`) alongside the `canonical_text` that is
   actually embedded.
6. Curate. Do not store every message; most turns produce none. Check for near-duplicates before writing.
7. One preprocessing path serves both writes and queries. It compresses to a semantic core while
   preserving names, dates, and project terms — no stemming, stop-word removal, or keyword-only
   normalization.
8. Retrieve few memories per inference, behind a similarity floor. Returning nothing is a correct result;
   never pad to `k`.
9. Keep the four stores separate. `current_state` is not episodic memory, and it is size-bounded.
10. Every stored vector records `embedding_model_id`, `embedding_dim`, and `preprocess_version`. Vectors
    from different models or preprocessing versions are not comparable — mixing them returns bad
    neighbors rather than raising, so these fields are what make a mismatch detectable.
11. Retrieval changes are justified by the committed eval set (recall@k), not by inspection.
12. Unit tests make no network calls — inject a fake embedder. Real-provider tests are a separate opt-in
    suite (`@pytest.mark.integration`).

Memory types for the MVP: `event`, `decision`, `project`, `relationship`, `preference`, `open_question`.

## Scope

Single identity, single thread. Out of scope per the spec: thread branching or fusion, multi-agent
identity, continuous inference, simulated needs, hard-coded personality, autonomous unrestricted tool use.

Milestone one, and the bar for the MVP: restart cold with no chat history loaded and recover prior
working context using only `current_state` + `memory.search()` — as an automated test with a stated
threshold, not a demo.
