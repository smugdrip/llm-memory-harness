# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Doc contract

This file is normative: rules and invariants live here, stated once, in checkable form. `docs/` explains
*why* and does not restate a rule. Keep it that way as the docs grow — a rule changes in one place.

- `docs/README.md` — the design: stores, the loop, record shapes, build order.
- `docs/architecture.md` — the object model: what each component is and what it promises.
- `docs/engineering-practices.md` — toolchain, testing, eval, observability, and the open decisions.
- `docs/user-guide.md` — how to install, configure, run, and troubleshoot it.

## Status: MVP implemented

`src/` implements the design (the layout in `docs/architecture.md`), `tests/` covers it including both
milestone tests, and `evals/` holds the committed eval set. Python 3.14; dependencies in
`pyproject.toml`, dev tools in its `dev` dependency group.

- Install: `.venv/bin/pip install -e . --group dev`
- Test: `.venv/bin/pytest` — unit suite, no network. Real-provider suite: `.venv/bin/pytest -m
  integration` (needs `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; skipped without one).
- Lint / format: `.venv/bin/ruff check src tests` · `.venv/bin/ruff format src tests`
- Eval: `.venv/bin/harness eval` (offline, hash embedder) · add `--real-embedder` for the configured
  embedding model.
- CLI: `harness chat | wake | rebuild --from-history | eval | state | log`. Configuration comes from
  `HARNESS_*` env vars (`src/runtime/config.py`); `data/` holds the SQLite file.

## What is being built

A continuity layer that gives an LLM durable memory across sessions, driven by a loop that can wake
without being spoken to. Four stores, deliberately not merged: **short-term** (the live thread),
**long-term** (curated embedded memories behind `memory.write()` / `memory.search()`), **`current_state`**
(a small object injected on every inference without a search), and **source history** (an immutable log of
every cognitive event — turns and wakes alike). Over them runs one **loop orchestrator**: a state machine,
not a store, entered by three triggers.

## Invariants

Design decisions that non-obviously constrain implementation, and the ones easiest to violate silently.

1. Source history is immutable ground truth, and it records every cognitive event — user turns and
   autonomous wakes alike, each with its trigger, retrieved memory ids, `current_state` snapshot, and the
   model's output text. Long-term memory and `current_state` are **derived** layers and must be fully
   regenerable by `rebuild --from-history`. A memory whose source is not in history is not regenerable.
2. `memory.write()` is idempotent on `source_id`, or a rebuild duplicates every memory.
3. Appending to history is the commit point and happens first — for a wake as much as for a turn; curation
   and embedding are downstream and retryable, because rebuild exists.
4. Memories are superseded — `superseded_by` set, row excluded from search — never deleted.
5. Every memory carries provenance (`raw_text` + `source_id` + `origin`) alongside the `canonical_text`
   that is actually embedded. `origin` names the trigger of the wake that wrote it, so memories the system
   produced about itself stay distinguishable from ones a conversation produced.
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
13. One state machine, three triggers — interaction, reflection, autonomous. They differ only in what
    wakes the cycle and what the decide step is biased toward; budgets, consolidation, and the state write
    are implemented once in the orchestrator, never per trigger.
14. The scheduler is outside the model. A cycle may *request* a successor; the orchestrator decides whether
    to grant one. That is the whole of the difference between a scheduled wake and continuous inference.
15. Every wake runs under explicit budgets — max iterations, tokens, tool calls, wall clock — plus a
    cooldown before the next autonomous wake. Budgets are per wake, not per iteration, so a granted
    successor spends from the same allowance rather than resetting it.
16. Doing nothing is a correct wake. A cycle that decides there is nothing worth doing writes no memories
    and no `current_state` change; its history record is its only effect. An idle wake that churns derived
    state is a bug, not activity.
17. Reflection consolidates. Its default output is supersession and merging of existing memories, not
    net-new claims — reflection reads what reflection wrote, so unchecked it fills the store with
    commentary about its own commentary.
18. The eval set carries a drift check: recall@k over the frozen corpus does not degrade after N reflection
    cycles. Without it, invariant 17 is a hope.
19. A `preference` memory needs repeated evidence, never a single statement. Promoting one remark to a
    durable trait is how a continuity layer becomes a hard-coded personality by accident — which is on the
    out-of-scope list, and will not announce itself when it happens.
20. Provider names, model ids, and litellm itself appear only in `LLMClient` and `Embedder`
    implementations. No other module imports litellm or names a model, and provider response objects do
    not cross that boundary — otherwise the dependency has quietly become the interface.
21. The model's tools are a binding over the same methods the system calls directly. There is no second
    retrieval path and no second write path; a similarity floor or duplicate check that only one caller
    goes through is not enforced.
22. Provenance is bound by the orchestrator, never supplied by the model. Tool schemas expose semantic
    arguments only — a model that can set `source_id` controls idempotency, rebuild, and every provenance
    claim in the system.
23. A model turn is appended to history before its tool calls execute, so a memory written mid-cycle
    always has a committed source to be replayed from.

Memory types for the MVP: `event`, `decision`, `project`, `relationship`, `preference`, `open_question`.

## Scope

Single identity, single thread. Out of scope per the spec: thread branching or fusion, multi-agent
identity, continuous inference, simulated needs, hard-coded personality, autonomous unrestricted tool use.

Two of those sit close to the loop and need a line drawn. *Continuous inference* means a process that is
always thinking; scheduled discrete wakes that sleep between cycles are in scope, and invariant 14 is what
holds the two apart. *Autonomous unrestricted tool use* means an open tool surface; the act branch reaches
only `memory.write`, `memory.supersede`, and `current_state` updates — no external tools, no workspace, no
environment store.

Milestone one, and the bar for the MVP: restart cold with no chat history loaded and recover prior
working context using only `current_state` + `memory.search()` — as an automated test with a stated
threshold, not a demo. It runs through the orchestrator's interaction trigger rather than calling
`memory.search()` directly, so the loop sits on the critical path.

Milestone two: over a static world, a run of idle wakes produces zero memories and zero `current_state`
changes, and a wake with a due `open_question` picks it up. Also a test.
