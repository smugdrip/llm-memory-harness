# AI-Native Memory Harness

A minimal continuity layer for LLMs.

The goal is to give an LLM durable memory across sessions without hard-coding a personality or simulating
a human body: short-term thread context, long-term semantic memory, persistent current state, and
immutable source history.

The rules are in `../CLAUDE.md`; this file is the design behind them.

## MVP

```text
LLM
├── Short-term memory
│   └── current thread / context window
│
├── Long-term memory
│   ├── memory.write()
│   └── memory.search()
│
├── Current state
│   └── active projects, priorities, unresolved questions
│
└── Source history
    └── immutable raw conversations/messages
```

## Core idea

```text
new input
→ preprocess search query
→ embed query
→ retrieve relevant memories
→ inject memories + current state
→ LLM responds / acts
→ select what is worth remembering
→ preprocess memory
→ embed + store
→ update current state
```

## Memory record

Store curated memories, not raw messages. `canonical_text` is what gets embedded; everything else is
metadata and provenance.

```json
{
  "id": "mem_...",
  "canonical_text": "Decision: store long-term memory outside the thread.",
  "raw_text": "...",
  "source_id": "thread/123/message/456",
  "type": "decision",
  "entities": ["long-term memory"],
  "importance": 0.8,
  "occurred_at": "2026-08-30T17:04:00Z",
  "created_at": "2026-08-30T17:05:12Z",
  "supersedes": null,
  "superseded_by": null,
  "embedding_model_id": "...",
  "embedding_dim": 1536,
  "preprocess_version": 1
}
```

- `type` is one of `event`, `decision`, `project`, `relationship`, `preference`, `open_question`.
- `occurred_at` is when the thing happened; `created_at` is when the memory was written. They diverge
  whenever a conversation recalls the past, and recency ranking wants `occurred_at`.
- `importance` is assigned by curation, in `[0, 1]`. A rerank signal, never a retrieval filter on its own
  — a low-importance memory is still the right answer to a question about it.
- `superseded_by` is the delete: set it, drop the row from search, keep it for history and rebuilds.
- The three embedding fields are what make a model or preprocessing change detectable rather than silent.

`memory.write()` canonicalizes a candidate, attaches this metadata, embeds, and stores — idempotent on
`source_id`.

## Preprocessing

One path, used for both memory writes and search queries — a query and the memory it should match have to
land in the same space.

```text
raw text
→ normalize whitespace
→ remove boilerplate
→ resolve obvious references when possible
→ compress to semantic core
→ preserve names, dates, and project terms
→ embed
```

Avoid aggressive stemming, stop-word removal, or keyword-only normalization. Bump `preprocess_version`
when the rules change; it is what tells a later reader that old vectors are stale.

## Curation

The step that decides what is worth remembering. It runs after a turn, over that turn only, and usually
returns nothing.

- Input: the turn's messages plus the memories that were retrieved for it.
- Output: 0..N candidate records, N bounded. A turn producing five memories is usually a prompt bug.
- Before writing, search for near-duplicates. Above the duplicate threshold, skip the candidate or write
  it and set `supersedes` on the row it replaces. Skipping this is the most likely way this design fails:
  conversation restates the same facts constantly, and an uncurated store fills with paraphrases that
  crowd each other out of the top `k`.

## Retrieval

`memory.search()` canonicalizes the query through the same path, embeds it, and returns a small ranked
set with provenance attached, so any claim can be traced back to the conversation that produced it.

- `k = 5` by default, applied after a similarity floor. Fewer results — including none — is correct.
- Exclude superseded rows; filter on `type` and rerank on recency and `importance` where it helps.

## `current_state`

A small object injected on every inference without a search. Derived like long-term memory, and
regenerable by `rebuild --from-history`.

```json
{
  "active_projects": [],
  "priorities": [],
  "open_questions": [],
  "recent_focus": []
}
```

- Bounded: a handful of one-line entries per list. It is in every prompt, so growth costs tokens on every
  request and dilutes the things that matter.
- Each entry cites the memory ids behind it — provenance applies here too.
- Updated by an explicit step after the turn, taking the prior state as input. It holds what is *current*:
  an entry that stops being true is dropped, not appended to. That is what keeps it from becoming a
  second, worse episodic store.

## Build order

1. **Source history** — append-only. The one store that cannot be regenerated.
2. **Preprocessing**, plus `Embedder` / `MemoryStore` protocols and a fake embedder.
3. **`memory.write()` / `memory.search()`** over the boring store.
4. **`rebuild --from-history`** — early, while the store is small enough that it is easy.
5. **Eval set + recall@k** — before tuning retrieval, not after.
6. **Curation**, including the duplicate check.
7. **`current_state`** and its update step.
8. **Runtime loop** tying them together.

## Repository structure

```text
.
├── CLAUDE.md
├── docs/
├── src/
│   ├── memory/          # store, search, preprocess, curate
│   ├── state/           # current_state and its updates
│   ├── history/         # append-only source history
│   ├── llm/             # provider adapters
│   └── runtime/         # inference loop, CLI
├── evals/               # frozen corpus + (query, expected_memory_ids)
├── tests/
├── data/                # gitignored: history, vectors, state
└── pyproject.toml
```

## First milestone

Restart with no chat history loaded and recover the important context of prior work using only
`current_state` + `memory.search()`. If that works reliably, the continuity layer is viable.

Make it a test, not a demo: a frozen corpus, a held-out question set, and a stated threshold (say, 8 of
10 facts recovered). Without a threshold it is an anecdote.

## Non-goals

Not part of the MVP: continuous 24/7 inference, a robot body, simulated biological needs, hard-coded
personality, consciousness detection, autonomous unrestricted tool use, thread branching or fusion.

## Guiding principle

**Build continuity, not a character.**

The system should increasingly behave in ways that make sense because of its own accumulated history, not
only because of the current prompt.
