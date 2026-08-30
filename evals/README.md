# Eval set

A frozen corpus of 30 hand-written memories about a fictional project (Dana's "Meridian"
photo archive) and 30 hand-labeled `(query, expected_keys)` pairs.

- `corpus.jsonl` — one candidate memory per line: `key`, `text`, `type`, `occurred_at`,
  `importance`, `entities`. Seeded through `memory.write()` (the same path everything
  else uses) with `source_id = eval/corpus/<key>`; expected ids are resolved by key so
  the set survives id-scheme and preprocessing changes.
- `queries.jsonl` — `query`, `expected_keys`, `set`. `set: "dev"` (20) backs the
  recall@k test that justifies retrieval changes; `set: "holdout"` (10) backs milestone
  one, driven through the orchestrator's interaction trigger with a threshold of 8/10.

Scoring (see `src/runtime/evals.py`) follows supersession chains: a memory that
supersedes the expected row counts as a hit. That is what lets the drift check tell
reflection that consolidates (recall flat) from reflection that editorializes (recall
falls).

Run offline (deterministic hash embedder) or against the real embedding model:

```sh
harness eval                  # HashEmbedder, no network
harness eval --real-embedder  # the configured embedding model
```

The queries were written for exact-token overlap so the offline run is meaningful with
the bag-of-words fake; the real-embedder run is the number that matters for tuning.
