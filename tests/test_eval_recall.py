"""The committed eval set: recall@k and MRR against Memory.search() directly, plus the
drift check — recall over the frozen corpus does not degrade after reflection cycles
(invariant 18), with the reflection-written ratio tracked beside it."""

from __future__ import annotations

from pathlib import Path

from fakes import ScriptedLLM, completion, finish, state_json, tool_call
from history.records import Trigger, TriggerKind
from memory.memory import Memory
from runtime.evals import load_corpus, load_queries, reflection_ratio, run_eval, seed_corpus
from runtime.orchestrator import Orchestrator

EVALS = Path(__file__).parent.parent / "evals"


def seeded_memory(store, embedder, clock):
    memory = Memory(store, embedder, clock=clock, similarity_floor=0.05, duplicate_threshold=0.9)
    key_to_id = seed_corpus(memory, load_corpus(EVALS / "corpus.jsonl"))
    return memory, key_to_id


def test_dev_recall_at_5_and_mrr(store, embedder, clock):
    memory, key_to_id = seeded_memory(store, embedder, clock)
    dev = [q for q in load_queries(EVALS / "queries.jsonl") if q.set == "dev"]
    assert len(dev) == 20
    report = run_eval(memory, dev, key_to_id, k=5)
    misses = [r.query for r in report.results if r.first_hit_rank is None]
    assert report.recall_at_k >= 0.8, f"recall@5={report.recall_at_k:.2f}, misses: {misses}"
    assert report.mrr >= 0.5


def test_returning_nothing_stays_correct_on_a_seeded_store(store, embedder, clock):
    seeded_memory(store, embedder, clock)
    # A production-like floor over the same seeded store: an off-topic query returns
    # nothing, and the result is never padded toward k. (The eval fixture's 0.05 floor
    # is tuned for the hash embedder's recall measurement, where collision noise is
    # part of the ranking; correctness of "nothing" belongs to the floor.)
    memory = Memory(store, embedder, clock=clock, similarity_floor=0.3)
    assert memory.search("zzz qqq xyzzy plugh") == []


def test_drift_check_reflection_does_not_degrade_recall(store, embedder, clock, history, state_store):
    memory, key_to_id = seeded_memory(store, embedder, clock)
    queries = load_queries(EVALS / "queries.jsonl")
    baseline = run_eval(memory, queries, key_to_id, k=5)

    # N reflection cycles that actually consolidate: each supersedes one corpus row
    # with a merged restatement that preserves its content.
    for key in ("dec-thumbnails", "dec-sqlite", "proj-map"):
        target = memory.store.get(key_to_id[key])
        merged = target.canonical_text + " (reconfirmed while consolidating overlapping notes.)"
        llm = ScriptedLLM(
            completion(tool_call("memory_supersede", {"memory_id": target.id, "text": merged})),
            finish("reflect"),
            state_json(state_store.load()),
        )
        Orchestrator(llm=llm, memory=memory, history=history, state=state_store, clock=clock).wake(
            Trigger(TriggerKind.REFLECTION, "cadence")
        )
        clock.advance(hours=1)

    after = run_eval(memory, queries, key_to_id, k=5)
    assert after.recall_at_k >= baseline.recall_at_k, (
        f"drift: recall fell from {baseline.recall_at_k:.2f} to {after.recall_at_k:.2f}"
        " after reflection cycles"
    )
    # the ratio that shows a store trending toward its own commentary
    ratio = reflection_ratio(memory.store)
    assert 0.0 < ratio <= 0.15
    live = [r for r in memory.store.records() if r.superseded_by is None]
    assert len(live) == 30  # consolidation replaced rows; it did not add opinions
