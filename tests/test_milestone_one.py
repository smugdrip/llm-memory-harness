"""Milestone one — continuity, as an automated test with a stated threshold.

Restart cold with no chat history loaded and recover prior working context using only
current_state + memory.search(), driven through the orchestrator's interaction trigger
(never memory.search() directly): at least 8 of the 10 held-out questions must surface
the expected memory in what the wake actually retrieved.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fakes import FakeClock, ScriptedLLM, completion, finish, tool_call
from history.history import History
from history.records import Trigger, TriggerKind
from llm.embedder import HashEmbedder
from memory.memory import Memory
from memory.store import SqliteStore
from runtime.evals import load_corpus, load_queries, seed_corpus
from runtime.orchestrator import Orchestrator
from state.state import CurrentState, Entry, StateStore

EVALS = Path(__file__).parent.parent / "evals"
THRESHOLD = 8


def test_cold_start_recovery_through_the_interaction_trigger(tmp_path):
    db = tmp_path / "continuity.db"
    clock = FakeClock()
    embedder = HashEmbedder()

    # --- session one: live work happens, then the process dies -----------------
    conn = sqlite3.connect(db)
    memory = Memory(SqliteStore(conn), embedder, clock=clock, similarity_floor=0.05)
    key_to_id = seed_corpus(memory, load_corpus(EVALS / "corpus.jsonl"))
    StateStore(conn).save(
        CurrentState(
            active_projects=(Entry("Meridian photo archive", (key_to_id["proj-meridian"],)),),
            open_questions=(Entry("B2 or Synology NAS backup", (key_to_id["oq-backup"],), due="2026-09-05"),),
        )
    )
    conn.close()

    # --- cold restart: fresh objects over the persisted stores, no thread ------
    conn = sqlite3.connect(db)
    memory = Memory(SqliteStore(conn), embedder, clock=clock, similarity_floor=0.05)
    history = History(conn, clock)
    state_store = StateStore(conn)

    holdout = [q for q in load_queries(EVALS / "queries.jsonl") if q.set == "holdout"]
    assert len(holdout) == 10

    recovered = 0
    prior_state_json = json.dumps(state_store.load().to_dict())
    for question in holdout:
        llm = ScriptedLLM(
            completion(tool_call("memory_search", {"query": question.query})),
            finish("respond", text="answered from memory"),
            completion(text=prior_state_json),  # update_state echoes the state unchanged
        )
        orchestrator = Orchestrator(llm=llm, memory=memory, history=history, state=state_store, clock=clock)
        record = orchestrator.wake(Trigger(TriggerKind.INTERACTION, payload=question.query))
        expected = {key_to_id[k] for k in question.expected_keys}
        if expected & set(record.retrieved_memory_ids):
            recovered += 1
        # the injected context came only from current_state + retrieval
        first_messages, _ = llm.calls[0]
        assert "Meridian photo archive" in first_messages[2].content
        clock.advance(minutes=1)

    assert recovered >= THRESHOLD, f"cold-start recovery {recovered}/10, threshold {THRESHOLD}/10"

    # the loop sat on the critical path: every question ran as an interaction wake
    wakes = list(history.replay())
    assert len(wakes) == 10
    assert all(w.trigger.kind is TriggerKind.INTERACTION for w in wakes)
    conn.close()
