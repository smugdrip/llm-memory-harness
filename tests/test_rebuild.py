"""rebuild --from-history: replays recorded tool calls with no model call, reproduces
ids and supersession chains exactly, and does not resurrect retired memories."""

from __future__ import annotations

import sqlite3

from fakes import ScriptedLLM, completion, finish, state_json, tool_call
from history.history import History
from history.records import Trigger, TriggerKind
from memory.memory import Memory
from memory.store import SqliteStore
from runtime.orchestrator import Orchestrator
from runtime.rebuild import rebuild
from state.state import CurrentState, Entry, StateStore

STATE_ONE = CurrentState(active_projects=(Entry("Meridian photo archive"),))
STATE_TWO = CurrentState(
    active_projects=(Entry("Meridian photo archive"),),
    recent_focus=(Entry("consolidated the thumbnail decision"),),
)


def build_world(conn, clock, embedder):
    """Two real wakes through the orchestrator: one writes two memories, a reflection
    wake supersedes one of them. Returns the shared components."""
    memory = Memory(SqliteStore(conn), embedder, clock=clock, similarity_floor=0.1)
    history = History(conn, clock)
    state_store = StateStore(conn)

    def orchestrator(llm):
        return Orchestrator(llm=llm, memory=memory, history=history, state=state_store, clock=clock)

    llm_one = ScriptedLLM(
        completion(
            tool_call(
                "memory_write", {"text": "Thumbnails use ImageMagick at 256 pixels", "type": "decision"}
            ),
            tool_call("memory_write", {"text": "Marco runs the Lumen Prints shop", "type": "relationship"}),
        ),
        finish("act", text="noted"),
        state_json(STATE_ONE),
    )
    orchestrator(llm_one).wake(Trigger(TriggerKind.INTERACTION, "please remember the setup"))

    old = next(r for r in memory.store.records() if "ImageMagick" in r.canonical_text)
    clock.advance(hours=1)
    llm_two = ScriptedLLM(
        completion(
            tool_call(
                "memory_supersede",
                {"memory_id": old.id, "text": "Thumbnails use libvips at 512 pixels, replacing ImageMagick"},
            )
        ),
        finish("reflect"),
        state_json(STATE_TWO),
    )
    orchestrator(llm_two).wake(Trigger(TriggerKind.REFLECTION, "cadence"))
    return memory, history, state_store


def fresh_target(embedder, clock):
    conn = sqlite3.connect(":memory:")
    return conn, Memory(SqliteStore(conn), embedder, clock=clock, similarity_floor=0.1), StateStore(conn)


def snapshot(memory):
    return {
        r.id: (r.canonical_text, r.supersedes, r.superseded_by, r.source_id, r.origin)
        for r in memory.store.records()
    }


def test_rebuild_reproduces_memories_chains_and_state(conn, clock, embedder):
    memory, history, state_store = build_world(conn, clock, embedder)
    _conn2, memory2, state_store2 = fresh_target(embedder, clock)

    report = rebuild(history, memory2, state_store2)

    assert report.errors == []
    assert (report.wakes, report.writes, report.supersedes) == (2, 2, 1)
    assert snapshot(memory2) == snapshot(memory)  # deterministic ids, identical chains
    assert state_store2.load() == state_store.load() == STATE_TWO
    assert report.state_restored


def test_rebuild_is_idempotent(conn, clock, embedder):
    _, history, _ = build_world(conn, clock, embedder)
    _conn2, memory2, state_store2 = fresh_target(embedder, clock)
    rebuild(history, memory2, state_store2)
    first = snapshot(memory2)
    report = rebuild(history, memory2, state_store2)  # replay again, same target
    assert snapshot(memory2) == first
    assert report.errors == []


def test_rebuild_does_not_resurrect_superseded_memories(conn, clock, embedder):
    """A rebuild that quietly resurrects retired memories looks exactly like one that
    worked — this is the round-trip guard over replay order."""
    _memory, history, _ = build_world(conn, clock, embedder)
    _conn2, memory2, state_store2 = fresh_target(embedder, clock)
    rebuild(history, memory2, state_store2)

    results = memory2.search("Thumbnails pixels ImageMagick libvips")
    assert results
    assert all(r.supersedes is not None or "libvips" in r.canonical_text for r in results)
    retired = next(r for r in memory2.store.records() if r.superseded_by is not None)
    assert "ImageMagick at 256" in retired.canonical_text
    assert retired.id not in [r.id for r in results]
