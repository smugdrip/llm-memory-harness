"""The orchestrator: commit ordering, the sleep short-circuit, budget stops at step
boundaries, provenance binding, and successor policy."""

from __future__ import annotations

import json

import pytest

from fakes import ScriptedLLM, completion, finish, state_json, tool_call
from history.history import History
from history.records import Decision, MemoryType, Trigger, TriggerKind
from memory.memory import Memory
from runtime.budget import BudgetLimits
from runtime.orchestrator import Orchestrator
from state.state import CurrentState, Entry

INTERACT = Trigger(TriggerKind.INTERACTION, "Why did we pick SQLite for Meridian?")
AUTO = Trigger(TriggerKind.AUTONOMOUS, "scheduled wake")


def test_interaction_respond_flow(make_orchestrator, state_store):
    llm = ScriptedLLM(
        finish("respond", text="Single-file deployment."),
        state_json(CurrentState(recent_focus=(Entry("SQLite question"),))),
    )
    record = make_orchestrator(llm).wake(INTERACT)
    assert record.decision is Decision.RESPOND
    assert record.stop_reason == "completed"
    assert [t.message.role for t in record.turns] == ["user", "assistant"]
    assert record.turns[1].message.content == "Single-file deployment."
    assert record.budget_used.iterations == 1
    assert record.state_after == state_store.load().to_dict()
    assert state_store.load().recent_focus == (Entry("SQLite question"),)
    # what the model saw: stable system prompt first, volatile block after the user turn
    messages, tools = llm.calls[0]
    assert [m.role for m in messages] == ["system", "user", "user"]
    assert "current state:" in messages[2].content
    assert {t.name for t in tools} == {"memory_search", "memory_write", "memory_supersede", "finish"}


def test_turn_is_committed_before_its_tools_dispatch(conn, clock, embedder, state_store):
    events = []

    class SpyHistory(History):
        def append_turn(self, wake_id, message):
            events.append(("append", message.role))
            return super().append_turn(wake_id, message)

    class SpyMemory(Memory):
        def dispatch(self, call, **kwargs):
            events.append(("dispatch", call.name))
            return super().dispatch(call, **kwargs)

    from memory.store import SqliteStore

    memory = SpyMemory(SqliteStore(conn), embedder, clock=clock, similarity_floor=0.1)
    history = SpyHistory(conn, clock)
    llm = ScriptedLLM(
        completion(
            tool_call("memory_write", {"text": "Meridian metadata lives in SQLite", "type": "decision"})
        ),
        finish("act"),
        state_json(CurrentState.empty()),
    )
    orchestrator = Orchestrator(llm=llm, memory=memory, history=history, state=state_store, clock=clock)
    orchestrator.wake(INTERACT)
    assert events.index(("append", "assistant")) < events.index(("dispatch", "memory_write"))
    assert events.index(("dispatch", "memory_write")) < events.index(("append", "tool"))


def test_provenance_is_bound_by_the_orchestrator(make_orchestrator, memory):
    llm = ScriptedLLM(
        completion(
            tool_call(
                "memory_write",
                {"text": "Thumbnails use libvips", "type": "decision", "source_id": "evil/spoof"},
            )
        ),
        finish("act"),
        state_json(CurrentState.empty()),
    )
    record = make_orchestrator(llm).wake(INTERACT)
    (written,) = list(memory.store.records())
    assert written.source_id == f"wake/{record.id}/turn/1"  # turn 0 is the user's message
    assert written.origin is TriggerKind.INTERACTION


def test_parallel_tool_results_return_in_one_message(make_orchestrator, memory):
    llm = ScriptedLLM(
        completion(
            tool_call("memory_write", {"text": "Marco handles the printing", "type": "relationship"}),
            tool_call("memory_write", {"text": "Priya designs the themes", "type": "relationship"}),
        ),
        finish("act"),
        state_json(CurrentState.empty()),
    )
    record = make_orchestrator(llm).wake(INTERACT)
    tool_turns = [t for t in record.turns if t.message.role == "tool"]
    assert len(tool_turns) == 1
    assert len(tool_turns[0].message.tool_results) == 2
    assert len(list(memory.store.records())) == 2
    # and the second completion request carried that single tool message
    messages, _ = llm.calls[1]
    assert [m.role for m in messages[-2:]] == ["assistant", "tool"]
    assert len(messages[-1].tool_results) == 2


def test_sleep_short_circuits_before_any_derived_write(make_orchestrator, memory, state_store, conn):
    state_store.save(CurrentState(priorities=(Entry("hold steady", ("mem_1",)),)))
    memory.write(
        "an existing fact about geohash maps",
        MemoryType.PROJECT,
        source_id="wake/w0/turn/1",
        origin=TriggerKind.INTERACTION,
    )
    before_state = conn.execute("SELECT value FROM current_state").fetchone()[0]
    before_count = len(list(memory.store.records()))

    llm = ScriptedLLM(finish("sleep"))  # nothing queued for update_state — it must not run
    record = make_orchestrator(llm).wake(AUTO)

    assert record.decision is Decision.SLEEP
    assert record.stop_reason == "idle"
    assert record.state_after is None
    assert len(list(memory.store.records())) == before_count
    assert conn.execute("SELECT value FROM current_state").fetchone()[0] == before_state
    assert llm.queue == []  # exactly one completion consumed
    # the committed record is the wake's entire effect
    assert record.budget_used.iterations == 1


def test_budget_exhaustion_stops_at_a_step_boundary(make_orchestrator, state_store, conn):
    llm = ScriptedLLM(
        completion(tool_call("memory_search", {"query": "anything at all"})),
        # nothing else: the loop must stop on budget, not ask for another completion
    )
    record = make_orchestrator(llm, limits=BudgetLimits(max_iterations=1)).wake(INTERACT)
    assert record.stop_reason == "budget:iterations"
    assert record.decision is None  # ran out is not a decision — the two stay distinguishable
    assert [t.message.role for t in record.turns] == ["user", "assistant", "tool"]
    assert record.state_after is None
    assert conn.execute("SELECT value FROM current_state").fetchone() is None


def test_error_closes_the_record_then_raises(make_orchestrator, history):
    llm = ScriptedLLM(RuntimeError("provider exploded"))
    with pytest.raises(RuntimeError):
        make_orchestrator(llm).wake(INTERACT)
    (record,) = list(history.replay())
    assert record.stop_reason == "error:RuntimeError"
    assert record.decision is None


def test_retrieval_is_recorded_from_observe_and_tool_searches(make_orchestrator, memory):
    seeded = memory.write(
        "Decision: Meridian stores gallery metadata in SQLite",
        MemoryType.DECISION,
        source_id="wake/w0/turn/1",
        origin=TriggerKind.INTERACTION,
    )
    other = memory.write(
        "Anneke curates the family album section",
        MemoryType.RELATIONSHIP,
        source_id="wake/w0/turn/2",
        origin=TriggerKind.INTERACTION,
    )
    llm = ScriptedLLM(
        completion(tool_call("memory_search", {"query": "Anneke family album"})),
        finish("respond", text="answer"),
        state_json(CurrentState.empty()),
    )
    record = make_orchestrator(llm).wake(INTERACT)
    assert seeded.id in record.retrieved_memory_ids  # from the observe step
    assert other.id in record.retrieved_memory_ids  # from the model's own search
    # and the retrieved memories were injected into the volatile block
    messages, _ = llm.calls[0]
    assert seeded.canonical_text in messages[2].content


def test_plain_text_with_no_tool_calls_finishes_as_respond(make_orchestrator):
    llm = ScriptedLLM(completion(text="just an answer"), state_json(CurrentState.empty()))
    record = make_orchestrator(llm).wake(INTERACT)
    assert record.decision is Decision.RESPOND
    assert record.stop_reason == "completed"


def test_write_alongside_finish_still_executes(make_orchestrator, memory):
    llm = ScriptedLLM(
        completion(
            tool_call("memory_write", {"text": "The backlog cap is 200 photos", "type": "decision"}),
            tool_call("finish", {"decision": "act"}),
        ),
        state_json(CurrentState.empty()),
    )
    record = make_orchestrator(llm).wake(INTERACT)
    assert record.decision is Decision.ACT
    assert record.budget_used.iterations == 1
    assert len(list(memory.store.records())) == 1


def test_successor_granted_spends_the_same_allowance(make_orchestrator):
    llm = ScriptedLLM(
        finish("sleep", request_successor=True),
        finish("sleep"),
    )
    records = make_orchestrator(llm).run(AUTO)
    assert len(records) == 2
    assert records[0].successor_requested is True
    assert records[0].budget_used.iterations == 1
    assert records[1].budget_used.iterations == 2  # cumulative: one allowance, not two


def test_successor_never_granted_for_interaction(make_orchestrator):
    llm = ScriptedLLM(
        finish("respond", text="done", request_successor=True),
        state_json(CurrentState.empty()),
    )
    records = make_orchestrator(llm).run(INTERACT)
    assert len(records) == 1
    assert records[0].successor_requested is True  # requested, recorded, not granted


def test_bad_decision_value_maps_to_sleep(make_orchestrator):
    llm = ScriptedLLM(completion(tool_call("finish", {"decision": "party"})))
    record = make_orchestrator(llm).wake(AUTO)
    assert record.decision is Decision.SLEEP


def test_tool_result_ids_parse_back(make_orchestrator, memory):
    """The search tool's JSON results are the contract _note_search_results parses."""
    memory.write(
        "rsync copies RAW files nightly",
        MemoryType.PROJECT,
        source_id="wake/w0/turn/1",
        origin=TriggerKind.INTERACTION,
    )
    llm = ScriptedLLM(
        completion(tool_call("memory_search", {"query": "rsync RAW files"})),
        finish("respond"),
        state_json(CurrentState.empty()),
    )
    record = make_orchestrator(llm).wake(Trigger(TriggerKind.INTERACTION, "unrelated zebra query"))
    tool_turn = next(t for t in record.turns if t.message.role == "tool")
    payload = json.loads(tool_turn.message.tool_results[0].content)
    assert payload["results"][0]["id"] in record.retrieved_memory_ids
