"""Milestone two — restraint, as a test.

Over a static world, a run of idle wakes produces zero memories and zero current_state
changes (each history record is its only effect), and a wake with a due open_question
picks it up.
"""

from __future__ import annotations

from datetime import timedelta

from fakes import ScriptedLLM, completion, finish, state_json, tool_call
from history.records import Decision, MemoryType, Trigger, TriggerKind
from runtime.triggers import AutonomousDue
from state.state import CurrentState, Entry


def seed_static_world(memory, state_store):
    memory.write(
        "Meridian is the photo archive pipeline",
        MemoryType.PROJECT,
        source_id="wake/w0/turn/1",
        origin=TriggerKind.INTERACTION,
    )
    state_store.save(CurrentState(active_projects=(Entry("Meridian photo archive"),)))


def test_idle_wakes_write_nothing_and_change_nothing(
    make_orchestrator, memory, state_store, history, conn, clock
):
    seed_static_world(memory, state_store)
    state_bytes_before = conn.execute("SELECT value FROM current_state").fetchone()[0]
    memories_before = len(list(memory.store.records()))
    source = AutonomousDue(
        history, state_store, clock, cooldown=timedelta(minutes=30), interval=timedelta(days=1)
    )

    for _ in range(5):
        clock.advance(days=2)
        trigger = source.next()
        assert trigger == Trigger(TriggerKind.AUTONOMOUS, "scheduled wake")
        record = make_orchestrator(ScriptedLLM(finish("sleep"))).wake(trigger)
        assert record.decision is Decision.SLEEP
        assert record.stop_reason == "idle"
        assert record.state_after is None

    assert len(list(memory.store.records())) == memories_before
    assert conn.execute("SELECT value FROM current_state").fetchone()[0] == state_bytes_before
    idle_wakes = [w for w in history.replay() if w.trigger.kind is TriggerKind.AUTONOMOUS]
    assert len(idle_wakes) == 5  # the records exist: doing nothing is auditable, not invisible


def test_due_open_question_is_picked_up(make_orchestrator, memory, state_store, history, clock):
    question_memory = memory.write(
        "Open question: back up original photos to Backblaze B2 or to the Synology NAS?",
        MemoryType.OPEN_QUESTION,
        source_id="wake/w0/turn/1",
        origin=TriggerKind.INTERACTION,
    )
    yesterday = (clock.now().date() - timedelta(days=1)).isoformat()
    state = CurrentState(
        open_questions=(
            Entry(
                "back up original photos to Backblaze B2 or the Synology NAS?",
                (question_memory.id,),
                due=yesterday,
            ),
        )
    )
    state_store.save(state)

    trigger = AutonomousDue(
        history, state_store, clock, cooldown=timedelta(minutes=30), interval=timedelta(days=1)
    ).next()
    assert trigger is not None
    assert trigger.payload.startswith("due open question:")
    assert "Backblaze B2" in trigger.payload

    llm = ScriptedLLM(
        completion(tool_call("memory_search", {"query": "Backblaze B2 Synology NAS backup"})),
        finish("act", text="picked a backup target"),
        state_json(state),
    )
    record = make_orchestrator(llm).wake(trigger)
    assert record.decision is Decision.ACT  # picked up, not slept through
    assert question_memory.id in record.retrieved_memory_ids
