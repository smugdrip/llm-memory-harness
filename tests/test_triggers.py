"""Trigger sources: cadence, cooldown persisted across restarts, and the due-question
path — asserted with an injected clock instead of waiting."""

from __future__ import annotations

from datetime import timedelta

from history.records import BudgetUsed, Decision, Message, Trigger, TriggerKind
from runtime.triggers import AutonomousDue, FirstDue, ReflectionDue
from state.state import CurrentState, Entry


def autonomous(history, state_store, clock, **kwargs) -> AutonomousDue:
    defaults = dict(cooldown=timedelta(minutes=30), interval=timedelta(days=1))
    defaults.update(kwargs)
    return AutonomousDue(history, state_store, clock, **defaults)


def close_idle(history, record):
    history.close_wake(record.id, Decision.SLEEP, "idle", BudgetUsed())


def test_first_run_fires_scheduled_wake(history, state_store, clock):
    trigger = autonomous(history, state_store, clock).next()
    assert trigger == Trigger(TriggerKind.AUTONOMOUS, "scheduled wake")


def test_cooldown_is_persisted_state_not_a_sleep(history, state_store, clock):
    record = history.open_wake(Trigger(TriggerKind.AUTONOMOUS, "scheduled wake"), {})
    close_idle(history, record)
    # A *fresh* source over the same history — a restarted process — still cools down.
    assert autonomous(history, state_store, clock).next() is None
    clock.advance(minutes=31)
    assert autonomous(history, state_store, clock).next() is None  # cooled, but interval not elapsed
    clock.advance(days=1)
    assert autonomous(history, state_store, clock).next() == Trigger(TriggerKind.AUTONOMOUS, "scheduled wake")


def test_due_open_question_wakes(history, state_store, clock):
    due = (clock.now().date() - timedelta(days=1)).isoformat()
    state_store.save(
        CurrentState(open_questions=(Entry("back up to Backblaze B2 or the NAS?", ("mem_7",), due=due),))
    )
    trigger = autonomous(history, state_store, clock).next()
    assert trigger is not None
    assert trigger.kind is TriggerKind.AUTONOMOUS
    assert trigger.payload == "due open question: back up to Backblaze B2 or the NAS?"


def test_future_or_malformed_due_dates_do_not_fire(history, state_store, clock):
    record = history.open_wake(Trigger(TriggerKind.AUTONOMOUS, "scheduled wake"), {})
    close_idle(history, record)
    clock.advance(hours=1)  # past cooldown, inside the interval
    future = (clock.now().date() + timedelta(days=3)).isoformat()
    state_store.save(
        CurrentState(
            open_questions=(
                Entry("later question", due=future),
                Entry("broken question", due="not-a-date"),
                Entry("question with no due date"),
            )
        )
    )
    assert autonomous(history, state_store, clock).next() is None


def test_reflection_cadence_counts_interaction_turns(history, clock):
    source = ReflectionDue(history, clock, turn_threshold=3, cooldown=timedelta(minutes=30))
    assert source.next() is None
    wake = history.open_wake(Trigger(TriggerKind.INTERACTION, "q"), {})
    for i in range(3):
        history.append_turn(wake.id, Message(role="user", content=f"turn {i}"))
    trigger = source.next()
    assert trigger is not None and trigger.kind is TriggerKind.REFLECTION
    assert "3 interaction turns" in trigger.payload

    reflection = history.open_wake(trigger, {})
    close_idle(history, reflection)
    assert source.next() is None  # cooldown and a fresh count both hold it back
    clock.advance(minutes=31)
    assert source.next() is None  # cooled down, but no new turns yet


def test_first_due_takes_the_first_source_with_something(history, state_store, clock):
    reflection = ReflectionDue(history, clock, turn_threshold=1)
    scheduled = autonomous(history, state_store, clock)
    assert FirstDue([reflection, scheduled]).next() == Trigger(TriggerKind.AUTONOMOUS, "scheduled wake")
    assert FirstDue([]).next() is None
