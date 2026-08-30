"""Source history: append-only, one record shape for turns and wakes, replayable."""

from __future__ import annotations

import pytest

from history.records import BudgetUsed, Decision, Message, ToolCall, Trigger, TriggerKind


def test_wake_roundtrip(history, clock):
    trigger = Trigger(TriggerKind.INTERACTION, "hello")
    record = history.open_wake(trigger, {"a": 1})
    assert record.occurred_at == clock.now()

    sid0 = history.append_turn(record.id, Message(role="user", content="hello"))
    assert sid0 == f"wake/{record.id}/turn/0"
    sid1 = history.append_turn(
        record.id,
        Message(
            role="assistant",
            content="hi",
            tool_calls=(ToolCall("c1", "memory_write", {"text": "x", "type": "event"}),),
        ),
    )
    assert sid1 == f"wake/{record.id}/turn/1"

    history.record_retrieval(record.id, ["mem_a", "mem_b"])
    history.record_retrieval(record.id, ["mem_b", "mem_c"])

    closed = history.close_wake(
        record.id,
        Decision.RESPOND,
        "completed",
        BudgetUsed(iterations=1, tokens=100, tool_calls=1, ms=5),
        state_after={"active_projects": []},
    )
    assert closed.retrieved_memory_ids == ("mem_a", "mem_b", "mem_c")
    assert closed.decision is Decision.RESPOND
    assert closed.stop_reason == "completed"
    assert closed.state_snapshot == {"a": 1}
    assert closed.state_after == {"active_projects": []}
    assert closed.budget_used == BudgetUsed(1, 100, 1, 5)
    assert closed.turns[1].message.tool_calls[0].arguments == {"text": "x", "type": "event"}
    assert [t.message.role for t in closed.turns] == ["user", "assistant"]


def test_append_only(history):
    record = history.open_wake(Trigger(TriggerKind.AUTONOMOUS, "scheduled"), {})
    history.close_wake(record.id, Decision.SLEEP, "idle", BudgetUsed())
    with pytest.raises(ValueError):
        history.append_turn(record.id, Message(role="assistant", content="late"))
    with pytest.raises(ValueError):
        history.close_wake(record.id, Decision.SLEEP, "idle", BudgetUsed())
    with pytest.raises(ValueError):
        history.record_retrieval(record.id, ["mem_x"])


def test_unknown_wake_raises(history):
    with pytest.raises(KeyError):
        history.append_turn("wake_missing", Message(role="user", content="?"))
    with pytest.raises(KeyError):
        history.get("wake_missing")


def test_replay_in_commit_order_including_unclosed(history):
    r1 = history.open_wake(Trigger(TriggerKind.INTERACTION, "one"), {})
    history.close_wake(r1.id, Decision.RESPOND, "completed", BudgetUsed())
    r2 = history.open_wake(Trigger(TriggerKind.AUTONOMOUS, "two"), {})
    replayed = list(history.replay())
    assert [w.id for w in replayed] == [r1.id, r2.id]
    assert replayed[1].decision is None  # a crash leaves an open record; replay still sees it


def test_last_wake_at_and_turns_since(history, clock):
    assert history.last_wake_at(TriggerKind.AUTONOMOUS) is None
    assert history.turns_since_last(TriggerKind.REFLECTION) == 0

    a = history.open_wake(Trigger(TriggerKind.INTERACTION, "q"), {})
    history.append_turn(a.id, Message(role="user", content="q"))
    history.append_turn(a.id, Message(role="assistant", content="a"))
    assert history.turns_since_last(TriggerKind.REFLECTION) == 2

    clock.advance(minutes=5)
    r = history.open_wake(Trigger(TriggerKind.REFLECTION, "cadence"), {})
    assert history.last_wake_at(TriggerKind.REFLECTION) == clock.now()
    assert history.turns_since_last(TriggerKind.REFLECTION) == 0

    b = history.open_wake(Trigger(TriggerKind.INTERACTION, "q2"), {})
    history.append_turn(b.id, Message(role="user", content="q2"))
    assert history.turns_since_last(TriggerKind.REFLECTION) == 1
    assert r.id != b.id
