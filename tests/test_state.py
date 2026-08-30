"""CurrentState: bounded, frozen, provenance-carrying; update_state drops rather than
appends and falls back to the prior state on garbage output."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fakes import ScriptedLLM, completion
from history.records import Trigger, TriggerKind, WakeRecord
from state.state import CurrentState, Entry
from state.update import update_state

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


def wake_record() -> WakeRecord:
    return WakeRecord(
        id="wake_test", trigger=Trigger(TriggerKind.INTERACTION, "hi"), occurred_at=NOW, state_snapshot={}
    )


def test_roundtrip_and_tolerant_parsing():
    state = CurrentState(
        active_projects=(Entry("Meridian", ("mem_1",)),),
        open_questions=(Entry("B2 or NAS?", ("mem_2",), due="2026-09-05"),),
    )
    again = CurrentState.from_dict(state.to_dict())
    assert again == state
    # bare strings and junk entries are tolerated
    parsed = CurrentState.from_dict({"priorities": ["ship the gallery", 42, {"text": ""}]})
    assert parsed.priorities == (Entry("ship the gallery"),)


def test_clamp_bounds_entries_and_chars():
    state = CurrentState(recent_focus=tuple(Entry(f"entry {i} " + "x" * 300) for i in range(9)))
    clamped = state.clamp(max_entries=5, max_chars=20)
    assert len(clamped.recent_focus) == 5
    assert all(len(e.text) <= 20 for e in clamped.recent_focus)


def test_render_shows_entries_provenance_and_due():
    state = CurrentState(
        active_projects=(Entry("Meridian photo archive", ("mem_1", "mem_2")),),
        open_questions=(Entry("B2 or NAS?", ("mem_3",), due="2026-09-05"),),
    )
    rendered = state.render()
    assert "Meridian photo archive" in rendered
    assert "[mem_1, mem_2]" in rendered
    assert "(due 2026-09-05)" in rendered
    assert CurrentState.empty().render() == "current state: (empty)"


def test_store_roundtrip(state_store):
    assert state_store.load() == CurrentState.empty()
    state = CurrentState(priorities=(Entry("edit the backlog", ("mem_9",)),))
    state_store.save(state)
    assert state_store.load() == state


def test_update_state_parses_fenced_json():
    payload = {
        "active_projects": [{"text": "Meridian", "memory_ids": ["mem_1"]}],
        "priorities": [],
        "open_questions": [],
        "recent_focus": [],
    }
    llm = ScriptedLLM(completion(text=f"```json\n{json.dumps(payload)}\n```"))
    new = update_state(llm, CurrentState.empty(), wake_record())
    assert new.active_projects == (Entry("Meridian", ("mem_1",)),)


def test_update_state_drops_stale_entries():
    prior = CurrentState(priorities=(Entry("finish captions"), Entry("fix the dedupe bug", ("mem_4",))))
    llm = ScriptedLLM(
        completion(text=json.dumps({"priorities": [{"text": "finish captions", "memory_ids": []}]}))
    )
    new = update_state(llm, prior, wake_record())
    assert new.priorities == (Entry("finish captions"),)  # the fixed bug is dropped, not annotated


def test_update_state_clamps_model_output():
    llm = ScriptedLLM(
        completion(text=json.dumps({"recent_focus": [{"text": f"item {i}"} for i in range(12)]}))
    )
    new = update_state(llm, CurrentState.empty(), wake_record(), max_entries=5)
    assert len(new.recent_focus) == 5


def test_update_state_returns_prior_on_garbage():
    prior = CurrentState(priorities=(Entry("hold steady"),))
    llm = ScriptedLLM(completion(text="I could not produce JSON, sorry."))
    assert update_state(llm, prior, wake_record()) == prior
