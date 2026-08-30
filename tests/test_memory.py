"""Memory: provenance, idempotency, curation-by-return-value, and the tool binding
being the same path the system calls."""

from __future__ import annotations

import json

import pytest

from history.records import MemoryType, ToolCall, TriggerKind
from memory.preprocess import PREPROCESS_VERSION

SRC = "wake/w1/turn/1"


def write(memory, text, type=MemoryType.EVENT, *, source_id=SRC, origin=TriggerKind.INTERACTION, **kw):
    return memory.write(text, type, source_id=source_id, origin=origin, **kw)


def test_write_builds_full_record(memory, clock):
    record = memory.write(
        "  Decision:   use SQLite for gallery metadata ",
        MemoryType.DECISION,
        importance=0.8,
        entities=["SQLite"],
        source_id=SRC,
        origin=TriggerKind.INTERACTION,
    )
    assert record.id.startswith("mem_")
    assert record.canonical_text == "Decision: use SQLite for gallery metadata"
    assert record.raw_text == "  Decision:   use SQLite for gallery metadata "
    assert record.source_id == SRC
    assert record.origin is TriggerKind.INTERACTION
    assert record.type is MemoryType.DECISION
    assert record.entities == ("SQLite",)
    assert record.importance == 0.8
    assert record.created_at == clock.now()
    assert record.occurred_at == clock.now()  # defaults to now when the model gives nothing
    assert record.embedding_model_id == memory.embedder.model_id
    assert record.embedding_dim == memory.embedder.dim
    assert record.preprocess_version == PREPROCESS_VERSION


def test_write_is_idempotent_on_source_and_text(memory):
    r1 = write(memory, "Marco runs Lumen Prints", MemoryType.RELATIONSHIP)
    r2 = write(memory, "Marco runs Lumen Prints", MemoryType.RELATIONSHIP)
    assert r1.id == r2.id
    assert len(list(memory.store.records())) == 1


def test_one_turn_can_write_two_memories(memory):
    r1 = write(memory, "Meridian runs on Hetzner", MemoryType.DECISION)
    r2 = write(memory, "Priya designs the CSS themes", MemoryType.RELATIONSHIP)
    assert r1.id != r2.id
    assert len(list(memory.store.records())) == 2


def test_near_duplicate_is_rejected(memory):
    first = memory.write(
        "Dana uses the Fuji X-T5 camera for Meridian",
        MemoryType.PROJECT,
        source_id="wake/w1/turn/1",
        origin=TriggerKind.INTERACTION,
    )
    assert first is not None
    dup = memory.write(
        "Dana uses the Fuji X-T5 camera for Meridian!",  # same token bag, different source
        MemoryType.PROJECT,
        source_id="wake/w2/turn/1",
        origin=TriggerKind.INTERACTION,
    )
    assert dup is None
    assert len(list(memory.store.records())) == 1


def test_empty_text_is_rejected(memory):
    assert write(memory, "   \n ") is None


def test_search_returns_nothing_below_floor(memory):
    write(memory, "Meridian publishes web galleries", MemoryType.PROJECT)
    assert memory.search("zebra xylophone quartz volcano") == []  # nothing is a correct result
    assert memory.search("") == []


def test_search_ranks_relevant_first(memory):
    a = memory.write(
        "the red fox jumped over the garden fence",
        MemoryType.EVENT,
        source_id="wake/w1/turn/1",
        origin=TriggerKind.INTERACTION,
    )
    memory.write(
        "quarterly accounting rules for tax season",
        MemoryType.EVENT,
        source_id="wake/w1/turn/2",
        origin=TriggerKind.INTERACTION,
    )
    results = memory.search("red fox jumped")
    assert results and results[0].id == a.id


def test_search_type_filter(memory):
    write(memory, "Decision: thumbnails use libvips", MemoryType.DECISION, source_id="wake/w1/turn/1")
    write(memory, "The thumbnails broke on Tuesday", MemoryType.EVENT, source_id="wake/w1/turn/2")
    results = memory.search("thumbnails", type=MemoryType.DECISION)
    assert results and all(r.type is MemoryType.DECISION for r in results)


def test_supersede_retires_old_and_inherits(memory, clock):
    old = memory.write(
        "Meridian thumbnails use ImageMagick at 256 pixels",
        MemoryType.DECISION,
        importance=0.7,
        source_id="wake/w1/turn/1",
        origin=TriggerKind.INTERACTION,
    )
    clock.advance(days=1)
    new = memory.supersede(
        old.id,
        "Meridian thumbnails use libvips at 512 pixels",
        source_id="wake/w2/turn/1",
        origin=TriggerKind.REFLECTION,
    )
    assert new.supersedes == old.id
    assert memory.store.get(old.id).superseded_by == new.id
    assert new.type is old.type
    assert new.importance == old.importance
    assert new.occurred_at == old.occurred_at
    assert new.origin is TriggerKind.REFLECTION
    results = memory.search("Meridian thumbnails pixels")
    ids = [r.id for r in results]
    assert new.id in ids and old.id not in ids


def test_supersede_unknown_raises(memory):
    with pytest.raises(KeyError):
        memory.supersede("mem_missing", "text", source_id=SRC, origin=TriggerKind.REFLECTION)


def test_supersede_replay_is_idempotent(memory):
    old = write(memory, "original fact")
    n1 = memory.supersede(old.id, "corrected fact", source_id="wake/w2/turn/1", origin=TriggerKind.REFLECTION)
    n2 = memory.supersede(old.id, "corrected fact", source_id="wake/w2/turn/1", origin=TriggerKind.REFLECTION)
    assert n1.id == n2.id
    assert len(list(memory.store.records())) == 2


def test_recent_returns_newest_live(memory, clock):
    a = write(memory, "first fact about apples", source_id="wake/w1/turn/1")
    clock.advance(minutes=1)
    b = write(memory, "second fact about bridges", source_id="wake/w1/turn/2")
    clock.advance(minutes=1)
    c = write(memory, "third fact about cranes", source_id="wake/w1/turn/3")
    recent = memory.recent(2)
    assert [r.id for r in recent] == [c.id, b.id]
    assert a.id not in [r.id for r in recent]


# ---------------------------------------------------------------- tool binding


def test_tool_schemas_never_expose_provenance(memory):
    for schema in memory.tool_schemas():
        blob = json.dumps(schema.parameters)
        assert "source_id" not in blob
        assert "origin" not in blob


def test_dispatch_binds_provenance_ignoring_model_supplied_values(memory):
    call = ToolCall(
        id="c1",
        name="memory_write",
        arguments={
            "text": "The domain is registered at Porkbun",
            "type": "decision",
            "source_id": "evil/spoof",  # a confused or adversarial model
            "origin": "reflection",
        },
    )
    out = memory.dispatch(call, source_id=SRC, origin=TriggerKind.INTERACTION)
    written = json.loads(out)["written"]
    record = memory.store.get(written)
    assert record.source_id == SRC
    assert record.origin is TriggerKind.INTERACTION


def test_dispatch_search_is_the_same_path(memory):
    write(memory, "Anneke curates the family album", MemoryType.RELATIONSHIP)
    direct = [r.id for r in memory.search("Anneke family album")]
    out = memory.dispatch(
        ToolCall(id="c1", name="memory_search", arguments={"query": "Anneke family album"}),
        source_id=SRC,
        origin=TriggerKind.INTERACTION,
    )
    via_tool = [r["id"] for r in json.loads(out)["results"]]
    assert via_tool == direct


def test_dispatch_write_reports_duplicate_rejection(memory):
    write(memory, "Backups go to the Synology NAS", MemoryType.DECISION, source_id="wake/w1/turn/1")
    arguments = {"text": "Backups go to the Synology NAS", "type": "decision"}
    out = memory.dispatch(
        ToolCall(id="c1", name="memory_write", arguments=arguments),
        source_id="wake/w2/turn/1",
        origin=TriggerKind.INTERACTION,
    )
    assert json.loads(out) == {"written": None, "reason": "rejected: empty or near-duplicate"}


def test_dispatch_errors_are_returned_not_raised(memory):
    unknown = memory.dispatch(
        ToolCall(id="c1", name="memory_forget", arguments={}), source_id=SRC, origin=TriggerKind.INTERACTION
    )
    assert "error" in json.loads(unknown)
    bad_type = memory.dispatch(
        ToolCall(id="c2", name="memory_write", arguments={"text": "x", "type": "bogus"}),
        source_id=SRC,
        origin=TriggerKind.INTERACTION,
    )
    assert "error" in json.loads(bad_type)
    missing = memory.dispatch(
        ToolCall(id="c3", name="memory_supersede", arguments={"memory_id": "mem_missing", "text": "y"}),
        source_id=SRC,
        origin=TriggerKind.INTERACTION,
    )
    assert "error" in json.loads(missing)
