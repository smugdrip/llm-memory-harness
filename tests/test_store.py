"""SqliteStore: the MemoryStore contract over SQLite with brute-force cosine."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from history.records import MemoryRecord, MemoryType, TriggerKind

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


def rec(id: str, text: str = "text", source: str = "src/1", **kwargs) -> MemoryRecord:
    defaults = dict(
        canonical_text=text,
        raw_text=text,
        source_id=source,
        origin=TriggerKind.INTERACTION,
        type=MemoryType.EVENT,
        occurred_at=NOW,
        created_at=NOW,
        embedding_model_id="fake/hash-bow-3",
        embedding_dim=3,
        preprocess_version=1,
    )
    defaults.update(kwargs)
    return MemoryRecord(id=id, **defaults)


def test_upsert_get_by_source(store):
    store.upsert(rec("mem_1", source="wake/w/turn/0"), [1.0, 0.0, 0.0])
    assert store.get("mem_1").canonical_text == "text"
    assert store.get("mem_nope") is None
    assert store.by_source_id("wake/w/turn/0").id == "mem_1"
    assert store.by_source_id("wake/w/turn/9") is None


def test_nearest_ranks_floors_and_never_pads(store):
    store.upsert(rec("mem_a"), [1.0, 0.0, 0.0])
    store.upsert(rec("mem_b"), [0.9, 0.1, 0.0])
    store.upsert(rec("mem_c"), [0.0, 1.0, 0.0])
    got = store.nearest([1.0, 0.0, 0.0], k=5, floor=0.5)
    assert [s.record.id for s in got] == ["mem_a", "mem_b"]  # c is below the floor; no padding
    assert got[0].score > got[1].score


def test_nearest_excludes_superseded(store):
    store.upsert(rec("mem_old"), [1.0, 0.0, 0.0])
    store.upsert(rec("mem_new", supersedes="mem_old"), [1.0, 0.0, 0.0])
    store.mark_superseded("mem_old", "mem_new")
    got = store.nearest([1.0, 0.0, 0.0], k=5, floor=0.1)
    assert [s.record.id for s in got] == ["mem_new"]
    assert store.get("mem_old").superseded_by == "mem_new"


def test_nearest_skips_dim_mismatch(store):
    store.upsert(rec("mem_3d"), [1.0, 0.0, 0.0])
    assert store.nearest([1.0, 0.0, 0.0, 0.0], k=5, floor=0.0) == []


def test_mark_superseded_errors(store):
    with pytest.raises(KeyError):
        store.mark_superseded("mem_missing", "mem_x")
    store.upsert(rec("mem_1"), [1.0, 0.0, 0.0])
    store.mark_superseded("mem_1", "mem_2")
    store.mark_superseded("mem_1", "mem_2")  # same value again is fine (replay)
    with pytest.raises(ValueError):
        store.mark_superseded("mem_1", "mem_3")


def test_records_and_clear(store):
    store.upsert(rec("mem_1", created_at=NOW), [1.0, 0.0, 0.0])
    store.upsert(rec("mem_2", created_at=NOW.replace(hour=13)), [0.0, 1.0, 0.0])
    assert [r.id for r in store.records()] == ["mem_1", "mem_2"]
    store.clear()
    assert list(store.records()) == []
