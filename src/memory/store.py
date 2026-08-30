"""MemoryStore protocol and the boring implementation: SQLite rows, vectors as JSON,
brute-force cosine. At MVP scale exact search over a few thousand vectors is fast and
has no index-tuning failure modes; swap in something heavier when measured volume
justifies it.

`by_source_id` is the cheap lookup idempotency rests on. `mark_superseded` rather than
a delete — the row leaves search and stays for history and rebuilds. `records()` is for
rebuild and the eval harness.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from history.records import MemoryRecord, MemoryType, TriggerKind, iso, parse_iso
from llm.embedder import Vector


@dataclass(frozen=True)
class Scored:
    record: MemoryRecord
    score: float


class MemoryStore(Protocol):
    def upsert(self, record: MemoryRecord, vector: Vector) -> None: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def by_source_id(self, source_id: str) -> MemoryRecord | None: ...

    def nearest(self, vector: Vector, k: int, floor: float) -> list[Scored]: ...

    def mark_superseded(self, memory_id: str, by: str) -> None: ...

    def records(self) -> Iterator[MemoryRecord]: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    canonical_text TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    source_id TEXT NOT NULL,
    origin TEXT NOT NULL,
    type TEXT NOT NULL,
    entities TEXT NOT NULL DEFAULT '[]',
    importance REAL NOT NULL DEFAULT 0.5,
    occurred_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    supersedes TEXT,
    superseded_by TEXT,
    embedding_model_id TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    preprocess_version INTEGER NOT NULL,
    vector TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_id);
CREATE INDEX IF NOT EXISTS idx_memories_live ON memories(superseded_by) WHERE superseded_by IS NULL;
"""


class SqliteStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def upsert(self, record: MemoryRecord, vector: Vector) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.id,
                record.canonical_text,
                record.raw_text,
                record.source_id,
                record.origin.value,
                record.type.value,
                json.dumps(list(record.entities)),
                record.importance,
                iso(record.occurred_at),
                iso(record.created_at),
                record.supersedes,
                record.superseded_by,
                record.embedding_model_id,
                record.embedding_dim,
                record.preprocess_version,
                json.dumps(vector),
            ),
        )
        self._conn.commit()

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _to_record(row) if row else None

    def by_source_id(self, source_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE source_id = ? ORDER BY created_at, id LIMIT 1", (source_id,)
        ).fetchone()
        return _to_record(row) if row else None

    def nearest(self, vector: Vector, k: int, floor: float) -> list[Scored]:
        """Exact cosine over live rows. Rows whose stored dim differs from the query's
        are skipped — vectors from different spaces are not comparable."""
        scored: list[Scored] = []
        for row in self._conn.execute(
            "SELECT * FROM memories WHERE superseded_by IS NULL AND embedding_dim = ?", (len(vector),)
        ):
            score = _cosine(vector, json.loads(row["vector"]))
            if score >= floor:
                scored.append(Scored(_to_record(row), score))
        scored.sort(key=lambda s: (-s.score, s.record.id))
        return scored[:k]

    def mark_superseded(self, memory_id: str, by: str) -> None:
        row = self._conn.execute("SELECT superseded_by FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(f"no memory {memory_id!r}")
        if row["superseded_by"] is not None and row["superseded_by"] != by:
            raise ValueError(f"memory {memory_id!r} already superseded by {row['superseded_by']!r}")
        self._conn.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (by, memory_id))
        self._conn.commit()

    def records(self) -> Iterator[MemoryRecord]:
        for row in self._conn.execute("SELECT * FROM memories ORDER BY created_at, id"):
            yield _to_record(row)

    def clear(self) -> None:
        """Not part of the protocol: rebuild's reset, used only by the CLI."""
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        canonical_text=row["canonical_text"],
        raw_text=row["raw_text"],
        source_id=row["source_id"],
        origin=TriggerKind(row["origin"]),
        type=MemoryType(row["type"]),
        entities=tuple(json.loads(row["entities"])),
        importance=row["importance"],
        occurred_at=parse_iso(row["occurred_at"]),
        created_at=parse_iso(row["created_at"]),
        supersedes=row["supersedes"],
        superseded_by=row["superseded_by"],
        embedding_model_id=row["embedding_model_id"],
        embedding_dim=row["embedding_dim"],
        preprocess_version=row["preprocess_version"],
    )
