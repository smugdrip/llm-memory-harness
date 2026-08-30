"""Append-only source history: the immutable ground truth, and the one store that
cannot be regenerated.

`open_wake` is the commit point — the record exists before anything derived from the
wake is written. `append_turn` commits each message as the cycle runs and returns the
`source_id` that memory writes from that turn's tool calls are bound to; the turn is in
history before its tools execute, so a memory written mid-cycle always has a committed
source a rebuild can replay.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any

from history.records import (
    BudgetUsed,
    Clock,
    Decision,
    Message,
    SystemClock,
    Trigger,
    TriggerKind,
    Turn,
    WakeRecord,
    iso,
    new_wake_id,
    parse_iso,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS wakes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    trigger_kind TEXT NOT NULL,
    trigger_payload TEXT,
    occurred_at TEXT NOT NULL,
    state_snapshot TEXT NOT NULL,
    retrieved_memory_ids TEXT NOT NULL DEFAULT '[]',
    decision TEXT,
    stop_reason TEXT,
    budget_used TEXT,
    state_after TEXT,
    successor_requested INTEGER NOT NULL DEFAULT 0,
    closed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turns (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    wake_id TEXT NOT NULL REFERENCES wakes(id),
    idx INTEGER NOT NULL,
    source_id TEXT UNIQUE NOT NULL,
    message TEXT NOT NULL,
    UNIQUE (wake_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_wakes_kind ON wakes(trigger_kind, seq);
"""


class History:
    def __init__(self, conn: sqlite3.Connection, clock: Clock | None = None) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._clock = clock or SystemClock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def open_wake(self, trigger: Trigger, state_snapshot: dict[str, Any]) -> WakeRecord:
        wake_id = new_wake_id()
        occurred_at = self._clock.now()
        self._conn.execute(
            "INSERT INTO wakes (id, trigger_kind, trigger_payload, occurred_at, state_snapshot)"
            " VALUES (?, ?, ?, ?, ?)",
            (wake_id, trigger.kind.value, trigger.payload, iso(occurred_at), json.dumps(state_snapshot)),
        )
        self._conn.commit()
        return WakeRecord(id=wake_id, trigger=trigger, occurred_at=occurred_at, state_snapshot=state_snapshot)

    def append_turn(self, wake_id: str, message: Message) -> str:
        self._require_open(wake_id)
        idx = self._conn.execute("SELECT COUNT(*) AS n FROM turns WHERE wake_id = ?", (wake_id,)).fetchone()[
            "n"
        ]
        source_id = f"wake/{wake_id}/turn/{idx}"
        self._conn.execute(
            "INSERT INTO turns (wake_id, idx, source_id, message) VALUES (?, ?, ?, ?)",
            (wake_id, idx, source_id, json.dumps(message.to_dict())),
        )
        self._conn.commit()
        return source_id

    def record_retrieval(self, wake_id: str, memory_ids: Sequence[str]) -> None:
        """Record what the model was actually looking at. Appends, deduplicates,
        preserves first-seen order; only valid while the wake is open."""
        row = self._require_open(wake_id)
        seen: list[str] = json.loads(row["retrieved_memory_ids"])
        for mid in memory_ids:
            if mid not in seen:
                seen.append(mid)
        self._conn.execute(
            "UPDATE wakes SET retrieved_memory_ids = ? WHERE id = ? AND closed = 0",
            (json.dumps(seen), wake_id),
        )
        self._conn.commit()

    def close_wake(
        self,
        wake_id: str,
        decision: Decision | None,
        stop_reason: str,
        budget: BudgetUsed,
        *,
        state_after: dict[str, Any] | None = None,
        successor_requested: bool = False,
    ) -> WakeRecord:
        self._require_open(wake_id)
        self._conn.execute(
            "UPDATE wakes SET decision = ?, stop_reason = ?, budget_used = ?, state_after = ?,"
            " successor_requested = ?, closed = 1 WHERE id = ? AND closed = 0",
            (
                decision.value if decision is not None else None,
                stop_reason,
                json.dumps(budget.to_dict()),
                json.dumps(state_after) if state_after is not None else None,
                1 if successor_requested else 0,
                wake_id,
            ),
        )
        self._conn.commit()
        return self.get(wake_id)

    def get(self, wake_id: str) -> WakeRecord:
        row = self._conn.execute("SELECT * FROM wakes WHERE id = ?", (wake_id,)).fetchone()
        if row is None:
            raise KeyError(f"no wake {wake_id!r}")
        return self._to_record(row)

    def replay(self) -> Iterator[WakeRecord]:
        """Every wake in commit order, with its turns — the input to rebuild."""
        for row in self._conn.execute("SELECT * FROM wakes ORDER BY seq"):
            yield self._to_record(row)

    def last_wake_at(self, kind: TriggerKind) -> datetime | None:
        row = self._conn.execute(
            "SELECT occurred_at FROM wakes WHERE trigger_kind = ? ORDER BY seq DESC LIMIT 1",
            (kind.value,),
        ).fetchone()
        return parse_iso(row["occurred_at"]) if row else None

    def turns_since_last(self, kind: TriggerKind) -> int:
        """Interaction turns accumulated since the most recent wake of `kind` —
        the reflection trigger's cadence input."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM turns t JOIN wakes w ON w.id = t.wake_id"
            " WHERE w.trigger_kind = ? AND w.seq >"
            " COALESCE((SELECT MAX(seq) FROM wakes WHERE trigger_kind = ?), -1)",
            (TriggerKind.INTERACTION.value, kind.value),
        ).fetchone()
        return int(row["n"])

    def _require_open(self, wake_id: str) -> sqlite3.Row:
        row = self._conn.execute("SELECT * FROM wakes WHERE id = ?", (wake_id,)).fetchone()
        if row is None:
            raise KeyError(f"no wake {wake_id!r}")
        if row["closed"]:
            raise ValueError(f"wake {wake_id!r} is closed; history is append-only")
        return row

    def _to_record(self, row: sqlite3.Row) -> WakeRecord:
        turns = tuple(
            Turn(source_id=t["source_id"], message=Message.from_dict(json.loads(t["message"])))
            for t in self._conn.execute("SELECT * FROM turns WHERE wake_id = ? ORDER BY idx", (row["id"],))
        )
        return WakeRecord(
            id=row["id"],
            trigger=Trigger(TriggerKind(row["trigger_kind"]), row["trigger_payload"]),
            occurred_at=parse_iso(row["occurred_at"]),
            state_snapshot=json.loads(row["state_snapshot"]),
            retrieved_memory_ids=tuple(json.loads(row["retrieved_memory_ids"])),
            turns=turns,
            decision=Decision(row["decision"]) if row["decision"] else None,
            stop_reason=row["stop_reason"],
            budget_used=BudgetUsed.from_dict(json.loads(row["budget_used"])) if row["budget_used"] else None,
            state_after=json.loads(row["state_after"]) if row["state_after"] else None,
            successor_requested=bool(row["successor_requested"]),
        )
