"""Memory: the one object the model talks to, and the only place text is preprocessed.

The tool surface is a binding over the same `search` / `write` / `supersede` the
orchestrator calls directly — one similarity floor, one duplicate check, one
preprocessing call site. Provenance (`source_id`, `origin`) is keyword-only, bound by
the caller, and absent from the tool schemas: a model that could set `source_id` would
control idempotency, rebuild, and every provenance claim in the system.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import structlog

from history.records import (
    Clock,
    MemoryRecord,
    MemoryType,
    SystemClock,
    ToolCall,
    TriggerKind,
    parse_iso,
)
from llm.client import ToolSchema
from llm.embedder import Embedder
from memory.preprocess import PREPROCESS_VERSION, preprocess
from memory.store import MemoryStore

log = structlog.get_logger(__name__)


def deterministic_memory_id(source_id: str, canonical_text: str) -> str:
    """Memory ids are a hash of (source_id, canonical_text): the same tool call replayed
    from history yields the same id. That is what makes rebuild idempotent and lets a
    recorded supersede call resolve its target after a rebuild."""
    digest = hashlib.sha256(f"{source_id}\n{canonical_text}".encode()).hexdigest()
    return f"mem_{digest[:16]}"


class Memory:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        *,
        clock: Clock | None = None,
        k: int = 5,
        similarity_floor: float = 0.3,
        duplicate_threshold: float = 0.9,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self._clock = clock or SystemClock()
        self.k = k
        self.similarity_floor = similarity_floor
        self.duplicate_threshold = duplicate_threshold

    # ------------------------------------------------------------------ search

    def search(
        self, query: str, k: int | None = None, *, type: MemoryType | None = None
    ) -> list[MemoryRecord]:
        """Small ranked set above the similarity floor. Fewer results — including
        none — is a correct outcome; the set is never padded to k."""
        k = k or self.k
        canonical = preprocess(query)
        if not canonical:
            return []
        vector = self.embedder.embed([canonical])[0]
        fetch = k if type is None else k * 4
        scored = self.store.nearest(vector, fetch, self.similarity_floor)
        if type is not None:
            scored = [s for s in scored if s.record.type == type][:k]
        for s in scored:
            if (
                s.record.embedding_model_id != self.embedder.model_id
                or s.record.preprocess_version != PREPROCESS_VERSION
            ):
                log.warning(
                    "memory.search.stale_vector",
                    memory_id=s.record.id,
                    stored_model=s.record.embedding_model_id,
                    stored_preprocess=s.record.preprocess_version,
                    hint="run rebuild --from-history",
                )
        log.info(
            "memory.search",
            query=canonical,
            results=[(s.record.id, round(s.score, 4)) for s in scored],
        )
        return [s.record for s in scored]

    # ------------------------------------------------------------------- write

    def write(
        self,
        text: str,
        type: MemoryType | str,
        *,
        # judgments, in the tool schema
        occurred_at: datetime | None = None,
        importance: float = 0.5,
        entities: Sequence[str] = (),
        # provenance, bound by the caller, absent from the schema
        source_id: str,
        origin: TriggerKind,
    ) -> MemoryRecord | None:
        """Canonicalize, dedupe, embed, store. Returns None when the near-duplicate
        check rejects the candidate — curation is this return value plus a prompt."""
        canonical = preprocess(text)
        if not canonical:
            return None
        memory_id = deterministic_memory_id(source_id, canonical)
        existing = self.store.get(memory_id)
        if existing is not None:
            return existing
        vector = self.embedder.embed([canonical])[0]
        duplicates = self.store.nearest(vector, 1, self.duplicate_threshold)
        if duplicates:
            log.info(
                "memory.write.duplicate",
                candidate=canonical,
                duplicate_of=duplicates[0].record.id,
                score=round(duplicates[0].score, 4),
            )
            return None
        now = self._clock.now()
        record = MemoryRecord(
            id=memory_id,
            canonical_text=canonical,
            raw_text=text,
            source_id=source_id,
            origin=origin,
            type=MemoryType(type),
            entities=tuple(entities),
            importance=min(max(float(importance), 0.0), 1.0),
            occurred_at=occurred_at or now,
            created_at=now,
            supersedes=None,
            superseded_by=None,
            embedding_model_id=self.embedder.model_id,
            embedding_dim=len(vector),
            preprocess_version=PREPROCESS_VERSION,
        )
        self.store.upsert(record, vector)
        log.info("memory.write", memory_id=memory_id, type=record.type.value, source_id=source_id)
        return record

    def supersede(self, memory_id: str, text: str, *, source_id: str, origin: TriggerKind) -> MemoryRecord:
        """Replace a memory: write the corrected/merged row, retire the old one from
        search, keep it for history. The duplicate check is deliberately skipped — a
        replacement is supposed to overlap what it replaces."""
        old = self.store.get(memory_id)
        if old is None:
            raise KeyError(f"no memory {memory_id!r} to supersede")
        canonical = preprocess(text)
        if not canonical:
            raise ValueError("replacement text is empty after preprocessing")
        new_id = deterministic_memory_id(source_id, canonical)
        existing = self.store.get(new_id)
        if existing is not None:  # replayed from history: reapply the mark, write nothing
            if old.superseded_by is None:
                self.store.mark_superseded(memory_id, new_id)
            return existing
        vector = self.embedder.embed([canonical])[0]
        now = self._clock.now()
        record = MemoryRecord(
            id=new_id,
            canonical_text=canonical,
            raw_text=text,
            source_id=source_id,
            origin=origin,
            type=old.type,
            entities=old.entities,
            importance=old.importance,
            occurred_at=old.occurred_at,
            created_at=now,
            supersedes=memory_id,
            superseded_by=None,
            embedding_model_id=self.embedder.model_id,
            embedding_dim=len(vector),
            preprocess_version=PREPROCESS_VERSION,
        )
        self.store.upsert(record, vector)
        self.store.mark_superseded(memory_id, new_id)
        log.info("memory.supersede", old=memory_id, new=new_id, source_id=source_id)
        return record

    def recent(self, n: int = 10) -> list[MemoryRecord]:
        """Newest live memories — the reflection wake's reading list."""
        live = [r for r in self.store.records() if r.superseded_by is None]
        live.sort(key=lambda r: (r.created_at, r.id), reverse=True)
        return live[:n]

    # ------------------------------------------------------------ tool binding

    def tool_schemas(self) -> list[ToolSchema]:
        """Semantic arguments only. source_id and origin are not here, by design."""
        types = [t.value for t in MemoryType]
        return [
            ToolSchema(
                name="memory_search",
                description=(
                    "Search long-term memory. Returns up to k results above a similarity"
                    " floor; zero results is a normal outcome, not an error."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "minimum": 1, "maximum": 10},
                        "type": {"type": "string", "enum": types},
                    },
                    "required": ["query"],
                },
            ),
            ToolSchema(
                name="memory_write",
                description=(
                    "Write one curated memory: a single self-contained factual sentence,"
                    " keeping names, dates, and project terms. Use sparingly — most turns"
                    " produce none, and a near-duplicate of an existing memory is rejected."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "type": {"type": "string", "enum": types},
                        "occurred_at": {
                            "type": "string",
                            "description": "ISO 8601 — when it happened, if not now",
                        },
                        "importance": {"type": "number", "minimum": 0, "maximum": 1},
                        "entities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "type"],
                },
            ),
            ToolSchema(
                name="memory_supersede",
                description=(
                    "Replace an existing memory with corrected or merged text. The old row"
                    " leaves search but is kept; use this to consolidate overlapping"
                    " memories instead of writing new near-duplicates."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["memory_id", "text"],
                },
            ),
        ]

    def dispatch(self, call: ToolCall, *, source_id: str, origin: TriggerKind) -> str:
        """Route a model tool call into the same methods the system calls directly.
        Only semantic arguments are read from the call; provenance comes from the
        keyword arguments the orchestrator binds."""
        try:
            if call.name == "memory_search":
                return self._dispatch_search(call.arguments)
            if call.name == "memory_write":
                return self._dispatch_write(call.arguments, source_id, origin)
            if call.name == "memory_supersede":
                return self._dispatch_supersede(call.arguments, source_id, origin)
            return json.dumps({"error": f"unknown tool {call.name!r}"})
        except (KeyError, ValueError, TypeError) as exc:
            return json.dumps({"error": str(exc)})

    def _dispatch_search(self, args: dict[str, Any]) -> str:
        type_filter = MemoryType(args["type"]) if args.get("type") else None
        records = self.search(str(args["query"]), k=args.get("k"), type=type_filter)
        return json.dumps(
            {
                "results": [
                    {
                        "id": r.id,
                        "type": r.type.value,
                        "text": r.canonical_text,
                        "occurred_at": r.occurred_at.date().isoformat(),
                        "importance": r.importance,
                    }
                    for r in records
                ]
            }
        )

    def _dispatch_write(self, args: dict[str, Any], source_id: str, origin: TriggerKind) -> str:
        occurred_at = parse_iso(str(args["occurred_at"])) if args.get("occurred_at") else None
        record = self.write(
            str(args["text"]),
            MemoryType(args["type"]),
            occurred_at=occurred_at,
            importance=args.get("importance", 0.5),
            entities=tuple(args.get("entities", ())),
            source_id=source_id,
            origin=origin,
        )
        if record is None:
            return json.dumps({"written": None, "reason": "rejected: empty or near-duplicate"})
        return json.dumps({"written": record.id})

    def _dispatch_supersede(self, args: dict[str, Any], source_id: str, origin: TriggerKind) -> str:
        record = self.supersede(str(args["memory_id"]), str(args["text"]), source_id=source_id, origin=origin)
        return json.dumps({"superseded": str(args["memory_id"]), "written": record.id})
