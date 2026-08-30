"""Eval harness: recall@k and MRR against Memory.search() directly.

The committed eval set under evals/ is what justifies retrieval changes (invariant 11)
— never inspection. Scoring follows supersession chains: a consolidated memory that
supersedes the expected row still counts as a hit, which is what lets the drift check
tell consolidation (recall flat) from editorializing (recall falls).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from history.records import MemoryType, TriggerKind, parse_iso
from memory.memory import Memory
from memory.store import MemoryStore


@dataclass(frozen=True)
class CorpusItem:
    key: str
    text: str
    type: str
    occurred_at: str | None = None
    importance: float = 0.5
    entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalQuery:
    query: str
    expected_keys: tuple[str, ...]
    set: str = "dev"  # "dev" | "holdout"


@dataclass
class QueryResult:
    query: str
    expected_ids: tuple[str, ...]
    returned_ids: tuple[str, ...]
    first_hit_rank: int | None  # 1-based


@dataclass
class EvalReport:
    k: int
    recall_at_k: float
    mrr: float
    results: list[QueryResult] = field(default_factory=list)


def load_corpus(path: Path) -> list[CorpusItem]:
    items = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        items.append(
            CorpusItem(
                key=d["key"],
                text=d["text"],
                type=d["type"],
                occurred_at=d.get("occurred_at"),
                importance=d.get("importance", 0.5),
                entities=tuple(d.get("entities", ())),
            )
        )
    return items


def load_queries(path: Path) -> list[EvalQuery]:
    queries = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        queries.append(
            EvalQuery(
                query=d["query"],
                expected_keys=tuple(d["expected_keys"]),
                set=d.get("set", "dev"),
            )
        )
    return queries


def seed_corpus(memory: Memory, corpus: list[CorpusItem]) -> dict[str, str]:
    """Write the frozen corpus through memory.write() — the same path everything else
    uses — and return key -> memory id. Expected ids are resolved through this map so
    the committed eval set survives id-scheme and preprocessing changes."""
    key_to_id: dict[str, str] = {}
    for item in corpus:
        record = memory.write(
            item.text,
            MemoryType(item.type),
            occurred_at=parse_iso(item.occurred_at) if item.occurred_at else None,
            importance=item.importance,
            entities=item.entities,
            source_id=f"eval/corpus/{item.key}",
            origin=TriggerKind.INTERACTION,
        )
        if record is None:
            raise ValueError(f"corpus item {item.key!r} was rejected — near-duplicate inside the corpus?")
        key_to_id[item.key] = record.id
    return key_to_id


def ancestor_ids(store: MemoryStore, memory_id: str) -> set[str]:
    """The id plus everything it (transitively) supersedes."""
    ids = {memory_id}
    current = store.get(memory_id)
    while current is not None and current.supersedes and current.supersedes not in ids:
        ids.add(current.supersedes)
        current = store.get(current.supersedes)
    return ids


def run_eval(
    memory: Memory, queries: list[EvalQuery], key_to_id: dict[str, str], *, k: int = 5
) -> EvalReport:
    hits = 0
    reciprocal_ranks: list[float] = []
    results: list[QueryResult] = []
    for q in queries:
        expected = tuple(key_to_id[key] for key in q.expected_keys)
        returned = memory.search(q.query, k=k)
        first_hit = None
        for rank, record in enumerate(returned, start=1):
            covered = ancestor_ids(memory.store, record.id)
            if any(e in covered for e in expected):
                first_hit = rank
                break
        if first_hit is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / first_hit)
        else:
            reciprocal_ranks.append(0.0)
        results.append(
            QueryResult(
                query=q.query,
                expected_ids=expected,
                returned_ids=tuple(r.id for r in returned),
                first_hit_rank=first_hit,
            )
        )
    n = len(queries) or 1
    return EvalReport(
        k=k,
        recall_at_k=hits / n,
        mrr=sum(reciprocal_ranks) / n,
        results=results,
    )


def reflection_ratio(store: MemoryStore) -> float:
    """Reflection-written share of live memories — the earliest signal that the store
    is filling with the system's commentary on itself (invariant 17's counter)."""
    live = [r for r in store.records() if r.superseded_by is None]
    if not live:
        return 0.0
    reflective = sum(1 for r in live if r.origin is TriggerKind.REFLECTION)
    return reflective / len(live)
