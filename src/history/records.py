"""Shared record shapes: the two durable records and the vocabulary they are written in.

WakeRecord is the history record — one per cognitive event, append-only, the same shape
for a user turn and an autonomous wake. MemoryRecord is the long-term memory record —
derived, and fully regenerable from history. The small enums and message types here are
the floor every other package stands on, which is why Clock also lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4


class TriggerKind(StrEnum):
    INTERACTION = "interaction"
    REFLECTION = "reflection"
    AUTONOMOUS = "autonomous"


class Decision(StrEnum):
    RESPOND = "respond"
    ACT = "act"
    REFLECT = "reflect"
    SLEEP = "sleep"


class MemoryType(StrEnum):
    EVENT = "event"
    DECISION = "decision"
    PROJECT = "project"
    RELATIONSHIP = "relationship"
    PREFERENCE = "preference"
    OPEN_QUESTION = "open_question"


@dataclass(frozen=True)
class Trigger:
    """What woke the cycle. `payload` is the user message for an interaction and the
    reason (schedule, due open question) for the other two kinds."""

    kind: TriggerKind
    payload: str | None = None


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(id=d["id"], name=d["name"], arguments=dict(d.get("arguments") or {}))


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"tool_call_id": self.tool_call_id, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolResult:
        return cls(tool_call_id=d["tool_call_id"], content=d["content"])


@dataclass(frozen=True)
class Message:
    """One entry in a wake's transcript. A `tool` message carries every result of one
    round of tool calls together — splitting them is what trains parallel calls away."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_results:
            d["tool_results"] = [r.to_dict() for r in self.tool_results]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            tool_calls=tuple(ToolCall.from_dict(c) for c in d.get("tool_calls", ())),
            tool_results=tuple(ToolResult.from_dict(r) for r in d.get("tool_results", ())),
        )


@dataclass(frozen=True)
class BudgetUsed:
    iterations: int = 0
    tokens: int = 0
    tool_calls: int = 0
    ms: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "iterations": self.iterations,
            "tokens": self.tokens,
            "tool_calls": self.tool_calls,
            "ms": self.ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> BudgetUsed:
        return cls(
            iterations=d.get("iterations", 0),
            tokens=d.get("tokens", 0),
            tool_calls=d.get("tool_calls", 0),
            ms=d.get("ms", 0),
        )


@dataclass(frozen=True)
class Turn:
    """A committed message plus the source_id it was committed under — the id a memory
    written by this turn's tool calls is bound to, and the id replay binds again."""

    source_id: str
    message: Message


@dataclass(frozen=True)
class WakeRecord:
    id: str
    trigger: Trigger
    occurred_at: datetime
    state_snapshot: dict[str, Any]
    retrieved_memory_ids: tuple[str, ...] = ()
    turns: tuple[Turn, ...] = ()
    decision: Decision | None = None
    stop_reason: str | None = None
    budget_used: BudgetUsed | None = None
    # The committed output of the state-update step. current_state must be regenerable
    # from history without a model call, and this field is what replay reads back.
    state_after: dict[str, Any] | None = None
    successor_requested: bool = False

    @property
    def messages(self) -> list[Message]:
        return [t.message for t in self.turns]


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    canonical_text: str
    raw_text: str
    source_id: str
    origin: TriggerKind
    type: MemoryType
    entities: tuple[str, ...] = ()
    importance: float = 0.5
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    supersedes: str | None = None
    superseded_by: str | None = None
    embedding_model_id: str = ""
    embedding_dim: int = 0
    preprocess_version: int = 0


def new_wake_id() -> str:
    return f"wake_{uuid4().hex[:12]}"


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
