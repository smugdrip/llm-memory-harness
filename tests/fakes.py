"""The four fakes the design calls for: a fake embedder (llm.embedder.HashEmbedder),
a fake clock, a scripted trigger source, and a scripted LLMClient — plus helpers for
building completions."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from itertools import count
from typing import Any

from history.records import Message, ToolCall, Trigger
from llm.client import Completion, ToolSchema, Usage
from state.state import CurrentState

_ids = count(1)


class FakeClock:
    def __init__(self, start: datetime = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, *, seconds: int = 0, minutes: int = 0, hours: int = 0, days: int = 0) -> None:
        self._now += timedelta(seconds=seconds, minutes=minutes, hours=hours, days=days)


def tool_call(name: str, arguments: dict[str, Any], id: str | None = None) -> ToolCall:
    return ToolCall(id=id or f"call_{next(_ids)}", name=name, arguments=arguments)


def completion(*calls: ToolCall, text: str = "", usage: Usage | None = None) -> Completion:
    return Completion(
        message=Message(role="assistant", content=text, tool_calls=tuple(calls)),
        usage=usage or Usage(input_tokens=100, output_tokens=50),
    )


def finish(decision: str = "respond", *, text: str = "", request_successor: bool = False) -> Completion:
    return completion(
        tool_call("finish", {"decision": decision, "request_successor": request_successor}),
        text=text,
    )


def state_json(state: CurrentState) -> Completion:
    """What update_state expects back: the given state echoed as JSON."""
    return completion(text=json.dumps(state.to_dict()))


class ScriptedLLM:
    """Plays a fixed queue of completions (or exceptions) and records every
    complete() call for assertions on what the model actually saw."""

    def __init__(self, *items: Completion | Exception) -> None:
        self.queue: list[Completion | Exception] = list(items)
        self.calls: list[tuple[list[Message], list[ToolSchema]]] = []

    def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> Completion:
        self.calls.append((list(messages), list(tools or [])))
        if not self.queue:
            raise AssertionError("ScriptedLLM exhausted: unexpected extra completion request")
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ScriptedTriggers:
    def __init__(self, *triggers: Trigger) -> None:
        self.queue = list(triggers)

    def next(self) -> Trigger | None:
        return self.queue.pop(0) if self.queue else None
