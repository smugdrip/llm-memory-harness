"""LLMClient: completions behind our own Completion type.

With llm.embedder, the only module that imports litellm or names a model. Completion is
ours, not litellm's — the moment a provider response object crosses this boundary,
litellm has stopped being a dependency and become the interface. One translation, one
place: message encoding, tool-call parsing, usage and cost normalization all live here.

`model` is a pinned `provider/model` string (e.g. "anthropic/claude-opus-5"), never a
floating alias and never a date-suffixed variant.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from history.records import Message, ToolCall

# The pinned default, overridable through Settings. Kept here so no module outside
# llm/ names a model (invariant 20).
DEFAULT_COMPLETION_MODEL = "anthropic/claude-opus-5"


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Completion:
    message: Message
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "stop"

    @property
    def text(self) -> str:
        return self.message.content

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.message.tool_calls


def encode_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Our transcript shape → the OpenAI-style shape litellm normalizes from. A single
    tool-results message fans out to one provider message per result here, and only
    here — the internal contract stays 'all results, one message'."""
    encoded: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            encoded.extend(
                {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
                for r in message.tool_results
            )
            continue
        d: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            d["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in message.tool_calls
            ]
        encoded.append(d)
    return encoded


def encode_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def decode_response(response: Any, cost_usd: float) -> Completion:
    choice = response.choices[0]
    raw = choice.message
    calls: list[ToolCall] = []
    for tc in raw.tool_calls or ():
        try:
            arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError, TypeError:
            arguments = {}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    # Cache-hit accounting may arrive under an OpenAI-shaped name or a provider one;
    # read whatever is there and expose it on our field.
    details = getattr(usage, "prompt_tokens_details", None)
    cache_read = getattr(details, "cached_tokens", None) or getattr(usage, "cache_read_input_tokens", 0) or 0
    return Completion(
        message=Message(role="assistant", content=raw.content or "", tool_calls=tuple(calls)),
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cost_usd=cost_usd,
        ),
        stop_reason=choice.finish_reason or "stop",
    )


class LLMClient:
    """Completions. The only module that imports litellm or names a model."""

    def __init__(
        self, model: str, *, max_tokens: int = 16_000, timeout: float = 120.0, retries: int = 3
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._retries = retries

    def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> Completion:
        import litellm

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": encode_messages(messages),
            "max_tokens": self._max_tokens,
            "timeout": self._timeout,
        }
        if tools:
            kwargs["tools"] = encode_tools(tools)
        response = self._with_retries(litellm, kwargs)
        try:
            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:  # cost accounting must never fail a wake
            cost = 0.0
        return decode_response(response, cost)

    def _with_retries(self, litellm, kwargs: dict[str, Any]):
        delay = 1.0
        for attempt in range(self._retries):
            # Most-specific-first: retryable classes are caught; anything else
            # (auth, bad request, APIStatusError) propagates immediately.
            try:
                return litellm.completion(**kwargs)
            except (
                litellm.exceptions.RateLimitError,
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.Timeout,
            ):
                if attempt == self._retries - 1:
                    raise
                time.sleep(delay + random.random())
                delay *= 2
        raise AssertionError("unreachable")
