"""update_state: one function, one model call, a new value out.

The result is committed to history on close_wake (`state_after`), which is what makes
current_state regenerable by rebuild without re-running this model call.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from history.records import Message, WakeRecord
from llm.client import LLMClient
from state.state import CurrentState

log = structlog.get_logger(__name__)

_SYSTEM = """You maintain the `current_state` object of a continuity layer: a small,
bounded summary of what is current, injected into every future inference.

Return ONLY a JSON object with exactly these keys: "active_projects", "priorities",
"open_questions", "recent_focus". Each is a list of at most {max_entries} entries.
Each entry is {{"text": "<one line>", "memory_ids": ["<ids it rests on>"],
"due": "<ISO date, open questions only, optional>"}}.

Rules:
- Start from the prior state and apply what this wake changed.
- Drop an entry that stopped being true. Never append a correction next to it.
- Keep entries that still hold, verbatim, including their memory_ids.
- Cite only memory ids that appear in the prior state or in this wake.
- This is what is *current*, not a log of what happened."""

_TRUNCATE = 500


def update_state(
    llm: LLMClient,
    prior: CurrentState,
    wake: WakeRecord,
    *,
    max_entries: int = 5,
    max_chars: int = 200,
) -> CurrentState:
    messages = [
        Message(role="system", content=_SYSTEM.format(max_entries=max_entries)),
        Message(
            role="user",
            content=(
                f"Prior state:\n{json.dumps(prior.to_dict(), indent=1)}\n\n"
                f"This wake:\n{_render_wake(wake)}\n\nReturn the updated state JSON."
            ),
        ),
    ]
    completion = llm.complete(messages)
    parsed = _parse_json_object(completion.text)
    if parsed is None:
        log.warning("state.update.unparseable", wake_id=wake.id, output=completion.text[:200])
        return prior
    return CurrentState.from_dict(parsed).clamp(max_entries, max_chars)


def _render_wake(wake: WakeRecord) -> str:
    lines = [f"trigger: {wake.trigger.kind.value}"]
    if wake.trigger.payload:
        lines.append(f"payload: {wake.trigger.payload[:_TRUNCATE]}")
    for turn in wake.turns:
        message = turn.message
        if message.content:
            lines.append(f"{message.role}: {message.content[:_TRUNCATE]}")
        for call in message.tool_calls:
            lines.append(f"{message.role} called {call.name}({json.dumps(call.arguments)[:_TRUNCATE]})")
        for result in message.tool_results:
            lines.append(f"tool result: {result.content[:_TRUNCATE]}")
    return "\n".join(lines)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start == -1:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
