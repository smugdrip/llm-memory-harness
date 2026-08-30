"""rebuild --from-history: regenerate the derived layers from immutable history.

Makes no model call. Writes and supersedes are recorded in history as tool calls, so
replay re-executes decisions rather than re-deriving them; supersession chains come
back for free because memory ids are deterministic in (source_id, canonical_text).
current_state comes back from the last committed `state_after`. The embedding backend
is the entire price of a rebuild.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import structlog

from history.history import History
from memory.memory import Memory
from state.state import CurrentState, StateStore

log = structlog.get_logger(__name__)

_REPLAYED_TOOLS = ("memory_write", "memory_supersede")


@dataclass
class RebuildReport:
    wakes: int = 0
    writes: int = 0
    supersedes: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    state_restored: bool = False


def rebuild(history: History, memory: Memory, state_store: StateStore) -> RebuildReport:
    """Replay history into `memory` and `state_store`. The caller decides whether the
    target store starts empty (a regeneration) or not (an idempotent top-up); writes
    are idempotent either way."""
    report = RebuildReport()
    final_state = None
    for wake in history.replay():
        report.wakes += 1
        for turn in wake.turns:
            for call in turn.message.tool_calls:
                if call.name not in _REPLAYED_TOOLS:
                    continue
                output = memory.dispatch(call, source_id=turn.source_id, origin=wake.trigger.kind)
                result = json.loads(output)
                if "error" in result:
                    report.errors.append(f"{turn.source_id} {call.name}: {result['error']}")
                elif "superseded" in result:
                    report.supersedes += 1
                elif result.get("written"):
                    report.writes += 1
                else:
                    report.skipped += 1
        if wake.state_after is not None:
            final_state = wake.state_after
    if final_state is not None:
        state_store.save(CurrentState.from_dict(final_state))
        report.state_restored = True
    log.info(
        "rebuild.done",
        wakes=report.wakes,
        writes=report.writes,
        supersedes=report.supersedes,
        skipped=report.skipped,
        errors=len(report.errors),
    )
    return report
