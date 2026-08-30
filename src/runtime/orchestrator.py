"""The orchestrator: one state machine, three triggers, and the only loop in the
system. It owns the cycle, the budget, and the append-turn-before-dispatch ordering.

The triggers differ only in what starts the cycle and how the decide prompt is framed;
budgets, consolidation, and the state write are implemented once, here, never per
trigger (invariant 13).
"""

from __future__ import annotations

import contextlib
import json

import structlog

from history.history import History
from history.records import (
    Clock,
    Decision,
    Message,
    ToolCall,
    ToolResult,
    Trigger,
    TriggerKind,
    WakeRecord,
)
from llm.client import LLMClient, ToolSchema
from memory.memory import Memory
from runtime.budget import Budget, BudgetLimits
from state.state import CurrentState, StateStore
from state.update import update_state

log = structlog.get_logger(__name__)

FINISH = ToolSchema(
    name="finish",
    description=(
        "End this wake cycle. decision is what this cycle amounted to: 'respond' (answered"
        " the user), 'act' (wrote or superseded memories), 'reflect' (consolidated existing"
        " memories), or 'sleep' (nothing worth doing — a correct and common outcome that"
        " writes nothing). Set request_successor=true only if another cycle soon would"
        " genuinely help; the orchestrator, not you, decides whether one runs."
    ),
    parameters={
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": [d.value for d in Decision]},
            "request_successor": {"type": "boolean"},
        },
        "required": ["decision"],
    },
)

_SYSTEM_COMMON = """You are the wake cycle of a continuity layer: an assistant with durable memory
across sessions. Your tools: memory_search, memory_write, memory_supersede, finish.

Memory rules:
- Write sparingly. Most turns produce no memories; more than two from one turn is almost always too many.
- One self-contained factual sentence per memory, keeping names, dates, and project terms.
- Never write a `preference` memory from a single remark — only from evidence repeated across sessions.
- If a candidate restates an existing memory, supersede that memory or write nothing.
- Doing nothing is a correct outcome. Always end the cycle by calling finish.
"""

_SYSTEM_BY_TRIGGER = {
    TriggerKind.INTERACTION: _SYSTEM_COMMON
    + """
A user message woke you. Answer it using the current state and retrieved memories in your
context; call memory_search if you need more. After answering, write at most the few
durable facts this exchange produced, then call finish with decision "respond".""",
    TriggerKind.REFLECTION: _SYSTEM_COMMON
    + """
A reflection wake — no user is present. Read the retrieved memories: your job is
consolidation. Merge overlapping rows and correct stale ones with memory_supersede.
Prefer supersession over new claims — a reflection whose output is all new rows has
added commentary, not memory. Call finish with decision "reflect", or "sleep" if
nothing needs consolidating.""",
    TriggerKind.AUTONOMOUS: _SYSTEM_COMMON
    + """
An autonomous wake — no user is present, and nothing may need doing; "sleep" is usually
the right decision and choosing it is correct behavior, not failure. Act only on
something concrete, such as a due open question named in the wake reason. If you do
act, call finish with decision "act".""",
}


class Orchestrator:
    def __init__(
        self,
        *,
        llm: LLMClient,
        memory: Memory,
        history: History,
        state: StateStore,
        clock: Clock,
        limits: BudgetLimits | None = None,
        max_successors: int = 2,
        state_max_entries: int = 5,
        state_max_chars: int = 200,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.history = history
        self.state = state
        self.clock = clock
        self.limits = limits or BudgetLimits()
        self.max_successors = max_successors
        self._state_max_entries = state_max_entries
        self._state_max_chars = state_max_chars

    def run(self, trigger: Trigger) -> list[WakeRecord]:
        """One wake, plus any granted successors — which spend from the same allowance."""
        budget = Budget(self.limits, self.clock)
        records = [self.wake(trigger, budget=budget)]
        while (
            records[-1].successor_requested
            and trigger.kind is not TriggerKind.INTERACTION
            and len(records) <= self.max_successors
            and budget.allows()
        ):
            successor = Trigger(trigger.kind, payload="granted successor cycle")
            records.append(self.wake(successor, budget=budget))
        return records

    def wake(self, trigger: Trigger, *, budget: Budget | None = None) -> WakeRecord:
        state = self.state.load()
        # Appending to history is the commit point and happens first; everything
        # after this line is derived, retryable, and reproducible by rebuild.
        record = self.history.open_wake(trigger, state.to_dict())
        wake_log = log.bind(wake_id=record.id, trigger=trigger.kind.value)
        budget = budget or Budget(self.limits, self.clock)
        if trigger.kind is TriggerKind.INTERACTION and trigger.payload:
            # The record holds both halves of the exchange; the user half goes in now.
            self.history.append_turn(record.id, Message(role="user", content=trigger.payload))
        retrieved = self._observe(trigger, state)
        if retrieved:
            self.history.record_retrieval(record.id, [r.id for r in retrieved])
        tools = [*self.memory.tool_schemas(), FINISH]
        messages = self._build_context(trigger, state, retrieved)
        decision: Decision | None = None
        successor = False
        stop_reason = "completed"

        try:
            while True:
                over = budget.exceeded()
                if over is not None:
                    stop_reason = over
                    wake_log.info("wake.budget_exhausted", reason=over)
                    break
                completion = self.llm.complete(messages, tools=tools)
                messages.append(completion.message)
                # The turn is committed before its tool calls execute (invariant 23):
                # a memory written below always has a source a rebuild can replay.
                source_id = self.history.append_turn(record.id, completion.message)
                budget.charge(completion.usage)

                memory_calls = [c for c in completion.tool_calls if c.name != FINISH.name]
                finish_call = next((c for c in completion.tool_calls if c.name == FINISH.name), None)

                if memory_calls:
                    budget.charge_tool_calls(len(memory_calls))
                    results = []
                    for call in memory_calls:
                        output = self.memory.dispatch(call, source_id=source_id, origin=trigger.kind)
                        results.append(ToolResult(tool_call_id=call.id, content=output))
                        self._note_search_results(record.id, call, output)
                    tool_message = Message(role="tool", tool_results=tuple(results))
                    messages.append(tool_message)  # all results, one message
                    self.history.append_turn(record.id, tool_message)

                if finish_call is not None:
                    decision = _parse_decision(finish_call, wake_log)
                    successor = bool(finish_call.arguments.get("request_successor", False))
                    break
                if not completion.tool_calls:
                    # Plain text and no finish call: the output is the act. Sleep stays
                    # explicit — it is a tool call, never inferred from silence.
                    decision = Decision.RESPOND
                    break

            if decision is Decision.SLEEP:
                # Early return before the state write: nothing derived is written, and
                # the committed record is the wake's entire effect (invariant 16).
                closed = self.history.close_wake(
                    record.id, Decision.SLEEP, "idle", budget.used(), successor_requested=successor
                )
                wake_log.info("wake.closed", decision="sleep", noop=True)
                return closed

            state_after = None
            if decision is not None and budget.allows():
                new_state = update_state(
                    self.llm,
                    state,
                    self.history.get(record.id),
                    max_entries=self._state_max_entries,
                    max_chars=self._state_max_chars,
                )
                self.state.save(new_state)
                state_after = new_state.to_dict()
            closed = self.history.close_wake(
                record.id,
                decision,
                stop_reason,
                budget.used(),
                state_after=state_after,
                successor_requested=successor,
            )
            wake_log.info(
                "wake.closed",
                decision=decision.value if decision else None,
                stop_reason=stop_reason,
                budget=budget.used().to_dict(),
                retrieved=list(closed.retrieved_memory_ids),
            )
            return closed
        except Exception as exc:
            # "Ran out of budget" and "crashed" share a recovery: the record is there
            # and rebuild re-derives the rest. Close it, then let the error surface.
            with contextlib.suppress(Exception):
                self.history.close_wake(record.id, decision, f"error:{type(exc).__name__}", budget.used())
            wake_log.error("wake.error", error=repr(exc))
            raise

    def _observe(self, trigger: Trigger, state: CurrentState) -> list:
        if trigger.kind is TriggerKind.INTERACTION and trigger.payload:
            return self.memory.search(trigger.payload)
        if trigger.kind is TriggerKind.REFLECTION:
            return self.memory.recent(self.memory.k * 2)
        if trigger.kind is TriggerKind.AUTONOMOUS and (trigger.payload or "").startswith("due open question"):
            return self.memory.search(trigger.payload or "")
        return []

    def _build_context(self, trigger: Trigger, state: CurrentState, retrieved: list) -> list[Message]:
        if trigger.kind is TriggerKind.INTERACTION:
            opening = trigger.payload or ""
        else:
            opening = f"wake trigger: {trigger.kind.value}\nreason: {trigger.payload or '(none)'}"
        # The volatile block sits after the turn it was retrieved for, in a user-role
        # message: provider-neutral, and it keeps the stable system prefix cacheable.
        return [
            Message(role="system", content=_SYSTEM_BY_TRIGGER[trigger.kind]),
            Message(role="user", content=opening),
            Message(role="user", content=self._volatile_block(state, retrieved)),
        ]

    def _volatile_block(self, state: CurrentState, retrieved: list) -> str:
        lines = ["[context refreshed for this wake]", state.render(), "", "retrieved memories:"]
        if retrieved:
            lines.extend(
                f"- {r.id} [{r.type.value}] ({r.occurred_at.date().isoformat()}): {r.canonical_text}"
                for r in retrieved
            )
        else:
            lines.append("(none above the similarity floor)")
        return "\n".join(lines)

    def _note_search_results(self, wake_id: str, call: ToolCall, output: str) -> None:
        """Tool searches are part of what the model actually saw; record their ids."""
        if call.name != "memory_search":
            return
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return
        ids = [r["id"] for r in payload.get("results", ()) if isinstance(r, dict) and r.get("id")]
        if ids:
            self.history.record_retrieval(wake_id, ids)


def _parse_decision(finish_call: ToolCall, wake_log) -> Decision:
    raw = finish_call.arguments.get("decision", Decision.SLEEP.value)
    try:
        return Decision(raw)
    except ValueError:
        wake_log.warning("wake.bad_decision", raw=raw)
        return Decision.SLEEP
