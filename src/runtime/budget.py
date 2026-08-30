"""Per-wake budget accounting: one object threaded through the cycle, decremented at
every model and tool call, checked at step boundaries rather than mid-write.

Budgets are per wake, not per iteration (invariant 15): a granted successor cycle is
handed this same object, so a chain of wakes spends one allowance instead of resetting
it — the loophole that would let the model set the pace while the scheduler nominally
stayed outside it.
"""

from __future__ import annotations

from dataclasses import dataclass

from history.records import BudgetUsed, Clock
from llm.client import Usage


@dataclass(frozen=True)
class BudgetLimits:
    max_iterations: int = 8
    max_tokens: int = 40_000
    max_tool_calls: int = 16
    max_wall_ms: int = 120_000


class Budget:
    def __init__(self, limits: BudgetLimits, clock: Clock) -> None:
        self._limits = limits
        self._clock = clock
        self._started = clock.now()
        self.iterations = 0
        self.tokens = 0
        self.tool_calls = 0
        self.cost_usd = 0.0

    def charge(self, usage: Usage) -> None:
        self.iterations += 1
        self.tokens += usage.total_tokens
        self.cost_usd += usage.cost_usd

    def charge_tool_calls(self, n: int) -> None:
        self.tool_calls += n

    def exceeded(self) -> str | None:
        if self.iterations >= self._limits.max_iterations:
            return "budget:iterations"
        if self.tokens >= self._limits.max_tokens:
            return "budget:tokens"
        if self.tool_calls >= self._limits.max_tool_calls:
            return "budget:tool_calls"
        if self._elapsed_ms() >= self._limits.max_wall_ms:
            return "budget:wall_clock"
        return None

    def allows(self) -> bool:
        return self.exceeded() is None

    def used(self) -> BudgetUsed:
        return BudgetUsed(
            iterations=self.iterations,
            tokens=self.tokens,
            tool_calls=self.tool_calls,
            ms=self._elapsed_ms(),
        )

    def _elapsed_ms(self) -> int:
        return int((self._clock.now() - self._started).total_seconds() * 1000)
