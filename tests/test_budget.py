"""Budget: per-wake accounting, checked at step boundaries, shared by successors."""

from __future__ import annotations

from fakes import FakeClock
from llm.client import Usage
from runtime.budget import Budget, BudgetLimits


def test_iteration_limit():
    budget = Budget(BudgetLimits(max_iterations=2), FakeClock())
    assert budget.allows()
    budget.charge(Usage(input_tokens=10, output_tokens=10))
    assert budget.allows()
    budget.charge(Usage(input_tokens=10, output_tokens=10))
    assert budget.exceeded() == "budget:iterations"


def test_token_limit():
    budget = Budget(BudgetLimits(max_tokens=100), FakeClock())
    budget.charge(Usage(input_tokens=80, output_tokens=30))
    assert budget.exceeded() == "budget:tokens"


def test_tool_call_limit():
    budget = Budget(BudgetLimits(max_tool_calls=3), FakeClock())
    budget.charge_tool_calls(2)
    assert budget.allows()
    budget.charge_tool_calls(1)
    assert budget.exceeded() == "budget:tool_calls"


def test_wall_clock_limit_uses_injected_clock():
    clock = FakeClock()
    budget = Budget(BudgetLimits(max_wall_ms=60_000), clock)
    assert budget.allows()
    clock.advance(minutes=2)
    assert budget.exceeded() == "budget:wall_clock"
    assert budget.used().ms == 120_000


def test_used_reports_everything():
    clock = FakeClock()
    budget = Budget(BudgetLimits(), clock)
    budget.charge(Usage(input_tokens=100, output_tokens=20, cost_usd=0.01))
    budget.charge_tool_calls(2)
    clock.advance(seconds=1)
    used = budget.used()
    assert (used.iterations, used.tokens, used.tool_calls, used.ms) == (1, 120, 2, 1000)


def test_one_allowance_across_a_successor_chain():
    # A granted successor spends from the same object; nothing resets (invariant 15).
    budget = Budget(BudgetLimits(max_iterations=3), FakeClock())
    budget.charge(Usage())  # wake one
    budget.charge(Usage())  # wake one, second iteration
    budget.charge(Usage())  # successor's first iteration exhausts the shared allowance
    assert budget.exceeded() == "budget:iterations"
