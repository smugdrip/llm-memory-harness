"""Typed settings from environment variables (prefix HARNESS_, .env supported).

Model ids are pinned exact strings — never a floating alias — so a provider-side
default change cannot silently alter retrieval behavior or output quality. Nothing
constructs an API client at import time; composition happens in cli.py.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from llm.client import DEFAULT_COMPLETION_MODEL
from llm.embedder import DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL
from runtime.budget import BudgetLimits


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HARNESS_", env_file=".env", extra="ignore")

    completion_model: str = DEFAULT_COMPLETION_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    max_completion_tokens: int = 16_000

    db_path: Path = Path("data/harness.db")

    k: int = 5
    similarity_floor: float = 0.30
    duplicate_threshold: float = 0.90

    max_iterations: int = 8
    max_tokens_per_wake: int = 40_000
    max_tool_calls: int = 16
    max_wall_ms: int = 120_000
    max_successors: int = 2

    cooldown_minutes: int = 30
    autonomous_interval_minutes: int = 1440
    reflection_turn_threshold: int = 20

    state_max_entries: int = 5
    state_max_chars: int = 200

    def budget_limits(self) -> BudgetLimits:
        return BudgetLimits(
            max_iterations=self.max_iterations,
            max_tokens=self.max_tokens_per_wake,
            max_tool_calls=self.max_tool_calls,
            max_wall_ms=self.max_wall_ms,
        )

    @property
    def cooldown(self) -> timedelta:
        return timedelta(minutes=self.cooldown_minutes)

    @property
    def autonomous_interval(self) -> timedelta:
        return timedelta(minutes=self.autonomous_interval_minutes)
