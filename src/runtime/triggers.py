"""Trigger sources: what may start a wake cycle.

The scheduler lives here, outside the model (invariant 14). A cycle can request a
successor; these sources and the orchestrator's grant policy are the only things that
actually start one. Cooldown is persisted state, not a sleep(): it is derived from the
wall-clock timestamps in immutable history, so a restarted process cannot forget the
cooldown it was in the middle of.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Protocol

from history.history import History
from history.records import Clock, SystemClock, Trigger, TriggerKind
from state.state import StateStore

__all__ = [
    "AutonomousDue",
    "Clock",
    "FirstDue",
    "ReflectionDue",
    "SystemClock",
    "Trigger",
    "TriggerKind",
    "TriggerSource",
]


class TriggerSource(Protocol):
    def next(self) -> Trigger | None: ...


class ReflectionDue:
    """Reflection cadence: interaction turns accumulated since the last reflection
    (open decision 6 — turns, because elapsed time reflects nothing having happened)."""

    def __init__(
        self,
        history: History,
        clock: Clock,
        *,
        turn_threshold: int = 20,
        cooldown: timedelta = timedelta(minutes=30),
    ) -> None:
        self._history = history
        self._clock = clock
        self._turn_threshold = turn_threshold
        self._cooldown = cooldown

    def next(self) -> Trigger | None:
        last = self._history.last_wake_at(TriggerKind.REFLECTION)
        if last is not None and self._clock.now() - last < self._cooldown:
            return None
        turns = self._history.turns_since_last(TriggerKind.REFLECTION)
        if turns >= self._turn_threshold:
            return Trigger(
                TriggerKind.REFLECTION,
                payload=f"{turns} interaction turns since the last reflection",
            )
        return None


class AutonomousDue:
    """Wakes on a due open_question, or on a schedule; never inside the cooldown."""

    def __init__(
        self,
        history: History,
        state_store: StateStore,
        clock: Clock,
        *,
        cooldown: timedelta = timedelta(minutes=30),
        interval: timedelta = timedelta(days=1),
    ) -> None:
        self._history = history
        self._state = state_store
        self._clock = clock
        self._cooldown = cooldown
        self._interval = interval

    def next(self) -> Trigger | None:
        now = self._clock.now()
        last = self._history.last_wake_at(TriggerKind.AUTONOMOUS)
        if last is not None and now - last < self._cooldown:
            return None
        for entry in self._state.load().open_questions:
            if entry.due is None:
                continue
            try:
                due = date.fromisoformat(entry.due)
            except ValueError:
                continue
            if due <= now.date():
                return Trigger(TriggerKind.AUTONOMOUS, payload=f"due open question: {entry.text}")
        if last is None or now - last >= self._interval:
            return Trigger(TriggerKind.AUTONOMOUS, payload="scheduled wake")
        return None


class FirstDue:
    """Composes sources; the first one with something due wins."""

    def __init__(self, sources: list[TriggerSource]) -> None:
        self._sources = sources

    def next(self) -> Trigger | None:
        for source in self._sources:
            trigger = source.next()
            if trigger is not None:
                return trigger
        return None
