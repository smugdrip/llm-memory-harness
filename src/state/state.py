"""CurrentState and its store.

A small object injected on every inference without a search. Frozen: the update step
takes the prior state and returns a new value, which is what makes "an entry that stops
being true is dropped" a visible diff instead of an absence nobody notices. Bounded: it
is in every prompt, so growth costs tokens on every request. Each entry cites the
memory ids behind it — provenance applies here too.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from typing import Any

LISTS = ("active_projects", "priorities", "open_questions", "recent_focus")


@dataclass(frozen=True)
class Entry:
    text: str
    memory_ids: tuple[str, ...] = ()
    due: str | None = None  # ISO date; meaningful on open_questions only

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"text": self.text, "memory_ids": list(self.memory_ids)}
        if self.due:
            d["due"] = self.due
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Entry:
        return cls(
            text=str(d.get("text", "")).strip(),
            memory_ids=tuple(str(m) for m in d.get("memory_ids", ()) if m),
            due=str(d["due"]) if d.get("due") else None,
        )


@dataclass(frozen=True)
class CurrentState:
    active_projects: tuple[Entry, ...] = ()
    priorities: tuple[Entry, ...] = ()
    open_questions: tuple[Entry, ...] = ()
    recent_focus: tuple[Entry, ...] = ()

    @classmethod
    def empty(cls) -> CurrentState:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {name: [e.to_dict() for e in getattr(self, name)] for name in LISTS}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CurrentState:
        kwargs = {}
        for name in LISTS:
            entries = []
            for item in d.get(name, ()):
                if isinstance(item, dict):
                    entry = Entry.from_dict(item)
                elif isinstance(item, str):
                    entry = Entry(text=item.strip())
                else:
                    continue
                if entry.text:
                    entries.append(entry)
            kwargs[name] = tuple(entries)
        return cls(**kwargs)

    def clamp(self, max_entries: int = 5, max_chars: int = 200) -> CurrentState:
        def cut(entries: tuple[Entry, ...]) -> tuple[Entry, ...]:
            return tuple(replace(e, text=e.text[:max_chars]) for e in entries[:max_entries])

        return CurrentState(**{name: cut(getattr(self, name)) for name in LISTS})

    def render(self) -> str:
        """The single definition of how state enters a prompt."""
        if all(not getattr(self, name) for name in LISTS):
            return "current state: (empty)"
        lines = ["current state:"]
        for name in LISTS:
            entries: tuple[Entry, ...] = getattr(self, name)
            label = name.replace("_", " ")
            if not entries:
                lines.append(f"  {label}: (none)")
                continue
            lines.append(f"  {label}:")
            for e in entries:
                due = f" (due {e.due})" if e.due else ""
                ids = f" [{', '.join(e.memory_ids)}]" if e.memory_ids else ""
                lines.append(f"  - {e.text}{due}{ids}")
        return "\n".join(lines)


class StateStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS current_state"
            " (id INTEGER PRIMARY KEY CHECK (id = 1), value TEXT NOT NULL)"
        )
        self._conn.commit()

    def load(self) -> CurrentState:
        row = self._conn.execute("SELECT value FROM current_state WHERE id = 1").fetchone()
        if row is None:
            return CurrentState.empty()
        return CurrentState.from_dict(json.loads(row["value"]))

    def save(self, state: CurrentState) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO current_state (id, value) VALUES (1, ?)",
            (json.dumps(state.to_dict()),),
        )
        self._conn.commit()
