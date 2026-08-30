from __future__ import annotations

import sqlite3

import pytest
import structlog

from fakes import FakeClock
from history.history import History
from llm.embedder import HashEmbedder
from memory.memory import Memory
from memory.store import SqliteStore
from runtime.orchestrator import Orchestrator
from state.state import StateStore

structlog.configure(logger_factory=structlog.ReturnLoggerFactory())


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def store(conn) -> SqliteStore:
    return SqliteStore(conn)


@pytest.fixture
def memory(store, embedder, clock) -> Memory:
    return Memory(store, embedder, clock=clock, similarity_floor=0.1, duplicate_threshold=0.9)


@pytest.fixture
def history(conn, clock) -> History:
    return History(conn, clock)


@pytest.fixture
def state_store(conn) -> StateStore:
    return StateStore(conn)


@pytest.fixture
def make_orchestrator(memory, history, state_store, clock):
    def make(llm, **kwargs) -> Orchestrator:
        return Orchestrator(llm=llm, memory=memory, history=history, state=state_store, clock=clock, **kwargs)

    return make
