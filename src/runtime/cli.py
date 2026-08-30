"""CLI entry point, and the composition root: every dependency is constructed here and
passed in, nothing at import time.

The runtime shape is one-shot (open decision 5): each command runs one thing and exits,
so cron or a systemd timer owns the clock. That makes invariant 14 structural — no
process is alive for a cycle to keep itself running in.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import structlog

from history.history import History
from history.records import SystemClock, Trigger, TriggerKind
from llm.client import LLMClient
from llm.embedder import Embedder, HashEmbedder, LiteLLMEmbedder
from memory.memory import Memory
from memory.store import SqliteStore
from runtime.config import Settings
from runtime.evals import load_corpus, load_queries, reflection_ratio, run_eval, seed_corpus
from runtime.orchestrator import Orchestrator
from runtime.rebuild import rebuild
from runtime.triggers import AutonomousDue, FirstDue, ReflectionDue
from state.state import StateStore

SCHEMA_VERSION = 1


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
    elif row["version"] != SCHEMA_VERSION:
        raise SystemExit(
            f"{path} has schema v{row['version']}, this code expects v{SCHEMA_VERSION};"
            " no migration exists yet"
        )
    return conn


def build_runtime(settings: Settings, conn: sqlite3.Connection, *, embedder: Embedder | None = None):
    clock = SystemClock()
    llm = LLMClient(settings.completion_model, max_tokens=settings.max_completion_tokens)
    embedder = embedder or LiteLLMEmbedder(settings.embedding_model, settings.embedding_dim)
    memory = Memory(
        SqliteStore(conn),
        embedder,
        clock=clock,
        k=settings.k,
        similarity_floor=settings.similarity_floor,
        duplicate_threshold=settings.duplicate_threshold,
    )
    history = History(conn, clock)
    state_store = StateStore(conn)
    orchestrator = Orchestrator(
        llm=llm,
        memory=memory,
        history=history,
        state=state_store,
        clock=clock,
        limits=settings.budget_limits(),
        max_successors=settings.max_successors,
        state_max_entries=settings.state_max_entries,
        state_max_chars=settings.state_max_chars,
    )
    return orchestrator, memory, history, state_store, clock


def _cmd_chat(args: argparse.Namespace, settings: Settings) -> int:
    conn = connect(settings.db_path)
    orchestrator, *_ = build_runtime(settings, conn)
    records = orchestrator.run(Trigger(TriggerKind.INTERACTION, payload=args.message))
    record = records[0]
    texts = (
        t.message.content
        for t in reversed(record.turns)
        if t.message.role == "assistant" and t.message.content
    )
    reply = next(texts, "(no text output)")
    print(reply)
    print(f"\n[{record.id} decision={record.decision} stop={record.stop_reason}]", file=sys.stderr)
    return 0


def _cmd_wake(args: argparse.Namespace, settings: Settings) -> int:
    conn = connect(settings.db_path)
    orchestrator, _memory, history, state_store, clock = build_runtime(settings, conn)
    if args.trigger:
        trigger = Trigger(TriggerKind(args.trigger), payload="manual wake")
    else:
        sources = FirstDue(
            [
                AutonomousDue(
                    history,
                    state_store,
                    clock,
                    cooldown=settings.cooldown,
                    interval=settings.autonomous_interval,
                ),
                ReflectionDue(
                    history,
                    clock,
                    turn_threshold=settings.reflection_turn_threshold,
                    cooldown=settings.cooldown,
                ),
            ]
        )
        maybe = sources.next()
        if maybe is None:
            print("nothing due")
            return 0
        trigger = maybe
    for record in orchestrator.run(trigger):
        print(
            f"{record.id} trigger={record.trigger.kind} decision={record.decision}"
            f" stop={record.stop_reason} budget={record.budget_used.to_dict() if record.budget_used else {}}"
        )
    return 0


def _cmd_rebuild(args: argparse.Namespace, settings: Settings) -> int:
    if not args.from_history:
        print("refusing: rebuild only works --from-history", file=sys.stderr)
        return 2
    conn = connect(settings.db_path)
    _, memory, history, state_store, _clock = build_runtime(settings, conn)
    store = memory.store
    assert isinstance(store, SqliteStore)
    store.clear()
    report = rebuild(history, memory, state_store)
    print(
        f"rebuilt from {report.wakes} wakes: {report.writes} writes,"
        f" {report.supersedes} supersedes, {report.skipped} skipped,"
        f" state_restored={report.state_restored}, errors={len(report.errors)}"
    )
    for error in report.errors:
        print(f"  error: {error}", file=sys.stderr)
    return 1 if report.errors else 0


def _cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    # Seeds an in-memory store — the eval never touches the data directory.
    conn = sqlite3.connect(":memory:")
    embedder: Embedder
    if args.real_embedder:
        embedder = LiteLLMEmbedder(settings.embedding_model, settings.embedding_dim)
    else:
        embedder = HashEmbedder()
    memory = Memory(
        SqliteStore(conn),
        embedder,
        k=settings.k,
        similarity_floor=settings.similarity_floor,
        duplicate_threshold=settings.duplicate_threshold,
    )
    corpus = load_corpus(Path(args.corpus))
    queries = [q for q in load_queries(Path(args.queries)) if args.set in ("all", q.set)]
    key_to_id = seed_corpus(memory, corpus)
    report = run_eval(memory, queries, key_to_id, k=settings.k)
    print(f"recall@{report.k}: {report.recall_at_k:.3f}   MRR: {report.mrr:.3f}   n={len(report.results)}")
    print(f"reflection ratio: {reflection_ratio(memory.store):.3f}")
    for r in report.results:
        if r.first_hit_rank is None:
            print(f"  MISS: {r.query}")
    return 0


def _cmd_state(_args: argparse.Namespace, settings: Settings) -> int:
    conn = connect(settings.db_path)
    print(json.dumps(StateStore(conn).load().to_dict(), indent=2))
    return 0


def _cmd_log(args: argparse.Namespace, settings: Settings) -> int:
    conn = connect(settings.db_path)
    history = History(conn)
    records = list(history.replay())[-args.n :]
    for r in records:
        used = r.budget_used.to_dict() if r.budget_used else {}
        print(
            f"{r.occurred_at.isoformat()} {r.id} trigger={r.trigger.kind}"
            f" decision={r.decision} stop={r.stop_reason} tokens={used.get('tokens', 0)}"
            f" retrieved={len(r.retrieved_memory_ids)}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
    )
    parser = argparse.ArgumentParser(prog="harness", description="LLM memory harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="one interaction wake")
    p_chat.add_argument("message")
    p_chat.set_defaults(fn=_cmd_chat)

    p_wake = sub.add_parser("wake", help="run whatever wake is due (for cron); or force one")
    p_wake.add_argument("--trigger", choices=["autonomous", "reflection"], default=None)
    p_wake.set_defaults(fn=_cmd_wake)

    p_rebuild = sub.add_parser("rebuild", help="regenerate memory and state from history")
    p_rebuild.add_argument("--from-history", action="store_true", dest="from_history")
    p_rebuild.set_defaults(fn=_cmd_rebuild)

    p_eval = sub.add_parser("eval", help="recall@k / MRR over the committed eval set")
    p_eval.add_argument("--corpus", default="evals/corpus.jsonl")
    p_eval.add_argument("--queries", default="evals/queries.jsonl")
    p_eval.add_argument("--set", choices=["dev", "holdout", "all"], default="all")
    p_eval.add_argument("--real-embedder", action="store_true")
    p_eval.set_defaults(fn=_cmd_eval)

    p_state = sub.add_parser("state", help="print current_state")
    p_state.set_defaults(fn=_cmd_state)

    p_log = sub.add_parser("log", help="list recent wakes")
    p_log.add_argument("-n", type=int, default=20)
    p_log.set_defaults(fn=_cmd_log)

    args = parser.parse_args(argv)
    settings = Settings()
    return args.fn(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
