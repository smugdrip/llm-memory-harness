# Architecture

The object model. `README.md` is the design and `../CLAUDE.md` holds the rules; this file says which
objects exist, what each one promises, and how they compose into a wake cycle. Signatures below are
contracts, not implementations — the point is the seams, not the bodies.

Nine types carry the system. Five are concrete, four are protocols, and the protocols exist only where a
swap is genuinely expected. Everything else is a function or a dataclass.

```text
Orchestrator ── owns the cycle, the budget, and the only loop in the system
│
├── Clock            (protocol)  now()
├── TriggerSource    (protocol)  next() -> Trigger | None
├── LLMClient                    complete()                   <- litellm lives here
├── History                      open_wake / append_turn / close_wake / replay
├── StateStore                   load() / save()
└── Memory                       search / write / supersede + the tool binding
     ├── Embedder    (protocol)  embed()
     └── MemoryStore (protocol)  upsert / nearest / mark_superseded
```

The model reaches exactly one of these: `Memory`, through tool calls. It never sees the store, the
embedder, or the history.

## LLMClient

```python
class LLMClient:
    """Completions. The only module that imports litellm or names a model."""

    def __init__(self, model: str, *, max_tokens: int = 16_000) -> None: ...

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> Completion: ...
```

`Completion` is ours, not litellm's: text, `tool_calls`, `usage`, `stop_reason`, and cost. The moment a
litellm response object reaches the orchestrator, litellm has stopped being a dependency and become the
interface, and swapping it means touching every caller. One translation, one place.

`model` is a pinned `provider/model` string — `anthropic/claude-opus-5`, never a floating alias and never
a date-suffixed variant. Provider-specific request parameters go through litellm's passthrough and stay
inside this class.

## Embedder

```python
class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, texts: list[str]) -> list[Vector]: ...
```

Separate from `LLMClient` even though litellm can serve both, because the two change on different clocks.
A completion model can be swapped between one wake and the next with no consequence. An embedding model
cannot be swapped at all without `rebuild --from-history`, which is why `model_id` and `dim` are on the
contract — they are what a stored vector records so a mismatch is detectable instead of silently
returning bad neighbors.

Implementations: litellm, or a local `sentence-transformers` model. Tests inject a fake that hashes to a
deterministic vector.

## MemoryStore

```python
class MemoryStore(Protocol):
    def upsert(self, record: MemoryRecord, vector: Vector) -> None: ...
    def by_source_id(self, source_id: str) -> MemoryRecord | None: ...
    def nearest(self, vector: Vector, k: int, floor: float) -> list[Scored]: ...
    def mark_superseded(self, memory_id: str, by: str) -> None: ...
    def records(self) -> Iterator[MemoryRecord]: ...
```

`by_source_id` is the whole of idempotency: a rebuild replays history, and without a cheap lookup on
`source_id` it writes every memory a second time. `mark_superseded` rather than a delete. `records()` is
for rebuild and for the eval harness.

## Memory

The object the model talks to, and the only object that preprocesses text.

```python
class Memory:
    def __init__(self, store: MemoryStore, embedder: Embedder) -> None: ...

    def search(self, query: str, k: int = 5) -> list[MemoryRecord]: ...

    def write(self, text: str, type: MemoryType,
              *,
              # judgments, in the tool schema
              occurred_at: datetime | None = None,
              importance: float = 0.5,
              entities: Sequence[str] = (),
              # provenance, bound by the caller, absent from the schema
              source_id: str, origin: Trigger) -> MemoryRecord | None: ...

    def supersede(self, memory_id: str, text: str, *,
                  source_id: str, origin: Trigger) -> MemoryRecord: ...

    def tool_schemas(self) -> list[ToolSchema]: ...
    def dispatch(self, call: ToolCall, *,
                 source_id: str, origin: Trigger) -> str: ...
```

Three properties are worth more than the signatures.

**The tool surface is a binding, not a second path.** `dispatch` routes into the same `search` and `write`
the orchestrator calls directly. There is one similarity floor, one duplicate check, one preprocessing
call site. A design where the model's search and the system's search are separate implementations is a
design where they drift apart and only one of them is the one the eval measures.

**A memory record has three sources, and the split is the contract.** Derived: `id`, `canonical_text`,
`created_at`, `supersedes` / `superseded_by`, and the three embedding fields. Bound by the orchestrator:
`source_id` and `origin`. Supplied by the model: `type`, `occurred_at`, `importance`, `entities`, and the
`text` that becomes `raw_text`. `occurred_at` and `importance` have to come from the model because nothing
else knows them — `occurred_at` diverges from `created_at` exactly when a conversation is recalling the
past, and importance is a judgment. Leave them off the signature and the record cannot be built.

**Provenance is bound by the caller, never supplied by the model.** `source_id` and `origin` are
keyword-only and absent from `tool_schemas()`, which is the line the split above draws. If the model could set `source_id`, then idempotency,
rebuild, and every provenance claim in the system would be model-controlled, and a confused or adversarial
turn could overwrite an unrelated memory by naming its source.

**`write` returns `None` when the duplicate check rejects the candidate.** Curation is not a separate
class; it is this return value plus a prompt. Most calls should produce a memory, but a wake that
restates something already stored should end with nothing written, and the caller has to be able to see
that happen.

## History

```python
class History:
    def open_wake(self, trigger: Trigger, state: CurrentState) -> WakeRecord: ...
    def append_turn(self, wake_id: str, message: Message) -> str: ...   # -> source_id
    def close_wake(self, wake_id: str, decision: Decision,
                   stop_reason: str, budget: Budget) -> None: ...
    def replay(self) -> Iterator[WakeRecord]: ...
```

`append_turn` returns the `source_id` for the turn it just committed, and that is the id the tool dispatch
binds. This is what keeps the commit-point rule true once the model can write memories mid-cycle: the
assistant turn carrying a `memory_write` call is in history **before** the call executes, so the memory it
produces always has a source that a rebuild can replay. Append the turn, then dispatch its tools — in that
order, every time.

`replay()` is `rebuild --from-history`. It is also why history is the one component with no protocol and
no swap story: there is nothing to swap it for.

## StateStore and CurrentState

```python
class StateStore:
    def load(self) -> CurrentState: ...
    def save(self, state: CurrentState) -> None: ...

@dataclass(frozen=True)
class CurrentState:
    active_projects: list[Entry]
    priorities: list[Entry]
    open_questions: list[Entry]
    recent_focus: list[Entry]

    def render(self) -> str: ...
```

Frozen, because the update step takes the prior state and returns a new one rather than mutating it in
place — that is what makes "an entry that stops being true is dropped" a visible diff instead of an
absence nobody notices. `render()` is the single definition of how state enters a prompt.

## Clock and TriggerSource

```python
class Clock(Protocol):
    def now(self) -> datetime: ...

class TriggerSource(Protocol):
    def next(self) -> Trigger | None: ...
```

Two one-method protocols that exist purely so the loop is testable. Cadence, cooldown and the idle path
are the behaviors most worth asserting and the ones a loop reading the system clock can only demonstrate
by waiting.

## Orchestrator

```python
class Orchestrator:
    def wake(self, trigger: Trigger) -> WakeRecord: ...
```

The whole cycle, and the only loop:

```python
def wake(self, trigger):
    state  = self.state.load()
    record = self.history.open_wake(trigger, state)
    budget = Budget.for_wake(trigger)
    tools  = self.memory.tool_schemas() + [FINISH]
    messages = self.build_context(trigger, state)

    while budget.allows():
        completion = self.llm.complete(messages, tools=tools)
        source_id  = self.history.append_turn(record.id, completion.message)
        budget.charge(completion.usage)

        if completion.finished:
            break

        results = [
            self.memory.dispatch(c, source_id=source_id, origin=trigger.kind)
            for c in completion.tool_calls
        ]
        messages.append(tool_results(results))       # all results, one message

    if completion.decision is Decision.SLEEP:
        return self.history.close_wake(record.id, Decision.SLEEP, "idle", budget)

    self.state.save(update_state(self.llm, state, record))
    return self.history.close_wake(record.id, completion.decision, budget.stop_reason, budget)
```

Three things this shape settles.

**Sleep is a tool call.** The model ends a cycle by calling `finish(decision=...)`, and `sleep` is one of
the values it can pass. Deciding there is nothing worth doing is therefore an act with a record, not an
absence inferred from the model having produced no output — which is the only version of it a test can
tell apart from a broken trigger.

**The sleep branch returns before the state write.** Not an `if` guarding two calls; an early return, so
the idle path cannot grow a state mutation by accident later.

**Every tool result goes back in one message.** Splitting parallel tool results across several messages
trains the model out of making parallel calls at all, and the failure is silent — the calls just stop
coming.

## Composition

```python
llm     = LLMClient(model=settings.completion_model)
memory  = Memory(store=SqliteStore(db), embedder=LiteLLMEmbedder(settings.embedding_model))
runtime = Orchestrator(llm=llm, memory=memory, history=History(db),
                       state=StateStore(db), clock=SystemClock(), budget=settings.budget)

while (trigger := triggers.next()) is not None:
    runtime.wake(trigger)
```

Every dependency is passed in. Nothing constructs a client at import time, which is what keeps the whole
thing testable with four fakes: a fake embedder, a fake clock, a scripted trigger source, and a scripted
`LLMClient`.

## What litellm buys, and what it costs

It buys one client shape across providers, cost accounting the budget object needs anyway
(`completion_cost`), and one tool-call format to parse. It is also not the kind of dependency the design
warns about elsewhere: litellm normalizes a single HTTP call and orchestrates nothing, so the rule against
retrieval frameworks is untouched.

The cost is real and specific, and all of it lands inside `LLMClient`:

- **The prompt-caching approach in `engineering-practices.md` may not survive it.** That guidance puts the
  volatile block — retrieved memories and `current_state` — into `messages[]` as a `{"role": "system"}`
  entry, which is an Anthropic behavior on Opus 5, Opus 4.8 and Fable 5. litellm normalizes to the OpenAI
  message shape, and for Anthropic it hoists system-role messages into the top-level `system` parameter,
  which is precisely what that guidance exists to avoid. **Verify before relying on it.** If it does not
  survive, put the volatile block in a user-role message instead: provider-neutral, same cache behavior,
  and it gives up the prompt-injection-safe operator channel that the system role was also buying.
- **Anthropic-specific request parameters reach the provider only through passthrough** — effort, adaptive
  thinking, `cache_control` breakpoints. Passthrough is the part of litellm most likely to shift under an
  upgrade, so pin the version and keep every passthrough argument in this one class.
- **Usage fields are normalized.** Cache-hit accounting may arrive under an OpenAI-shaped name, or not at
  all. Read whatever the provider gives, expose it on our `Completion`, and let the rest of the system ask
  our field. A zero cache-hit rate still means something upstream is changing on every request.
- **Model ids become `provider/model`.** The pinning rule is unchanged; the string is longer.

## Deliberately not objects

- **`preprocess(text) -> str`**, with a module-level `PREPROCESS_VERSION`. A pure function, called only by
  `Memory`. That single call site is what makes "one path serves writes and queries" structural rather
  than a thing to remember.
- **`update_state(llm, prior, wake) -> CurrentState`**. One function, one model call, a new value out.
- **Curation.** Not a class and not a pass. On the interaction trigger it is the model calling
  `memory_write`; on the reflection trigger it is the same tool under a prompt that asks for
  consolidation. The near-duplicate check lives in `Memory.write`, where both reach it.
- **A tool registry.** Two providers contribute schemas and a dict dispatches. A registry would be a layer
  over a four-entry list.

## Where the eval attaches

Retrieval quality and the model's willingness to search are different things, and a tool-gated design lets
them hide behind each other. Measure them separately: recall@k against `Memory.search()` directly, which
is the number that justifies a retrieval change, and cold-start recovery end to end through
`Orchestrator.wake()`, which is the milestone. A drop in the second with the first flat means the model is
not calling the tool, and no amount of retrieval tuning will move it.

## Layout

```text
src/
├── llm/         client.py (LLMClient), embedder.py (Embedder, LiteLLMEmbedder)
├── memory/      memory.py (Memory), store.py (MemoryStore, SqliteStore), preprocess.py
├── history/     history.py (History), records.py (WakeRecord, MemoryRecord)
├── state/       state.py (CurrentState, StateStore), update.py (update_state)
└── runtime/     orchestrator.py (Orchestrator), triggers.py, budget.py, cli.py
```
