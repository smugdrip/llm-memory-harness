# AI-Native Memory Harness

A minimal continuity layer for LLMs.

The goal is to give an LLM durable memory across sessions without hard-coding a personality or simulating
a human body: short-term thread context, long-term semantic memory, persistent current state, and
immutable source history — plus a loop over them that can start a cycle without being spoken to.

The rules are in `../CLAUDE.md`; this file is the design behind them.

## MVP

```text
LLM
├── Short-term memory
│   └── current thread / context window
│
├── Long-term memory
│   ├── memory.write()
│   └── memory.search()
│
├── Current state
│   └── active projects, priorities, unresolved questions
│
├── Source history
│   └── immutable log of cognitive events: turns and wakes
│
└── Loop orchestrator                       (a process, not a store)
    └── triggers: interaction · reflection · autonomous
```

Four stores and one process. The orchestrator is listed among them because it is a component with rules of
its own rather than glue, but it holds nothing, and the stores stay separate underneath it.

## Core idea

Every cognitive event is a wake cycle. A user message is one kind of trigger, not the only one.

```text
wake                       trigger: user input | schedule | due open_question
→ open a history record    trigger, current_state snapshot
→ observe                  what is new since the last wake, what is pending
→ retrieve memories        ids recorded on the record
→ decide                   respond · act · reflect · sleep
→ commit the record        the model's output text   ← the commit point
→ consolidate              curate, dedupe, supersede, write memories
→ update current_state
→ sleep
```

The record is committed before anything derived from it is written. That is what makes the last two steps
retryable, and what makes a reflection's conclusions survive a rebuild. On `sleep` those two steps do not
run at all: the committed record is the entire effect of the wake.

The interaction trigger is the familiar path through that cycle, and the one to build first.

```text
new input
→ preprocess search query
→ embed query
→ retrieve relevant memories
→ inject memories + current state
→ LLM responds / acts
→ select what is worth remembering
→ preprocess memory
→ embed + store
→ update current state
```

## History record

One record per cognitive event, appended and never modified. A user turn and an autonomous wake produce the
same shape, which is what lets `rebuild --from-history` replay them the same way.

```json
{
  "id": "wake_...",
  "trigger": "interaction",
  "occurred_at": "2026-08-30T17:04:00Z",
  "state_snapshot": { "...": "current_state as it was loaded" },
  "retrieved_memory_ids": ["mem_...", "mem_..."],
  "messages": [{ "role": "user", "content": "..." }],
  "output_text": "...",
  "decision": "respond",
  "stop_reason": "completed",
  "budget_used": { "iterations": 1, "tokens": 4120, "tool_calls": 0, "ms": 2310 }
}
```

- `trigger` is `interaction`, `reflection`, or `autonomous`, and it is what a derived memory's `origin`
  copies.
- `output_text` is the reason a reflection memory is regenerable at all. Rebuild does not re-run the
  model's reasoning any more than it re-runs the user's half of a conversation; it re-derives memories from
  recorded text. Leave the model's output out of history and everything reflection ever concluded
  disappears on the first rebuild — silently, because the rebuild will look like it succeeded.
- `retrieved_memory_ids` and `state_snapshot` record what the model was actually looking at. Retrieval
  quality is undiagnosable after the fact otherwise, and unlike a conversation, a wake has no user who
  remembers what it saw.
- `decision`, `stop_reason` and `budget_used` make an idle wake auditable. A run of wakes that all decided
  `sleep` and wrote nothing is the system working correctly, and that should be cheap to demonstrate.
  `stop_reason` is separate from `decision` because *chose to sleep*, *ran out of budget* and *raised* all
  look identical from outside — nothing happened — and only one of them is the system working.

## Memory record

Store curated memories, not raw messages. `canonical_text` is what gets embedded; everything else is
metadata and provenance.

```json
{
  "id": "mem_...",
  "canonical_text": "Decision: store long-term memory outside the thread.",
  "raw_text": "...",
  "source_id": "thread/123/message/456",
  "origin": "interaction",
  "type": "decision",
  "entities": ["long-term memory"],
  "importance": 0.8,
  "occurred_at": "2026-08-30T17:04:00Z",
  "created_at": "2026-08-30T17:05:12Z",
  "supersedes": null,
  "superseded_by": null,
  "embedding_model_id": "...",
  "embedding_dim": 1536,
  "preprocess_version": 1
}
```

- `type` is one of `event`, `decision`, `project`, `relationship`, `preference`, `open_question`.
- `origin` is the trigger of the wake that produced the memory. It is already encoded in the `source_id`
  prefix, but a field is what lets the drift check count reflection-written memories without parsing ids,
  and that ratio is the earliest signal that the store is filling with the system talking to itself.
- `occurred_at` is when the thing happened; `created_at` is when the memory was written. They diverge
  whenever a conversation recalls the past, and recency ranking wants `occurred_at`.
- `importance` is assigned by curation, in `[0, 1]`. A rerank signal, never a retrieval filter on its own
  — a low-importance memory is still the right answer to a question about it.
- `superseded_by` is the delete: set it, drop the row from search, keep it for history and rebuilds.
- The three embedding fields are what make a model or preprocessing change detectable rather than silent.

`memory.write()` canonicalizes a candidate, attaches this metadata, embeds, and stores — idempotent on
`source_id`.

## Preprocessing

One path, used for both memory writes and search queries — a query and the memory it should match have to
land in the same space.

```text
raw text
→ normalize whitespace
→ remove boilerplate
→ resolve obvious references when possible
→ compress to semantic core
→ preserve names, dates, and project terms
→ embed
```

Avoid aggressive stemming, stop-word removal, or keyword-only normalization. Bump `preprocess_version`
when the rules change; it is what tells a later reader that old vectors are stale.

## Curation

The step that decides what is worth remembering. On the interaction path it runs after a turn, over that
turn only, and usually returns nothing.

- Input: the turn's messages plus the memories that were retrieved for it.
- Output: 0..N candidate records, N bounded. A turn producing five memories is usually a prompt bug.
- Before writing, search for near-duplicates. Above the duplicate threshold, skip the candidate or write
  it and set `supersedes` on the row it replaces. Skipping this is the most likely way this design fails:
  conversation restates the same facts constantly, and an uncurated store fills with paraphrases that
  crowd each other out of the top `k`.

Reflection curates differently. An interaction turn asks what in this exchange is worth keeping; a
reflection wake asks what a set of existing memories, taken together, now says — and the usual answer is
that three overlapping rows should become one, with `supersedes` set on the ones it replaces. A reflection
wake whose output is all new rows has not consolidated anything, it has added opinions.

## Retrieval

`memory.search()` canonicalizes the query through the same path, embeds it, and returns a small ranked
set with provenance attached, so any claim can be traced back to the event that produced it.

- `k = 5` by default, applied after a similarity floor. Fewer results — including none — is correct.
- Exclude superseded rows; filter on `type` and rerank on recency and `importance` where it helps.

## `current_state`

A small object injected on every inference without a search. Derived like long-term memory, and
regenerable by `rebuild --from-history`.

```json
{
  "active_projects": [],
  "priorities": [],
  "open_questions": [],
  "recent_focus": []
}
```

- Bounded: a handful of one-line entries per list. It is in every prompt, so growth costs tokens on every
  request and dilutes the things that matter.
- Each entry cites the memory ids behind it — provenance applies here too.
- Updated by an explicit step at the end of a cycle, taking the prior state as input. It holds what is *current*:
  an entry that stops being true is dropped, not appended to. That is what keeps it from becoming a
  second, worse episodic store.

## The loop

The cycle in *Core idea* is the reason this is more than a memory system. A design whose only entry point
is `new input` thinks solely when spoken to: the one thing that can cause a thought is a person typing.
Continuity retrievable on demand is a better chatbot; continuity that something acts on between
conversations is what the project was for.

Closing that gap does not require continuous inference. It requires a wake cycle that something other than
a user can start.

```text
IDLE
 │ trigger
 ▼
LOAD_STATE
 │
 ▼
OBSERVE            retrieve memories, read pending questions
 │
 ▼
DECIDE ─────────────────── SLEEP ──▶ IDLE
 │                         nothing derived is written
 ├─ RESPOND
 ├─ ACT
 └─ REFLECT
 │
 ▼
CONSOLIDATE        curate, dedupe, supersede, write memories
 │
 ▼
SAVE_STATE         update current_state
 │
 ▼
IDLE
```

`SLEEP` short-circuits deliberately. A wake with nothing to do that still ran consolidation and saved state
would churn `recent_focus` on a timer, and every idle wake would dirty a derived store — which is how a
system convinces itself it is busy.

### Three triggers, one machine

| Trigger | Wakes on | `DECIDE` leans toward |
| --- | --- | --- |
| Interaction | a user message | `RESPOND` |
| Reflection | elapsed time, or turns accumulated since the last one | `REFLECT` |
| Autonomous | a schedule, or an `open_question` coming due | `SLEEP`, and usually it should pick it |

They are not three loops. They differ in what starts the cycle and how the decide prompt is framed;
everything from `DECIDE` onward is the same code. Building them separately means three copies of the budget
accounting, the consolidation step, and the state write, and within a month those copies disagree.

### Why the scheduler lives outside the model

"Decide whether another cycle is useful" is the step that turns a discrete wake into continuous inference
without anyone choosing to. A cycle that schedules its own successor, whose successor can do the same, is
an always-running process with extra steps — and it arrives by accident, one reasonable-looking prompt at a
time.

So a cycle may *request* a successor and the orchestrator decides. Keeping that decision outside the model
is cheap now and very hard to retrofit later, once prompts exist that assume they can keep going.

### Budgets

Two reasons every wake runs against a fixed allowance, and the second is what makes enforcing one cheap:

- An unbounded reflection cycle is a bill that arrives without any user having asked for anything.
- History is committed before consolidation runs, so exhausting a budget can end the cycle at a step
  boundary and lose nothing. `rebuild --from-history` reproduces whatever consolidation did not reach.

Accounting per wake rather than per iteration is the part that is easy to get wrong. A cycle that keeps
requesting successors, each with a fresh allowance, is unbounded no matter how tight any single iteration's
limit is — and that is the loophole that would leave the scheduler nominally outside the model while the
model set the pace anyway.

### Reflection feeds on itself

Reflection reads memories and writes memories. What it writes is retrieved by the next reflection, which
writes memories about those. Left alone, the store fills with the system's commentary on its own
commentary, and those paraphrases crowd real facts out of the top `k` — the failure mode curation already
exists to prevent, except with something actively producing it rather than a conversation happening to.

Three things hold it: reflection is pointed at supersession and merging rather than new claims, `origin`
marks which memories a wake wrote so the ratio is measurable, and the eval set carries a drift check, so
degradation shows up as a number instead of as a vague sense that the answers got worse.

## Build order

1. **Source history** — append-only, one record shape for turns and wakes. The one store that cannot be
   regenerated.
2. **Preprocessing**, plus `Embedder` / `MemoryStore` protocols and a fake embedder.
3. **`memory.write()` / `memory.search()`** over the boring store.
4. **`rebuild --from-history`** — early, while the store is small enough that it is easy.
5. **Eval set + recall@k** — before tuning retrieval, not after.
6. **Curation**, including the duplicate check.
7. **`current_state`** and its update step.
8. **The orchestrator**, interaction trigger only — the state machine, the budgets, and the
   consolidate/save steps, driving the path that already worked in steps 3–7. Milestone one runs through
   this.
9. **Reflection and autonomous triggers**, plus cooldown and the drift check. The parts that let a cycle
   begin without a user.

A throwaway script can drive steps 3–7 by hand long before step 8. That script is not the orchestrator, and
it should be deleted when the orchestrator exists rather than allowed to grow into a second one.

## Repository structure

```text
.
├── CLAUDE.md
├── docs/
├── src/
│   ├── memory/          # store, search, preprocess, curate
│   ├── state/           # current_state and its updates
│   ├── history/         # append-only source history
│   ├── llm/             # provider adapters
│   └── runtime/         # orchestrator, triggers, budgets, CLI
├── evals/               # frozen corpus + (query, expected_memory_ids)
├── tests/
├── data/                # gitignored: history, vectors, state
└── pyproject.toml
```

## Milestones

**One — continuity.** Restart with no chat history loaded and recover the important context of prior work
using only `current_state` + `memory.search()`. If that works reliably, the continuity layer is viable.

Make it a test, not a demo: a frozen corpus, a held-out question set, and a stated threshold (say, 8 of
10 facts recovered). Without a threshold it is an anecdote. Drive it through the orchestrator's interaction
trigger rather than calling `memory.search()` directly — a loop the bar routes around stays plumbing, which
is how it came to be a single line at the bottom of the build order in the first place.

**Two — restraint.** Over a static world, a run of idle wakes produces no memories and no state changes,
and a wake with an `open_question` coming due picks it up. This is the loop's version of *returning nothing
is a correct result*. The interesting property of an autonomous cycle is not that it can act; it is that it
can decline to, and only a test separates a system exercising judgment from one that has not yet been given
anything to do.

## Non-goals

Not part of the MVP: continuous 24/7 inference, a robot body, simulated biological needs, hard-coded
personality, consciousness detection, autonomous unrestricted tool use, thread branching or fusion.

The loop presses on two of those, so it is worth saying where the line falls. An always-running process and
one that wakes on a schedule differ in exactly one place: who decides when the next thought happens. And
the act branch exists but reaches only memory and `current_state` — an open tool surface is a separate
design with its own failure modes, and folding it in here would make the MVP's riskiest component the one
nobody set out to build.

## Guiding principle

**Build continuity, not a character.**

The system should increasingly behave in ways that make sense because of its own accumulated history, not
only because of the current prompt. Memory is what makes that possible; the loop is what gives it somewhere
to go. And a system free to decide there is nothing worth doing reveals more of that history than one that
always produces something.
