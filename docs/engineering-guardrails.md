# Engineering Guidance — MVP Guardrails

This document captures the few implementation details that matter most for the first version of the AI-native continuity harness. The goal is to keep the system observable and trustworthy without adding unnecessary engineering work.

## 1. Log the exact context sent to the LLM

For every inference, record the final assembled input:

- system/developer instructions
- current thread context
- retrieved long-term memories
- current state
- wake reason / trigger

This is the single most important debugging tool. If behavior changes, we need to know exactly what the model saw.

## 2. Keep raw history immutable

Conversation turns and autonomous wakes should be written to an append-only source history.

Long-term memory is a **derived layer**, not the source of truth.

Every memory should keep provenance back to the source event/message that created it.

## 3. Make memory behavior inspectable

For every memory write and retrieval, log:

- memory text
- type
- source ID
- retrieval score
- whether it was written, updated, or superseded

Avoid automatically turning one-off statements into permanent personality traits. Persistent preferences should emerge from repeated behavior, not one generated sentence.

## 4. Instrument the loop lifecycle

Every wake should record:

- trigger: user / schedule / due item
- start and stop time
- iteration count
- token/tool budget used
- stop reason
- state changes made

A wake is allowed to do nothing.

`idle -> no-op -> sleep` is valid behavior and should not be treated as a failure.

## 5. Keep durable state changes explicit

For MVP, autonomous actions should be limited to:

- `memory.write`
- `memory.supersede`
- `current_state.update`
- `open_question.update`

Do not add external tools or a persistent workspace yet.

Every durable state change should have provenance and be auditable.

## 6. Add simple hard limits

Each autonomous wake should have:

- max iterations
- token budget
- timeout
- cooldown

This prevents accidental infinite loops and keeps cost predictable.

## 7. Test continuity, not personality

The first useful evaluation is simple:

> Restart the system with no thread history loaded and verify that it can recover important prior context from `current_state` and long-term memory.

Also test that retrieved memories are relevant and that old/incorrect memories can be superseded without deleting source history.

## MVP architecture

```text
LLM
├── short-term context
├── long-term memory
├── current_state
├── source history
└── loop orchestrator
        ^
        └── user input | schedule | due item
```

## Priority

Engineering time is limited, prioritize these three areas:

1. **Context construction logging**
2. **Memory provenance + inspectability**
3. **Loop lifecycle logging + hard budgets**

Everything else can wait.
