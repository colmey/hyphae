# 02 — Anatomy of a Harness & the Agent Loop

This is the heart of the harness. Get the loop and state model right and everything else is a
plug-in. Get it wrong and no amount of tooling saves you.

## 1. The minimal viable harness

A working agent harness needs only five things:

```
┌─────────────────────────────────────────────────────────┐
│                       AGENT LOOP                          │
│                                                           │
│   ┌────────────┐   tool calls   ┌──────────────────┐      │
│   │   MODEL    │ ─────────────▶ │  TOOL DISPATCH   │      │
│   │   CLIENT   │ ◀───────────── │  (registry+exec) │      │
│   └────────────┘  tool results  └──────────────────┘      │
│         ▲                                                 │
│         │ messages (curated)                              │
│   ┌────────────┐                ┌──────────────────┐      │
│   │  CONTEXT   │                │     LIMITS /      │      │
│   │  ASSEMBLY  │                │   GUARDRAILS      │      │
│   └────────────┘                └──────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

1. **Model client** — sends messages + tool schemas, returns text and/or tool calls. ([03](03_model_interface.md))
2. **Tool registry + dispatcher** — holds tool definitions, validates args, executes, returns results. ([04](04_tool_design.md))
3. **Context assembly** — builds the message list for each model call. ([05](05_context_and_memory.md))
4. **Limits/guardrails** — iteration cap, token/spend budget, timeouts, permissions. ([07](07_reliability_and_safety.md))
5. **The loop** — ties them together (below).

Everything else — memory, subagents, MCP, tracing UI, RAG — is **optional** and bolts onto these
five. If you can't draw your harness this simply, you've over-built it.

## 2. The core loop

Pseudocode (language-neutral; assume an OpenAI-compatible client from [03](03_model_interface.md)):

```python
def run(task, tools, limits):
    state = State(messages=[system_prompt(), user(task)], step=0, spend=0)

    while True:
        guard(state, limits)                      # raise if over caps (step/spend/time)

        messages = assemble_context(state)        # curate what the model sees (doc 05)
        response = model.complete(messages, tools.schemas())

        state.append(response.message)            # record assistant turn (incl. tool calls)
        state.spend += response.usage.cost
        state.step += 1

        if not response.tool_calls:               # model answered with no tool call → done
            return response.text

        for call in response.tool_calls:          # take action
            result = tools.execute(call)          # validate args, run, capture errors
            state.append(tool_result(call.id, result))   # feed result back (doc 04 §errors)
        # loop continues: model sees results, decides next action (verify / continue / finish)
```

That's the whole engine. Note what's *not* here: no graph, no planner module, no orchestration DSL.
The model does the planning; the loop just executes and feeds back.

### Three phases inside the loop

| Phase | In the loop above | Done well in |
|-------|-------------------|--------------|
| **Gather context** | `assemble_context(state)` | [05](05_context_and_memory.md) |
| **Take action** | `model.complete(...)` + `tools.execute(...)` | [03](03_model_interface.md), [04](04_tool_design.md) |
| **Verify work** | model inspects tool results next iteration; optional explicit checks | [04](04_tool_design.md) §errors, [08](08_observability_and_evals.md) |

## 3. State as a stateless reducer (12-Factor #12)

Model the run as `next_state = step(state, event)` — a **pure reduction** over an append-only event
log, not mutable globals. This buys you, almost for free:

- **Resumability / pause-resume** — persist `state`, reload, continue. (12-Factor #6)
- **Testability** — replay an event log to reproduce any run deterministically (mock the model).
- **Observability** — the event log *is* the trace. ([08](08_observability_and_evals.md))
- **Time-travel debugging** — fork from any prior state.

Minimal state:

```python
State = {
  "messages": [...],     # the conversation / event log (append-only)
  "step": int,           # iteration counter (for the cap)
  "spend": float,        # accumulated cost/tokens (for the budget)
  "started_at": ts,      # for the wall-clock timeout
  "scratch": {...},      # optional: todo list, notes, artifacts (doc 05 memory)
}
```

**Unify execution state and business state (12-Factor #5):** keep one source of truth. If the agent
"thinks" a file was written, the file must actually exist. Drift between the model's belief and
reality is the root of a whole class of failures ([09](09_antipatterns_and_checklist.md)).

## 4. Termination — the loop *must* stop

A loop that can't reliably stop is the most common way a harness burns money and trust. Provide
**multiple independent stop conditions**, checked every iteration:

| Condition | Why | Default |
|-----------|-----|---------|
| **Natural completion** | Model responds with no tool call | primary exit |
| **Max iterations** | Cap runaway loops | 20–50 (task-dependent) |
| **Token / spend budget** | Cap cost | per-task ceiling |
| **Wall-clock timeout** | Cap latency / hung tools | task-dependent |
| **No-progress detection** | Same tool+args repeated, or repeated failures | abort or escalate to human |
| **Explicit `finish`/`done` tool** | Gives the model a clean, unambiguous way to end | optional but recommended |
| **External cancel** | User/operator interrupt | always wire this up |

When a limit trips, **don't fail silently** — return a partial result with the reason, and feed a
clear message back so the model (or a human) can react. (See "retry loops" and "compounding error
cascade" in [09](09_antipatterns_and_checklist.md).)

## 5. Single-turn vs multi-turn vs background

The same loop serves all three; only the entry/exit differs (12-Factor #11, "trigger from
anywhere"):

- **Single-shot** — run to completion, return result (CLI, API call).
- **Interactive** — pause for user input, resume with the same `state` (chat).
- **Background / long-horizon** — checkpoint `state` to durable storage, survive restarts, run for
  hours. Requires memory/compaction ([05](05_context_and_memory.md)).

Keep the loop itself unaware of *which* mode it's in — that's the caller's concern. This is the
modularity payoff: one engine, many front ends.

## What works / What doesn't

| ✅ Works | ❌ Doesn't |
|---------|-----------|
| A ~one-page loop the whole team can read | A planner/orchestrator subsystem before you need it |
| State as an append-only log + pure reducer | Mutable global state scattered across the codebase |
| Multiple independent stop conditions | A single "max iterations" as the only safety net |
| Letting the model plan; the loop just executes | Encoding a fixed plan the model must follow |
| One engine behind CLI/API/background front ends | Coupling the loop to a single invocation mode |
| Feeding tool results straight back each turn | Hiding/discarding results, so the model can't verify |

**Next:** [03 — Model Interface](03_model_interface.md)
