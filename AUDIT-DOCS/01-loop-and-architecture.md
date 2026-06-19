# 01 — Loop & Architecture

**Research basis:** Doc 02 (Anatomy & the Agent Loop), Doc 10 (Reference Architecture), Doc 01 (12-Factor Agents).
**Verdict:** ✅ **Strong** — with two structural gaps (state-as-reducer, incomplete stop conditions) that unlock observability and safety wins downstream.

The agent loop is the heart of the harness and the part the research is most prescriptive about: "a ~one-page loop the whole team can read," append-only state, multiple independent stop conditions, and "let the model plan; the loop just executes." PyAiHarness gets the spirit right.

---

## What the research wants (Doc 02)

The MUST-have core: model client + tool dispatcher + context assembly + limits/guardrails + the loop. The loop drives `gather context → take action → feed results back → repeat`. State should be an **append-only event log** reduced by a pure function (`next_state = f(state, event)`), which buys resumability, testability, observability, and time-travel debugging. The loop **must stop** via *multiple independent* conditions and must **fail loud with partial results**, never silently.

---

## What the code does

### ✅ A single, readable bridge loop
`run_agent()` in [agent/loop.py:231-475](../agent/loop.py#L231-L475) is the only module importing both `LLMClient` and `MCPManager` ([agent/loop.py:62-64](../agent/loop.py#L62-L64)). The docstring states the invariant plainly: "This is the only module in the harness that imports both… the loop is the bridge." The body is a clean `while iteration < max_iterations` with the canonical shape: complete → record assistant turn → stream text → extract tool calls → execute sequentially → append results → loop. This is exactly the research's "thin deterministic shell."

### ✅ The model plans; the loop executes
There is no planner subsystem, no graph, no orchestration DSL inside the loop. Tool calls come from the model and are dispatched verbatim ([agent/loop.py:379-417](../agent/loop.py#L379-L417)). This matches Doc 02's "What works: letting the model plan; the loop just executes" and avoids the "encoding a fixed plan" anti-pattern.

### ✅ Clear, distinguished termination set
The loop yields a `DoneEvent` with an explicit reason in every exit path ([agent/loop.py:382-394](../agent/loop.py#L382-L394)):
- `end_turn` — model produced no tool calls (primary exit),
- `truncated` — stopped on `max_tokens` mid-answer (the response is clipped — the loop *does not* pretend it's complete),
- `max_iterations` — hit the cap; on the final iteration tools are withheld and a wrap-up note coaxes a best-effort answer ([agent/loop.py:316-327](../agent/loop.py#L316-L327)),
- `empty` — no candidates,
- `llm_error` — unrecoverable LLM failure after retries ([agent/loop.py:346-352](../agent/loop.py#L346-L352)).

This honors Doc 02's "when a limit trips, don't fail silently — return a partial result with the reason." The `truncated` vs `end_turn` distinction is a nice touch most harnesses miss.

### ✅ No-progress detection
Stall detection short-circuits a byte-for-byte-identical repeat call to a synthetic `is_error` result instead of re-executing ([agent/loop.py:405-409](../agent/loop.py#L405-L409)), with a stable canonical-args key ([agent/loop.py:115-126](../agent/loop.py#L115-L126)). A consecutive-failure nudge is appended once at a threshold ([agent/loop.py:441-447](../agent/loop.py#L441-L447)). This directly implements Doc 02/07's "no-progress detection: same tool+args repeated, or repeated failures."

### ✅ One engine, many front ends
The loop is an `AsyncIterator[Event]` ([agent/loop.py:247](../agent/loop.py#L247)) — caller-agnostic. The `/chat` route consumes it; smoke tests consume it directly. Doc 02's "one engine behind CLI/API/background front ends" is satisfied; streaming would be a trivial third consumer.

### ✅ Save-after-each-iteration + helper-only mutation
The session is `save()`'d after the assistant turn and after tool results ([agent/loop.py:370-371](../agent/loop.py#L370-L371), [agent/loop.py:464-465](../agent/loop.py#L464-L465)), and mutated only via `append_assistant` / `append_tool_results`. This is a checkpoint *hook* (a no-op for the in-memory store today, but the seam exists for durable backends).

---

## Gaps & recommendations

### 🟡 (a) State is not a pure append-only reducer
Doc 02 makes "state as a stateless reducer over an append-only event log" a **critical** rule, because it's what makes runs replayable, forkable, and diffable (Doc 08). Here, `session.messages` is append-only, but the *run* carries mutable local state: `iteration`, `cumulative`, `seen_calls`, and `consecutive_tool_errors` ([agent/loop.py:294-301](../agent/loop.py#L294-L301)). The loop is therefore not `next_state = f(state, event)` — you cannot replay a recorded run deterministically or fork from step *k*.

> **Recommendation (P2):** model a run as `RunState{messages, step, spend, started_at, seen_calls, consecutive_errors}` advanced by a pure `step(state, event)`. This is the prerequisite for replay/fork/diff debugging (see `07-observability-evals.md`). A lighter intermediate win — persisting the yielded event stream to JSONL — gives most of the debugging value without the refactor; do that first (P0).

### 🟡 (b) Stop conditions are incomplete
Doc 02's "What doesn't: a single 'max iterations' as the only safety net." The loop gates only on `iteration < max_iterations` ([agent/loop.py:313](../agent/loop.py#L313)). There is **no spend/token-budget cap and no overall wall-clock cap** — `cumulative` tokens are tracked ([agent/loop.py:359-360](../agent/loop.py#L359-L360)) but never compared against a ceiling, and per-call timeouts bound each call but not the whole request. A misbehaving model can burn `max_iterations` of bounded-but-real calls.

> **Recommendation (P0):** add `max_spend`/token ceiling and `wall_clock_seconds` to the loop guard (detail in `06-reliability-safety.md`). These are the two missing "multiple independent stop conditions."

### ⬜ (c) No explicit `finish`/`done` tool
Doc 02 lists an explicit `finish` tool as "optional but recommended" — a clean, unambiguous end signal. The loop infers completion from the absence of tool calls, which is fine for this design but means "the model is still thinking but emitted no tool call" is indistinguishable from "done." Low priority given the orchestrator-curated toolsets.

### 🟡 (d) No durable checkpoint / resume
Doc 02 (single-turn vs multi-turn vs background) and 12-Factor #6 (launch/pause/resume) want state persisted so a crash resumes rather than restarts. The `SessionStore` ABC and save-hooks exist, but the only implementation is in-memory ([agent/session.py](../agent/session.py)). Acknowledged as a deliberate v1 simplification (LibreChat holds durable context); the seam is correctly placed for a future durable backend (see roadmap P2 #11).

---

## What works / what doesn't — scored

| Doc 02 criterion | Status | Evidence |
|---|---|---|
| ~one-page loop the team can read | ✅ | [agent/loop.py:231-475](../agent/loop.py#L231-L475) |
| append-only log + pure reducer | 🟡 | history append-only; run uses mutable locals ([agent/loop.py:294-301](../agent/loop.py#L294-L301)) |
| multiple independent stop conditions | 🟡 | iterations + per-call timeouts only; no spend/wall-clock |
| model plans, loop executes | ✅ | [agent/loop.py:379-417](../agent/loop.py#L379-L417) |
| one engine, many front ends | ✅ | async-generator events |
| feed tool results straight back | ✅ | [agent/loop.py:455-463](../agent/loop.py#L455-L463) |
| fail loud, return partials | ✅ | reason on every exit ([agent/loop.py:382-394](../agent/loop.py#L382-L394)) |
