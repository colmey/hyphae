# 07 — Observability & Evals

**Research basis:** Doc 08 (Observability & Evals), Doc 10 (Trace seam).
**Verdict:** 🔴 **Gap** — the second-largest divergence. The harness already *produces* the raw material for observability (a per-step event stream) but never **persists it as a trace**, and it has **no evals and no CI**. Doc 08 calls a small eval suite "the single highest-leverage habit" and "the model-swap safety net" — directly relevant to a harness that swaps between Gemini and a local Qwen.

---

## What the research wants (Doc 08)

"You cannot improve — or trust — what you can't see and measure." Instrument the loop so every run emits a connected **trace** (one input → each model call → each tool call → results → final answer) with **timing, tokens, and cost per step** — and note the trace is *mostly free* because it's just **serializing the append-only event log**. Start with **JSONL** (greppable, diffable, replayable), add **OpenTelemetry** for vendor-neutral export later. The event log enables **replay/fork/diff** debugging. **Evals:** outcome-based + programmatic, a small versioned dataset (20–100 cases) grown from real failures, **run in CI** to gate changes — "the single highest-leverage habit."

---

## What the code does

### ✅ The trace primitive already exists
The loop yields a typed event stream — `UsageEvent`, `TextEvent`, `ToolCallEvent`, `ToolResultEvent`, `DoneEvent`, `ErrorEvent` (`agent/events.py`, emitted throughout [agent/loop.py](../agent/loop.py)). Per-iteration and cumulative token usage are captured ([agent/loop.py:359-368](../agent/loop.py#L359-L368)) and returned in the HTTP response. This is *exactly* the append-only event log Doc 08 says a trace should serialize — the hard part is done.

### ✅ Reasonable structured logging
Key seams log at INFO: per-request model selection ("orchestrator picked model=… tools=… fallback=…"), the done-reason/iterations/tokens summary, MCP connect results ([mcp_layer/manager.py:76-79](../mcp_layer/manager.py#L76-L79)), per-iteration debug ([agent/loop.py:329-331](../agent/loop.py#L329-L331)), stall and timeout warnings. `LOG_LEVEL=DEBUG` exposes per-request orchestration detail.

---

## Gaps & recommendations

### 🔴 (a) No persisted trace
The event stream is returned in the `/chat` response body and then **discarded** — it is never written to durable storage. There is no JSONL trace file, no OTel exporter, no Trace seam. Doc 08's "trace = serialized event log (free from the state model)" is unrealized; "logging only the final answer" is the named anti-pattern, and the harness is one short step from avoiding it.

> **Recommendation (P0 — highest ROI in the whole audit):** add a **Trace seam** — `Tracer.emit(event)` with a JSONL sink (one file per run, or append with a `run_id`). Wire it where events are already produced in `run_agent`; the events carry tokens/latency already. This is a few dozen lines and unlocks grep/diff/replay debugging immediately. Add an OpenTelemetry exporter behind the same seam later (Doc 08 / Doc 10: JSONL ↔ OTel is a swap).

### 🔴 (b) No request/run correlation IDs
Logs have no `request_id`/`run_id`, so interleaved concurrent requests can't be untangled and a log line can't be tied to a trace. Doc 08 lists "step index, parent/run id" as the first per-step field.

> **Recommendation (P0, pairs with a):** generate a `run_id` per `/chat` call, attach it to every log line (logging filter/adapter) and every emitted trace event.

### 🔴 (c) No evals — the highest-leverage missing habit
`tests/` contains 11 **smoke tests** that verify the harness *runs* (config parses, MCP connects, the loop iterates, the guard rejects) — valuable, but they are not **evals**: none judges whether a run produced the *right outcome*. Doc 08: outcome-based, programmatic, versioned, in CI. The absence is especially costly here because the harness routinely **swaps models** (Gemini ↔ local Qwen) and Doc 08 names evals "your model-swap safety net… without evals, 'does this work on the local model?' is unanswerable." Every recommendation elsewhere in this audit (compaction, capability profiles, routing changes) is also a change you currently can't measure.

> **Recommendation (P1):** a small **eval runner** in `tests/` (standalone, per the project's no-pytest convention) over a versioned dataset of 20–100 real cases with **programmatic checks** (did the expected tool get called? does the final answer contain the known fact? did `done_reason` match?). Grow it from real failures. This is the gate for trusting every other improvement.

### 🔴 (d) No CI
No `.github/` or any automation runs the smoke tests or evals. Doc 08: "Run in CI… block merges that drop success/safety below a threshold — the single highest-leverage habit."

> **Recommendation (P1, pairs with c):** a minimal CI workflow that runs the smoke tests and the eval suite on push/PR, gating on a success threshold.

### 🟡 (e) No replay / fork / diff
Doc 08's debugging superpowers (replay a recorded run by mocking the model; fork from step *k*; diff two runs) depend on (a) a persisted event log and on the loop being a pure reducer (`01-loop-and-architecture.md` gap (a)). Persisting the trace (a) delivers replay/diff at the data level; full fork-from-state needs the reducer refactor (roadmap P2 #10).

---

## What works / what doesn't — scored

| Doc 08 criterion | Status | Evidence |
|---|---|---|
| Trace = serialized event log | 🔴 | events produced but never persisted |
| Per-step tokens/cost/latency/tool I/O captured | 🟡 | tokens yes; latency/cost not recorded |
| JSONL trace as v0 | 🔴 | none |
| OpenTelemetry export | 🔴 | none |
| Replay/fork/diff via the log | 🔴 | not possible (no persisted log + mutable run state) |
| Outcome-based programmatic evals | 🔴 | smoke tests only |
| Small versioned dataset grown from failures | 🔴 | none |
| Evals in CI gate changes | 🔴 | no CI |
| Evals as model-swap safety net | 🔴 | none — yet the harness swaps models by design |

**Bottom line:** this is the cheapest high-impact area to fix. Persisting the already-emitted event stream as JSONL with a `run_id` (a, b) is a quick win; a small programmatic eval suite + CI (c, d) is the discipline that makes every *other* recommendation in this audit measurable and safe to ship.
