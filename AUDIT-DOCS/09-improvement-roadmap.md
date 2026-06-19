# 09 — Improvement Roadmap

A prioritized, actionable plan that collects every recommendation from `01`–`08`. Sequenced against the research's phased build path (Doc 10): the harness has effectively shipped **v0 (walking skeleton)** and most of **v1 (make it safe)**; this roadmap finishes v1 and reaches **v2 (durable)** and **v3 (measurable)**. v4 items are à la carte.

**Rating key:** Impact (🟢 high / 🟡 med / ⚪ low) · Effort (S / M / L).
Each item: *finding → research basis → where → sketch*.

---

## P0 — Quick wins (high impact, low effort) — *finish v1*

### 1. Persist the event log as a JSONL trace + `run_id` — 🟢 / S
**Finding (`07`):** events with token usage are emitted per step but discarded after the HTTP response — no durable trace. **Basis:** Doc 08 ("trace = serialized event log," nearly free). **Where:** `run_agent` ([agent/loop.py](../agent/loop.py)) where events are yielded; the `/chat` route in `api/routes.py` to mint the id. **Sketch:** a `Tracer` seam with `emit(event)`; default JSONL sink writing one record per event tagged with `run_id` + step index + timestamp. Generate `run_id` per `/chat` call, thread it into a logging adapter so every log line carries it. Add latency by timestamping each `complete()`/`call_tool`. *This is the single highest-ROI change in the audit.*

### 2. Spend/token cap + wall-clock cap in the loop guard — 🟢 / S
**Finding (`06`):** only `iteration < max_iterations` and per-call timeouts bound a run; `cumulative` tokens are tracked but never capped. **Basis:** Doc 07 layered guardrails; Doc 02 multiple independent stop conditions. **Where:** the `while` guard at [agent/loop.py:313](../agent/loop.py#L313); new `Settings` fields (`max_run_tokens`, `wall_clock_seconds`) in `harness_config.py`. **Sketch:** record `started_at`; before each iteration, exit with a distinct `done_reason` (`budget_exceeded` / `deadline_exceeded`) + partial result if `cumulative.total_tokens >= cap` or `now - started_at >= deadline`. Never silent (Doc 07 "fail loud, return partials").

### 3. Validate tool args against `input_schema` before execution — 🟢 / S
**Finding (`03`):** model args pass straight to the MCP server; hallucinated args (Doc 04/09 #6) caught only remotely, opaquely. **Basis:** Doc 04 "validate before you execute." **Where:** a dispatch seam between [agent/loop.py:410-417](../agent/loop.py#L410-L417) and `MCPManager.call_tool`; schemas already on `Tool.input_schema`. **Sketch:** `jsonschema`-validate `tu.input`; on failure short-circuit to an `is_error` result naming the bad/missing field and showing the expected shape (a *teaching* message). Adopt the validator lib (Doc 10 build-vs-adopt).

### 4. Abort/escalate on N consecutive failures — 🟢 / S
**Finding (`06`):** the consecutive-failure nudge informs the model but never stops the cascade. **Basis:** Doc 07 "no-progress detection that changes strategy… abort or escalate." **Where:** extend the counter at [agent/loop.py:445-447](../agent/loop.py#L445-L447). **Sketch:** configurable `abort_after_consecutive_failures` (default ~5) → exit `done_reason="no_progress"` with the partial result.

### 5. Correct the stale docs — 🟡 / S
**Finding (`02` (d)):** `docs/architecture.md` + `docs/operations.md` claim OpenAI raises `NotImplementedError`; it is fully implemented and is the **default** (local Qwen). **Basis:** doc accuracy / Doc 09 (hidden behavior). **Where:** `docs/architecture.md`, `docs/operations.md`, and the "One LLM provider implemented" limitation. **Sketch:** "Gemini and OpenAI-compatible (incl. local Ollama/vLLM) implemented; Anthropic stubbed. Default model is a local OpenAI-compatible Qwen3.6."

---

## P1 — Medium (close the core gaps) — *reach v2 + v3*

### 6. Context-assembly seam: token budget + naive compaction — 🟢 / M
**Finding (`04`):** no Context manager — full `session.messages` sent every call ([agent/loop.py:337](../agent/loop.py#L337)), no budget, no compaction. **Most consequential gap given the local small-window default.** **Basis:** Doc 05 (correctness, not optimization); Doc 10 Context seam. **Where:** new `agent/context.py` with `assemble_context(state, model_profile) -> messages`; loop calls it instead of reading `session.messages`. **Sketch:** v0 = pass-through + budget guard (`budget = context_window − max_output − margin`, using the token-estimate fallback from #9); when over budget, compaction strategy keeps last *N* turns verbatim, summarizes the middle via a cheap model call, always retains system prompt + pinned task header. Make strategy swappable (`naive | compaction`) per Doc 10. Add `context_window` to model profiles (#9).

### 7. Minimal programmatic eval suite + CI — 🟢 / M
**Finding (`07`):** smoke tests verify "it runs," not "it's right"; no CI. **Basis:** Doc 08 ("single highest-leverage habit"; model-swap safety net). **Where:** `tests/eval_*.py` (standalone, per the no-pytest convention); a `.github/workflows/ci.yml`. **Sketch:** a versioned dataset (20–100 real cases) + a runner doing **programmatic** checks (expected tool called? known fact present? `done_reason` correct?). CI runs smoke tests + evals on PR, gating on a success threshold. *Prerequisite for trusting #6, #8, #9 and any routing change (`05`).*

### 8. Policy/permissions seam + auth on `/chat` — 🟢 / M
**Finding (`06` (c,d), `03` (b)):** every tool executes unconditionally; no read-only default, no approval, no auth. Highest *safety* gap for a tool-executing HTTP service. **Basis:** Doc 07 ("highest-stakes part"); Doc 10 Policy seam. **Where:** same dispatch seam as #3; a FastAPI auth `Depends` on the `/chat` route. **Sketch:** `policy.check(state, tool_call) -> allow | deny | ask`; start minimal — read-only default + explicit write/destructive allow-list; `ask` pauses for human approval reusing the existing event/session machinery (human-in-the-loop as a tool result). Swappable dev-permissive ↔ prod-sandboxed. Add bearer/API-key auth before any untrusted exposure.

### 9. Per-model capability profiles + prompted-tool & token-estimate fallbacks — 🟡 / M
**Finding (`02` (a,b)):** no capability detection (`thinking_level` ignored, no prompted-tool fallback); no token estimate when `usage` is missing. **Basis:** Doc 03 capability table + weak-model tactics. **Where:** extend `models.yaml` entries + `orchestrator/schemas.py`; a prompted-tool encoder behind the existing registry. **Sketch:** profile fields `supports_native_tools`, `json_mode`, `thinking`, `context_window`; when `supports_native_tools=false`, render tools into the prompt and parse a JSON action (same registry/dispatcher, different encoder — Doc 04); add a local tokenizer estimate used whenever provider `usage` is absent/zero (feeds #2 and #6).

---

## P2 — Larger / optional — *v4, à la carte*

### 10. Pure-reducer run state → replay/fork/diff + OTel exporter — 🟡 / L
**Finding (`01` (a), `07` (e)):** run carries mutable locals; can't replay/fork deterministically. **Basis:** Doc 02 stateless reducer; Doc 08 replay/fork/diff. **Sketch:** model a run as `RunState` advanced by a pure `step(state, event)`; once events are persisted (#1), replay by mocking the model. Add an OpenTelemetry exporter behind the Trace seam (JSONL ↔ OTel swap).

### 11. Durable `SessionStore` for checkpoint/resume + multi-worker — 🟡 / L
**Finding (`01` (d), `06`):** in-memory only; per-worker isolation; no resume. **Basis:** Doc 02 background/long-horizon; Doc 07 checkpoint & resume; 12-Factor #6. **Where:** the `SessionStore` ABC + save-hooks already at [agent/loop.py:370-371](../agent/loop.py#L370-L371),[464-465](../agent/loop.py#L464-L465). **Sketch:** a Redis/SQL implementation behind the ABC; shared session claim to make `SessionGuard` correct across workers. *Build only if use cases outgrow the LibreChat-fronts-durability assumption.*

### 12. Prompt-injection guard for tool output + sandboxing — 🟡 / L
**Finding (`06` (e)):** untrusted MCP/web-search output flows into context raw. **Basis:** Doc 07 prompt-injection awareness; sandbox destructive tools. **Sketch:** keep the policy layer (#8) authoritative — never let tool-output content widen permissions; label tool output as untrusted in the prompt; sandbox any future shell/file/destructive tools (containers/jails — adopt, not build).

### 13. Working memory / long-term memory; read-only subagents — ⚪ / L
**Finding (`04` (d), `05` (d)):** no files-as-memory; no subagents. **Basis:** Doc 05 (files-first memory), Doc 06 (subagent-as-a-tool). **Sketch:** add only when a task demands it — files-as-memory pairs naturally with compaction (#6) for long-horizon runs; a **read-only `spawn_subagent` tool** (fresh context, returns a ~1–2k-token summary, reuses the one loop) is the *only* sanctioned multi-agent form, and only for genuinely parallel read-heavy fan-out. Do **not** build a multi-agent framework.

---

## Sequencing at a glance

| Phase | Items | Outcome (Doc 10) |
|---|---|---|
| **P0** | 1–5 | Finishes **v1 "make it safe"**: bounded runs, validated args, observable traces, accurate docs |
| **P1** | 6–9 | Reaches **v2 "durable"** (#6) + **v3 "measurable"** (#7), plus the safety seam (#8) and weak-model support (#9) |
| **P2** | 10–13 | **v4 extensions** — add the one a task demands; skip the rest |

**The two changes to make first:** **#1 (persist the trace)** and **#7 (a small eval suite)** — together they turn every subsequent item from a guess into a measured engineering change, which is the research's fifth principle: *measure before you trust.*
