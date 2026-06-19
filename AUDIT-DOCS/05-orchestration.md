# 05 — Orchestration: Single vs Multi-Agent

**Research basis:** Doc 06 (Single vs Multi-Agent), Doc 01 (workflow patterns, 12-Factor #8).
**Verdict:** ✅ **Strong** — a model of the research's "healthy restraint." The harness defaults to a single agent and uses a deterministic **routing workflow** for model/tool selection, exactly the lighter-than-multi-agent pattern Doc 06 prescribes. This dimension is more *exemplary* than *deficient*; the notes below are refinements, not corrections.

---

## What the research wants (Doc 06 / 01)

> "Default to a single agent. Add subagents only for read-heavy, parallelizable fan-out where context isolation clearly helps — and make them an optional module, not the core architecture."

Multi-agent is risky by default (context fragmentation is the #1 cause of failure; coordination cost, token/latency multiplier, debuggability collapse). Before reaching for "agents talking to agents," prefer a deterministic **workflow pattern** — **routing** (classify → dispatch to specialized handler) is the lightest. Keep control flow in *your* code (12-Factor #8), not emergent agent negotiation. On local/small models, *strongly* prefer single-agent + workflows.

---

## What the code does

### ✅ Single-agent by default — the #1 risk avoided
There is exactly one agent loop and one conversation thread per request. No subagents, no agents negotiating, no fan-out merge. Context stays **continuous in a single thread** — Cognition's prescription and Doc 06's default. The harness therefore structurally cannot suffer the "context fragmentation" failure that Doc 06 calls the top cause of multi-agent breakage.

### ✅ The orchestrator is a clean routing workflow
`Orchestrator.decide()` is one LLM call that classifies the request and returns a structured `OrchestrationDecision` — selected model, tool subset, system prompt, thinking level ([orchestrator/orchestrator.py:89-130](../orchestrator/orchestrator.py#L89-L130)). This is Doc 01's **routing** pattern precisely: classify input → dispatch to the right configuration. The control flow lives in deterministic Python (the route calls the orchestrator, then `run_agent`), not in emergent multi-agent behavior — 12-Factor #8 satisfied.

### ✅ The router decides; it does not act
The orchestrator is given **no tools** — it "must decide, not act" (per `docs/architecture.md`'s stated principle, confirmed by `_call_orchestrator_llm` exposing no tool list). It reads inventories and emits a JSON decision. This keeps the routing step pure and side-effect-free.

### ✅ It never breaks a request — degrades to a safe default
*Any* failure (LLM error, parse error, validation error) is caught and replaced with a fallback decision carrying `fallback_used=True` + `fallback_reason` ([orchestrator/orchestrator.py:114-130](../orchestrator/orchestrator.py#L114-L130)). The whole layer is optional — disable it (`ORCHESTRATION_ENABLED=false` or omit `models.yaml`) and the harness runs in legacy single-model mode. Doc 06's "optional module, not the core architecture" is honored to the letter, and the `fallback_used` flag makes the degradation **observable** rather than silent.

### ✅ Actively fights tool bloat
The orchestrator selects the *smallest tool subset* for the job and disciplines its own decision prompt (history clipped to 6 messages / 500 chars each — [orchestrator/orchestrator.py:46-50](../orchestrator/orchestrator.py#L46-L50), [167-179](../orchestrator/orchestrator.py#L167-L179)). It is, in effect, the harness's phase-gating mechanism (Doc 04/05).

### ✅ Right call for local models
Doc 06: "On local/small models, strongly prefer single-agent + workflows." The default model *is* local (Qwen3.6), and the architecture is single-agent + routing workflow — perfectly matched.

---

## Notes & refinements (not gaps)

### 🟡 (a) Per-request routing cost
Orchestration adds one LLM call per request (acknowledged in `docs/operations.md`). Mitigations already exist (`ORCHESTRATOR_MODEL_ID` to run routing on a cheap model) but aren't emphasized.

> **Refinement (P2):** document the cheap-orchestrator-model recommendation prominently; optionally add a heuristic fast-path that skips orchestration for trivial prompts (e.g. no tools needed), or a short-TTL cache keyed on a normalized prompt. Tie any such change to evals (see `07`) so you can prove routing quality didn't regress.

### 🟡 (b) No budget-aware routing
Doc 06 notes routing "picks the best model for the job, not the cheapest acceptable one." The orchestrator's only steering is the `description` text in `models.yaml`. A future `max_spend`-aware decision (cross-ref `06-reliability-safety.md`) could pick the cheapest model that clears a quality bar.

### 🟡 (c) No A/B / shadow routing
Acknowledged limitation. This is really an **evals** dependency (`07-observability-evals.md`): without an eval harness you can't measure whether one routing policy beats another, so shadow-routing has nowhere to report to. Sequence it after evals exist.

### ✅ (d) Multi-agent correctly *absent*
There is no premature multi-agent machinery to critique — which is the right answer. *If* a genuinely parallel, read-heavy, independent task ever appears (Doc 06's only "favors subagents" case), the lightest addition is a **read-only `spawn_subagent` tool** that reuses the same loop with a fresh context and returns a ~1–2k-token summary — *not* a new orchestration framework. Captured as roadmap P2 #13, explicitly "only if a task demands it."

---

## What works / what doesn't — scored

| Doc 06 criterion | Status | Evidence |
|---|---|---|
| Single agent as the default | ✅ | one loop, one thread per request |
| Workflow (routing) over agents-negotiating | ✅ | [orchestrator/orchestrator.py:89-130](../orchestrator/orchestrator.py#L89-L130) |
| Orchestration is optional / deletable | ✅ | `ORCHESTRATION_ENABLED`, fallback path |
| Degrades gracefully, observably | ✅ | `fallback_used` ([orchestrator.py:114-130](../orchestrator/orchestrator.py#L114-L130)) |
| Router decides, doesn't act (no tools) | ✅ | orchestrator gets no tool list |
| Single-agent on local models | ✅ | local Qwen default + single agent |
| Subagents only for read-only fan-out | ✅ (n/a) | none present — correct |
