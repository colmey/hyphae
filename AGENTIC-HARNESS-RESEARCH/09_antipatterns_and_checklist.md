# 09 — Anti-Patterns & the Include/Avoid Checklist

A consolidated catalogue of **what doesn't work**, plus a practical checklist of what to
include and what to leave out. This is the doc to re-read before adding any new piece to the
harness. Sourced partly from Atlan's
[13 agent-harness anti-patterns](https://atlan.com/know/agent-harness-failures-anti-patterns/) and
the failure modes referenced throughout these docs.

## 1. The 13 harness anti-patterns (by where failures originate)

Empirically, the *majority* of agent failures are **context/data-layer**, not architecture. Don't
spend all your effort on clever orchestration while feeding the model stale or wrong context.

### Architectural failures (~20%)

| # | Anti-pattern | What it is | Fix → |
|---|-------------|-----------|-------|
| 1 | **Monolithic mega-prompt** | All behavior crammed into one giant system prompt; model loses coherence, ignores constraints | Right-altitude prompt; load detail on demand ([05](05_context_and_memory.md)) |
| 2 | **Invisible state (LLM-as-memory)** | Relying on the context window to carry state; ~2% retention loss per step | Persist state explicitly; unify exec & business state ([02](02_anatomy_and_the_loop.md)) |
| 3 | **All-or-nothing autonomy** | Full autonomy, no approval gates → errors cascade | Stake-scaled approval gates ([07](07_reliability_and_safety.md)) |
| 4 | **Compounding error cascade** | Per-step errors multiply (0.85¹⁰ ≈ 20%) | Raise per-step reliability + stop cascades early ([07](07_reliability_and_safety.md)) |

### Execution / tool-layer failures (~25%)

| # | Anti-pattern | What it is | Fix → |
|---|-------------|-----------|-------|
| 5 | **Tool bloat** | 30–50 tools when <10 are relevant; selection degrades past ~20 | Few high-leverage tools; phase-gate ([04](04_tool_design.md)) |
| 6 | **Hallucinated tool arguments** | Right tool, invented argument values | Validate args against schema before executing ([04](04_tool_design.md) §7) |
| 7 | **Schema drift in tool calls** | Harness built against outdated tool schemas keeps calling obsolete args | Version tool schemas; surface changes |
| 8 | **Chronic tool-call failure rate** | Even good systems see 3–15% per-call failures that compound | Errors-as-feedback + idempotency + retries ([07](07_reliability_and_safety.md)) |

### Data / context-layer failures (~55% — the silent majority)

| # | Anti-pattern | What it is | Fix → |
|---|-------------|-----------|-------|
| 9 | **Dumb RAG / context flooding** | Indiscriminate retrieval buries high-signal tokens | Just-in-time, filtered retrieval ([05](05_context_and_memory.md)) |
| 10 | **Stale context / context drift** | Context outdated when underlying data changes | Freshness signals; retrieve at use-time, not pre-load |
| 11 | **Schema-drift blindness** | No signal when upstream schemas change | Validate against live schema; alert on change |
| 12 | **Uncertified source selection** | Agent uses deprecated/provisional data sources | Mark trusted sources; constrain retrieval |
| 13 | **Missing business context** | Technical schema known, semantic meaning absent | Provide semantics/ownership in tool/data descriptions |

> These are *silent* — they produce no exceptions, just wrong answers. Catch them with evals
> ([08](08_observability_and_evals.md)), not by waiting for a crash.

## 2. Cross-cutting failure modes (named throughout these docs)

- **Context rot** — recall/reasoning degrades as the window fills ([05](05_context_and_memory.md)).
- **Goal drift** — the agent loses track of the original objective over many turns → keep a
  persistent task header ([05](05_context_and_memory.md)).
- **Retry loops / in-context locking** — the model repeats the same failing action or fixates on an
  initial pattern → no-progress detection ([07](07_reliability_and_safety.md)).
- **Unverified progress** — the agent assumes a step succeeded when it didn't → verify work
  ([02](02_anatomy_and_the_loop.md), [04](04_tool_design.md)).
- **Context fragmentation** — multi-agent handoffs lose context → default single-agent
  ([06](06_orchestration_single_vs_multi.md)).

## 3. Engineering anti-patterns (the "over-engineering" failures)

These don't crash; they slowly make the harness unmaintainable and violate "light and modular":

| Anti-pattern | Why it hurts | Instead |
|--------------|-------------|---------|
| **Framework lock-in** | A heavy framework owns your control flow/prompts; you can't see or change behavior | Own the loop, prompts, control flow (12-Factor #2/#8) |
| **Premature multi-agent** | Cost, latency, fragmentation before it's justified | Single agent + workflow patterns first ([06](06_orchestration_single_vs_multi.md)) |
| **Over-abstraction** | Layers/plugins/DSLs nobody needs; every layer is tokens + maintenance | Add a seam only when a second concrete use appears |
| **Hidden control flow** | "Magic" routing you can't debug | Explicit, inspectable code paths |
| **Speculative generality** | Building for imagined future requirements | Build for the task in front of you; YAGNI |
| **Vendor coupling** | Hardcoded to one model/SDK/observability tool | Narrow interfaces + config ([03](03_model_interface.md), [08](08_observability_and_evals.md)) |
| **No evals** | Can't tell improvement from regression; can't swap models safely | A small eval suite in CI ([08](08_observability_and_evals.md)) |

## 4. The Include / Avoid checklist

**Include (the lean core):**

- [ ] A single, readable **agent loop** with append-only state ([02](02_anatomy_and_the_loop.md))
- [ ] A narrow **model-client interface**; provider/model via config; OpenAI-compatible ([03](03_model_interface.md))
- [ ] **Capability detection + graceful degradation** for weak/local models ([03](03_model_interface.md))
- [ ] A small **tool registry** with schema validation and errors-as-feedback ([04](04_tool_design.md))
- [ ] A **context manager** enforcing a token budget; compaction when needed ([05](05_context_and_memory.md))
- [ ] **Guardrails**: iteration cap, spend cap, timeout, no-progress detection ([07](07_reliability_and_safety.md))
- [ ] **Least-privilege** tools + a policy seam for dangerous actions ([07](07_reliability_and_safety.md))
- [ ] **Tracing** (= serialized event log) + a **small eval suite** ([08](08_observability_and_evals.md))

**Add only when justified (optional modules):**

- [ ] Long-term memory / RAG (when tasks outlive one window) ([05](05_context_and_memory.md))
- [ ] Subagents (read-only fan-out, as a tool) ([06](06_orchestration_single_vs_multi.md))
- [ ] MCP adapter (curated external tools) ([04](04_tool_design.md))
- [ ] OTel export + trace UI; production sampling ([08](08_observability_and_evals.md))
- [ ] Sandboxing / approval UI (when tools warrant it) ([07](07_reliability_and_safety.md))

**Avoid (until proven necessary, if ever):**

- [ ] ❌ Multi-agent orchestration by default
- [ ] ❌ A heavyweight framework that owns control flow/prompts
- [ ] ❌ A mega-prompt; a 30+ tool registry
- [ ] ❌ Pre-loading large context "just in case"; dumb RAG
- [ ] ❌ Vendor/model hardcoding; secrets in context
- [ ] ❌ Abstractions/DSLs with a single use site
- [ ] ❌ Shipping without any evals

## What works / What doesn't (summary of summaries)

| ✅ Works | ❌ Doesn't |
|---------|-----------|
| Fix the **context layer** first (most failures live there) | Polishing orchestration while feeding stale context |
| Lean core + clearly-optional modules | Speculative generality and hidden magic |
| Owning loop/prompts/control flow | Framework lock-in |
| Evals as the gate for every change | "Looks fine" + no measurement |

**Next:** [10 — Reference Architecture](10_reference_architecture.md)
