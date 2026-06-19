# 04 — Context Engineering & Memory

**Research basis:** Doc 05 (Context & Memory), Doc 10 (Context seam).
**Verdict:** 🔴 **Gap** — the **largest** divergence from the research, and the most consequential one given the harness runs a **local, small-context model by default**. The Context seam (Doc 10's fourth *core* module) is effectively absent: there is no token budget and no compaction.

> Why this matters more here than the project docs suggest: `config/models.yaml` defaults to a local Qwen3.6 over an OpenAI-compatible endpoint. Doc 05 is explicit that "context rot arrives fast on small local-model windows (4k–32k)" and that on those windows "context management isn't an optimization; it's correctness."

---

## What the research wants (Doc 05)

Find the **smallest set of high-signal tokens** per call. The **context manager** is the single component that assembles the message list each turn and **enforces an explicit token budget** (`budget = context_window − max_output_tokens − safety_margin`); it should **never send a request it knows will overflow**. For long runs, **compaction** summarizes older history while preserving decisions/constraints and dropping redundant tool output. Keep a small always-present **task header** so the goal never rots away. Memory is optional (files-first), retrieval is just-in-time, not pre-loaded.

---

## What the code does

### ✅ Tool output truncated at the source
The one Doc 05 lever fully present: `tool_result_max_chars` clips each tool result before it enters history ([agent/loop.py:136-146](../agent/loop.py#L136-L146)). This prevents a single large output from flooding the window — "the cheapest, most effective tactic before reaching for summarization."

### ✅ Right-altitude system prompts (not a mega-prompt)
The orchestrator generates a **2–6 sentence** system prompt per request (`config/orchestrator_prompt.md` instructs exactly this) rather than a monolithic mega-prompt. Doc 05's "right altitude… grown from real failures" and Doc 09's avoidance of the mega-prompt anti-pattern are honored — the prompt is short, role-oriented, and per-request.

### ✅ The orchestrator disciplines *its own* context
When building its decision prompt, the orchestrator clips conversation history to the **last 6 messages, text blocks only, each ≤500 chars** (`orchestrator/orchestrator.py` `_build_prompt`). The router practices context engineering even though the main loop does not — a telling asymmetry.

---

## Gaps & recommendations

### 🔴 (a) No context manager / no token budget — the missing core module
Doc 10 lists the Context manager as one of four **core** modules, behind the `assemble_context(state) -> messages` seam. The harness has no such seam. `run_agent` passes the **entire** `session.messages` to every `complete()` call ([agent/loop.py:337](../agent/loop.py#L337)) with no budget check anywhere. There is no notion of `context_window`, no `budget = context_window − max_output − margin`, and therefore no deterministic handling of overflow — the failure mode is "discover the limit via a 400 at runtime" (Doc 05's explicit anti-pattern). On a 32k (or smaller) local Qwen window, a moderately long multi-tool run will hit this.

> **Recommendation (P1):** introduce a Context seam — a function `assemble_context(state, model_profile) -> messages` the loop calls instead of reading `session.messages` directly. v0 behavior: pass-through *plus* a budget guard that, when `estimated_tokens > budget`, triggers (b). Pair with the token-estimate fallback from `02-model-interface.md` (local servers under-report usage). Add `context_window` to the per-model profile in `models.yaml`.

### 🔴 (b) No compaction — unbounded history → context rot
Session history grows without bound ([agent/session.py](../agent/session.py) appends only; the in-memory store bounds the *number of sessions*, not the length of any one). Doc 05's compaction recipe — "keep the most recent turns verbatim, summarize the older middle, preserve decisions/constraints, discard redundant tool output" — has no implementation. For the LibreChat-fronted, mostly-short-turn usage the docs describe this is survivable, but any agentic run that takes many tool round-trips on the local model will degrade (forgetting earlier constraints, re-doing work — Doc 05's named symptoms).

> **Recommendation (P1):** implement compaction as a Context-seam strategy: keep the last *N* turns verbatim, summarize the middle via a cheap model call, always retain the system prompt and a task header. Make it a swappable strategy (`naive | compaction | retrieval`) per Doc 10 so the seam stays modular.

### 🟡 (c) No persistent task header (goal-drift risk)
Doc 05 recommends a small always-present "task header (goal + constraints) so it never rots away," and lists **goal drift** as a cross-cutting failure mode. The per-call system prompt partially serves this (it's re-sent every iteration), but the *user's actual goal* lives only in the first user message and can be pushed down by tool output. Low cost to add: pin the original task in the system prompt or as a retained header during compaction.

### ⬜ (d) No working memory / files-as-memory
Doc 05 calls memory optional ("don't build until a task needs to outlive one window"). There is no scratchpad/notes mechanism. Correct to defer — but note it's the natural companion to compaction for long-horizon tasks (roadmap P2 #13).

### ⬜ (e) Retrieval / RAG — N/A
No RAG is present, which is fine — Doc 05 prefers agent-driven retrieval (a `search` tool) over pre-loaded chunks, and the web-search MCP server already provides exactly that model. No "dumb RAG" anti-pattern here. ✅ by absence.

---

## What works / what doesn't — scored

| Doc 05 criterion | Status | Evidence |
|---|---|---|
| Smallest high-signal token set per call | 🔴 | full history every call ([agent/loop.py:337](../agent/loop.py#L337)) |
| Right-altitude system prompt | ✅ | `orchestrator_prompt.md` (2–6 sentences) |
| Explicit, enforced token budget | 🔴 | none |
| Compaction preserving decisions | 🔴 | none; unbounded history |
| Truncate tool output at source | ✅ | [agent/loop.py:136-146](../agent/loop.py#L136-L146) |
| Persistent task header (anti goal-drift) | 🟡 | implicit via re-sent system prompt only |
| Just-in-time, filtered retrieval (no dumb RAG) | ✅ | agent-driven via web-search tool |
| Files-as-memory for long runs | ⬜ | absent (acceptable to defer) |

**Bottom line:** this dimension is where a senior reviewer would push hardest, precisely because the default deployment is the small-window local-model case the research warns about. The fix is *additive* — one new seam (`assemble_context`) the loop calls instead of reading `session.messages` — and it unlocks the budget cap that `06` also needs.
