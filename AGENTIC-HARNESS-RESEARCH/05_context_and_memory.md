# 05 — Context Engineering & Memory

If the loop ([02](02_anatomy_and_the_loop.md)) is the heart, **context is the bloodstream.**
[Context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
is *"curating and maintaining the optimal set of tokens during inference"* — and it matters more
than prompt wording. Owning your context window is 12-Factor #3.

## 1. The governing principle

> **Find the smallest possible set of high-signal tokens that maximize the likelihood of the
> desired outcome.** — Anthropic

The context window is not free storage. The model has a limited **attention budget** (like working
memory). More tokens ≠ more capability — past a point it's the opposite.

### Context rot — the failure you're fighting

As the window fills, the model's ability to accurately recall and reason over its contents
**degrades** — even well within the nominal limit. Symptoms: forgetting earlier constraints,
re-doing work, fixating on stale information. **More relevant on small local-model windows
(4k–32k), where rot arrives fast.** Context management isn't an optimization; it's correctness.

## 2. What's in the window (and who owns each part)

| Component | Owned by | Keep it lean by… |
|-----------|----------|------------------|
| System prompt | Harness | Right "altitude" (§3); load skills/policies on demand |
| Tool schemas | Tool registry ([04](04_tool_design.md)) | Few tools; gate by phase |
| Retrieved knowledge | Memory/RAG (§5) | Retrieve just-in-time; filter for relevance |
| Message history | Context manager (§4) | Compaction; drop redundant tool output |
| Tool results | Dispatcher ([04](04_tool_design.md)) | Truncate/paginate at the source |
| Working memory / notes | Memory (§5) | Externalize to files; reference, don't inline |

The **context manager** is the harness component that assembles this list each turn (`assemble_context`
in [02](02_anatomy_and_the_loop.md)). It is the single place that enforces the token budget.

## 3. System prompt: get the altitude right

Two failure modes, one target:

- **Too specific** — hardcoded if/else logic and brittle rules. Fragile, high-maintenance, fights
  the model.
- **Too vague** — "be helpful" with no concrete behavioral signals.
- **Right altitude** — clear guidance that gives strong signals without micromanaging.

Practices: simple direct language; distinct sections (*background, instructions, tool guidance,
output format*); Markdown headers or XML tags to delineate; **start minimal on the best model, then
add instructions only in response to observed failure modes** — not speculatively. (Weak local
models need a *lower* altitude / more explicit instructions — make that an optional prompt profile,
[03](03_model_interface.md) §4.)

## 4. The four levers (write / select / compress / isolate)

LangChain's taxonomy is the cleanest mental model; map each to a harness mechanism:

| Lever | Meaning | Harness mechanism |
|-------|---------|-------------------|
| **Write** | Author/persist context outside the window | Scratchpad files, todo lists, notes (§5) |
| **Select** | Pull in only what's relevant now | Retrieval, tool-result filtering, phase-gated tools |
| **Compress** | Reduce token cost of what you keep | **Compaction**/summarization, truncation |
| **Isolate** | Keep unrelated context separate | **Subagents** with fresh windows ([06](06_orchestration_single_vs_multi.md)) |

### Compaction (the workhorse for long runs)

When history approaches the budget, summarize it: **preserve** decisions, constraints, open
problems, and key artifacts; **discard** redundant tool output and superseded reasoning. Practical
recipe: *maximize recall first (keep too much), then tighten precision* once you see what the model
actually needs downstream. Keep the most recent turns verbatim; summarize the older middle.

Cheap, effective tactics before you reach for summarization:
- Truncate/paginate tool output **at the source** ([04](04_tool_design.md)).
- Drop or collapse stale tool results once their information is captured elsewhere.
- Keep a small, always-present "task header" (goal + constraints) so it never rots away.

## 5. Memory: working vs long-term

Memory is **optional** — don't build it until a task needs to outlive one window. Two kinds:

| | Working memory | Long-term memory |
|---|----------------|------------------|
| **Scope** | Within a run / across compactions | Across runs / sessions |
| **Form** | Todo list, scratchpad, notes file | Files, KV store, vector DB |
| **Pattern** | **Structured note-taking** — the agent writes notes/todos to a file and re-reads them after a context reset | Write durable facts; retrieve on demand |
| **Why** | Survive compaction; multi-hour coherence | Personalization, accumulated knowledge |

**Filesystem-as-memory** is the lightest durable option and fits our ethos: the agent's notes,
todos, and artifacts are just files it can `read`/`write` via tools. No database required to start.
This is also how subagents hand large artifacts back without flooding the parent's context
([06](06_orchestration_single_vs_multi.md)).

### Retrieval (RAG) — just-in-time, not just-in-case

Pull knowledge into context **when needed**, filtered for relevance/recency — don't pre-load
everything. Indiscriminate retrieval ("dumb RAG"/context flooding) is a named anti-pattern
([09](09_antipatterns_and_checklist.md)): it buries the high-signal tokens. Prefer agent-driven
retrieval (the model calls a `search` tool) over stuffing top-k chunks blindly into the prompt;
it's simpler and the model can iterate.

## 6. Budgeting (make the limit explicit)

The context manager should track an explicit **token budget** derived from the model's context
window ([03](03_model_interface.md) config) minus headroom for the response:

```
budget = context_window - max_output_tokens - safety_margin
if assembled_tokens > budget:  trigger compaction / drop / summarize
```

This is also a guardrail ([02](02_anatomy_and_the_loop.md) §4): never send a request you know will
overflow — handle it deterministically first. Because local servers under-report tokens
([03](03_model_interface.md) §3), keep a local estimate so budgeting still works offline.

## What works / What doesn't

| ✅ Works | ❌ Doesn't |
|---------|-----------|
| Smallest high-signal token set per call | "More context is better"; stuffing the window |
| Right-altitude system prompt, grown from real failures | Mega-prompt with speculative rules ([09](09_antipatterns_and_checklist.md)) |
| Compaction that preserves decisions, drops redundancy | Truncating blindly / never compacting |
| Truncating tool output at the source | Dumping raw tool output into history |
| Structured note-taking + files-as-memory | Relying on the window to "remember" everything (12-Factor #5 drift) |
| Just-in-time, filtered retrieval | Pre-loading top-k chunks ("dumb RAG") |
| An explicit, enforced token budget | Discovering the limit via a 400 at runtime |

**Next:** [06 — Orchestration: Single vs Multi-Agent](06_orchestration_single_vs_multi.md)
