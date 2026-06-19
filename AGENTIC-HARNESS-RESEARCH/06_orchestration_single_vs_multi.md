# 06 — Orchestration: Single vs Multi-Agent

The most over-reached-for complexity in agent engineering is multi-agent. This doc gives you a
default and the criteria to deviate from it.

## TL;DR

> **Default to a single agent. Add subagents only for read-heavy, parallelizable fan-out where
> context isolation clearly helps — and make them an optional module, not the core architecture.**

## 1. The debate (and why both sides are right)

Two respected teams published seemingly opposite advice within months:

- **Cognition — [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents):**
  multi-agent systems are fragile because **context fragments**. When you split work across agents,
  decisions and assumptions made by one are invisible to another → conflicting actions, lost
  context, brittle results. Their prescription: keep context **continuous** in a single thread;
  invest in context engineering.
- **Anthropic — multi-agent research system:** a lead + parallel subagents **outperformed** a
  single agent on broad research tasks, by letting each subagent explore with a **clean, isolated
  context** and return a condensed summary.

These aren't actually contradictory. The deciding variable is **whether subtasks are independent and
read-mostly**:

| | Favors single agent | Favors subagents |
|---|--------------------|------------------|
| Task shape | Sequential, interdependent, stateful **writes** | Broad, parallel, **independent** exploration |
| Context need | Shared, continuous thread | Isolated per branch; only summaries merge |
| Failure cost | Coordination errors are catastrophic | Branches fail independently |
| Example | Editing a codebase; a multi-step transaction | "Research these 8 topics"; search/gather/summarize |

Cognition's coding agent writes interdependent changes (single agent wins). Anthropic's research
agent fans out independent reads (subagents win). **Match the architecture to the task, not the
hype.**

## 2. Why multi-agent is risky by default

- **Context fragmentation** — the #1 cause of multi-agent failure. Sub-results assembled from agents
  working on conflicting assumptions don't cohere.
- **Coordination cost** — every handoff is a place to lose information; merging outputs is its own
  hard problem.
- **Token/latency multiplier** — running N agents multiplies cost and adds orchestration latency.
- **Debuggability collapses** — interleaved traces across agents are far harder to reason about
  ([08](08_observability_and_evals.md)).
- **Compounding errors across boundaries** — a named anti-pattern ([09](09_antipatterns_and_checklist.md)).

You pay all of this up front, often before you have evals to prove it helps.

## 3. The recommended pattern: subagents as an optional tool

When you *do* need parallel exploration, the lightest safe form is a **read-only subagent exposed as
a tool** — it reuses the same loop ([02](02_anatomy_and_the_loop.md)), nothing new architecturally:

```
parent agent
   └─ calls tool: spawn_subagent(task, allowed_tools=READ_ONLY)
         └─ runs the SAME loop with a FRESH context window
         └─ returns a CONDENSED summary (~1–2k tokens), not its transcript
```

Design rules that keep it safe and light:

- **Fresh, isolated context** per subagent (the whole point — "isolate" lever, [05](05_context_and_memory.md)).
- **Read-only by default.** Writes are interdependent and belong in the main thread (Cognition's
  lesson). Let subagents gather; let the parent decide and act.
- **Return summaries, not transcripts.** Hand back ~1–2k tokens, or write large artifacts to files
  and return a path ([05](05_context_and_memory.md) §5).
- **Just another tool.** No new orchestration engine — `spawn_subagent` is a tool in the registry,
  using the existing loop with a tool-subset. This is the modular way to add the capability.
- **Bounded fan-out.** Cap concurrent subagents (cost + rate limits).

## 4. Prefer workflow patterns to "agents talking to agents"

Before any multi-*agent* design, ask whether a deterministic **workflow** pattern
([01](01_landscape_and_principles.md)) does the job with far less risk:

| Need | Lighter alternative than multi-agent |
|------|--------------------------------------|
| Different handling per input type | **Routing** (classify → handler) |
| Independent subtasks | **Parallelization** (fan out tool calls / subagents, aggregate) |
| Unknown subtasks at runtime | **Orchestrator–worker** (one lead, ephemeral workers) |
| Quality via iteration | **Evaluator–optimizer** (generate → critique → revise) |

These keep control flow in *your* deterministic code (12-Factor #8) instead of emergent agent
negotiation. The orchestrator–worker pattern is essentially the subagent-as-tool pattern above.

## 5. Local-model note

Multi-agent multiplies the weak points of weak models ([03](03_model_interface.md)): more tool
calls to get wrong, more summaries to mangle, more coordination to drop. **On local/small models,
strongly prefer single-agent + workflows.** If you fan out, keep each subagent's job tiny and
heavily validated.

## What works / What doesn't

| ✅ Works | ❌ Doesn't |
|---------|-----------|
| Single agent as the default | Multi-agent because it sounds more capable |
| Subagents for parallel, **read-only** fan-out | Subagents performing interdependent writes |
| Subagents returning condensed summaries / file paths | Merging full subagent transcripts into the parent |
| Subagent-as-a-tool reusing the one loop | A separate orchestration framework/DSL |
| Workflow patterns for predictable structure | "Agents negotiating" for tasks a router would solve |
| Bounded, single-agent designs on local models | Heavy multi-agent topologies on small models |

**Next:** [07 — Reliability & Safety](07_reliability_and_safety.md)
