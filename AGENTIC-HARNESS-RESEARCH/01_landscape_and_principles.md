# 01 — Landscape & Foundational Principles

Before designing anything, understand the design space and the principles that separate harnesses
that survive contact with production from ones that don't.

## 1. What kind of thing are we building?

"Agent harness" is overloaded. There are three layers, and we are building the **first** one:

| Layer | What it is | Examples | Our stance |
|-------|-----------|----------|-----------|
| **Harness / runtime** (our target) | The loop + tool execution + context + limits. The engine. | Claude Agent SDK, OpenAI Agents SDK, smolagents, the core of any coding agent | **Build this**, light and modular. |
| **Framework** | Opinionated orchestration, graphs, state machines, abstractions on top of the runtime | LangGraph, CrewAI, AutoGen | Borrow ideas; **don't adopt wholesale** — they trade control for convenience. |
| **Application** | A finished product wrapping a harness | Coding assistants, research agents, support bots | Out of scope; it's what you build *with* the harness. |

A harness can be as small as ~200 lines: a loop, a model client, a tool dispatcher, and some
limits. Everything else is optional and should be added only when justified.

## 2. The workflow ↔ agent spectrum

Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
draws the most useful distinction in the field:

- **Workflow** — LLMs and tools orchestrated through **predefined code paths**. Predictable,
  cheaper, easier to debug. Best when you can enumerate the steps.
- **Agent** — the LLM **dynamically directs its own process** and tool use over many turns. Best
  for open-ended problems where you can't predict the steps in advance.

They are a spectrum, not a binary. **Most production "agents" are mostly workflow with a small
agentic core.** The mistake is reaching for full autonomy when a fixed pipeline would be more
reliable and cheaper.

### The five workflow patterns (use before reaching for full autonomy)

| Pattern | Shape | Use when |
|---------|-------|----------|
| **Prompt chaining** | Output of step N → input of step N+1 | Task decomposes into fixed sequential subtasks |
| **Routing** | Classify input → dispatch to a specialized handler | Distinct input categories need different handling |
| **Parallelization** | Fan out subtasks → aggregate | Independent subtasks, or voting/consensus |
| **Orchestrator–worker** | A lead delegates dynamic subtasks to workers | Subtasks unknown until runtime |
| **Evaluator–optimizer** | Generate → critique → revise loop | Quality improves with iterative feedback |

**Design rule:** reach for a workflow pattern first. Use a full agent loop only when the steps
genuinely cannot be predetermined. (See [06](06_orchestration_single_vs_multi.md).)

## 3. Foundational principles

### 3.1 Start with the simplest thing that works

> "Find the simplest solution possible, and only increase complexity when needed." — Anthropic

Agentic systems trade **latency and cost** for flexibility. If a single model call with a good
prompt solves the task, don't build an agent. If a fixed three-step chain works, don't add a loop.
The harness should make the simple case trivial and the complex case *possible* — not force
ceremony on every task.

### 3.2 The 12-Factor Agents discipline

[12-Factor Agents](https://github.com/humanlayer/12-factor-agents) (HumanLayer) is the most
practical articulation of "agents are mostly software." The factors most load-bearing for a harness:

| Factor | Why it matters for the harness |
|--------|--------------------------------|
| 1. Natural language → tool calls | The core transduction: model emits structured calls, your code executes them. |
| 2. **Own your prompts** | Prompts are source code. Version them; don't bury them in a framework. |
| 3. **Own your context window** | You decide exactly what tokens go in each call. (See [05](05_context_and_memory.md).) |
| 4. Tools are just structured outputs | A "tool call" is just JSON the model produced; treat it as data. |
| 5. Unify execution state & business state | One source of truth; don't let the model's context drift from reality. |
| 6. Launch/Pause/Resume with simple APIs | Make runs interruptible and resumable. |
| 7. Contact humans with tool calls | Human-in-the-loop is just another tool. (See [07](07_reliability_and_safety.md).) |
| 8. **Own your control flow** | Explicit code decides what happens next — not hidden framework magic. |
| 9. Compact errors into context | Feed failures back to the model concisely so it can self-correct. |
| 10. **Small, focused agents** | Narrow scope beats sprawling autonomy. |
| 11. Trigger from anywhere | CLI, API, queue, cron — entry points are decoupled from the core. |
| 12. **Stateless reducer** | `next_state = f(state, event)`. Pure, testable, resumable. (See [02](02_anatomy_and_the_loop.md).) |

Core thesis: a good agent is *"mostly deterministic code, with LLM steps sprinkled in at just the
right points."* This is the single most important framing in this guide.

### 3.3 The agent loop as the universal skeleton

Independent of vendor, the loop reduces to three repeating phases (the framing the Claude Agent SDK
makes explicit, but it's universal):

```
gather context  →  take action (call model + run tools)  →  verify work  →  repeat until done
```

Everything in docs 03–08 is about doing one of these three phases well. (Detailed in [02](02_anatomy_and_the_loop.md).)

### 3.4 The "bitter lesson" applied to harnesses

General methods that leverage the model's own capability beat hand-engineered scaffolding that
encodes our assumptions. As models improve, **scaffolding that compensated for weak models becomes
dead weight** — and actively harmful, because it adds context and constrains the model. Prefer
giving the model good tools and clear context over encoding rigid procedures. (The exception: weak
*local* models genuinely need more guardrails — see [03](03_model_interface.md). Make that
scaffolding *optional and configurable*, not baked into the core.)

### 3.5 Modularity is a context-budget decision, not just clean code

Every tool, instruction, and abstraction you add is **tokens the model must process and code you
must maintain**. Modularity here isn't aesthetic — it's how you keep the context lean and the
system debuggable. A module you can delete is a module that isn't costing you tokens.

## What works / What doesn't

| ✅ Works | ❌ Doesn't |
|---------|-----------|
| Starting from a workflow and adding autonomy only where needed | Building full autonomy first because it sounds powerful |
| Treating the harness as mostly deterministic software | Letting a framework own your control flow and prompts |
| A small core (loop + tools + limits) with optional modules | A monolithic framework you adopt wholesale |
| Owning prompts, context, and control flow explicitly | Hidden orchestration "magic" you can't inspect or debug |
| Picking the simplest pattern that solves the task | Reaching for multi-agent / graphs by default |

**Next:** [02 — Anatomy & the Agent Loop](02_anatomy_and_the_loop.md)
