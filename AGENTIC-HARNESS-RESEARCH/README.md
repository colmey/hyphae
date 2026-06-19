# Building a Light, Modular Agentic Harness — Research & Build Guide

This is a research-backed **build guide** for engineering our own agentic AI harness. It distills
what works and what doesn't from production agent systems (2024–2026) into prescriptive guidance,
ending in an opinionated minimal reference design.

> **A "harness" is the deterministic software that wraps a language model and turns it into an
> agent**: the loop that calls the model, parses its tool calls, executes them, feeds results back,
> manages context, and enforces limits. The model supplies intelligence; the harness supplies
> structure, safety, and integration.

## The thesis (read this first)

Five principles drive every recommendation in these docs:

1. **Thin deterministic shell + capable model.** The best harnesses are *mostly ordinary software*
   with LLM calls placed at a few high-leverage points — not clever frameworks that try to "think"
   for the model. Own your control flow, state, and prompts explicitly.
2. **Lean on the model; add scaffolding only when it pays for itself.** Start with the simplest
   thing that works (a single loop, a handful of tools). Every abstraction you add is context the
   model must navigate and code you must maintain. Complexity must earn its place with a measurable
   win.
3. **Light and modular.** Small, swappable components behind narrow interfaces (model client, tool
   registry, context manager, policy/guardrails, observability). You should be able to replace any
   one without touching the others — and delete any one you don't need.
4. **Provider- and model-agnostic.** Target the **OpenAI-compatible API** as the integration
   lingua franca so the same harness runs against hosted models *and* local models via **Ollama**
   (and vLLM, LM Studio, OpenRouter, …) by swapping a `base_url`. Assume some models will be weak:
   **degrade gracefully** for unreliable tool-calling and small context windows.
5. **Measure before you trust.** Traces and evals are how you know a change helped. Build the
   cheapest version of this early; it's the difference between engineering and guessing.

## How to read these docs

Read top-to-bottom for a full build guide, or jump to the topic you're designing.

| # | Doc | What it answers |
|---|-----|-----------------|
| — | [README.md](README.md) | The thesis, glossary, navigation (you are here). |
| 01 | [Landscape & Principles](01_landscape_and_principles.md) | What kinds of harnesses exist, the workflow↔agent spectrum, the foundational principles. |
| 02 | [Anatomy & the Agent Loop](02_anatomy_and_the_loop.md) | The minimal viable harness: the loop, state, termination. MUST-have vs optional. |
| 03 | [Model Interface](03_model_interface.md) | Provider-agnostic model client; OpenAI-compatible API; Ollama/local; weak-model degradation. |
| 04 | [Tool Design](04_tool_design.md) | The agent–computer interface; effective tools; native vs prompted tool-calling; MCP. |
| 05 | [Context & Memory](05_context_and_memory.md) | Context engineering, compaction, memory, retrieval, context rot. |
| 06 | [Orchestration: Single vs Multi-Agent](06_orchestration_single_vs_multi.md) | When (not) to use subagents; the Cognition⇄Anthropic debate; workflow patterns. |
| 07 | [Reliability & Safety](07_reliability_and_safety.md) | Error handling, retries, guardrails, sandboxing, human-in-the-loop. |
| 08 | [Observability & Evals](08_observability_and_evals.md) | Tracing, logging, eval-driven development, debugging. |
| 09 | [Anti-Patterns & Checklist](09_antipatterns_and_checklist.md) | What *not* to do; a practical include/avoid checklist. |
| 10 | [Reference Architecture](10_reference_architecture.md) | The payoff: an opinionated minimal modular design + phased build path. |

Each topical doc (01–09) ends with a compact **"What works / What doesn't"** box.

## Glossary

- **Harness** — the deterministic software around the model (loop, tools, context, limits).
- **Agent** — a system where the model directs its own actions over multiple turns via tools.
- **Workflow** — a system where the model is orchestrated through *predefined* code paths.
- **Agent loop** — the cycle: gather context → call model → execute tool calls → feed results back → repeat until done.
- **Tool / function** — a typed capability the model can invoke (e.g. `read_file`, `http_get`).
- **Tool call** — a structured request emitted by the model to invoke a tool with arguments.
- **Context window** — the token budget the model sees on each call (system prompt + history + tools + results).
- **Context engineering** — curating the smallest high-signal set of tokens for each model call.
- **Context rot** — degraded recall/reasoning as the window fills with low-signal tokens.
- **Compaction** — summarizing history to reclaim context while preserving decisions.
- **Subagent** — a child agent invoked with its own fresh context that returns a condensed result.
- **MCP** — Model Context Protocol, an open standard for exposing tools/data to agents as plug-ins.
- **OpenAI-compatible API** — the `/v1/chat/completions` request/response shape that Ollama, vLLM, LM Studio, OpenRouter, and others implement, enabling provider portability.

## Scope (and deliberate non-scope)

These are **research/design docs**, not an implementation. We deliberately avoid an exhaustive
framework-by-framework survey and vendor-specific deep dives — this is a build guide, and keeping
*the guide itself* light is part of the point.

## Primary sources

- Anthropic — [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- Anthropic — [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- Anthropic — [Writing Effective Tools for AI Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Cognition — [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents)
- HumanLayer — [12-Factor Agents](https://github.com/humanlayer/12-factor-agents)
- [Model Context Protocol](https://modelcontextprotocol.io)
- Atlan — [Agent Harness Failures: 13 Anti-Patterns](https://atlan.com/know/agent-harness-failures-anti-patterns/)
