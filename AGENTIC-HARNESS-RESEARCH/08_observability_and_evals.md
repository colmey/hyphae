# 08 — Observability & Evals

You cannot improve — or trust — what you can't see and measure. This is the discipline that turns
agent development from guessing into engineering. The good news: the **append-only state log**
([02](02_anatomy_and_the_loop.md) §3) gives you most of it almost for free.

## 1. Observability: the trace is the state log

Instrument the loop so every run produces a connected **trace**: one user input → planning → each
model call → each tool call → results → final answer, with **timing, token counts, and cost** at
each step. Because the harness already keeps an append-only event log as its state, **emitting a
trace is mostly serializing that log** — don't build a parallel logging system.

Capture per step:

| Field | Why |
|-------|-----|
| Step index, parent/run id | Reconstruct the loop; group multi-turn / subagent runs |
| Model request (messages, tools) & response | Debug what the model actually saw and produced |
| Tool name, args, result, success/failure | Find tool misuse / failures / hallucinated args |
| Input/output tokens, cost, latency | Spot context bloat, expensive steps, slow tools |
| Finish reason, guardrail trips | Explain why a run stopped |

### Use OpenTelemetry (portability)

Emit traces via **OpenTelemetry** (with GenAI semantic conventions). It's vendor-neutral: instrument
once, send to any backend (Langfuse, Braintrust, Phoenix, Grafana, …) without re-instrumenting. This
keeps the harness from being welded to one observability vendor — same provider-agnostic ethos as
the model client ([03](03_model_interface.md)).

> Start cheap: structured **JSONL logs of the event log** are a perfectly good v0 and are greppable,
> diffable, and replayable. Add an OTel exporter and a trace UI when manual log-reading hurts.

## 2. Debugging with the log: replay & time-travel

Because state is a pure reduction over an event log:

- **Replay** a recorded run by feeding the logged model responses back (mock the model) →
  deterministic reproduction of any bug.
- **Fork** from any prior step to test a fix ("what if the prompt said X here?").
- **Diff** two runs to see where behavior diverged.

This makes agent bugs — usually maddeningly non-deterministic — tractable.

## 3. Evals: measure outcomes, not vibes

Evals are how you know a prompt/tool/model change helped instead of silently regressing. The most
effective teams run a **continuous loop**: production traces → build eval datasets from real
scenarios → evals drive targeted fixes → fixes generate new traces.

### What to measure

- **Outcome-based first.** Judge the *final result* (did the task succeed?), not the exact path.
  This is essential for non-deterministic agents — many trajectories are valid. Prefer
  programmatic checks (did the file get written correctly? does the code pass tests? is the API
  result right?) over judging prose.
- **Trajectory metrics where they matter** — tool-selection accuracy, step count, token/cost,
  loop/failure rates. Useful for *diagnosis*, not as the primary pass/fail.
- **LLM-as-judge sparingly** — for fuzzy quality (helpfulness, format). Validate the judge against
  human labels; don't trust it blindly.

### How to run them

| Practice | Detail |
|----------|--------|
| **Small, real, versioned dataset** | 20–100 cases beats 0; grow it from real failures. Version it like code. |
| **Run in CI** | Block merges that drop success/safety below a threshold. This is the single highest-leverage habit. |
| **Score next to traces** | Treat eval scores as part of observability — jump from a bad score straight to the failing trace. |
| **Sample production** | Re-run the same evals on sampled live traffic to catch drift over time. |

### Provider-agnostic payoff

A solid eval suite is also your **model-swap safety net**: it's how you compare a frontier hosted
model vs a local Ollama model objectively, and how you catch the quality cliff when you downgrade
for cost/privacy ([03](03_model_interface.md)). Without evals, "does this work on the local model?"
is unanswerable.

## 4. Keep it light

The trap here is buying a heavyweight platform before you have anything to observe. Proportional path:

1. **v0:** structured JSONL trace = the event log; a handful of asserted eval cases run by a script.
2. **v1:** OTel exporter → a trace UI; evals in CI with a threshold gate.
3. **later:** production sampling, drift dashboards, LLM-judge scoring — only if scale demands it.

Each step adds value on its own; none is a prerequisite for shipping.

## What works / What doesn't

| ✅ Works | ❌ Doesn't |
|---------|-----------|
| Trace = serialized event log (free from the state model) | A separate, hand-rolled logging subsystem |
| Per-step tokens/cost/latency/tool I/O captured | Logging only the final answer |
| OpenTelemetry for vendor-neutral export | Welding the harness to one observability SaaS |
| Replay/fork/diff via the event log | Re-running live and hoping to reproduce a bug |
| Outcome-based, programmatic evals in CI | "Looks good to me"; eval only on prose vibes |
| A small real dataset grown from failures, versioned | No evals, or a giant synthetic set nobody trusts |
| Evals as the model-swap safety net | Swapping models with no way to measure regression |
| Starting with JSONL + a script | Buying a platform before you have traces to view |

**Next:** [09 — Anti-Patterns & Checklist](09_antipatterns_and_checklist.md)
