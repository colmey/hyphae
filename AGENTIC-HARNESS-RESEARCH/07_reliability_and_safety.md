# 07 — Reliability & Safety

Agents act in the world, non-deterministically, in a loop. Without guardrails, small per-step error
rates **compound** and autonomy becomes a liability. This doc covers making the harness robust and
safe — cheaply.

## 1. The core threat: compounding errors

Per-step errors multiply across a multi-step task:

> At 85% per-step success, a 10-step task succeeds only **0.85¹⁰ ≈ 20%** of the time.

Two implications:
1. **Raise per-step reliability** (good tools, validation, verification — [04](04_tool_design.md)).
2. **Stop cascades early** (the guardrails below). You cannot eliminate per-step error; you can stop
   it from snowballing.

## 2. Layered guardrails (defense in depth)

These are cheap, deterministic, and belong to the harness — not the model's discretion. Most were
introduced as stop conditions in [02](02_anatomy_and_the_loop.md) §4; collected here as the safety
layer:

| Guardrail | Protects against | Default |
|-----------|------------------|---------|
| **Max iterations** | Infinite/runaway loops | 20–50, task-tuned |
| **Token / spend budget** | Cost blowups | hard per-task ceiling |
| **Wall-clock timeout** (run + per-tool) | Hung tools, latency | task-tuned |
| **No-progress / loop detection** | Repeated identical calls, repeated failures | abort or escalate |
| **Rate limiting** | API throttling, runaway external calls | per-tool / per-provider |
| **Output validation** | Malformed/unsafe outputs | schema-validate |

**Loop/no-progress detection** deserves emphasis — it catches the "retry loop" and "in-context
locking" failure modes ([09](09_antipatterns_and_checklist.md)): if the model repeats the same
tool+args or fails the same way twice, change strategy — inject a hint, escalate to a human, or
abort. Don't let it grind.

## 3. Error handling & recovery

- **Errors are feedback** ([04](04_tool_design.md) §6, 12-Factor #9): compact the failure into a
  clear, actionable result and let the model retry differently. This is the first line of recovery.
- **Classify errors** (transient vs permanent):
  - *Transient* (429, 5xx, timeouts) → retry with **exponential backoff + jitter**, capped. Lives
    in the model client / tool layer ([03](03_model_interface.md) §5), not the loop.
  - *Permanent* (bad args, not-found, unauthorized) → don't retry blindly; feed back so the model
    corrects, or escalate.
- **Idempotency** — design write-tools so a retry doesn't double-apply (idempotency keys, "create if
  not exists"). Critical because the model *will* retry.
- **Checkpoint & resume** — persist `state` ([02](02_anatomy_and_the_loop.md) §3) so a crash or
  timeout resumes rather than restarts. Pairs with 12-Factor #6 (launch/pause/resume).
- **Fail loud, return partials** — when a cap trips, return what was accomplished + the reason.
  Never pretend success (the "unverified progress" failure mode).

## 4. Permissions & sandboxing (the highest-stakes part)

Tools like shell, file-write/delete, network, and anything that spends money are where an agent can
do real damage. Treat tool execution as **untrusted by default**.

- **Least privilege** — each run gets only the tools and scopes it needs. Read-only by default;
  grant write/exec explicitly. Subagents inherit a *narrower* set ([06](06_orchestration_single_vs_multi.md)).
- **A policy layer at the dispatch seam** ([04](04_tool_design.md) §7) decides allow / deny /
  ask-human per tool call, by tool + arguments (e.g. allow writes under `./workspace`, deny outside).
- **Sandbox destructive tools** — run shell/code in a container, VM, or jailed working directory
  with no host credentials and a scoped filesystem. Network egress allow-listed.
- **Protect secrets** — never put API keys/tokens in the model's context. Tools hold credentials;
  the model references resources by name. (Prevents leakage via output and prompt injection.)
- **Prompt-injection awareness** — tool *outputs* (web pages, files, tickets) are untrusted input
  that may try to hijack the agent. Don't auto-escalate privileges based on content the model read;
  keep the policy layer authoritative, not the model.

## 5. Human-in-the-loop (12-Factor #7)

Human oversight is **just another tool call** — uniform, inspectable, resumable:

```
model → request_approval(action="delete 3 files", detail=...)   # pauses the run
human → approve / reject / edit
loop  → resumes from persisted state with the decision as a tool result
```

Calibrate autonomy to stakes — avoid "all-or-nothing autonomy" ([09](09_antipatterns_and_checklist.md)):

| Action stakes | Mode |
|---------------|------|
| Read-only, reversible | Full autonomy |
| Writes, moderate cost | Autonomous within bounds; log + allow review |
| Destructive, expensive, irreversible | **Approval gate** before execution |

Because human contact is a tool, the same pause/resume machinery serves "ask the user a question,"
"approve this action," and "escalate on repeated failure."

## 6. Keep it proportional

Don't build a policy engine, sandbox cluster, and approval workflow on day one. **Start with: iteration cap +
spend cap + timeout + read-only-by-default + a hardcoded allow-list for write tools.** That covers
the vast majority of risk in a few dozen lines. Add the heavier machinery (containers, fine-grained
policy, approval UI) when the tools or the deployment actually warrant it. Over-built safety is
still over-engineering.

## What works / What doesn't

| ✅ Works | ❌ Doesn't |
|---------|-----------|
| Layered caps (iterations, spend, time) checked every turn | A single safety net (or none) |
| Loop/no-progress detection that changes strategy | Letting the model retry the same failing call forever |
| Errors fed back as compact, actionable signals | Crashing, or dumping tracebacks into context |
| Backoff+jitter for transient errors; idempotent writes | Naive immediate retries; non-idempotent side effects |
| Least-privilege tools + policy at the dispatch seam | Full filesystem/shell access by default |
| Secrets in tools, never in context | API keys in the prompt/history |
| Approval gates scaled to action stakes | All-or-nothing autonomy on destructive actions |
| Minimal guardrails first, heavier ones when warranted | A full policy/sandbox platform before it's needed |

**Next:** [08 — Observability & Evals](08_observability_and_evals.md)
