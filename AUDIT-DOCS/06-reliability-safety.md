# 06 — Reliability & Safety

**Research basis:** Doc 07 (Reliability & Safety), Doc 10 (Policy/Guardrails seam).
**Verdict:** 🟡 **Partial** — the **reliability** half (retries, timeouts, error-as-feedback, no-progress detection) is strong and well-engineered. The **safety** half (spend/wall-clock caps, least-privilege, policy/approval, auth, prompt-injection) is largely absent — a real concern for an HTTP service that executes arbitrary MCP tools.

---

## What the research wants (Doc 07)

The core threat is **compounding errors** (0.85¹⁰ ≈ 20%): raise per-step reliability *and* stop cascades early. Use **layered guardrails** checked every turn — max iterations, **token/spend budget**, **wall-clock timeout**, **no-progress detection**, rate limiting, output validation. Classify errors (transient → backoff+jitter; permanent → feed back). Make writes idempotent; checkpoint & resume. Treat tool execution as **untrusted**: least-privilege/read-only by default, a **policy layer at the dispatch seam** (allow/deny/ask), sandbox destructive tools, keep secrets out of context, stay prompt-injection-aware. **Approval gates scaled to stakes.** Start minimal (iter cap + spend cap + timeout + read-only + write allow-list) — a few dozen lines.

---

## What the code does — reliability (strong)

### ✅ Retries with backoff+jitter and error classification
`_complete_with_retry` retries only on conditions the client deems transient and on empty responses, with jittered exponential backoff capped at 30s ([agent/loop.py:149-228](../agent/loop.py#L149-L228)); non-transient exceptions re-raise immediately ([agent/loop.py:195-198](../agent/loop.py#L195-L198)). Classification is delegated to the provider's `is_transient_error` (`02-model-interface.md`). This is Doc 07's "backoff+jitter for transient errors" and "transient vs permanent" split, in the right layer.

### ✅ Layered timeouts
Per-attempt LLM timeout wraps each `complete()` ([agent/loop.py:178-186](../agent/loop.py#L178-L186)); per-tool timeout wraps each `call_tool`, converting a hung tool into an `is_error` result so the loop keeps going ([agent/loop.py:413-426](../agent/loop.py#L413-L426)). A hung call can't stall the request (or lock the session via `SessionGuard`) indefinitely.

### ✅ Errors as feedback + clean fatal path
Tool failures stay in-conversation as `is_error` results; an unrecoverable LLM failure (after retries) yields `ErrorEvent` + `DoneEvent("llm_error")` and stops cleanly ([agent/loop.py:346-352](../agent/loop.py#L346-L352)). Doc 07's "errors fed back as compact, actionable signals" and "fail loud" are honored.

### ✅ No-progress detection
Stall short-circuit on identical tool+args ([agent/loop.py:405-409](../agent/loop.py#L405-L409)) and a consecutive-failure nudge ([agent/loop.py:441-447](../agent/loop.py#L441-L447)) implement Doc 07's loop-detection guardrail.

### ✅ Strict boundary validation + graceful degradation
`extra="forbid"` on request schemas rejects unknown fields with 422 ("the bouncer"); orchestration and MCP startup both degrade rather than fail. `SessionGuard` rejects same-session concurrency with 409.

### ✅ Secrets are not in context
API keys live in env/`Settings` and are referenced by provider builders ([llm/client.py:98-119](../llm/client.py#L98-L119)); they are never injected into messages or tool descriptions. Doc 07's "secrets in tools, never in context" holds. (One thing to keep verifying as config grows: `${ENV_VAR}` interpolation in `mcp_config.yaml` should never resolve a secret into a value that ends up in a tool description sent to the model.)

---

## What the code does — safety (largely absent)

### 🔴 (a) Missing caps: no spend/token budget, no wall-clock
Doc 07's guardrail table lists **token/spend budget** and **wall-clock timeout** as first-class, checked-every-turn caps. The loop gates only on `iteration < max_iterations` ([agent/loop.py:313](../agent/loop.py#L313)). `cumulative` tokens are tracked but never compared to a ceiling ([agent/loop.py:359-360](../agent/loop.py#L359-L360)); there is no run-level wall-clock. `docs/operations.md` acknowledges "no overall wall-clock cap." A misbehaving model burns the full `max_iterations` of real (if individually bounded) calls.

> **Recommendation (P0):** add `max_spend` (token-based when cost is unknown — see token-estimate fallback in `02`) and `wall_clock_seconds` to the loop guard at [agent/loop.py:313](../agent/loop.py#L313). On trip, exit with a distinct `done_reason` (`budget_exceeded` / `deadline_exceeded`) and the partial result — never silently.

### 🔴 (b) No-progress only nudges; never aborts/escalates
Doc 07: detection should "abort or escalate," "change strategy, don't let it grind." The consecutive-failure path appends one nudge at threshold 3 ([agent/loop.py:441-447](../agent/loop.py#L441-L447)) and then keeps looping to `max_iterations`. It informs the model but never stops the cascade.

> **Recommendation (P0):** add a configurable `abort_after_consecutive_failures` (e.g. 5) that exits with `done_reason="no_progress"` and the partial result. Cheap extension of the counter already in place.

### 🔴 (c) No policy / permissions seam, no least-privilege, no approval, no sandbox
This is the largest *safety* gap. Every tool the orchestrator exposed executes unconditionally ([agent/loop.py:412-417](../agent/loop.py#L412-L417), [mcp_layer/manager.py:125-139](../mcp_layer/manager.py#L125-L139)). There is no read-only default, no write allow-list, no `check(state, action) -> allow|deny|ask`, no approval gate for destructive actions, no sandboxing. For a service whose entire purpose is to run third-party MCP tools, Doc 07 calls this "the highest-stakes part."

> **Recommendation (P1):** a Policy seam at the dispatch point (same seam as arg validation in `03-tool-design.md`): `policy.check(state, tool_call)`. Start minimal per Doc 07 — read-only default, an explicit write/destructive allow-list, and an optional `ask` that pauses for approval (human-in-the-loop as a tool result, reusing the existing event/session machinery). Swappable dev-permissive ↔ prod-sandboxed per Doc 10.

### 🔴 (d) No authentication on `/chat`
`docs/operations.md`: "No authentication. `/chat` is wide open." Combined with (c), an unauthenticated caller can drive arbitrary MCP tool execution. Deployment-level concern but real.

> **Recommendation (P1):** API-key/bearer auth dependency on the route (FastAPI `Depends`), even if just a shared secret, before any production exposure.

### 🟡 (e) Prompt-injection unaddressed
MCP tool outputs — including the configured **web-search** server — flow untrusted straight into context ([agent/loop.py:449-463](../agent/loop.py#L449-L463)). Doc 07: "tool outputs are untrusted input that may try to hijack the agent… keep the policy layer authoritative, not the model." There's no marking of tool output as untrusted and no guard against content-driven privilege escalation.

> **Recommendation (P2):** once a policy layer exists (c), keep it authoritative (never let tool-output content widen permissions); consider lightly delimiting/labeling tool output as untrusted in the prompt.

### 🟡 (f) Write-tool idempotency delegated
Doc 07 wants write tools idempotent because "the model will retry." This is delegated entirely to MCP servers; the harness's stall/retry logic could re-drive a non-idempotent write. Note for whoever owns the MCP servers.

---

## What works / what doesn't — scored

| Doc 07 criterion | Status | Evidence |
|---|---|---|
| Layered caps checked every turn | 🟡 | iterations + per-call timeouts; no spend/wall-clock |
| No-progress detection that changes strategy | 🟡 | nudges, never aborts ([agent/loop.py:441-447](../agent/loop.py#L441-L447)) |
| Errors fed back as compact signals | ✅ | [agent/loop.py:418-460](../agent/loop.py#L418-L460) |
| Backoff+jitter for transient errors | ✅ | [agent/loop.py:149-228](../agent/loop.py#L149-L228) |
| Least-privilege + policy at dispatch seam | 🔴 | none |
| Approval gates scaled to stakes | 🔴 | none |
| Secrets in tools, never in context | ✅ | [llm/client.py:98-119](../llm/client.py#L98-L119) |
| Prompt-injection awareness | 🔴 | untrusted tool output flows in raw |
| Checkpoint & resume | 🟡 | save-hooks present; in-memory only |
| Auth on the public endpoint | 🔴 | none |

**Bottom line:** reliability is genuinely strong; the safety gaps are the ones to close before any untrusted/production exposure. The good news per Doc 07 — "minimal guardrails first" — is that caps (a,b) and a basic policy seam (c) are a few dozen lines each and reuse seams the harness already has.
