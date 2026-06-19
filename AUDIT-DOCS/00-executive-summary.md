# PyAiHarness — Audit Against the Agentic-Harness Research

**Auditor lens:** senior software engineer.
**Rubric:** `AGENTIC-HARNESS-RESEARCH/` docs 01–10 — the five-principle thesis, the per-doc "What works / What doesn't" boxes, the 13 anti-patterns, the include/avoid checklist, and the reference architecture's 5 seams + phased build path.
**Scope:** the implementation in `agent/`, `llm/`, `mcp_layer/`, `orchestrator/`, `api/`, plus `config/`, `harness_config.py`, `main.py`. Advisory only — no code was changed.

> Each themed file (`01`–`08`) maps one research dimension to the code with `file:line` evidence and a ✅/🟡/🔴 verdict. `09-improvement-roadmap.md` collects every recommendation into a prioritized, impact/effort-rated plan.

---

## Verdict

PyAiHarness is a genuinely **thin deterministic shell with a capable model placed at a few high-leverage points** — the central thesis of the research. It honors the top-line ethos better than most hand-rolled harnesses: a single readable bridge loop, a provider-agnostic OpenAI-compatible client, single-agent-by-default with a *routing-workflow* orchestrator, errors-as-feedback, graceful degradation, and strict input validation. The code is small (~5,200 LOC core), the seams are clean, and the design decisions are documented and deliberate.

The real gaps cluster in **three research dimensions** — and they matter *more* than the project's own docs imply, because **`config/models.yaml` defaults to a local Qwen3.6 model served over an OpenAI-compatible endpoint** (`config/models.yaml`). That is precisely the *weak / small-context model* scenario the research repeatedly warns about. Against that reality:

1. **Context engineering (🔴 Doc 05)** — no token budget, no compaction; full session history is re-sent every iteration. The research frames this as *correctness*, not optimization, and "context rot arrives fast on small windows."
2. **Observability & evals (🔴 Doc 08)** — structured logging exists, but the event stream is never persisted as a trace, and there are **zero evals and no CI**. The research calls a small eval suite "the single highest-leverage habit" and "the model-swap safety net" — exactly the Gemini↔local-Qwen situation this repo lives in.
3. **Reliability / safety (🟡 Doc 07)** — strong on retries/timeouts, but no spend or wall-clock cap, no policy/permissions seam, no pre-execution argument validation, and no auth on a tool-executing HTTP endpoint.

**Secondary finding — stale docs:** `docs/architecture.md` and `docs/operations.md` state that "only Gemini is implemented; OpenAI/Anthropic raise `NotImplementedError`." This is no longer true — `llm/providers/openai.py` (353 LOC) is fully implemented and is the **active default provider**. The docs understate the harness's own provider-agnosticism and misdescribe its default runtime. See `02-model-interface.md`.

---

## Scorecard

| Dimension (research doc) | Verdict | One-line |
|---|---|---|
| Loop & architecture (02, 10) | ✅ **Strong** | One-page bridge loop, clear termination set, no-progress detection; state is not yet a replayable reducer |
| Model interface / provider-agnostic (03) | ✅ **Strong** | Narrow ABC, OpenAI-compatible path live; missing capability detection + token-count fallback |
| Tool design & registry (04) | 🟡 **Partial** | MCP adapter + errors-as-feedback + source truncation; no arg validation, no policy seam |
| Context & memory (05) | 🔴 **Gap** | No token budget, no compaction — acute given the local-model default |
| Orchestration: single vs multi (06, 01) | ✅ **Strong** | Single-agent default; orchestrator is a clean routing workflow that never breaks a request |
| Reliability & safety (07) | 🟡 **Partial** | Great retries/timeouts/stall detection; no spend/wall-clock cap, no policy/auth/sandbox |
| Observability & evals (08) | 🔴 **Gap** | Logging + usage events only; no persisted trace, no request IDs, no evals, no CI |
| Anti-patterns (09) | ✅ **Mostly clean** | Avoids the big ones; live exceptions: no evals, unbounded history, no approval gates |

---

## Top findings (preview — full detail in `09-improvement-roadmap.md`)

**P0 — quick wins (high impact / low effort):**
1. **Persist the event log as a JSONL trace** + add a `request_id` — the events already exist; this is the cheapest path to observability (Doc 08).
2. **Add a spend/token cap and a wall-clock cap** to the loop guard — today only `max_iterations` and per-call timeouts bound a run (Doc 07).
3. **Validate tool arguments against `input_schema` before executing** — catch hallucinated args with a teaching message instead of an opaque MCP error (Doc 04).
4. **Abort/escalate on N consecutive failures** — the nudge exists but never stops the loop (Doc 07).
5. **Fix the stale docs** — OpenAI is implemented; the default is a local model (doc accuracy).

**P1 — medium:**
6. **Context-assembly seam** with a token budget + naive compaction — the one missing *core* module, most needed because of the local-model default (Doc 05).
7. **A minimal programmatic eval suite + CI** — the model-swap safety net (Doc 08).
8. **A policy/permissions seam at dispatch** (read-only default, write allow-list, optional approval) + auth on `/chat` (Doc 07).
9. **Per-model capability profiles + prompted-tool & token-estimate fallbacks** for weak models (Doc 03).

---

## What the harness gets right (credit where due)

A balanced senior-eng read has to start here, because the foundation is strong and most of the gaps are *additive*, not *rework*:

- **The loop is the only bridge** between `LLMClient` and `MCPManager` ([agent/loop.py:62-64](../agent/loop.py#L62-L64)) — a textbook thin-shell boundary; neither subsystem knows the other exists.
- **Provider-agnostic by construction** — a narrow `LLMClient` ABC, lazy SDK imports, a `_PROVIDERS` registry, and a working OpenAI-compatible path that reaches Ollama/vLLM by `base_url`. This is the research's #1 portability recommendation, *implemented*.
- **Single-agent default with a routing workflow** — no multi-agent, no "agents negotiating." The orchestrator is a deterministic router that picks model + tool subset + system prompt in one LLM call and **degrades to a safe default, never 5xx-ing a request**.
- **Errors are feedback, not failures** — tool errors, timeouts, and stalls all return `is_error` results the model can react to ([agent/loop.py:418-460](../agent/loop.py#L418-L460)); only an unrecoverable LLM call kills the loop, cleanly, with `done_reason="llm_error"`.
- **Real reliability primitives** — capped jittered backoff, transient-vs-permanent classification, per-attempt LLM timeout, per-tool timeout, stall detection, final-iteration wrap-up.
- **Strict boundary validation** — `extra="forbid"` on every request schema ("the bouncer").
- **Curated toolset** — the orchestrator selects the *smallest tool subset* per request, actively fighting tool bloat — a discipline most harnesses lack.

The harness has effectively shipped the research's **v0 (walking skeleton)** and most of **v1 (make it safe)**. The roadmap in `09` is about finishing v1 and reaching v2–v3 (durable, measurable) — the phases the research says "anything serious" needs.
