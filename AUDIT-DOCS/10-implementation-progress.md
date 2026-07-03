# 10 — Implementation Progress

> **⚠️ Superseded (2026-07-01):** active tracking has moved to
> [`EXECUTION_PLAN/README.md`](../EXECUTION_PLAN/README.md), which sequences the
> remaining roadmap items plus the adapter-architecture work as per-session phase
> files. This file remains as the historical record of milestones 1–3 (and the
> decided API contract below still stands). Start new sessions from EXECUTION_PLAN.

The execution tracker for the audit-driven improvements. **Every working session
starts by reading this file's *Start here* and ends by updating the *Status* table
+ *Session log* below.** Each milestone is a separate session that ships clean code
*and* the docs it touches.

---

## Start here (cold-start onboarding)

You are improving PyAiHarness against the `AGENTIC-HARNESS-RESEARCH/` build guide,
following the audit in `AUDIT-DOCS/`. Read in this order:

1. `AUDIT-DOCS/00-executive-summary.md` — the verdict + scorecard.
2. `AUDIT-DOCS/09-improvement-roadmap.md` — the prioritized plan (items #1–#13).
3. The `AGENTIC-HARNESS-RESEARCH/0X_*.md` doc for the milestone you're on (each
   roadmap item cites its basis doc).
4. `CLAUDE.md` — conventions and hard rules (do not violate).
5. This file's *Status* + *Session log* — what's done and what's next.

**Two invariants that must survive every change:**
- `agent/loop.py` is the **only** bridge between the LLM client and the MCP
  manager — neither knows the other exists.
- Orchestration is optional and **degrades to a safe default; it must never
  break a request.**

**The decided API contract (don't re-litigate):** two thin HTTP adapters over
one shared core (`api/routes.py::_run_turn`).
- **`POST /chat`** — plain text in / plain text out. Body = prompt; answer =
  body. `X-Session-Id` header continues a session (echoed back, with
  `X-Done-Reason`). No per-call JSON knobs.
- **`POST /v1/chat/completions`** *(Milestone 2, not yet built)* — OpenAI-
  compatible JSON + SSE, plus `GET /v1/models`, so OpenWebUI/LibreChat connect
  natively. Stateless: client re-feeds `messages`.
- `GET /health` stays JSON.
- Per-call `system` → a `/v1` system message; `max_iterations` → config
  (`settings.max_loop_iterations`); `MCP` tool prefs → off the wire
  (orchestrator selects tools; `ToolPreferences` plumbing kept in-process).

**Key file map:**
| Path | What |
|---|---|
| `agent/loop.py` | `run_agent()` — the reasoning loop (the one LLM⇄MCP bridge) |
| `agent/session.py` | `Session`, `SessionStore`, `SessionGuard` |
| `agent/events.py` | typed loop events |
| `api/routes.py` | `/chat`, `/health`, `_run_turn` shared core, `_resolve_routing` |
| `api/schemas.py` | `HealthResponse` + internal `OrchestrationInfo`/`TokenUsage` |
| `orchestrator/` | per-request model + tool + system-prompt router |
| `llm/client.py` | `LLMClient` ABC + `_PROVIDERS` registry (SDK-free) |
| `llm/providers/` | `gemini.py`, `openai.py` (OpenAI-compatible incl. local) |
| `mcp_layer/` | MCP client + manager (`call_tool`, `get_tools_for_llm`) |
| `harness_config.py` | `Settings` + `get_settings()` + MCP config loader |
| `config/models.yaml` | routable model registry (default = local Qwen) |

**Run it:**
```bash
./runscript.sh -m uvicorn main:app --host 0.0.0.0 --port 8000   # server
./runscript.sh tests/smoke_test_http.py                          # one smoke test
for t in tests/smoke_test_*.py; do ./runscript.sh "$t" || break; done
```
Smoke/eval tests are **standalone runnables, never pytest**; each calls
`load_secrets()` first. Full smoke runs need a live LLM + MCP backend.

**Definition of done (every milestone):** clean senior-level code (no bloat /
minimal comments); affected docs updated; the affected smoke test(s) pass;
`.env.example` synced with any new setting; this file updated.

---

## Status

| # | Milestone | Roadmap | State | Updated |
|---|---|---|---|---|
| 1 | Plain-text `/chat` + drop bespoke JSON + doc accuracy | API + #5 | ✅ Done | 2026-06-17 |
| 2 | OpenAI-compatible `/v1` endpoint (+SSE, `/v1/models`) | API | ✅ Done | 2026-06-17 |
| 3 | Observability: JSONL trace + `run_id` | #1 | ✅ Done | 2026-06-18 |
| 4 | Bounded & safe runs (caps, arg-validate, abort) | #2,#3,#4 | ⬜ Not started | — |
| 5 | Evals + CI | #7 | ⬜ Not started | — |
| 6 | Context-assembly seam (budget + compaction) | #6 | ⬜ Not started | — |
| 7 | Policy/auth + capability profiles | #8,#9 | ⬜ Not started | — |

---

## Session log

### Milestone 1 — Plain-text `/chat` + drop bespoke JSON + doc accuracy — 2026-06-17

**Done**
- `api/routes.py`: `/chat` now reads a plain-text body as the prompt and returns
  `PlainTextResponse` + `X-Session-Id` / `X-Done-Reason` headers; `X-Session-Id`
  request header continues a session (empty body → 400, unknown session → 404).
  Extracted `_run_turn()` (the orchestrate→loop→collect core, reused by the
  Milestone 2 `/v1` adapter) and refactored `_resolve_routing()` to take
  `prompt`/`system_override`/`preferences` instead of the old request model.
  Removed `_event_to_dict` and all response-event serialization.
- `api/schemas.py`: removed `ChatRequest`, `Commands`, `MCPServerPreference`,
  `ChatResponse`. Kept `HealthResponse` and the internal `OrchestrationInfo` /
  `TokenUsage` value objects.
- `tests/smoke_test_http.py`: rewritten for the plain-text contract (answer body
  + headers, session continuation, 400 empty body, 404 unknown session).
- Docs: rewrote `docs/api.md` `POST /chat`; updated `docs/architecture.md`
  (request-flow diagram, HTTP-layer flow, principle #7, decisions #19/#21,
  provider listing, events table) and `docs/operations.md` (smoke-test row,
  failure-mode rows) and `README.md` (quick start, API section, features,
  status). Fixed roadmap **#5**: docs now state Gemini **and** OpenAI-compatible
  are implemented and the default runtime is a local Qwen (Anthropic stubbed).

**Verified**
- App + changed modules import cleanly; no dangling refs to removed symbols.
- `/chat` 400 (empty body) and 404 (unknown session) paths confirmed in-process.
- ⚠️ Full end-to-end `smoke_test_http.py` not runnable in this sandbox: the MCP
  backend connected but exposed 0 tools (a `streamablehttp_client` async-generator
  error at startup), so the `tool_count > 0` assertion fails before `/chat` is
  exercised. Re-run against a working LLM + MCP backend to confirm the live path.

**Known follow-ups (not blocking)**
- `README.md`/`docs/architecture.md` reference `harness_client.py` / `chat_client.py`;
  verify these exist or prune the references in a later doc pass.

**Next:** Milestone 2 — add `POST /v1/chat/completions` (non-stream + SSE) and
`GET /v1/models` over `_run_turn`, seeding an ephemeral session from `messages[]`.

### Milestone 2 — OpenAI-compatible `/v1` endpoint (+SSE, `/v1/models`) — 2026-06-17

**Done**
- `api/routes.py`: extracted `_turn_events()` — the single orchestrate→loop seam
  (resolve routing → guard.claim → `append_user` → `run_agent`, yielding the
  loop's events, with the routing + "done" logging moved into it). `_run_turn()`
  is now a thin non-streaming collector over it (`**kwargs` passthrough). Added a
  `model_id` hint to `_resolve_routing`/`_turn_events`: when it names a registered
  model it pins that model while the orchestrator still picks tools + system
  prompt; unknown/absent → orchestrator decides. `/chat` passes `model_id=None`.
- `api/openai_compat.py` (new): the OpenAI adapter — tolerant request parsing
  (`extra="ignore"`, so `temperature` et al. are accepted+ignored), `messages[]`
  → (system override, history, final-user prompt) mapping, fresh ephemeral
  session per request seeded via the `append_*` helpers. `POST
  /v1/chat/completions` returns a `chat.completion` (non-stream) or an
  `EventSourceResponse` of `chat.completion.chunk` frames + `data: [DONE]`
  (stream), mapping each `TextEvent` explicitly to a `delta.content` (never
  `dataclasses.asdict`). `GET /v1/models` lists `registry.model_ids` (or the
  single default when orchestration is off). Failures use the OpenAI error
  envelope; streaming errors become a final SSE `error` frame.
- `api/__init__.py`: combine `routes` + `openai_compat` routers into the one
  exported `router` (main.py unchanged).
- `requirements.txt`: added `sse-starlette>=2.0` (already present in the venv at
  3.4.4). No new settings → `.env.example` unchanged.
- Docs: `docs/api.md` (new `/v1/chat/completions` + `/v1/models` sections + a
  "Connecting OpenWebUI" snippet; de-"planned"-ified the `/chat` cross-refs);
  `README.md` (features, `/v1` API section, streaming/limitation flips, extension
  note); `docs/operations.md` (rewrote "Adding streaming" → "Streaming" as
  shipped, new smoke-test row, "No streaming endpoint" limitation resolved
  for `/v1`).

**Verified**
- `tests/smoke_test_openai_api.py` (new, hermetic — scripted fake LLM + empty
  fake MCP, app.state wired by hand, no lifespan/backend) passes: `/v1/models`
  list shape, non-stream `chat.completion` (content + `usage` + `finish_reason`),
  SSE chunk deltas reassembling to the full answer + `[DONE]`, and OpenAI-style
  400 on empty `messages`.
- App imports cleanly; all four routes register (`/health`, `/chat`,
  `/v1/chat/completions`, `/v1/models`).
- ⚠️ Full e2e against a real model still needs a live LLM+MCP backend (this
  sandbox exposes 0 MCP tools). Point an OpenWebUI "OpenAI API" connection at
  `http://<host>:8000/v1` to confirm the live streaming path.

**Invariants preserved:** the LLM⇄MCP bridge stays solely in `agent/loop.py`
(the adapter only translates wire format and drives `_turn_events`); orchestration
still degrades — the adapter never forces a model and routing falls back exactly
as `/chat` does.

**Next:** Milestone 3 — observability: JSONL trace + `run_id`.

### Milestone 3 — Observability: JSONL trace + `run_id` — 2026-06-18

**Done**
- `agent/tracing.py` (new): the `Tracer` seam, mirroring the `SessionStore` ABC
  — `Tracer` ABC (`emit(record)` + `close()`), a `NoOpTracer`, and a
  `JSONLTracer` (append-only single file, line-per-event, flushed; on the
  single-threaded loop synchronous emits never interleave mid-line). Plus
  `build_tracer(enabled, path)` → tracer or `None` (None when off/unbuildable,
  so the hot path skips emit entirely; a bad sink degrades to `None` with a
  warning, never blocks startup); `run_logger(logger, run_id)` (a
  `LoggerAdapter` that prefixes `[run <id>]`); `event_record(event, run_id,
  step)` (explicit per-type event→JSON map, **never** `dataclasses.asdict`); and
  a `_json_default` that base64-encodes stray `bytes` so `provider_metadata`
  signatures can't crash the serializer. provider_metadata itself is omitted
  from records (opaque round-trip state, not trace signal).
- `agent/events.py`: added `latency_ms: float | None` to `UsageEvent` (LLM
  `complete()` duration incl. retries) and `ToolResultEvent` (`call_tool`
  duration; `None` when stall-skipped). Default-`None` fields, so existing
  consumers (`_run_turn`, the `/v1` stream mapper, `_turn_events`) are undisturbed.
- `agent/loop.py`: `run_agent` takes `tracer` + `run_id`. A single in-loop
  `_emit(event)` helper serializes every yielded event through the tracer
  (wrapped in try/except — tracing never breaks a run) with a monotonic `step`
  counter, then returns the event to `yield`. Wrapped all yield sites
  (`yield await _emit(...)`). Timed `llm.complete()` and `mcp.call_tool()` with
  `time.perf_counter()` and set the results on the usage/tool-result events.
- `harness_config.py` + `.env.example`: new `TRACE_ENABLED` (default `false`)
  and `TRACE_PATH` (default `traces/harness.jsonl`).
- `main.py`: lifespan builds `app.state.tracer = build_tracer(...)` and
  `tracer.close()` on shutdown. `api/dependencies.py`: `get_tracer`. `api/routes.py`:
  `_turn_events` mints the per-request `run_id` (one place — covers `/chat` **and**
  `/v1`, a Milestone 2 payoff), routes its `chat:` log lines through the
  `run_id`-tagged adapter, and threads `tracer` + `run_id` into `run_agent`;
  `/chat` injects `get_tracer`. `api/openai_compat.py`: injects `get_tracer` and
  passes it into the shared `turn_kwargs`.
- **Sensitivity (decided + documented):** the trace captures full prompt text,
  tool args, and tool results by default — appropriate for the single-operator
  dev harness, called out as sensitive in docs/`.env` since there's no auth yet
  (roadmap #8); a metadata-only mode is flagged as the next toggle. The OTel
  exporter is named as the seam's reason, not built (Doc 08 "keep it light").
- Docs: `docs/operations.md` (rewrote "Adding observability" from the decorator
  hand-wave to the shipped trace + a smoke-test row), `docs/architecture.md`
  (events table `latency_ms`, dir listing, `_turn_events` run_id/tracer wiring,
  principle #19 flipped to "serialize explicitly"), `docs/configuration.md` (the
  two env vars), `README.md` (Observable-runs feature, extending bullet, the
  trace-sensitivity limitation).

**Verified**
- `tests/smoke_test_tracing.py` (new, hermetic — scripted fake LLM + fake MCP
  driving `run_agent` directly, JSONLTracer → temp file) passes: one record per
  emitted event in order, stable `run_id`, strictly increasing step indices,
  ISO timestamps on every record, `latency_ms` on the `usage`/`tool_result`
  records, bytes-in-args serialize to base64 without error, and the `tracer=None`
  default path writes nothing and leaves the event stream unchanged.
- Re-ran the hermetic suite green: `smoke_test_reliability`,
  `smoke_test_loop_intelligence`, `smoke_test_openai_api`, `smoke_test_session`,
  `smoke_test_config`. App imports cleanly; all four routes still register.
- ⚠️ Live `/chat`/`/v1` end-to-end still needs a real LLM+MCP backend (this
  sandbox exposes 0 MCP tools). To confirm the on-disk trace, set
  `TRACE_ENABLED=true` against a working backend and tail `traces/harness.jsonl`.

**Invariants preserved:** a `Tracer` is neither the LLM client nor the MCP
manager, so threading it through `run_agent` keeps `agent/loop.py` the sole
bridge; tracing is optional and best-effort (`tracer=None` default, emit failures
swallowed), degrading exactly like orchestration and leaving the no-trace hot
path untouched.

**Next:** Milestone 4 — bounded & safe runs (spend/token + wall-clock caps,
tool-arg validation, abort on consecutive failures).
