# hyphae — Operations & Extending

How to run the harness, read its logs, diagnose failures, run its tests,
and extend it along its designed seams. Plus the known v1
limitations and their extension paths.

> See also: [README.md](README.md) (overview + quick start),
> [architecture.md](architecture.md) (subsystems & design decisions),
> [configuration.md](configuration.md) (env vars and config files),
> [api.md](api.md) (HTTP contract).

## Contents

1. [Extending the Harness](#extending-the-harness)
2. [Operations](#operations)
3. [Tests](#tests)
4. [Eval Suite](#eval-suite)
5. [Known Limitations](#known-limitations)

---

## Extending the Harness

### Adding a routable model

If the provider is already registered in `llm/client.py`'s `_PROVIDERS`
(implementation under `llm/providers/`), adding a new model is
**configuration-only**:

1. Open `config/models.yaml`.
2. Add an entry:
   ```yaml
   models:
     # ... existing entries ...
     qwen3-local:
       provider: openai
       model: qwen3.6-35b-a3b
       description: >
         Local Qwen3.6 model; strong for agentic coding and deliberate
         tool-using workflows.
       context_window: 65536
       max_tokens: 16384
       supports_native_tools: true
       thinking: think-tags
       sampling:
         temperature: 0.6
         top_p: 0.95
         top_k: 20
   ```
3. Restart the harness. The orchestrator can now route to it.

The orchestrator's selection criteria are governed entirely by what's in
the `description` field — write a description that explains when this
model should win. Capability-profile fields (`supports_native_tools`,
`thinking`, and `sampling`) describe how the provider should call the served
model; they do not replace the description used for routing.

### Adding a new LLM provider

Two steps — the harness is provider-blind everywhere else:

1. **Write `llm/providers/<name>.py`** with a class that subclasses
   `LLMClient` and implements `async complete(request: GenerationRequest)`
   (handle `request.response_schema` if you want orchestrator support, and
   `request.thinking_level` if the provider has a deliberation control —
   ignore it if not). Translate the request's internal `Message` sequence to the
   provider's request format and the response back to `AssistantMessage`
   with `TextBlock`s / `ToolUseBlock`s. If the provider has opaque
   round-trip state (signatures, reasoning traces, etc.), stash it in
   `TextBlock.provider_metadata` / `ToolUseBlock.provider_metadata` on
   parse and re-attach it on the next request. Override `is_transient_error`
   for the provider's retryable failures. Keep the provider SDK import inside
   this file only. If native streaming is implemented, accept the same request
   and reject a non-`None` `response_schema` before calling the SDK.
2. **Register it** in `llm/client.py`: add a small lazy builder (3 lines,
   `from llm.providers.<name> import ...` inside the function) and one entry
   to `_PROVIDERS`.

That's it. `models.yaml` accepts the new provider automatically (validated
against `supported_providers()`), credentials resolve via the typed
`<provider>_api_key` field or the `<PROVIDER>_API_KEY` env var fallback (or
not at all, for a key-less local provider). Optionally add a typed key field
in `Settings`, set `LLM_PROVIDER=...` as the legacy default, and/or add
`models.yaml` entries to route specific requests to it.

### Connecting a local Ollama (OpenAI-compatible) model

The `openai` provider (`llm/providers/openai.py`) speaks the OpenAI wire
protocol, so it also drives any OpenAI-compatible server — including a local
Ollama instance — by pointing it at an alternate endpoint:

1. Export `OPENAI_BASE_URL=http://localhost:11434/v1` and `OPENAI_API_KEY=<key>`
   (Ollama may ignore the key, but the SDK requires a non-empty value).
2. Add a `models.yaml` entry with `provider: openai` and a `model` matching an
   `ollama list` tag, e.g. `qwen3.6-35b-a3b`.
3. Restart the harness.

Caveats: tool calling only works with tools-capable models. Reasoning models
(e.g. Qwen3) may inline a leading `<think>...</think>` block; declare
`thinking: think-tags` so the OpenAI-compatible client removes it from visible
answer text and records it as trace-only reasoning. Endpoints that accept a
request hint such as `reasoning_effort` should use `thinking: hint-param`.
See [configuration.md](configuration.md) for the env vars and profile fields.

`/v1/chat/completions` with `stream:true` uses the provider's native token
stream when the selected OpenAI-compatible model client supports it. The
existing `LLM_TIMEOUT_SECONDS` caps each incremental provider read for this
path, so the idle timeout resets after every chunk while `MAX_RUN_SECONDS`
continues shrinking absolutely. After the first text delta is emitted, the
harness does not retry or resume a broken stream. A later provider failure is
surfaced as partial text followed by an error event and
`done_reason=llm_error`.

Local harness verification:

```bash
./runscript.sh -m uvicorn main:app --host 127.0.0.1 --port 8000
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"gpt-oss-20b","stream":true,"messages":[{"role":"user","content":"Write three short sentences slowly."}]}'
```

If OpenWebUI reaches the harness through Nginx Proxy Manager, proxy buffering
can make correct harness-local token streaming appear all-at-once. Disable
buffering for the proxy host (`proxy_buffering off` / `X-Accel-Buffering: no`)
before treating an end-to-end OpenWebUI test as authoritative.

### Connecting a weak prose-only tool model

If a served model cannot reliably emit native OpenAI/Gemini tool calls but can
follow JSON-in-prose instructions, keep the same provider and add a model row
with `supports_native_tools: false`:

```yaml
models:
  weak-local-prompted:
    provider: openai
    model: small-local-model
    description: >
      Local prose-only model with no reliable native tool calling. Use for
      lightweight tasks where prompted JSON tool actions are acceptable.
    supports_native_tools: false
    thinking: none
    context_window: 32768
    max_tokens: 4096
```

No provider code or loop changes are required. The registry builds the normal
provider client, wraps it with the prompted-tool adapter, renders the
orchestrator-selected tools into the system prompt, and parses one JSON action
back into the same `ToolUseBlock` shape native models produce. Do not use a
prompted-only row as the orchestrator control model; startup will log a warning
and disable orchestration because the control path depends on structured output.

### Adding persistence

Not needed for the current use case — LibreChat is the system of record and
re-feeds conversation context, so server-side sessions are short-lived
scratchpads. The `SessionStore` ABC is kept solely as the seam for if that
ever changes. To add a durable backend:

1. Write a new class in `agent/session.py` (or a new file) that
   implements `SessionStore.create / get / save` against a real backend
   (SQLite, Postgres, Redis). Note `provider_metadata` holds raw **bytes**
   (Gemini's `thought_signature`), so the serializer needs base64/binary
   handling, not naive JSON.
2. Replace `InMemorySessionStore(...)` in `main.py`'s lifespan with the new
   class.
3. For a multi-process / multi-host deployment, promote `SessionGuard` from
   its in-process `set` to a DB/Redis-level claim (or use optimistic
   versioning) so the same-session guard holds across workers. Single
   process needs no change — the existing guard already covers it.

That's it. No other code changes.

### Streaming

Streaming is implemented on the OpenAI-compatible adapter: `POST
/v1/chat/completions` with `stream: true` returns an SSE stream of
`chat.completion.chunk` frames terminated by `data: [DONE]` (see
[api.md](api.md#post-v1chatcompletions)). It is built on `sse-starlette`'s
`EventSourceResponse` and iterates the shared `_turn_events` generator in
`api/routes.py`, mapping each `TextEvent` to a `delta.content`. The loop is
already an async generator, so no loop change was needed — orchestration runs
once (inside `_turn_events`) before the first frame is emitted.

For a *live activity feed* — seeing the agent's tool calls and results as they
happen, not just the final answer — use the native **`POST /chat/stream`** route
(`api/routes.py`). Same dumb-pipe contract as `/chat` (plain-text body = prompt,
optional `X-Session-Id`), but instead of collecting the events it forwards the
loop's typed events over SSE as they occur: `text`, `tool_call`, `tool_result`,
`usage`, `done`, `error` (see [api.md](api.md#post-chatstream)). It is the same
kind of thin renderer as the `/v1` SSE branch — it iterates the shared
`_turn_events` generator and serializes each event with the bytes-safe
`event_record` mapping (the same one the tracer uses; never `dataclasses.asdict`,
since `provider_metadata` can hold bytes — events don't carry it, but reusing the
explicit mapping keeps one source of truth). Reasoning extracted from model
responses is included here as a `reasoning` event because this route is the raw
debug event stream; `/chat` and `/v1` hide it. No orchestration or loop logic
is duplicated; orchestration still runs once inside `_turn_events` before the
first frame.

This is the design seam for *any* live-update consumer: a custom dashboard, a
voice assistant, or an OpenWebUI **pipe** (a plug-in that lives inside OpenWebUI,
not the harness) that renders `tool_call`/`tool_result` as status updates. The
harness stays a dumb event source; each consumer is a renderer at the edge.

Note: text is **not** token-streamed — assistant text arrives as one `text`
event per loop iteration, not token-by-token. The harness deliberately has no
provider-level token streaming (it would mean per-provider stream plumbing and a
reasoning-tag stripper in the core); `/chat/stream` trades token-smooth text for
a simple core plus live *activity* visibility. The OpenAI `/v1` SSE path is
unaffected and still chunks text per `TextEvent`.

### Choosing a context strategy (`naive` vs `compaction`)

The agent loop shapes each LLM call's message view through `agent/context.py`
(see [architecture.md](architecture.md), *Context assembly*). Which strategy
to run:

- **`naive`** (default) — pass-through, behavior-preserving. Right for
  large-window cloud models and short conversations. Over budget it only
  warns (`context over budget under 'naive'`), so watch the logs.
- **`compaction`** — opt in (`CONTEXT_STRATEGY=compaction`) when runs are
  long/tool-heavy on a small-window model (the local-Qwen/gpt-oss case).
  Over-budget history is summarized once per LLM call by the same selected
  model.

What compaction **preserves**: the system prompt (never in history), the
first user message verbatim (the task header), the last
`CONTEXT_RECENT_MESSAGES` protocol-safe units verbatim, tool-use/tool-result
pairing (never split), and — via the summary — decisions, constraints, facts
learned, failed approaches, and open questions.

What it does **not** preserve: verbatim middle-of-conversation text, full
tool outputs already superseded, and provider-specific `provider_metadata` on
summarized (dropped) messages. The summary is only as good as the selected
model's summarization; raise `CONTEXT_SUMMARY_MAX_TOKENS` if it's dropping
detail.

Failure behavior: any compaction failure (summarizer error/empty, malformed
history, history too short) logs a warning and sends the full history —
requests are never failed by the context layer. Costs: one extra LLM call per
over-budget iteration. Session history is never rewritten; continuing a
session sees the original messages.

### Adding new loop strategies

Drop a new file in `agent/` (e.g. `planner_loop.py`) with its own
`run_*` function. The route layer picks which loop to use (via a
request field, a config setting, etc.). No changes to existing code.
The orchestrator's decision can include a hint about which loop to use
if you add that field to `OrchestrationResult`.

### Observability (run tracing)

The loop already keeps an append-only event log as its run state; tracing
**serializes that log to disk** rather than running a parallel logging
subsystem. Set `TRACE_ENABLED=true` (and optionally `TRACE_PATH`, default
`traces/harness.jsonl`) and every run appends one JSON record per event:

```jsonl
{"run_id":"a1b2…","step":1,"ts":"2026-06-18T…","type":"usage","input_tokens":10,"output_tokens":5,"total_tokens":15,"latency_ms":812.4,"iteration":1}
{"run_id":"a1b2…","step":2,"ts":"…","type":"reasoning","reasoning":"..."}
{"run_id":"a1b2…","step":3,"ts":"…","type":"tool_call","tool_use_id":"call_1","name":"toolbox__lookup","args":{…}}
{"run_id":"a1b2…","step":4,"ts":"…","type":"tool_result","name":"toolbox__lookup","is_error":false,"latency_ms":41.0,"content":"…"}
{"run_id":"a1b2…","step":8,"ts":"…","type":"done","reason":"end_turn","iterations":2,"total_tokens":39}
```

Each record carries the per-request `run_id` (minted in `_turn_events`, so it
covers both `/chat` and `/v1`), a monotonic `step` index, an ISO `ts`, and
`latency_ms` on the LLM (`usage`) and `tool_result` records. The same `run_id`
prefixes every log line for that turn (`[run a1b2…] chat: …`), so a log line
points straight at its trace. Grep one run with `grep '"run_id":"a1b2…"'`.

The seam is `agent/tracing.py`: a `Tracer` ABC (`emit(record)` + `close()`),
a `NoOpTracer`, and the `JSONLTracer`. It mirrors `SessionStore` — swap in an
OpenTelemetry exporter later by adding one `Tracer` subclass and one line in
`build_tracer`, without touching the loop. Wiring: `main.py`'s lifespan builds
the tracer onto `app.state.tracer`; `get_tracer` injects it; `run_agent` takes
it as an optional arg and emits every event through it. **Tracing is optional
and best-effort:** disabled (the default) means `tracer=None` and zero hot-path
cost; a failing or slow sink is logged and swallowed and never breaks a request.

Events map to JSON **explicitly** (never `dataclasses.asdict`) and the JSONL
writer base64-encodes any stray `bytes` (e.g. a provider's `thought_signature`)
so the serializer can't crash. **Sensitivity:** the trace captures full prompt
text, tool args, and tool results by default — appropriate for the single-
operator dev harness, but treat the file as sensitive. The harness endpoints can
now be gated with `HARNESS_API_KEY`, but the trace **file** is not covered by
that — guard it at the filesystem level. A metadata-only mode (names + usage +
latency, bodies omitted) is the natural next toggle once the harness is
multi-tenant.

For per-request orchestration visibility without a trace, set `LOG_LEVEL=DEBUG`
— the route emits the orchestrator's full generated system prompt and selected
tool list at DEBUG.

### Parallel tool execution

In `agent/loop.py`, replace the sequential `for tu in tool_uses` block
with:

```python
results = await asyncio.gather(*(
    _execute_one(tu, mcp) for tu in tool_uses
), return_exceptions=False)
```

where `_execute_one` does the yield + call_tool + result block. Be aware:
events come out in completion order, not call order, which may confuse
streaming UIs.

### Tuning orchestration behavior

Three knobs, all configuration-only:

1. **Add or remove models** in `config/models.yaml`. Update descriptions
   to nudge the orchestrator toward / away from a given model.
2. **Rewrite `config/orchestrator_prompt.md`** to change selection
   heuristics globally. The file is loaded at startup; restart to pick
   up edits.
3. **Override the orchestrator's own model** via
   `ORCHESTRATOR_MODEL_ID` if the default is too cheap (orchestration
   making poor decisions) or too expensive (orchestration costing more
   than the work it routes).

### Disabling orchestration

Two ways:

1. Set `ORCHESTRATION_ENABLED=false`. The harness logs the disable and
   runs in legacy mode: default LLM + all tools + `request.system` (or
   none). Useful for dev environments or when comparing orchestrated
   vs. unorchestrated behavior.
2. Don't ship `config/models.yaml`. The lifespan will log a warning
   and degrade to the same legacy mode.

---

## Operations

### First-time setup

```bash
# Creates .venv, installs requirements.txt, and seeds .env from .env.example
./setup.sh

# Then edit .env and set your API keys (e.g. GEMINI_API_KEY) and MCP URLs.
```

`runscript.sh` (used everywhere below) activates `.venv`, prepends the
project root to `PYTHONPATH`, and runs Python. Bootstrap loads `.env` into
`os.environ` at the top of each entry point.

### Running the server

```bash
./runscript.sh -m uvicorn main:app --host 0.0.0.0 --port 8000

# With auto-reload (dev only)
./runscript.sh -m uvicorn main:app --reload

# With workers (production)
./runscript.sh -m uvicorn main:app --workers 4
```

**Note on workers**: each uvicorn worker is a separate Python process with
its own `InMemorySessionStore` **and** its own in-process `SessionGuard`.
Two consequences for multi-worker deployments: a session created on one
worker isn't visible on another (fine here — LibreChat re-feeds context),
and the same-session 409 guard only holds *within* a worker. If you ever
need the guard to span workers, promote it to a shared claim (see "Adding
persistence"). The orchestrator's LLMRegistry is per-worker too; that's fine
because client construction is idempotent. **Single process (the default)
is fully covered** — distinct sessions run concurrently, same-session
overlaps get 409.

### Graceful shutdown

Uvicorn lifespan shutdown closes every constructed LLM SDK client, then every
retained MCP client (healthy or failed), and the tracer. MCP inventory is
cleared and records transition to `closed` before client cleanup. The default
LLM and lazy registry cache are
combined by object identity, so a client reachable through both paths is closed
once. Prompted-tool wrappers forward lifecycle ownership to their inner
provider. One cleanup failure is logged and does not skip the remaining LLMs or
the MCP/tracer owners; an active task cancellation is preserved.

For OpenAI streaming, the agent loop closes the provider generator and the
provider generator closes its inner SDK stream. This releases the HTTP response
on normal completion, timeout, provider failure, cancellation, and clients that
stop reading early. Repeated graceful shutdown is safe. A forced process kill
(`SIGKILL`, container hard-stop, or equivalent) bypasses Python lifespan hooks,
so the operating system must reclaim any remaining sockets and file handles.

### Logs

Logging is `INFO` by default. Key log lines to know:

| Line                                                                              | When                                                |
|-----------------------------------------------------------------------------------|-----------------------------------------------------|
| `harness ready: provider=... \| orchestration=... \| mcp=...`                     | End of lifespan startup. One greppable summary.    |
| `chat: orchestrator picked model=... tools=N fallback=false`                     | Per request, when orchestration is on.            |
| `chat: legacy mode (orchestration disabled), tools=N`                             | Per request, when orchestration is off.            |
| `chat: done reason=... iterations=... tokens=... session=...`                    | At the end of every request.                       |
| `orchestration fallback in effect: ...`                                          | Per request when the orchestrator's LLM call fails.|

`httpx` and `mcp.client.*` are chatty — each MCP request and each LLM
API call shows up at INFO. For production quieting:

```python
# in main.py lifespan, after logging.basicConfig:
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
```

For per-request orchestration detail (full system prompt and tool list),
set `LOG_LEVEL=DEBUG`. The route logs the generated system prompt at
DEBUG so you can audit what the agent was told without spamming INFO.

### Health checks

`GET /health` returns 200 as long as the app is up. `mcp_servers` reports every
configured enabled server's current state, sanitized last error, and advertised
tool count. `connected_servers` remains the healthy-only compatibility list and
`tool_count` remains the aggregate healthy inventory. Top-level status is `ok`
when all enabled servers are healthy (or none are enabled) and `degraded`
otherwise.

This is a truthful passive snapshot, not an active reachability probe. A
transport/protocol tool-call failure marks the server unhealthy and removes its
advertised tools. A later call to a formerly known tool triggers one bounded,
per-server reconnect and inventory refresh; the ambiguous failed call is never
replayed. `/health` itself does not reconnect or poll and does not probe the
orchestrator's LLM provider. Add a separate liveness/readiness policy if
deployment requirements need active probes.

### Failure modes

| Symptom                                       | Likely cause                                                 |
|-----------------------------------------------|--------------------------------------------------------------|
| App fails to start with missing-key error     | API key missing from `.env`, `config.load_secrets()` didn't run, or wrong `LLM_PROVIDER` |
| `/health` is `degraded` or an MCP server is `unhealthy` | The server failed, was cancelled, exceeded `MCP_CONNECT_TIMEOUT_SECONDS` during startup/recovery, or raised a transport/protocol failure during dispatch; healthy siblings remain usable |
| Tool result says its outcome is unknown and was not replayed | The MCP call crossed the remote invocation boundary and then failed. The server was invalidated; retry only if the operation is safe to issue as a new invocation. |
| `/health` shows `orchestration_enabled: false` | `models.yaml` or `orchestrator_prompt.md` missing/unparseable, or `ORCHESTRATION_ENABLED=false`. Lifespan logs the reason. |
| Every response has `orchestration.fallback_used: true` | Orchestrator's LLM call is failing. Check the `orchestration fallback in effect:` warning logs for the underlying provider error. |
| 400 from `/chat`                              | Empty request body. `/chat` is plain text — send the prompt as the body. |
| 400 from Gemini after first tool result       | `provider_metadata` round-trip broken somewhere              |
| `done_reason: "max_iterations"`               | Run hit the iteration cap. The loop forces a best-effort answer on the last step (tools withheld + wrap-up prompt), so `response` is populated — but recurring `max_iterations` means the model is churning; review the prompt or raise `MAX_LOOP_ITERATIONS`. Repeated-identical tool calls are already short-circuited (stall detection); look for genuinely distinct-but-unproductive calls. |
| `done_reason: "llm_error"`                    | API call failed (after retries); see logs for the underlying exception |
| `done_reason: "truncated"`                    | Model hit `LLM_MAX_TOKENS` mid-answer; `response` is clipped. Raise `LLM_MAX_TOKENS` or the model's per-entry `max_tokens`. |
| `tool_result` with `is_error` "timed out"     | A tool exceeded `TOOL_TIMEOUT_SECONDS`. The MCP server is slow/hung; the loop continues and the model sees the error. |
| `tool_result` with `is_error` "invalid arguments for field ..." | The model's tool call failed `jsonschema` validation against the tool's `input_schema`; `mcp.call_tool` was never reached. The model sees the offending field + expected shape and can retry. |
| `done_reason: "budget_exceeded"`              | Run hit `MAX_RUN_TOKENS`. Disabled (`0`) by default; raise or disable the cap. Works even when the provider reports all-zero usage — the local estimator fills in. |
| `context over budget under 'naive'` warnings  | The estimated request exceeds `context_window − max output − margin`. Set the model's `context_window` accurately in `models.yaml`, and consider `CONTEXT_STRATEGY=compaction` for long tool-heavy runs. |
| `done_reason: "deadline_exceeded"`            | Run hit `MAX_RUN_SECONDS`, including while an LLM/tool call or retry sleep was in flight. Disabled (`0`) by default; raise or disable the cap, or investigate why the run is slow (retries, a slow provider). |
| `done_reason: "no_progress"`                  | `ABORT_AFTER_CONSECUTIVE_TOOL_FAILURES` consecutive tool-call failures (including validation failures). Disabled (`0`) by default. Review the failing tool/args in the logs — the consecutive-failure nudge already tried to steer the model before the abort fired. |
| 404 on `/chat`                                | The `X-Session-Id` header names a session that doesn't exist (e.g. evicted by TTL/max-size, or after a restart wiped state)|
| 409 on `/chat`                                | The `X-Session-Id` session already has a request in flight; `SessionGuard` rejects the concurrent turn. Retry after the first completes. |

---

## Tests

Pytest is the canonical regression runner. Its default selection is hermetic: tests
use scripted provider/MCP fakes, temporary configuration, and in-process ASGI wiring,
so a normal run neither loads developer credentials nor contacts configured services.

```bash
./runscript.sh -m pytest
```

Configured-backend checks live behind an explicit `live` marker and are never part of
that default signal. Select all live checks, or narrow them by capability:

```bash
./runscript.sh -m pytest -m live
./runscript.sh -m pytest -m "live and model"
./runscript.sh -m pytest -m "live and mcp"
./runscript.sh -m pytest -m "live and http_server"
```

Live checks load `.env`/`Settings`, can contact configured MCP and model endpoints,
and may spend model tokens. The `http_server` marker identifies checks of the fully
configured FastAPI surface; these currently run the app in-process with its lifespan
and do not require a separately launched Uvicorn process.

All former smoke coverage is now pytest-native. Configured MCP, model, agent,
orchestrator, HTTP, concurrency, and OpenAI-provider checks live in focused
`test_*_live.py` modules with explicit markers. See `tests/README.md` for the
current organization and focused commands.

---

## Eval Suite

The deterministic eval suite is the harness's regression net for "did the
agent produce the right outcome?", separate from smoke tests that prove
subsystems run. It is intentionally cheap: no pytest, no LLM judge, no external
eval framework.

Run the hermetic tier:

```bash
./runscript.sh tests/eval_agent.py
```

Hermetic evals load `tests/eval_data/agent_eval_v1.yaml`, drive the agent loop
or `TurnRunner` with scripted LLM/MCP fakes, print a per-case pass/fail table,
and exit non-zero if the pass rate is below the dataset threshold. The live tier
is skipped loudly by default.

Run live evals manually:

```bash
EVAL_LIVE=1 ./runscript.sh tests/eval_agent.py
```

Live evals use the configured real LLM backend from `.env`/`Settings` and an
empty MCP inventory unless a future live case says otherwise. They are
best-effort and should not be treated as hermetic CI signal.

Add a new eval case by editing only `tests/eval_data/agent_eval_v1.yaml`:

1. Add an entry under `cases` with a unique `id`, `tier`, `prompt`, scripted
   `llm.script` responses, optional `mcp.tools`/`mcp.results`, optional `run`
   knobs, and an `expect` block.
2. Prefer programmatic expectations: `done_reason`, `answer_contains`,
   `tool_called`, `tool_not_called`, `tool_call_count`, `mcp_call_count`,
   `tool_result_contains`, `tool_result_error_count`, `llm_calls`, and
   `max_iterations`.
3. Keep the case hermetic unless it genuinely needs a real model; mark live
   cases with `tier: live`.

The dataset threshold is the minimum pass rate required for the selected tier.
The current hermetic dataset uses `threshold: 1.0`, meaning any hermetic
regression fails the runner. If the suite grows to include known-flaky live
cases, keep that tolerance in the live tier, not in hermetic checks.

---

## Known Limitations

These are deliberate v1 simplifications, not bugs. Each has a clear
extension path described above.

- **Sessions are in-memory by design.** Conversation history dies with the
  process — intentional, since LibreChat holds the durable context and
  re-feeds it. The store is bounded (`SESSION_TTL_SECONDS`,
  `SESSION_MAX_COUNT`) so it can't grow without limit. Durable persistence
  is the `SessionStore` ABC's job if the use case ever changes.
- **Streaming is available on `/v1`.** `POST /v1/chat/completions` with
  `stream: true` returns an SSE stream of `chat.completion.chunk` frames. The
  native `/chat` endpoint is still non-streaming; use `/chat/stream` for the
  native live activity feed with tool and reasoning events.
- **Optional API-key auth.** Set `HARNESS_API_KEY` to require a key on `/chat`,
  `/chat/stream`, and `/v1/*` (via `X-API-Key` or `Authorization: Bearer`);
  `/health` stays open. Unset = auth off (dev default), the pre-existing wide-open
  behavior. Enforced at the route layer only — see
  [api.md](api.md#authentication). No per-user/multi-tenant policy, rate limiting,
  or trace-file protection is included.
- **Optional tool-dispatch policy.** `tool_policy` in `mcp_config.yaml`
  (`allow_all` default, or `allow_list` with patterns) gates which tools may
  *execute* at dispatch; a denied call never reaches MCP and comes back as a
  teaching `is_error` result. Distinct from `disabled_tools` (which controls tool
  *visibility*). See [configuration.md](configuration.md).
- **Capability profiles are declarative.** `models.yaml` can declare
  `supports_native_tools`, `thinking`, and `sampling` per model. Sampling and
  supported thinking hints are consumed by providers; setting
  `supports_native_tools` to `false` selects the prompted-tool wrapper for
  downstream agent turns.
- **Reasoning is trace/debug data, not answer text.** OpenAI-compatible
  leading `<think>...</think>` content is separated from visible text and
  emitted as a `reasoning` trace/event record. No HTTP route renders it, and
  session replay excludes it.
  Gemini reasoning remains `None` unless the SDK exposes thought content in a
  detectable form; Gemini `thought_signature` still round-trips through
  `provider_metadata`.
- **Per-call timeouts, plus optional overall token/wall-clock caps.** Each
  `llm.complete()` attempt is bounded by `LLM_TIMEOUT_SECONDS` and each
  `mcp.call_tool()` by `TOOL_TIMEOUT_SECONDS`, so a single hung call can't
  stall a request indefinitely. A timeout cancels the in-flight call and
  abandons it; for the LLM that feeds the retry path, for a tool it becomes an
  `is_error` result. `MAX_RUN_TOKENS` and `MAX_RUN_SECONDS` (both `0`/disabled
  by default) additionally bound *total* run cost/duration across all
  iterations; the wall-clock budget is also applied to in-flight LLM/tool calls
  and retry sleeps. See [configuration.md](configuration.md) and the *Bounded &
  safe runs* section of [architecture.md](architecture.md). The token cap
  works even against a provider that reports all-zero usage — the local
  token estimator (`agent/context.py`) fills in from the outgoing messages
  and response. Estimates are heuristic (chars/4), so treat the cap as a
  guard rail, not billing-grade accounting, on such servers.
- **No parallel tool execution.** Sequential is safer; switch when you
  need it.
- **No token/cost tracking aggregated across requests.** Per-request
  usage is in the response; aggregation is the caller's job.
- **Concurrent `/chat` on the same session_id is rejected, not raced.**
  `SessionGuard` returns **409** for a second in-flight request on a session
  (distinct sessions are fully isolated and run in parallel freely). The
  guard is in-process; a multi-worker deployment would need a shared claim
  (see "Adding persistence") to cover the same id across workers.
- **`/health` doesn't probe MCP.** It truthfully reports every enabled server's
  current retained state and healthy advertised inventory, but does not
  reconnect or poll. Lazy recovery occurs only on a later call to a formerly
  known tool. It also doesn't probe the orchestrator's LLM provider.
- **Two LLM providers implemented: Gemini and OpenAI-compatible.** The OpenAI
  client also drives any OpenAI-compatible server (local Ollama/vLLM) via
  `base_url`, and is the **default runtime** (a local Qwen, per
  `config/models.yaml`). Anthropic keys are recognized but its client is still
  stubbed (`build_llm_client` / `build_llm_client_from_entry` raise
  `NotImplementedError`).
- **Per-worker session isolation.** Multi-worker uvicorn deployments
  have independent in-memory stores per worker.
- **Orchestration adds an LLM call per request.** That's the cost of
  dynamic routing. Use a cheap model for the orchestrator
  (`ORCHESTRATOR_MODEL_ID`) to keep it affordable.
- **Switching models mid-session can be lossy.** Provider-specific
  conversation state (Gemini's `thought_signature`, OpenAI's reasoning
  traces) doesn't always cross providers cleanly. The orchestrator may
  pick different models on different turns of the same session; in
  practice modern models tolerate this but pathological cases exist.
- **The orchestrator has no per-request budget control.** It picks the
  best model for the job, not the cheapest acceptable one — descriptions
  in `models.yaml` are the only steering. A future change could add
  explicit cost ceilings.
- **No A/B testing or routing experimentation.** The orchestrator is one
  LLM call making one decision; there's no shadow-routing or sampling
  policy.
