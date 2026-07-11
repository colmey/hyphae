# PyAiHarness

A minimal, extendable AI harness in Python. Receives a prompt over HTTP,
runs an **orchestrator** to pick a model + tool subset + system prompt
for the request, runs an agent reasoning loop against the chosen LLM,
lets the model call tools exposed by external MCP (Model Context
Protocol) servers, and returns the answer.

Built on **FastAPI**, the official **`mcp`** Python SDK, and
**`google-genai`** (default provider; provider-agnostic by design).

## Features

- 🎯 **Orchestration layer** — per-request LLM-driven router picks the
  cheapest-capable model, the smallest tool subset, and a task-specific
  system prompt. Configured by a YAML model registry + a markdown
  prompt, no code changes to retune.
- 🔌 **MCP-native** — connect to any MCP server (streamable-http, SSE, stdio).
- 🔁 **Agent loop** — multi-step reasoning with autonomous tool selection
  and chaining.
- 🧩 **Provider-agnostic core** — each provider is one file under
  `llm/providers/` behind a registry; the core never imports a provider
  SDK. Gemini and OpenAI-compatible (incl. local Ollama/vLLM via `base_url`)
  today — the default runtime is a local Qwen; adding another is a file + one
  registry line. One process can hold clients for multiple providers.
- 🛡️ **Minimal HTTP surface** — `/chat` is plain text in / plain text out;
  the body is the prompt, the answer is the body, session continuation rides
  on a header. Nothing to over-accept.
- 🔌 **OpenAI-compatible `/v1`** — `POST /v1/chat/completions` (+ SSE) and
  `GET /v1/models` let OpenWebUI / LibreChat / the `openai` SDK connect
  natively by pointing `base_url` at `/v1`. A thin translator over the same
  shared core — no duplicated orchestration or loop logic.
- 🧠 **Session memory** — multi-turn conversations via a swappable
  `SessionStore` interface. In-memory and bounded (TTL + max-size eviction);
  the ABC is the seam for a durable backend if ever needed.
- 🚦 **Concurrency-safe** — `async def` handlers service many requests at
  once; distinct sessions are fully isolated, and `SessionGuard` returns 409
  if two requests overlap on the same `session_id`.
- 🌊 **Streaming** — the loop is an async generator yielding typed events.
  `/v1/chat/completions` streams them as SSE `chat.completion.chunk` frames
  (`stream: true`); the native `POST /chat/stream` forwards the raw typed events
  (`tool_call`, `tool_result`, `text`, `done`, …) as an SSE activity feed. Plain
  `/chat` stays non-streaming.
- 🔭 **Observable runs** — opt-in JSONL tracing (`TRACE_ENABLED=true`) serializes
  the loop's event stream: one record per event, tagged with a per-request
  `run_id` (also stamped on every log line), a step index, a timestamp, and
  per-step LLM/tool latency. The trace *is* the event log, behind a swappable
  `Tracer` seam (JSONL now, OpenTelemetry later). Off by default and best-effort
  — a failing sink never breaks a request.
- ⚙️ **YAML-configured MCP servers** — pasteable from Claude Desktop /
  Cursor / Roo.
- ♻️ **Graceful degradation** — orchestration is fully optional. If
  `models.yaml` is missing, the orchestrator's LLM call fails, or any
  config is bad, the harness drops back to legacy (default model + all
  tools) behavior. Requests keep working.
- ✅ **Verified end-to-end** — a hermetic pytest regression suite plus explicitly
  selected live model, MCP, and HTTP integration checks.

## Quick start

```bash
# 1. Create the venv, install dependencies, and seed .env from .env.example
./setup.sh

# 2. Set your secrets in .env (API keys, MCP server URLs)
#    Configure your MCP servers in config/mcp_config.yaml
#    Configure your model registry in config/models.yaml
#    Tune the orchestrator's behavior in config/orchestrator_prompt.md

# 3. Run the server
./runscript.sh -m uvicorn main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
# Check it's alive
curl http://localhost:8000/health

# Talk to it (plain text in, plain text out; orchestrator picks model + tools).
# -i shows the X-Session-Id response header you reuse to continue.
curl -i -X POST http://localhost:8000/chat \
    -H 'Content-Type: text/plain' \
    --data 'What tools do you have access to?'

# Continue the conversation by passing the session id back as a header
curl -X POST http://localhost:8000/chat \
    -H 'X-Session-Id: sess_...' \
    --data 'Use one of them to...'
```

## How it works

```
HTTP request
    │
    ▼
POST /chat ──► orchestrator ──► agent loop ──┬──► chosen LLM
               (pick            (chosen       │
               model + tools    model +       ├──► MCP manager
               + system)        chosen        │
                                tools)        └──► session store
```

The **orchestrator** runs once per request and is itself an LLM call. Its
structured output dictates how `run_agent()` is invoked.

> **Caching, if needed, is handled by a separate layer/service in front
> of this harness, not inside it.** The request body is deliberately small
> (`prompt` + optional `commands` + optional `MCP`) so a fronting cache can
> forward it unchanged.

The **agent loop** is the only module that knows about both the LLM and
the MCP layer. Per iteration: ask the model what to do → if it asked for
tools, run them via MCP and feed the results back → loop until the model
produces a final answer.

Every architectural seam (LLM provider, MCP transport, session store,
loop strategy, orchestration policy) is designed to be swappable without
touching the rest.

## Configuration

All runtime config lives under `config/`:

**`config/mcp_config.yaml`** — MCP server definitions:

```yaml
mcpServers:
  my-toolbox:
    transport: streamable-http
    url: http://example.local:5002/mcp

  web-search:
    transport: sse
    url: http://example.local:5003/sse
```

Supported transports: `streamable-http`, `sse`, `stdio`. Server-side
auth, disabled tools, and `${ENV_VAR}` interpolation are all supported.

**`config/models.yaml`** — routable models the orchestrator can pick:

```yaml
models:
  gemini-flash:
    provider: gemini
    model: gemini-3-flash-preview
    description: Fast, cost-efficient. Best for lookups and single-tool tasks.
    default: true
  gemini-pro:
    provider: gemini
    model: gemini-3-pro-preview
    description: Higher reasoning. Multi-step plans, complex analytics.
```

The `description` field is what the orchestrator LLM reads when picking
a model — write it for an LLM audience.

**`config/orchestrator_prompt.md`** — the orchestrator's system prompt.
Edit this to tune routing behavior; no code changes needed.

**Secrets** live in a `.env` file at the project root. `bootstrap.load_secrets()`
loads it into `os.environ` before the harness starts; bootstrap runs
in-process at the top of every entry point. Copy `.env.example` to `.env`
(`setup.sh` does this) and fill in your values. Real environment variables
already set in the process take precedence over `.env`.

Relevant env vars:

| Variable                     | Default                          | Purpose                                  |
|------------------------------|----------------------------------|------------------------------------------|
| `LLM_PROVIDER`               | `gemini`                         | Legacy default; orchestrator overrides per request |
| `LLM_MODEL`                  | `gemini-3-flash-preview`         | Legacy default model identifier          |
| `GEMINI_API_KEY`             | _(required for Gemini)_          | Set in `.env`                            |
| `MCP_CONFIG_PATH`            | `config/mcp_config.yaml`         | Path to MCP server config                |
| `MODELS_CONFIG_PATH`         | `config/models.yaml`             | Path to the model registry               |
| `ORCHESTRATOR_PROMPT_PATH`   | `config/orchestrator_prompt.md`  | Path to orchestrator system prompt       |
| `ORCHESTRATION_ENABLED`      | `true`                           | Master toggle for the orchestration layer|
| `ORCHESTRATOR_MODEL_ID`      | _(empty)_                        | Override the orchestrator's own model    |
| `MAX_LOOP_ITERATIONS`        | `25`                             | Agent loop iteration cap                 |
| `LOG_LEVEL`                  | `INFO`                           | Python logging level (`DEBUG` opens per-request orchestration detail) |

See [`docs/configuration.md`](./docs/configuration.md) for the full list.

## API

### `GET /health`

```json
{
  "status": "ok",
  "provider": "gemini",
  "model": "gemini-3-flash-preview",
  "connected_servers": ["my-toolbox", "web-search"],
  "tool_count": 15,
  "orchestration_enabled": true,
  "available_model_ids": ["gemini-flash", "gemini-pro"]
}
```

### `POST /chat`

Run one user prompt through the orchestrator + agent loop. **Plain text in,
plain text out** — the request body is the prompt, the response body is the
answer. Creates a new session unless `X-Session-Id` is supplied.

**Request:**

```
POST /chat
Content-Type: text/plain
X-Session-Id: sess_abc123...        (optional)

List the tables in the customer database.
```

The body is the prompt (empty → 400). `X-Session-Id` continues a session
(unknown → 404). The orchestrator picks the model, tool subset, and system
prompt; the iteration cap is config-driven. There are no per-call JSON knobs on
this endpoint — for OpenAI-client tooling and a per-call system prompt, use the
`/v1` adapter below. Full detail in [docs/api.md](docs/api.md).

**Response:**

```
HTTP/1.1 200 OK
Content-Type: text/plain; charset=utf-8
X-Session-Id: sess_abc123...
X-Done-Reason: end_turn

The customer database contains tables including customers, orders, payments.
```

The body is the final human-readable answer. `X-Session-Id` is the session to
reuse for the next turn; `X-Done-Reason` says why the loop stopped
(`end_turn`, `max_iterations`, `truncated`, ...). Which model handled the
request and what it spent is recorded in the server logs, not the response.

### `POST /chat/stream`

The same turn as `/chat`, but streamed live as the loop's typed events instead
of one collected answer — for clients that want to show *what the agent is
doing* (each `tool_call`/`tool_result` as it happens, then the answer). Same
plain-text body + `X-Session-Id` contract; the response is an SSE stream of
`{type, …}` event frames ending on a `done` event. Text arrives per loop
iteration, not token-by-token. Full detail in
[docs/api.md](docs/api.md#post-chatstream).

### `POST /v1/chat/completions` + `GET /v1/models`

OpenAI-compatible adapter so OpenWebUI / LibreChat / the `openai` SDK connect
natively — point the client's base URL at `http://<host>:8000/v1`. Standard
OpenAI JSON in (a `system` message becomes the per-call system prompt; `model`
matching a registry id pins that model); a `chat.completion` object out, or an
SSE stream of `chat.completion.chunk` frames when `stream: true`. Stateless: the
client re-feeds `messages` each turn. `GET /v1/models` lists the routable
registry. Full detail in [docs/api.md](docs/api.md#post-v1chatcompletions).

## Project layout

```
PyAiHarness/
├── main.py                 # FastAPI app + lifespan
├── bootstrap.py            # load_secrets(): .env → os.environ
├── harness_config.py       # Settings + MCPConfig
├── harness_client.py       # Reference async Python client
├── .env.example            # Env var template (copy to .env)
├── setup.sh                # Create .venv + install deps + seed .env
│
├── config/                 # All runtime YAML/text config
│   ├── mcp_config.yaml
│   ├── models.yaml
│   └── orchestrator_prompt.md
│
├── tests/                  # Hermetic pytest suite + explicit live checks
│
├── mcp_layer/              # MCP client + manager (renamed to avoid SDK shadow)
├── llm/                    # Provider-agnostic schemas + LLMClient ABC + registry
│   └── providers/          #   One file per provider (gemini.py); imported lazily
├── agent/                  # Session, events, the reasoning loop
├── orchestrator/           # Model + tool + system-prompt router
├── api/                    # HTTP renderers (native /chat + /chat/stream, OpenAI /v1) over a shared turn core (turn.py)
└── docs/                   # Reference docs (README router + architecture/api/configuration/operations)
```

## Tests

Run every hermetic regression test with one command. This path uses fakes and
temporary configuration and does not contact configured model or MCP services:

```bash
./runscript.sh -m pytest
```

Configured-backend checks are marked `live` and excluded by default. Opt in when
credentials and services are available (these checks may spend model tokens):

```bash
./runscript.sh -m pytest -m live
```

Configured integrations are native `test_*_live.py` modules. See
[`tests/README.md`](tests/README.md) for marker selection and focused commands.

## Extending

The harness is designed for the following extensions to land without
disrupting existing code. Each is documented in detail in
[`docs/operations.md`](./docs/operations.md):

- **Add a routable model** — add an entry to `config/models.yaml`. No
  code changes needed if the provider is already implemented.
- **Add an LLM provider** — drop a `llm/providers/<name>.py` implementing
  the `LLMClient` ABC, then add one entry to `_PROVIDERS` in
  `llm/client.py`. Config validation, key resolution, and routing pick it
  up automatically — nothing else changes.
- **Tune orchestration behavior** — edit
  `config/orchestrator_prompt.md` and/or model `description`s in
  `models.yaml`. Restart to pick up changes.
- **Disable orchestration** — set `ORCHESTRATION_ENABLED=false` for
  legacy behavior (default LLM, all tools).
- **Add session persistence** — implement `SessionStore` against a real DB,
  swap the one-liner in `main.py`'s lifespan.
- **Streaming** — `/v1/chat/completions` (`stream: true`) and the native
  `/chat/stream` event feed both render the shared `_turn_events` generator as
  SSE via `sse-starlette`; the loop is already an async generator. Adding
  another transport is one more renderer over the same core (`api/turn.py`), no
  loop change.
- **Tracing / observability** — set `TRACE_ENABLED=true` for a JSONL trace of
  every run. Implement another `Tracer` (e.g. an OpenTelemetry exporter) in
  `agent/tracing.py` and return it from `build_tracer` — the loop is unchanged.
- **Add a new loop strategy** — drop a new file in `agent/`; the route
  picks which loop to use.
- **Parallel tool execution** — `asyncio.gather` in the tool-execution
  block of `agent/loop.py`.

## Documentation

- **[`docs/`](./docs/)** — comprehensive reference, split for targeted
  reading. Start at [`docs/README.md`](./docs/README.md), which routes to
  [`architecture.md`](./docs/architecture.md) (design + subsystems),
  [`configuration.md`](./docs/configuration.md),
  [`api.md`](./docs/api.md), and
  [`operations.md`](./docs/operations.md) (running, extending, limits).

If you're handing this codebase to an AI assistant or new contributor,
point them at `docs/README.md` — it indexes the complete context.

## Status

**Working prototype.** Hermetic regression coverage is collected by pytest. The
configured-backend suite has been verified end-to-end
against real MCP servers (streamable-http and SSE), both a hosted Gemini 3
backend and a local OpenAI-compatible Qwen with multi-step tool chaining, the
orchestrator routing across model tiers, and concurrent requests (isolation +
same-session guard).

Known limitations (intentional scope, each with a documented extension
path):

- Sessions are in-memory by design (LibreChat holds durable context); the
  store is bounded by TTL + max-size so it can't grow without limit.
- Streaming: `/v1/chat/completions` forwards provider text deltas as token-level
  SSE, while `/chat/stream` exposes the same core event stream as an activity feed;
  plain `/chat` stays non-streaming.
- No authentication on `/chat` or `/v1`. Relatedly, the JSONL trace
  (`TRACE_ENABLED`) captures full prompts/args/results by default — fine for a
  single-operator dev harness, but a metadata-only mode is future work once the
  harness is multi-tenant.
- Configurable per-read LLM timeouts and absolute run deadlines bound stalled calls.
- Tool execution is sequential.
- Two LLM providers implemented (Gemini and OpenAI-compatible, each isolated
  in `llm/providers/`); Anthropic stubbed. Registry + ABC ready for more.
- Orchestration adds one LLM call per request; mitigate by using a
  cheap model as the orchestrator (`ORCHESTRATOR_MODEL_ID`).

## License

_TBD_
