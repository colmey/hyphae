# hyphae — HTTP API

Request/response contracts for the native routes and the OpenAI-compatible
adapter. All chat surfaces drive the same shared turn core
(`api/turn.py::TurnRunner`). Accepted turns run under one session claim,
absolute deadline, run ID, immutable tool snapshot, and resolved model identity.

> See also: [README.md](README.md) (overview + quick start),
> [architecture.md](architecture.md) (how the route is wired,
> [the "Bouncer" rationale](architecture.md#design-decisions)),
> [configuration.md](configuration.md) (env vars and config files).

## Authentication

Optional and off by default. When `HYPHAE_API_KEY` is **unset**, every route is
open (single-operator dev default). When it is **set**, the chat routes —
`POST /chat`, `POST /chat/stream`, `POST /v1/chat/completions`, `GET /v1/models` —
require the key and return **401** without it. `GET /health` is **always open**.

Present the key either way:

- `X-API-Key: <key>`, or
- `Authorization: Bearer <key>` (what OpenWebUI/LibreChat send on an OpenAI
  connection — set the key in that connection's config).

The comparison is constant-time and stays at the route layer.

```bash
# with a key configured:
curl -sS -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $HYPHAE_API_KEY" --data 'what is 2+2?'
```

## `GET /health`

```json
{
  "status": "degraded",
  "provider": "openai_compatible",
  "model": "qwen",
  "connected_servers": ["my-toolbox"],
  "tool_count": 8,
  "mcp_servers": [
    {
      "name": "my-toolbox",
      "state": "healthy",
      "last_error": null,
      "tool_count": 8
    },
    {
      "name": "web-search",
      "state": "unhealthy",
      "last_error": "connection timed out after 30 seconds",
      "tool_count": 0
    }
  ],
  "orchestration_enabled": true,
  "available_model_ids": ["qwen-local"]
}
```

`orchestration_enabled` is true if and only if an orchestrator was built
during startup. `available_model_ids` lists the model IDs from
`config/models.yaml` (empty when orchestration is off).

`mcp_servers` contains every configured enabled server in configuration order.
Its state is one of `disconnected`, `connecting`, `healthy`, `unhealthy`, or
`closed`; errors are sanitized and unhealthy servers always advertise zero
tools. The compatibility fields remain: `connected_servers` contains healthy
servers only and `tool_count` is the aggregate healthy inventory. Top-level
`status` is `ok` when every enabled server is healthy (or none are enabled), and
`degraded` otherwise. This is a passive current-state snapshot, not an active
reachability probe; `/health` never initiates recovery and remains HTTP 200 in
either state. Recovery is attempted only when a later call names a tool formerly
known to an unhealthy server.

## `POST /chat`

**Plain text in, plain text out.** The request body **is** the prompt and the
response body **is** the answer. Session continuation and result metadata travel
as headers. For OpenAI-client tooling, use the
[`/v1/chat/completions`](#post-v1chatcompletions) adapter instead.

**Request:**

```
POST /chat
Content-Type: text/plain
X-Session-Id: sess_abc123...        (optional)

List the tables in the customer database.
```

- The body is the prompt. An empty body returns **400**.
- `X-Session-Id` (optional request header) continues an existing session. If
  omitted, a new session is created; if provided but unknown, returns **404**.
- There are no per-call `system`, `max_iterations`, or tool-preference knobs on
  this endpoint. The orchestrator picks model/tools/system; the iteration cap
  comes from settings. A per-call system prompt is available on the
  [`/v1/chat/completions`](#post-v1chatcompletions) endpoint (as a
  `role: "system"` message).

**Response (200):**

```
HTTP/1.1 200 OK
Content-Type: text/plain; charset=utf-8
X-Session-Id: sess_abc123...
X-Done-Reason: end_turn

The customer database contains tables including customers, orders, payments.
```

- The body is the final human-readable answer (all `TextEvent.text` joined).
  Provider-extracted reasoning is not included in this body.
- `X-Session-Id` is the session this turn ran in — pass it back on the next
  request to continue the conversation.
- `X-Done-Reason` ∈ `{"end_turn", "max_iterations", "llm_error", "empty", "truncated",
  "budget_exceeded", "deadline_exceeded", "no_progress", "content_filter",
  "refusal", "provider_error", "incomplete_stream"}`.
  `"truncated"` means the model stopped on `max_tokens` mid-answer. `"max_iterations"`
  means the loop hit its iteration cap and forced a best-effort wrap-up.
  `"budget_exceeded"`, `"deadline_exceeded"`, and `"no_progress"` are optional
  guard exits; they include answer text already produced. See
  [configuration.md](configuration.md) and architecture.md's *Bounded & safe runs*
  section.

Routing and token-usage detail (which model handled the request, how many
tokens it spent) is recorded in the per-request server logs, not the response.

**Response (400) — empty body:**

```json
{"detail": "empty body; send the prompt as plain text"}
```

**Response (404) — unknown session:**

```json
{"detail": "session 'sess_doesnotexist' not found"}
```

**Response (409) — session busy:**

```json
{"detail": "session 'sess_abc123...' is processing another request"}
```

Returned when a second `/chat` request targets an `X-Session-Id` that already
has a request in flight. Distinct sessions run concurrently without
restriction; only same-session overlap is rejected. Retry once the first
request completes.

## `POST /chat/stream`

The **live activity feed**: same turn as `/chat`, streamed as typed loop events
instead of one collected answer. Tool calls/results appear as the loop reaches
them; clients that only want the final answer should use `/chat` or `/v1`.

Same dumb-pipe contract as `/chat`: the request body **is** the prompt;
`X-Session-Id` (optional) continues a session; an empty body returns **400**, an
unknown session **404**.

**Response (200):** a `text/event-stream` (`sse-starlette`). Each frame is one
loop event, JSON in the `data:` field, with a `type` discriminator. The stream
ends after the `done` event. `X-Session-Id` is returned as a response header.

```
data: {"type":"usage","iteration":1,"total_tokens":42,"latency_ms":120.4, ...}

data: {"type":"tool_call","tool_use_id":"call_1","name":"web_search","args":{"q":"..."}}

data: {"type":"tool_result","tool_use_id":"call_1","name":"web_search","content":"...","is_error":false,"latency_ms":1830.2}

data: {"type":"text","text":"SpaceX launched ..."}

data: {"type":"done","reason":"end_turn","iterations":2,"total_tokens":1875}
```

Event `type`s: `orchestration`, `text`, `tool_call`, `tool_result`, `usage`,
`done`, `error`. When orchestration is active its sanitized decision is the
first event and carries the same resolved model ID recorded in turn metadata.
Provider reasoning is not rendered by this native route or `/chat`; the
OpenAI-compatible streaming route may expose sanitized reasoning through its
optional reasoning channel. A failure mid-turn is delivered as a terminal error
frame because the SSE response is already open. Provider policy and failure
outcomes remain explicit in `done.reason`: `content_filter`, `refusal`,
`provider_error`, and `incomplete_stream` are never collapsed to `end_turn`.

Providers with native streaming emit incremental `text` events; complete-only
providers emit final coarse text blocks through the common streaming fallback.

## `POST /v1/chat/completions`

An **OpenAI-compatible** adapter for OpenWebUI, LibreChat, and the `openai` SDK.
Point the client's base URL at `/v1`.

**Stateless.** Each request seeds a new ephemeral session from `messages`; the
client owns durable history, and `/v1` never creates, saves, or evicts entries
in the native bounded session store. Same-session 409 therefore never applies.
The adapter validates message ordering and model IDs before constructing that
ephemeral session. It reports the model ID resolved by `TurnRunner`, never an
unverified request label.

**Request** (standard OpenAI body; unknown fields like `temperature`, `top_p`
are tolerated and ignored):

```json
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "gemini-pro",
  "stream": false,
  "messages": [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "List the tables in the customer database."}
  ]
}
```

- A `role: "system"` message becomes the per-call system override (multiple are
  concatenated). Ordered `user`/`assistant` messages become conversation
  history, and the final supported conversational message must be `user`; that
  message is the active turn. Assistant-prefill ordering is rejected with 400
  rather than moved ahead of an earlier user message.
- `model`, when supplied, must exactly match an ID advertised by
  [`GET /v1/models`](#get-v1models) and pins that model (the orchestrator still
  selects tools and the system prompt). An unknown explicit ID returns 400;
  omitted `model` leaves model selection to the orchestrator.
- `messages` must be a non-empty array containing at least one `user` message.

**Response (200) — non-stream** (`object: "chat.completion"`):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1718600000,
  "model": "gemini-pro",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
}
```

`finish_reason` maps from the loop's done reason: `end_turn`/`empty → stop`;
`truncated`/`max_tokens`/`max_iterations`/`budget_exceeded`/`deadline_exceeded → length`;
`no_progress → stop`; and `content_filter`/`refusal → content_filter`.
`provider_error`, `incomplete_stream`, unrecoverable, and unknown terminal
reasons fail closed instead of being presented as a successful stop.
Provider-extracted reasoning is not included in the `message.content` payload.
The response `model` is the registry ID that actually executed, including when
the request omitted `model` and orchestration selected it.

**Response (200) — stream** (`stream: true`, OpenWebUI's default): a
`text/event-stream` of `chat.completion.chunk` frames, terminated by
`data: [DONE]`. For providers with native streaming, `/v1` emits genuinely
incremental model text as it arrives. Providers without native streaming still
work through the `LLMClient.stream()` fallback, but their text arrives as final
coarse blocks. The first frame carries `delta.role = "assistant"`, each visible
text delta becomes `delta.content`, and the final frame carries
`finish_reason`:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Provider `ReasoningEvent` values are emitted as the optional
`delta.reasoning_content` extension when reasoning mode is enabled. Raw
`<think>` wrappers and opaque provider metadata such as Gemini thought
signatures remain absent. Only visible assistant text becomes `delta.content`. Every chunk's
`model` is the registry ID that actually executed. The turn is resolved before
the initial assistant-role chunk is emitted. If the model fails after the
stream opens, any prior content remains visible, followed by one OpenAI error
envelope and `[DONE]`; no successful `finish_reason` frame is emitted. Policy
stops and refusals instead end normally with `finish_reason: "content_filter"`.

**Reasoning and tool activity (stream only).** By default, sanitized model
reasoning plus server-side `ToolCallEvent` and `ToolResultEvent` progress is
rendered in the optional `delta.reasoning_content` extension. Calls appear when
they start and results when they complete. The default
`OPENAI_COMPAT_TOOL_ACTIVITY_MODE=reasoning` renders portable, emoji-free status
lines such as `Tool web.search started` and `Tool web.search completed`. Set the
mode to `reasoning_full` to add arguments and results in fenced code blocks;
these bodies are bounded by `OPENAI_COMPAT_TOOL_ACTIVITY_MAX_CHARS`. Displayed MCP
separators are normalized from `server__tool` to `server.tool`; internal tool
names are unchanged. Hyphae remains the sole tool executor and never emits standard
`delta.tool_calls`.

Clients that ignore unknown delta fields still reconstruct the complete answer
from `delta.content`. Set `OPENAI_COMPAT_TOOL_ACTIVITY_MODE=hidden` for strict
clients that reject or mishandle `reasoning_content`. This is a deployment
policy, not a per-request option. Non-stream JSON carries plain answer text only.

The adapter still removes old Hyphae-marked `<details>` tool blocks from
replayed assistant history for conversations saved by earlier versions. It
does not remove arbitrary model- or client-authored `<details>` or `<think>`
markup.

```json
{"error": {"message": "'messages' must be a non-empty array", "type": "invalid_request_error", "param": null, "code": null}}
```

An unknown explicit model is also a 400 and names both the invalid ID and the
advertised inventory:

```json
{"error": {"message": "invalid model 'bogus'; available model IDs: gemini-flash, gemini-pro", "type": "invalid_request_error", "param": null, "code": null}}
```

Bad input returns **400**; unexpected internal failure, including an
unrecoverable non-streaming LLM call, returns **500** with
`type: "server_error"`. Streaming errors are emitted in-band as SSE error
frames and still terminate with `[DONE]`.

Native persistent turns publish only protocol-safe checkpoints: a completed
assistant response or a complete assistant-tool-call/result batch. Cancellation
preserves earlier safe checkpoints, balances an interrupted tool batch with
explicit synthetic results when possible, re-raises cancellation, and never
persists a prompt-only or unmatched-tool transcript. Ephemeral `/v1` turns use
the same staging rules but never publish to the native store.

## `GET /v1/models`

OpenAI list shape, sourced from the model registry:

```json
{
  "object": "list",
  "data": [
    {"id": "gemini-flash", "object": "model", "created": 1718600000, "owned_by": "hyphae"},
    {"id": "gemini-pro",   "object": "model", "created": 1718600000, "owned_by": "hyphae"}
  ]
}
```

`data` lists `config/models.yaml`'s model_ids when orchestration is on, or just
the single default model when orchestration is off.

## Connecting OpenWebUI

In OpenWebUI, add an **OpenAI API** connection:

- **API Base URL:** `http://<host>:8000/v1`
- **API Key:** if `HYPHAE_API_KEY` is set, use that value (OpenWebUI sends it as
  `Authorization: Bearer`); if auth is off, any non-empty placeholder works.

OpenWebUI calls `GET /v1/models` to populate its model dropdown and
`POST /v1/chat/completions` (with `stream: true`) for chat. LibreChat connects
the same way via a custom OpenAI-compatible endpoint pointed at `/v1`.
