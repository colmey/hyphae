# PyAiHarness — HTTP API

The full request/response contract for the endpoints. Native surfaces —
`GET /health`, the plain-text `POST /chat`, and the live event feed
`POST /chat/stream` — plus an OpenAI-compatible adapter
(`POST /v1/chat/completions`, `GET /v1/models`) so tools like OpenWebUI and
LibreChat connect natively. All of them are thin shells over one shared core
(`api/turn.py::_turn_events`).

> See also: [README.md](README.md) (overview + quick start),
> [architecture.md](architecture.md) (how the route is wired,
> [the "Bouncer" rationale](architecture.md#design-decisions)),
> [configuration.md](configuration.md) (env vars and config files).

## `GET /health`

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

`orchestration_enabled` is true if and only if an orchestrator was built
during startup. `available_model_ids` lists the model IDs from
`config/models.yaml` (empty when orchestration is off).

## `POST /chat`

**Plain text in, plain text out.** The request body **is** the prompt and
the response body **is** the answer. There is no JSON wire schema — this is a
deliberate dumb-pipe contract that any client (curl, a script, a custom
adapter) can drive with no serialization. Session continuation and the result
metadata travel as headers. For OpenAI-client tooling, use the
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
  this endpoint. The orchestrator picks the model, the tool subset, and the
  system prompt; the iteration cap comes from `settings.max_loop_iterations`. A
  per-call system prompt is available on the
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
- `X-Session-Id` is the session this turn ran in — pass it back on the next
  request to continue the conversation.
- `X-Done-Reason` ∈ `{"end_turn", "max_iterations", "llm_error", "empty", "truncated",
  "budget_exceeded", "deadline_exceeded", "no_progress"}`.
  `"truncated"` means the model stopped on `max_tokens` mid-answer (the body is
  clipped). `"max_iterations"` means the run hit the iteration cap; the loop
  withholds tools on that last step and asks the model to wrap up, so the body
  carries a best-effort final answer rather than mid-investigation fragments. A
  tool that exceeds `TOOL_TIMEOUT_SECONDS` does not end the run — the model sees
  the error and reacts. `"budget_exceeded"`/`"deadline_exceeded"` mean the run hit
  `MAX_RUN_TOKENS`/`MAX_RUN_SECONDS` (both disabled by default); `"no_progress"`
  means `ABORT_AFTER_CONSECUTIVE_TOOL_FAILURES` consecutive tool-call failures
  ended the run early. All three carry whatever answer text had already been
  produced. See [configuration.md](configuration.md) and architecture.md's
  *Bounded & safe runs* section.

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

The **live activity feed**: same turn as `/chat`, but streamed as the loop's
typed events instead of one collected answer. Use it to show *what the agent is
doing* — tool calls and results appear the instant the loop reaches them, then
the answer text follows. Clients that only want the final answer should use
`/chat` (or `/v1`) instead.

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

Event `type`s: `text`, `tool_call`, `tool_result`, `usage`, `done`, `error`
(payload fields mirror `agent/events.py`, serialized by the same bytes-safe
mapping the tracer uses). A failure mid-turn — including the same-session 409 or
an LLM error — is delivered as a terminal `{"type":"error","message":...}` frame
rather than an HTTP status, since the SSE response is already open.

**Text is not token-streamed:** assistant text arrives as one `text` event per
loop iteration, not token-by-token. This endpoint streams *activity*, not tokens
— the harness has no provider-level token streaming by design (it would push
per-provider stream plumbing and reasoning-tag stripping into the core). It is
the seam any live-update consumer plugs into: a dashboard, a voice assistant, or
an OpenWebUI pipe that renders tool events as status updates.

## `POST /v1/chat/completions`

An **OpenAI-compatible** adapter (`api/openai_compatible.py`) so any OpenAI client —
OpenWebUI, LibreChat, the `openai` SDK — drives the harness by pointing its
base URL at `/v1`. It is a thin wire-format translator over the same shared core
as `/chat`; no orchestration or loop logic is duplicated.

**Stateless.** Each request seeds a fresh ephemeral session from the `messages`
array (the client re-feeds the full history every turn), so there is no
server-side session and the same-session 409 guard never applies.

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
  concatenated). `user`/`assistant` messages become the conversation history;
  the final `user` message is the turn that runs.
- `model`, when it matches a `config/models.yaml` model_id, pins that model
  (the orchestrator still selects tools and the system prompt). Otherwise it is
  a free-form label and the orchestrator decides everything. Use
  [`GET /v1/models`](#get-v1models) to discover routable ids.
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

`finish_reason` maps from the loop's done reason: `end_turn → stop`;
`truncated`/`max_tokens`/`max_iterations`/`budget_exceeded`/`deadline_exceeded → length`;
`no_progress → stop`; anything unmapped falls back to `stop`.

**Response (200) — stream** (`stream: true`, OpenWebUI's default): a
`text/event-stream` of `chat.completion.chunk` frames, terminated by
`data: [DONE]`. The first frame carries `delta.role = "assistant"`, each
subsequent frame maps one model text block to a `delta.content`, and the final
frame carries `finish_reason`:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Tool-call visibility (stream only).** Because OpenAI clients render only
`delta.content`, the harness folds each completed server-side tool call into a
collapsible `<details>` block emitted as a `delta.content` chunk (tool name,
arguments, result, and `✅`/`❌` + latency). UIs like OpenWebUI render it as an
expandable "tool" section inline with the answer. These blocks are part of the
assistant message, so a stateless client re-feeds them on the next turn — the
adapter strips them back out of assistant history on the inbound path
(`_strip_tool_blocks`) so they never re-enter the agent's context. The
non-stream JSON response carries plain text only (no blocks).

```json
{"error": {"message": "'messages' must be a non-empty array", "type": "invalid_request_error", "param": null, "code": null}}
```

Bad input (malformed JSON, empty/`user`-less `messages`) returns **400**; an
unexpected internal failure returns **500** with `type: "server_error"`. In
streaming mode the connection is already open, so an error is emitted as a final
SSE `error` frame before `[DONE]` rather than an HTTP status.

## `GET /v1/models`

OpenAI list shape, sourced from the model registry:

```json
{
  "object": "list",
  "data": [
    {"id": "gemini-flash", "object": "model", "created": 1718600000, "owned_by": "pyaiharness"},
    {"id": "gemini-pro",   "object": "model", "created": 1718600000, "owned_by": "pyaiharness"}
  ]
}
```

`data` lists `config/models.yaml`'s model_ids when orchestration is on, or just
the single default model when orchestration is off.

## Connecting OpenWebUI

In OpenWebUI, add an **OpenAI API** connection:

- **API Base URL:** `http://<host>:8000/v1`
- **API Key:** any non-empty value (the harness does not authenticate `/v1`).

OpenWebUI calls `GET /v1/models` to populate its model dropdown and
`POST /v1/chat/completions` (with `stream: true`) for chat. LibreChat connects
the same way via a custom OpenAI-compatible endpoint pointed at `/v1`.
