# HTTP API

Hyphae exposes one native API and a text-oriented OpenAI-compatible adapter.
Both execute through the same `ApplicationRuntime` and `TurnRunner`; the HTTP
layer only validates and renders wire contracts.

## Common behavior

The fixed request-body limit is 1 MiB (1,048,576 bytes). It is checked against
both a usable `Content-Length` and the bytes actually streamed. There is no
configuration override.

`HYPHAE_API_KEY` is an optional gate on `/chat`, `/chat/stream`, and `/v1/*`.
When set, callers may send either `X-API-Key: <key>` or `Authorization: Bearer
<key>`; `X-API-Key` takes precedence. `/health` is always public.

## Native API

### `GET /health`

Returns process, routing, and MCP catalog health:

```json
{
  "status": "ok",
  "provider": "gemini",
  "model": "gemini-3-flash-preview",
  "connected_servers": ["web-search"],
  "tool_count": 4,
  "mcp_servers": [
    {
      "name": "web-search",
      "state": "healthy",
      "last_error": null,
      "tool_count": 4,
      "catalog_revision": 1,
      "last_discovered_at": "2026-08-06T12:00:00Z",
      "next_refresh_at": "2026-08-06T12:05:00Z",
      "active_leases": 0
    }
  ],
  "orchestration_enabled": true,
  "available_model_ids": ["model-a", "model-b"]
}
```

`status` is `degraded` if any enabled server is not healthy. In orchestrated
mode, `available_model_ids` contains ready registry models. A degraded fixed
route advertises its configured fixed model. Ordinary direct mode leaves the
health inventory empty; `/v1/models` remains the authoritative model list for
OpenAI clients.

### `POST /chat`

The body is one UTF-8 prompt. Missing `Content-Type`, or `text/plain` with
optional parameters, is accepted; another media type returns 415. Invalid
UTF-8 and empty/whitespace-only prompts return 400.

Send `X-Session-Id` to continue a native session. Omitting it creates a new
session. An unknown supplied ID returns 404. A successful response is plain
text with:

- `X-Session-Id`: the resolved session;
- `X-Done-Reason`: the terminal reason.

### `POST /chat/stream`

Input and session behavior match `/chat`. The response is server-sent events;
each `data:` value is JSON produced by the same explicit event mapping used by
tracing. Possible successful records include `orchestration`, `text`,
`tool_call`, `tool_result`, `usage`, and `done`. Provider reasoning is not
exposed on the native API.

If the stream has already opened and execution fails, it emits one safe frame:

```json
{"type":"error","code":"provider_failure","message":"The model provider failed to complete the request."}
```

There is no following successful `done` frame for that failure.
Authentication, body/media validation, and native session lookup happen before
the response opens; those failures retain their HTTP status and return the
normal native JSON error body. Failures during `TurnRunner.open()` or execution
after the SSE response opens are error frames because the HTTP status can no
longer change.

Native errors use this envelope:

```json
{"code":"invalid_request","message":"The request is invalid."}
```

## OpenAI-compatible API

### `POST /v1/chat/completions`

The supported request is intentionally narrow:

```json
{
  "model": "optional-model-id",
  "stream": false,
  "messages": [
    {"role": "system", "content": "optional caller-owned behavior"},
    {"role": "user", "content": "question"}
  ]
}
```

`messages` must be nonempty and contain at least one user message; the final
non-system message must be a user message with nonempty text. Roles are only
`system`, `user`, and `assistant`. Content is either a strict string or a
nonempty list of objects shaped exactly as `{"type":"text","text":"..."}`.
Unknown message/content-part fields and unsupported roles or content are
rejected. Unknown top-level fields are ignored. `model` must be a string when
present, and `stream` must be a boolean.

All system messages are joined into the authorized downstream system override.
Earlier user/assistant messages seed a fresh ephemeral session; the final user
message becomes the active prompt. Legacy Hyphae-rendered tool detail blocks
are stripped from assistant history before replay. Every `/v1` request is
ephemeral—clients remain responsible for durable conversation history.

The adapter parses JSON regardless of the request's declared media type. An
invalid UTF-8 body, invalid JSON, unsupported message shape, or invalid
conversation returns `invalid_request`.

Buffered success is a standard `chat.completion` object with one assistant
choice and usage. The executing model—not merely the requested value—is
returned in `model`.

With `stream: true`, the response is SSE `chat.completion.chunk` values followed
by `data: [DONE]`. The first chunk establishes the assistant role. Answer text
uses `delta.content`. Depending on `OPENAI_COMPAT_TOOL_ACTIVITY_MODE`, sanitized
provider reasoning and compact tool activity may use the nonstandard optional
`delta.reasoning_content` channel; `hidden` omits it and `reasoning_full` adds
bounded arguments/results.

Authentication, bounded-body/JSON/schema validation, conversation validation,
and explicit model validation happen before an OpenAI streaming response opens;
those failures retain their HTTP status and return a JSON OpenAI error envelope.
Failures while opening or executing the turn after SSE has opened are OpenAI
error frames followed by `[DONE]`, without a successful finish chunk.

Successful internal terminal reasons map as follows:

| Hyphae done reason | OpenAI `finish_reason` |
|---|---|
| `end_turn`, `empty`, `no_progress` | `stop` |
| `truncated`, `max_tokens`, `max_iterations`, `budget_exceeded`, `deadline_exceeded` | `length` |
| `content_filter`, `refusal` | `content_filter` |

### `GET /v1/models`

Returns an OpenAI-style model list. Orchestrated mode exposes ready registry
IDs. Direct mode exposes the configured global model. A degraded fixed route
advertises only the ready registry default selected during startup.

## Error matrix

OpenAI endpoints wrap the same safe values as:

```json
{
  "error": {
    "message": "The request is invalid.",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_request"
  }
}
```

| HTTP | Code | Safe message | Typical cause |
|---:|---|---|---|
| 400 | `invalid_request` | The request is invalid. | Empty native prompt, invalid UTF-8/JSON, or unsupported `/v1` input |
| 400 | `invalid_model` | The requested model is not available. | Unknown or inadmissible model ID |
| 401 | `authentication_failed` | Invalid or missing API key. | Configured API key did not match |
| 404 | `session_not_found` | Session not found. | Unknown native session ID |
| 409 | `session_busy` | Session is processing another request. | Concurrent turn for one native session |
| 409 | `session_history_limit` | Session history limit reached; start a new session. | Persistent transcript would exceed its cap |
| 413 | `request_too_large` | Request body too large. | Body exceeds 1 MiB |
| 415 | `unsupported_media_type` | Content-Type must be text/plain. | Native route received a different media type |
| 500 | `execution_protocol_error` | The service could not complete the request. | Invalid terminal/event protocol |
| 500 | `internal_error` | An internal server error occurred. | Internal configuration/inventory failure |
| 502 | `provider_failure` | The model provider failed to complete the request. | Provider/stream failure |
| 503 | `model_unavailable` | The requested model is temporarily unavailable. | Configured model failed readiness |
| 503 | `session_capacity_unavailable` | Session capacity is temporarily unavailable. | No idle session can be admitted/evicted |

Internal exception text, provider diagnostics, paths, and configuration values
are logged privately and are not copied into public bodies.
