# PyAiHarness — Architecture

How the harness is put together: the request flow, the stack, the
directory layout, every subsystem, and the architectural principles and
design decisions behind them.

> New here? Start at [README.md](README.md) for the project overview and
> quick start. For the HTTP contract see [api.md](api.md); for config see
> [configuration.md](configuration.md); for running/extending see
> [operations.md](operations.md).

## Contents

1. [System Overview](#system-overview)
2. [Stack and Conventions](#stack-and-conventions)
3. [Directory Layout](#directory-layout)
4. [The Subsystems](#the-subsystems)
5. [Architectural Principles](#architectural-principles)
6. [Design Decisions](#design-decisions)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HTTP Client                                    │
│                                  │                                          │
│                                  ▼                                          │
│                          POST /chat, GET /health                            │
│                                                                             │
│  ┌──────────────────────── api/routes.py ───────────────────────┐           │
│  │                                                                │         │
│  │  1. Read the plain-text prompt; resolve/create the session     │         │
│  │  2. Orchestrate -> pick model + tools + system                 │         │
│  │  3. Append user message, run agent loop with the selections    │         │
│  │  4. Stream-collect the answer text                             │         │
│  │  5. Return plain text + X-Session-Id / X-Done-Reason headers   │         │
│  │                                                                │         │
│  └────────────┬─────────────────────────────────┬────────────────┘          │
│               │                                 │                           │
│               ▼                                 ▼                           │
│  ┌───── orchestrator/orchestrator.py ──┐  ┌── agent/loop.py: run_agent()    │
│  │  one LLM call w/ response_schema    │  │                            │    │
│  │  -> OrchestrationDecision {result,  │  │  ┌── one iteration ─────┐  │    │
│  │     fallback_used, fallback_reason} │  │  │ llm.complete(...)    │  │    │
│  └─────────────┬───────────────────────┘  │  │ for each tool_use:   │  │    │
│                │                          │  │   mcp.call_tool(...) │  │    │
│                ▼                          │  │ session save         │  │    │
│  ┌────────── LLMRegistry ─────────────┐   │  │ if no tool_uses:done │  │    │
│  │  model_id -> LLMClient (lazy)      │   │  └──────────────────────┘  │    │
│  └─────────────────────────────────────┘  └────────────────────────────┘    │
│                                                                             │
│       LLMs ◄────────────── used by both ────────────► MCP servers           │
└─────────────────────────────────────────────────────────────────────────────┘
```

The **orchestrator** is an LLM-driven router that runs once per request
and outputs a structured decision: which model handles the request,
which subset of the MCP tool inventory the agent will see, and what
system prompt the agent will run with. Its output is then fed into
`run_agent()`, which is the reasoning loop and the **one bridge between
worlds** — the only module that touches both the LLM client and the MCP
manager.

> **Caching lives outside the harness.** Response caching is handled by a
> separate layer/service in front of this harness, not inside it. The
> request body is kept small and declarative (`prompt` + optional
> `commands` + optional `MCP`) so a fronting cache can forward it unchanged.

---

## Stack and Conventions

- **Python** 3.11+
- **FastAPI** for the HTTP layer (async-native, fits MCP's async model).
- **`mcp`** official Python SDK for MCP client functionality.
- **`google-genai`** for the LLM (not the deprecated `google-generativeai`).
- **Pydantic + pydantic-settings** for config and request/response schemas.
- **PyYAML** for the MCP config file and the models registry.
- **`jsonschema`** to validate tool-call arguments against each tool's
  declared `input_schema` at the dispatch seam (Phase 1 bounded runs).
- **`asgi-lifespan`** (test-only) for running FastAPI lifespan in
  in-process httpx tests.

### Runtime conventions

- **Bootstrap is in-process and explicit.** `bootstrap.load_secrets()`
  loads the project's `.env` file into `os.environ` (via `python-dotenv`).
  Every entry point (`main.py`, `tests/smoke_test_*.py`, any future CLI)
  calls `load_secrets()` as its first action.
- **Config lives in `.env`.** All environment variables are kept in a
  `.env` file at the project root (template: `.env.example`). Real
  environment variables already set in the process win over `.env`, so the
  same file works for local dev, containers, and CI.
- **Settings are pulled lazily** via `harness_config.get_settings()`.
  Nothing in the codebase should `from harness_config import settings` at
  module level — the global instance doesn't exist. Lazy + cached means
  bootstrap runs first and is picked up correctly.
- **`runscript.sh`** is the canonical launcher; it cd's to the project
  root, activates the venv, prepends `.` to `PYTHONPATH` (so scripts in
  subdirs like `tests/` can still import `bootstrap`, `harness_config`,
  etc.), and runs Python.
- **LLM providers implemented: Gemini and OpenAI-compatible**, each isolated in
  `llm/providers/`. The OpenAI client (`llm/providers/openai.py`) speaks the
  OpenAI wire protocol, so it also drives any OpenAI-compatible server (local
  Ollama/vLLM) via `base_url`. The **default runtime** is a local
  OpenAI-compatible Qwen (see `config/models.yaml`). Anthropic key fields exist
  in `Settings` but its client is still stubbed. Architecture is
  provider-agnostic; adding a provider is "a file in `llm/providers/` + one
  `_PROVIDERS` entry" (see the LLM Layer section).
- **All runtime config lives under `config/`.** Three files: MCP server
  definitions, model registry, orchestrator system prompt. See
  [configuration.md](configuration.md).

---

## Directory Layout

```
PyAiHarness/
├── main.py                   # FastAPI app, lifespan, wires everything
├── bootstrap.py              # load_secrets() - loads .env into os.environ
├── harness_config.py         # Settings (env) + MCPConfig (YAML loader)
├── requirements.txt
├── .env.example              # Template for .env (copy and fill in)
├── setup.sh                  # Creates .venv + installs deps + seeds .env
├── runscript.sh              # Standard launcher (venv + PYTHONPATH + python)
├── chat_client.py            # Optional interactive REPL client (uses HTTP)
├── harness_client.py         # Reference async Python client (orchestration-aware)
│
├── config/                   # All runtime YAML/text config
│   ├── mcp_config.yaml       # MCP server definitions
│   ├── models.yaml           # Routable model registry for the orchestrator
│   └── orchestrator_prompt.md  # Orchestrator's own system prompt
│
├── tests/                    # Standalone smoke scripts (NOT pytest)
│   ├── README.md
│   ├── smoke_test_config.py        # Step 1: settings + MCP config parsing
│   ├── smoke_test_mcp.py           # Step 2: MCP connectivity
│   ├── smoke_test_llm.py           # Step 3: LLM client round-trips
│   ├── smoke_test_session.py       # Step 4: session store, eviction, guard
│   ├── smoke_test_agent.py         # Step 5: full agent loop (Python API)
│   ├── smoke_test_orchestrator.py  # Step 6: orchestration decisions
│   ├── smoke_test_http.py          # Step 7: full HTTP surface in-process
│   └── smoke_test_concurrency.py   # Concurrent requests: isolation + 409 guard
│
├── mcp_layer/                # NOT `mcp/` - shadow-free name for the SDK
│   ├── __init__.py
│   ├── client.py             # MCPClient: one server, one ClientSession
│   └── manager.py            # MCPManager: aggregate, namespace, route
│
├── llm/
│   ├── __init__.py
│   ├── schemas.py            # Provider-agnostic Message / *Block / AssistantMessage
│   ├── client.py             # LLMClient ABC + _PROVIDERS registry + factories (SDK-free)
│   └── providers/            # One file per provider; imported lazily by the registry
│       ├── __init__.py
│       ├── gemini.py         # GeminiLLMClient (owns the google-genai SDK)
│       └── openai.py         # OpenAILLMClient (OpenAI + OpenAI-compatible, e.g. Ollama)
│
├── agent/
│   ├── __init__.py
│   ├── session.py            # Session, SessionStore ABC, InMemorySessionStore
│   ├── events.py             # TextEvent / ToolCallEvent / ToolResultEvent /
│   │                         #   UsageEvent / OrchestrationDecisionEvent /
│   │                         #   DoneEvent / ErrorEvent
│   ├── context.py            # assemble_context() seam: token estimator, budget,
│   │                         #   naive / compaction strategies (view-only)
│   ├── tracing.py            # Tracer ABC / NoOpTracer / JSONLTracer + run_id log adapter
│   └── loop.py               # run_agent() - the async-generator reasoning loop
│
├── orchestrator/             # Routes requests to model + tool subset + system
│   ├── __init__.py
│   ├── schemas.py            # ModelEntry, ModelsConfig, OrchestrationResult,
│   │                         #   OrchestrationDecision
│   ├── config.py             # load_models_config, load_orchestrator_prompt
│   ├── registry.py           # LLMRegistry (lazy LLMClient cache per model_id)
│   └── orchestrator.py       # Orchestrator.decide() -> OrchestrationDecision
│
├── api/
│   ├── __init__.py           # Combines the routers into one
│   ├── schemas.py            # HealthResponse, OrchestrationInfo, TokenUsage (Pydantic)
│   ├── dependencies.py       # FastAPI Depends() providers (pull from app.state)
│   ├── turn.py               # Shared orchestrate→loop core + TurnRunner seam (events/run)
│   ├── routes.py             # Native: GET /health, POST /chat, POST /chat/stream
│   └── openai_compatible.py  # OpenAI adapter: POST /v1/chat/completions, GET /v1/models
│
└── docs/                     # Split reference docs (read on demand)
    ├── README.md             # Index / router + quick start
    ├── architecture.md       # This file: overview, subsystems, principles, decisions
    ├── configuration.md      # Env vars + config/*.yaml + orchestrator prompt
    ├── api.md                # HTTP /chat, /chat/stream, /health, and /v1 contract
    └── operations.md         # Running, extending, smoke tests, limitations
```

---

## The Subsystems

### Config Layer

`harness_config.py` exposes runtime `Settings` and typed MCP config loading.
`Settings` is always accessed via `get_settings()`, never instantiated at
module import.

- `Settings.api_key_for_provider(provider)` returns the key for an
  arbitrary provider — used by each provider's builder so the orchestrator
  can spin up clients for multiple providers in one process. It checks the
  typed key fields first, then falls back to the conventional
  `<PROVIDER>_API_KEY` env var, so a newly registered credentialed provider
  needs no change here (and key-less providers like a local Ollama never
  call it).
- `Settings.required_api_key()` is a thin wrapper over
  `api_key_for_provider(llm_provider)`, preserved for backward compatibility.

`MCPConfig` + `load_mcp_config(path)` parse `config/mcp_config.yaml` with a
discriminated union for server transports and `enabled_servers()` for filtering.

### MCP Layer

`mcp_layer/` (not `mcp/`, to avoid shadowing the official SDK package).

**`MCPClient`** wraps one server session and transport. It connects, lists tools,
filters `disabled_tools`, calls raw tool names, and flattens MCP content blocks
to text for v1.

**`MCPManager`** aggregates clients, connects enabled servers in parallel, and
indexes tools as `{server}__{tool}`. It exposes provider-agnostic tool schemas
to LLM clients and routes namespaced calls back to the owning server. Unknown or
disconnected tools return `is_error=True`.

### LLM Layer

`llm/schemas.py` — provider-agnostic types everyone outside the LLM client
speaks:

- **`Role`** enum: `USER`, `ASSISTANT`, `SYSTEM`, `TOOL`. `TOOL` is
  internal; providers translate (e.g. Gemini wraps function_response in
  user-role content).
- **`Message(role, content: list[ContentBlock])`** with classmethod
  constructors `user(text)`, `assistant(blocks)`, `tool_results(results)`.
- Content blocks:
  - `TextBlock(text, provider_metadata)`
  - `ToolUseBlock(id, name, input, provider_metadata, parse_error)`
  - `ToolResultBlock(tool_use_id, name, content, is_error)`
- **`AssistantMessage(content, stop_reason, model, reasoning)`** with helpers
  `text_blocks()`, `tool_uses()`, `to_message()`. `reasoning` is trace-only
  data extracted by the provider; it is not replayed through `to_message()`
  and is not part of user-facing answer text.
- **`ModelProfile`**: the immutable, provider-agnostic capability profile
  resolved from one `models.yaml` row. It carries
  `supports_native_tools`, `thinking`, and optional sampling
  (`temperature`, `top_p`, `top_k`) across the client build chain as one value.

**Two non-obvious fields** that exist for specific reasons:

1. **`provider_metadata: dict`** on `TextBlock` and `ToolUseBlock`. Holds
   opaque per-provider state that must round-trip back to the model in
   subsequent turns. Specifically: Gemini 3+ attaches a `thought_signature`
   to every Part and *requires* it to be echoed back on function_call
   parts in conversation history — without it, turn 2 of a tool-using
   conversation returns 400. Generic mechanism; other providers ignore it.
2. **`name`** on `ToolResultBlock`. Anthropic's tool_result only needs
   the `tool_use_id`, but Gemini's `function_response` requires the
   function name. The agent loop populates it from the matching
   `ToolUseBlock`.
3. **`parse_error`** on `ToolUseBlock`. Providers set this when the model
   emitted a tool call whose argument string was not valid JSON. The input
   remains `{}` for shape compatibility, but the loop turns the parse failure
   into a teaching `is_error` result instead of silently executing an empty
   call.
4. **`reasoning`** on `AssistantMessage`. Provider-extracted thinking lives as
   a sibling of content, not inside `provider_metadata`. The loop may emit it
   as a `ReasoningEvent` for trace/debug surfaces, but session replay and
   OpenAI-compatible responses stay clean.

`llm/client.py` — the abstraction + the provider registry, and **nothing
SDK-specific** (importing it never pulls in a provider SDK):

- **`LLMClient`** ABC with two call modes:
  ```python
  async complete(
      messages: list[Message],
      tools: list[dict] | None = None,
      system: str | None = None,
      max_tokens: int | None = None,
      response_schema: type | None = None,
      thinking_level: str | None = None,
  ) -> AssistantMessage

  async stream(
      messages: list[Message],
      tools: list[dict] | None = None,
      system: str | None = None,
      max_tokens: int | None = None,
      thinking_level: str | None = None,
  ) -> AsyncIterator[StreamChunk]
  ```
  `complete()` is the canonical completed-turn API and remains the path for
  structured-output calls such as orchestration. `stream()` is an optional
  token-streaming call mode for ordinary agent turns; the ABC fallback calls
  `complete()`, emits each final text block as a coarse `TextDelta`, then
  emits `StreamEnd(AssistantMessage)`. Native streaming providers override it.
- **`StreamChunk`** is provider-agnostic and SDK-free:
  `TextDelta(text=...)` carries visible assistant text during generation, and
  `StreamEnd(message=...)` carries the fully assembled `AssistantMessage`.
  The agent loop streams deltas to callers immediately, then reuses the normal
  assistant/session/usage/reasoning/tool tail once `StreamEnd` arrives.
- **`_PROVIDERS`** — the single source of truth mapping a provider name onto a
  builder. Each builder imports its provider module *lazily* (inside the
  function), so the ABC can be imported without dragging in any SDK, and each
  builder pulls what it needs from the duck-typed `settings` itself (an API
  key, a `base_url`, nothing).
- **`supported_providers()`** exposes the registry's keys; config validators
  (`ModelEntry.provider`) key off it so no other place enumerates providers.
- **`build_llm_client(settings)`** factory dispatches on
  `settings.llm_provider`. Used by `main.py` to build the legacy/default
  client at startup.
- **`build_llm_client_from_entry(entry, settings)`** is the multi-model
  variant. Takes a `ModelEntry` (from `models.yaml`), resolves its
  `ModelProfile`, and pulls the API key by `entry.provider` (not by
  `settings.llm_provider`), so one process can hold clients for multiple
  providers simultaneously. Used by `LLMRegistry`. If the profile declares
  `supports_native_tools: false`, this factory wraps the provider client in
  `PromptedToolLLMClient`; omitted or `true` profiles are not wrapped.

`llm/prompted_tools.py` — the prompted-tool dialect adapter for weak/prose
models:

- Renders the already-filtered tool list into compact system-prompt text:
  tool name, one-line description, and compressed JSON schema.
- Calls the wrapped provider with `tools=None`, so endpoints without native
  tool calling see only ordinary text.
- Parses one fenced or whole-response JSON action from visible prose and
  returns a normal `ToolUseBlock`; final prose remains normal `TextBlock`
  content.
- Bad JSON gets one repair prompt. Parsed semantic errors such as an unknown
  tool or non-object arguments become `ToolUseBlock(parse_error=...)`, which
  the unchanged loop turns into a model-facing `is_error` result instead of
  executing `{}`.

This is the second-dialect drop-in proof: native function calls and prompted
JSON actions differ at the model-interface edge, but both normalize to the
same `AssistantMessage` contract before the agent loop sees them.

**Adding a provider is two steps:** add `llm/providers/<name>.py` implementing
`LLMClient`, then add one `_PROVIDERS` entry. The loop, orchestrator, session
store, MCP layer, and `models.yaml` validation remain provider-blind.

**Structured output (`response_schema`).** Providers that support
structured output (Gemini, OpenAI) honor a Pydantic class passed here
and return JSON conforming to its schema. The orchestrator uses this
for its routing decision; the agent loop does not. Providers without
native support may ignore the kwarg.

**Thinking level (`thinking_level`).** `"low" | "medium" | "high"` (or
`None` to leave the model default). The agent loop passes the orchestrator's
chosen level through on every iteration. Providers consult the selected
model's `ModelProfile`: `hint-param` profiles may map it to a request field
such as `reasoning_effort`, `think-tags` profiles do not add a request knob,
and `none` profiles log once that the knob is inert.

**Gemini specifics** (isolated in `llm/providers/gemini.py`):

- Async via `client.aio.models.generate_content`.
- **Automatic function calling is disabled** — the agent loop is the
  orchestrator, not the SDK.
- **System prompt** is passed via `GenerateContentConfig.system_instruction`,
  not as a message. Internal `Role.SYSTEM` entries in history are
  logged and skipped (they're a bug if they appear).
- **Tool schema** is a no-op transformation: MCP's `input_schema` is
  already JSON Schema, and Gemini's `FunctionDeclaration.parameters_json_schema`
  accepts JSON Schema directly.
- **Tool call IDs** are minted client-side (`call_{uuid}`) because
  Gemini doesn't return one.
- **Role mapping**: `USER` → `"user"`, `ASSISTANT` → `"model"`,
  `TOOL` → `"user"` (Gemini wraps function_response in user-role
  content).
- **`response_schema` caveat**: Gemini's schema dialect is a restricted
  subset of OpenAPI 3 — it does **not** accept `additionalProperties`.
  Pydantic emits that field when a model has `ConfigDict(extra="forbid")`.
  Schemas you pass as `response_schema` must therefore avoid `extra="forbid"`
  (see `OrchestrationResult` for the reference pattern: lenient at the
  parse boundary, then sanitized in code).
- **Reasoning extraction is conservative**: Gemini returns `reasoning=None`
  unless the SDK exposes thought content in a form the provider can identify
  without guessing. Gemini `thought_signature` still uses `provider_metadata`
  for round-trip state.

**OpenAI-compatible specifics** (isolated in `llm/providers/openai.py`):

- Per-model sampling from `ModelProfile` is copied into the chat-completions
  request when present.
- For `thinking: hint-param`, `thinking_level` is passed as
  `reasoning_effort`. For `thinking: think-tags`, a leading
  `<think>...</think>` block is split into `AssistantMessage.reasoning` and
  removed from visible content. For `thinking: none`, the knob is logged as
  inert once and omitted from the request.
- Malformed tool-call argument JSON is surfaced as `ToolUseBlock.parse_error`
  instead of disappearing into an empty argument object.

### Agent Layer

**`agent/session.py`**

- `Session(session_id, messages, created_at, updated_at, metadata)`.
  Auto-generated `sess_<16-hex>` IDs.
- Mutation API: `append_user(text)`, `append_assistant(response)`,
  `append_tool_results(results)`. The agent loop uses these rather than
  poking `.messages` directly — one seam for future invariant checks.
- `last_assistant_tool_uses()` — convenience for "what tools did the
  model just ask me to run?"
- `SessionStore` ABC: `create(metadata)`, `get(session_id)`,
  `save(session)`. Async throughout, even though `InMemorySessionStore`
  doesn't need to be — keeps call sites unchanged when a durable backend
  lands. The harness keeps no durable copy (LibreChat re-feeds context),
  so the ABC is retained purely as that future seam.
- `InMemorySessionStore` is **bounded**: it evicts on idle TTL
  (`SESSION_TTL_SECONDS`) and on a max-size cap (`SESSION_MAX_COUNT`,
  oldest-updated first) so it can't grow without limit under concurrent
  load. Eviction is lazy (swept on `create()`), not a background task.
  Active/in-flight sessions stay "young" because `save()` bumps
  `updated_at` every turn, so they aren't evicted out from under a request.
- `SessionNotFoundError(KeyError)` — subclassing `KeyError` means
  existing `except KeyError` catches still work; callers wanting
  specificity have it.

**Concurrency model.** FastAPI interleaves async handlers on one event loop.
Distinct sessions use distinct `Session` objects; shared `app.state` singletons
carry no per-user state. The only crossover vector is two concurrent requests on
the same `session_id`.

- `SessionGuard` (`agent/session.py`) closes that vector: `claim(session_id)`
  is an async context manager that registers the id as in-flight; a second
  concurrent `claim` of the same id raises `SessionBusyError`, which the
  route maps to **HTTP 409**. New sessions get a fresh id and never contend;
  only client-supplied `commands.session` ids can collide.
- It is **lock-free**: on the single-threaded event loop the membership
  check and the add happen with no `await` between them, so they're atomic
  relative to other tasks. To switch to wait-semantics (queue instead of
  reject), swap the in-flight `set` for a `dict[str, asyncio.Lock]`.

**`agent/events.py`** — dataclass event types with `type: Literal[...]`
discriminators for JSON serialization at the API boundary:

| Event                          | Fields                                       | Emitted when                              |
|--------------------------------|----------------------------------------------|-------------------------------------------|
| `ReasoningEvent`               | `text`                                       | Provider extracted trace-only reasoning from a model response |
| `TextEvent`                    | `text`                                       | Model produced a text block               |
| `ToolCallEvent`                | `id, name, input`                            | Model decided to call a tool (pre-call)   |
| `ToolResultEvent`              | `id, name, content, is_error, latency_ms`    | Tool call completed (`latency_ms` = `call_tool` duration; `None` if stall-skipped) |
| `UsageEvent`                   | `input_tokens, output_tokens, total_tokens, thinking_tokens, cached_tokens, iteration, latency_ms` | One LLM completion finished (`latency_ms` = `complete()` duration incl. retries) |
| `OrchestrationDecisionEvent`   | `model_id, tools, system_prompt, fallback_used` | Defined for tracing/SSE; not emitted into the plain-text `/chat` response today |
| `DoneEvent`                    | `reason, iterations, total_tokens, input_tokens, output_tokens, thinking_tokens` | Loop finished |
| `ErrorEvent`                   | `message`                                    | Unrecoverable internal failure            |

`DoneEvent.reason` ∈ `{"end_turn", "max_iterations", "llm_error", "empty",
"truncated", "budget_exceeded", "deadline_exceeded", "no_progress"}`. Guard
exits carry any answer text already accrued.

Tool execution failures become `ToolResultEvent(is_error=True)` so the model can
recover. `ErrorEvent` is only for failures the model never sees.

**`agent/loop.py`**

```python
async def run_agent(
    session: Session,
    llm: LLMClient,
    mcp: MCPManager,
    *,
    store: SessionStore | None = None,
    system: str | None = None,
    max_iterations: int = 25,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    llm_timeout_seconds: float | None = None,
    tool_timeout_seconds: float | None = None,
    max_retries: int = 0,
    retry_base_delay: float = 0.5,
    tool_result_max_chars: int | None = None,
    max_run_tokens: int | None = None,
    max_run_seconds: float | None = None,
    abort_after_consecutive_tool_failures: int | None = None,
    thinking_level: str | None = None,
    context_strategy: str = "naive",
    context_window: int | None = None,
    context_safety_margin_tokens: int = 1024,
    context_recent_messages: int = 6,
    context_summary_max_tokens: int = 512,
) -> AsyncIterator[Event]:
    ...
```

Reliability params default to legacy behavior for direct callers; routes opt in
by passing `Settings` values. The loop owns timeout/retry policy while providers
classify transient errors.

**The caller appends the user message before invoking `run_agent`.** The
loop owns assistant turns and tool round-trips. This keeps the loop
callable identically from a CLI, a FastAPI route, or a test.

The **`tools` parameter** is the orchestration seam. When `None`, the
loop pulls the full MCP inventory (`mcp.get_tools_for_llm()`) — legacy
behavior, used by code paths that aren't orchestration-aware. When
provided, the loop uses the list verbatim — this is how the route
hands the orchestrator's filtered tool subset to the model.

The **`thinking_level` parameter** is the deliberation seam. It is passed
straight through to every `llm.complete()` call of the run; the loop never
inspects it. `None` (the default) leaves the model's own default. The route
supplies the orchestrator's chosen level here; see *Thinking level* under
the Orchestration Layer.

**The bounded-run guard params** (`max_run_tokens`, `max_run_seconds`,
`abort_after_consecutive_tool_failures`) follow the same default-disabled
pattern as the reliability params: `None`/`<=0` disables each dimension, so
direct callers are unaffected. The `/chat` and `/v1` routes opt in via the
corresponding `Settings` fields, themselves `0` (disabled) by default — an
operator sets them explicitly. See *Bounded & safe runs* below.

Per-iteration algorithm:

0. Check bounded-run guards before another LLM call.
1. Assemble the outgoing context view, then call the LLM with timeout/retry.
   Exhausted LLM failures yield `ErrorEvent` + `DoneEvent("llm_error")`.
2. `session.append_assistant(response)` immediately — a later crash in
   this iteration still leaves the session consistent.
3. Yield a `UsageEvent` for this iteration's tokens. Provider-reported usage
   is used as-is; absent/all-zero usage is filled by the local estimator
   (`estimate_usage_tokens`, chars/4 heuristic over the outgoing view +
   system + response) so the token cap works against local servers that
   report zero usage. Never double-counted.
4. If `store` was provided, `await store.save(session)`.
5. If the response carries `reasoning`, yield a `ReasoningEvent` for trace and
   raw debug renderers. Reasoning is not appended to session content.
6. Yield a `TextEvent` for each non-empty text block.
7. Collect tool calls; if no tools were requested, finish with the appropriate
   done reason (`end_turn`, `truncated`, or `max_iterations`).
8. For each tool call, sequentially: emit `ToolCallEvent`, run repeat-call
   detection, convert provider parse errors or schema validation failures into
   teaching `is_error` results, enforce `ToolPolicy`, call MCP with timeout,
   clip the result, update failure counters, emit `ToolResultEvent`, and build
   the matching `ToolResultBlock`.
9. `session.append_tool_results(results)` and save again.
10. Loop. **Final-iteration wrap-up:** on the last allowed iteration the
   loop withholds tools and appends a wrap-up note to the per-call system prompt
   so the model produces a best-effort final answer instead of dying
   mid-investigation; the run reports `DoneEvent("max_iterations")`.

### Bounded & safe runs (Phase 1)

Three **independent stop conditions**, layered on top of `max_iterations`,
each with its own `done_reason` and the partial answer accrued so far —
never a silent truncation:

| Guard | Settings field | Checked | `done_reason` |
|---|---|---|---|
| Token budget | `max_run_tokens` | Before each iteration and immediately after each reported `Usage.total_tokens` update | `budget_exceeded` |
| Wall clock | `max_run_seconds` | Before each iteration and around in-flight LLM/tool calls/retry sleeps | `deadline_exceeded` |
| No-progress abort | `abort_after_consecutive_tool_failures` | After each tool result, against the consecutive-failure counter | `no_progress` |

All three default to `0` (disabled) in `Settings`, matching the repo's
established pattern for new safety knobs (e.g. `trace_enabled`) — installing
the harness doesn't change behavior until an operator opts in.

**Token cap and zero-usage providers:** when a provider reports absent or
all-zero `Usage` (common on local OpenAI-compatible servers), the loop fills
in a local estimate from the outgoing messages + system prompt + response
(`agent/context.py: estimate_usage_tokens`, chars/4 heuristic), so
`max_run_tokens` still trips. Non-zero provider usage is authoritative and
never mixed with estimates; a missing `total_tokens` is filled from
`input + output`.

**Tool-argument validation** lives at the same dispatch seam (see step 8
above) but is always on — it isn't a stop condition, it's a per-call check
that turns a would-be opaque MCP error into a teaching `is_error` result
before the call ever reaches the server. Provider parse failures
(`ToolUseBlock.parse_error`) take the same path, so invalid JSON arguments are
visible feedback rather than a silent `{}` execution. Validation and parse
failures count toward the consecutive-failure counter like any other tool
error, so a model stuck sending malformed args still trips the no-progress
abort if one is configured.

If any bounded-run guard trips after an assistant has requested tools, the loop
emits and persists compact synthetic `is_error=True` results for skipped tool
calls before `DoneEvent`. That keeps provider histories well-formed for a later
turn: every assistant tool call still has a matching tool result.

The loop-intelligence behaviors (stall check, final-iteration wrap-up, and the
consecutive-failure nudge) are **always on** — pure
steering with no failure mode that warrants a kill switch, so unlike the
reliability params they carry no `Settings` toggle.

### Context assembly (Phase 3)

`agent/context.py` shapes session history into the outgoing LLM view. The loop
calls `assemble_context()` per LLM call against
`context_window − max_output_tokens − safety_margin`, using a cheap local token
estimate that includes messages, tool schemas, and the effective system prompt.

Two strategies, selected by `CONTEXT_STRATEGY` (config selects strategies;
code implements them):

- **`naive`** (default, behavior-preserving): pass-through. Over budget only
  logs a warning — no message is altered, no request is blocked.
- **`compaction`** (opt-in): when the estimate exceeds the budget, the view
  becomes *task header (first user message, verbatim) + one summary message +
  the last N protocol-safe units (verbatim)*. The middle is summarized by one
  tool-less LLM call on the same selected client (bounded `max_tokens`, no
  recursive assembly; the transcript is clipped to roughly the input budget).
  The summary is a plain user message prefixed `Conversation summary so
  far:` — no new role or block type.

**View-only, always.** Compaction shapes the outgoing view for one call;
`session.messages` remains the append-only source of truth. The view is
reassembled every iteration as history grows.

**Protocol safety.** Assistant tool-use messages travel with their tool results,
so compaction does not split provider-required pairs. Malformed history degrades
to pass-through.

**Budget inputs.** Selected model entries provide `context_window` and
`max_tokens`; Settings fill gaps. Direct `run_agent` callers that pass no
`context_window` skip assembly.

**Degrade, never break.** Summarizer failures, malformed boundaries, or
too-short histories fall back to full history with a warning.

The module also owns `estimate_usage_tokens()`, the estimator behind the
token-cap fallback described under *Bounded & safe runs*.

### Orchestration Layer

`orchestrator/` is a router that sits in front of the agent loop. Per
request, it makes one LLM call to decide:

- **which model** in the registry handles this request,
- **which subset of MCP tools** the downstream agent will see,
- **what system prompt** the downstream agent runs with,
- **how hard the model should think** — a `thinking_level` of `low`,
  `medium`, or `high`.

Its output is handed to `run_agent()` as concrete arguments; the loop remains
provider- and orchestrator-blind.

**`orchestrator/schemas.py`**

```python
class ModelEntry(BaseModel):
    provider: str          # validated against llm.client.supported_providers()
    model: str
    description: str       # what the orchestrator LLM sees
    max_tokens: int | None = None
    context_window: int | None = None  # feeds the loop's context budget
    supports_native_tools: bool = True
    thinking: Literal["none","hint-param","think-tags"] = "none"
    sampling: SamplingParams | None = None
    default: bool = False  # exactly one entry should be default

    def to_profile(self) -> ModelProfile: ...

class ModelsConfig(BaseModel):
    models: dict[str, ModelEntry]

class OrchestrationResult(BaseModel):
    selected_model_id: str
    selected_tools: list[str]              # namespaced tool names
    generated_system_prompt: str
    thinking_level: Literal["low","medium","high"] = "medium"

@dataclass
class OrchestrationDecision:
    result: OrchestrationResult
    fallback_used: bool = False
    fallback_reason: str | None = None
```

`OrchestrationResult` is the LLM's structured output. `OrchestrationDecision`
adds fallback metadata for in-process callers.

**`orchestrator/config.py`** — `load_models_config(path)` and
`load_orchestrator_prompt(path)`. Mirrors the loader pattern in
`harness_config.load_mcp_config`.

**`orchestrator/registry.py`** — `LLMRegistry`:

- Lazy-builds an `LLMClient` per `model_id` on first use, caches it.
- `registry.get(model_id)` returns a built client (raises `KeyError`
  on unknown id).
- `registry.get_or_default(model_id)` returns `(resolved_id, client)`;
  silently falls back to default on unknown id.
- `registry.describe_for_prompt()` formats the model inventory as the
  orchestrator-prompt block ("AVAILABLE MODELS").
- Concurrency: clients are built on first use and stashed in a dict
  with no lock. Client construction is idempotent, so a rare double-build
  wastes a few cycles but cannot produce wrong behavior.

**`orchestrator/orchestrator.py`** —
`Orchestrator.decide(user_message, preferences=None, history=None)`:

1. Builds the prompt: orchestrator system instruction (from the
   `orchestrator_prompt.md` file) + AVAILABLE MODELS block + AVAILABLE
   TOOLS block (live MCP inventory) + an optional CONVERSATION SO FAR
   block (see *Context-aware routing* below) + USER MESSAGE.
2. Calls the orchestrator's own LLM client with
   `response_schema=OrchestrationResult`. Tools are **not** exposed —
   the orchestrator must decide, not act.
3. Parses the JSON response into `OrchestrationResult`. Strips any
   stray markdown fences defensively. `thinking_level` is coerced
   leniently (a stray/unknown value clamps to `"medium"`) so one odd
   field can't sink an otherwise-valid decision.
4. Sanitizes the result: drops tool names not in the live MCP inventory;
   rewrites an unknown `selected_model_id` to the registry default.
   Sanitization is **not** counted as fallback — the orchestrator made
   a real decision, we just trimmed it.
5. Returns `OrchestrationDecision(result=..., fallback_used=False)`.

**Thinking level.** The route passes `result.thinking_level` to `run_agent()`,
which forwards it to every `llm.complete()` call. The selected client's
`ModelProfile` decides whether that value becomes a provider request hint
(`hint-param`), is intentionally inert (`none`), or is represented by
self-emitted reasoning tags (`think-tags`). The loop never branches on provider
or model name.

**Context-aware routing.** Continued sessions pass prior messages to the
orchestrator as a compact text-only tail, so follow-up turns route with enough
context to resolve references and keep thread-critical tools.

**Failure handling.** ANY failure (LLM call error, parse error,
validation error) is caught and replaced with a safe fallback:

```python
OrchestrationDecision(
    result=OrchestrationResult(
        selected_model_id=registry.default_id(),
        selected_tools=[all available tool names],
        generated_system_prompt="You are a helpful assistant. Use the available tools when relevant.",
    ),
    fallback_used=True,
    fallback_reason="<short description of why>",
)
```

The fallback preserves pre-orchestrator behavior while `fallback_used` makes the
degradation observable.

Startup also rejects a prompted-only orchestrator control model. If
`ORCHESTRATOR_MODEL_ID` or the registry default points to a row with
`supports_native_tools: false`, `_try_build_orchestration()` logs a warning and
runs in legacy mode. Prompted-tool models may still be selected for downstream
agent turns; they are not supported as the structured-output control model in
this v1 adapter.

**The system-prompt override.** The route, not the orchestrator,
implements precedence: if the request sets `commands.system`, that wins
over `result.generated_system_prompt`. Orchestration still runs (to pick
model + tools), only the system prompt is overridden. This lets a client
pin an explicit system prompt while keeping orchestration on.

**MCP tool preferences.** The route normalizes the request's `MCP` list
into a `ToolPreferences` value object and hands it to
`orchestrator.decide()`. Preferred tools are rendered into the
orchestrator prompt as a priority hint, and `_sanitize()` unions the
valid ones into the final selection so they're *guaranteed* exposed —
while the orchestrator's own picks remain as fallback. It's a priority,
not a lock-out. Per-tool argument lists are informational context only.

**Disabled mode.** If orchestration is disabled or its config cannot load, routes
use the default LLM, all MCP tools, and any request system override directly.

### HTTP Layer

**`main.py`** — FastAPI entry point with a `lifespan` context manager.

Startup order:
1. `bootstrap.load_secrets()` (called at the top of `main.py`).
2. `get_settings()` — reads from env.
3. `build_llm_client(settings)` — fails fast if API key missing.
   This is the *legacy/default* client used when orchestration is
   disabled.
4. `load_mcp_config(settings.mcp_config_path)`.
5. `MCPManager(mcp_config).startup()` — connects to all enabled servers
   in parallel.
6. `InMemorySessionStore()`.
7. `_try_build_orchestration(settings, mcp)` — returns registry/orchestrator or
   `(None, None)` on optional-layer failure.
8. All values stashed on `app.state` (`settings`, `llm`, `mcp`, `store`,
   `guard`, `registry`, `orchestrator`).
9. A single consolidated "harness ready" INFO log line is emitted.

Shutdown closes MCP connections; other singletons currently need no teardown.

**`api/dependencies.py`** — `Depends()` providers that pull from `app.state`.
`require_api_key` is the optional route-layer auth gate for `/chat`,
`/chat/stream`, and `/v1/*`; `/health` stays open.

**`api/routes.py`** — request flow:

1. **Read the prompt** from the plain-text request body (empty → 400).
   **Resolve/create the session first** (the `X-Session-Id` header). The session
   is resolved *before* routing so the orchestrator can route a follow-up turn
   with the conversation in view. The new prompt is **not** appended yet — it's
   passed to the orchestrator separately, so `session.messages` at routing time
   is the prior history only.
2. **`runner.run(...)`** — every renderer reaches the shared core through
   **`TurnRunner`**, which bundles process-wide singletons and exposes `events()`
   and `run()`. `_resolve_routing(...)` selects LLM, tools, system prompt,
   thinking level, and optional orchestration metadata.
3. **Per-request log line** at INFO level summarizing the routing decision
   (model, tool count, thinking level, fallback flag).
4. Under the same-session guard, append the prompt and iterate `run_agent(...)`
   with the orchestrator's selections — including `thinking_level` — collecting
   `TextEvent` text into the answer and reading the final `DoneEvent` for the
   done-reason and token usage.
5. Return `PlainTextResponse(answer, headers={X-Session-Id, X-Done-Reason})`.

The full event stream is no longer serialized into the response (the body is
plain text); routing/usage detail lives in the logs and the JSONL trace.

`_turn_events` also mints the per-request **`run_id`**, tags route logs, and
threads optional tracing into `run_agent`. Tracing is best-effort and mirrors the
event stream; see `agent/tracing.py` and operations docs.

---

## Architectural Principles

1. **Separate the agent loop from the LLM client.** The LLM client only
   knows how to send messages and get a response back. The loop is the
   only place that knows about tools, MCP, and iteration. This is the
   #1 thing that keeps the harness extendable.
2. **MCP servers are long-lived.** Connect on startup via the FastAPI
   lifespan; do not spawn per-request.
3. **One bridge between worlds.** `agent/loop.py` is the only module
   that touches both the LLM client and the MCP manager. Neither of
   those two knows the other exists.
4. **The loop is an async generator** yielding typed events. Each HTTP
   surface is a thin renderer over the same generator (`api/turn.py`):
   `/chat` collects all events into one plain-text response, `/chat/stream`
   forwards them as native SSE event frames, and the `/v1` adapter maps them
   to OpenAI `chat.completion.chunk` frames. Adding a transport is a new
   renderer, not a loop change.
5. **Session abstraction from day one.** Even v1 threads message history
   through loop iterations via a `Session` object. A minimal abstraction
   now beats retrofitting it later.
6. **Orchestration is a router, not a part of the loop.** It runs ONCE
   per request, takes no tools (it must decide, not act), and produces
   a structured decision that's fed into `run_agent()` as concrete
   arguments. The loop is unchanged from the pre-orchestrator design.
7. **A minimal, explicit boundary.** `/chat` is a plain-text dumb pipe — the
   body *is* the prompt, so there is nothing to over-accept; an empty body is
   the only rejection (400). Optional behavior travels as named headers
   (`X-Session-Id`), never as free-form fields. JSON request schemas that do
   survive (e.g. the planned OpenAI-compatible endpoint, `/health`) stay strict.
   The surface is small and explicit; new capabilities are added as new typed
   inputs, not by accepting unknown ones.
8. **Orchestration must never break a request.** Every failure path in
   the orchestrator (LLM error, parse error, validation error) degrades
   to a safe default (the registry's default model, all tools, generic
   system prompt) with `fallback_used=true` so the degradation is
   observable. The harness can always serve traffic even when
   orchestration is failing.

---

## Design Decisions

The accumulated rules-of-the-road. Each was a real choice, often made to
resolve a problem we hit; don't change them without understanding why.

1. **Provider-agnostic until the provider client.** The MCP manager doesn't
   know what model is being used. Provider-specific tool schema shaping
   happens only inside the provider's `llm/providers/<name>.py` (the LLM
   layer's ABC and registry in `llm/client.py` stay SDK-free).
2. **Settings are pulled lazily**, not snapshotted at import. Bootstrap
   (which loads `.env` into `os.environ`) and the harness run in the same
   process; lazy + cached means env vars are read after bootstrap, not before.
3. **`mcp_layer/` not `mcp/`** to avoid shadowing the SDK's `mcp` package.
4. **`__` is the tool-namespacing separator.** Config validation enforces
   that server names can't contain it.
5. **Graceful MCP startup**: one bad server doesn't kill the harness.
6. **`provider_metadata` round-trips opaque per-provider state.** Required
   for Gemini 3+'s `thought_signature`. Generic mechanism — other providers
   ignore it. Never strip this field anywhere in the pipeline.
7. **`ToolResultBlock.name` is mandatory.** Gemini needs the function name
   on the response; Anthropic doesn't but tolerates it. The agent loop
   copies it from the matching `ToolUseBlock`.
8. **System messages travel via the `system=` param**, not in the message
   list. A `Role.SYSTEM` entry in history is a bug; providers should log
   and skip.
9. **Automatic function calling is disabled** in the Gemini client. The
   agent loop is the orchestrator. Don't re-enable.
10. **`google-genai`, not `google-generativeai`.** The latter is deprecated.
11. **Session mutation goes through helpers** (`append_user`,
    `append_assistant`, `append_tool_results`), not direct `.messages.append()`.
12. **`SessionStore.save()` is explicit, not auto-on-mutate.** In-memory
    save is a no-op today; writing the calls explicitly now means durable
    storage can drop in without call-site changes.
13. **The loop saves after every iteration**, not just at the end. Keeps
    the session consistent through partial failures.
14. **Tool execution is sequential, not parallel.** Some MCP tools have
    side effects; parallel makes failure modes harder to reason about.
    Wrap in `asyncio.gather` later if needed.
15. **Tool failures stay in-conversation; LLM failures kill the loop.**
    A `mcp.call_tool` error becomes `ToolResultBlock(is_error=True)` and
    goes back to the model. An `llm.complete` exception yields
    `ErrorEvent` + `DoneEvent("llm_error")` and stops.
16. **Caller appends the user message before invoking `run_agent`.** The
    loop owns only assistant turns and tool round-trips.
17. **System prompt is a per-call argument, not stored on the Session.**
    Session = conversation state; system prompt = call-time configuration.
18. **Routes use `Depends()`, not `request.app.state` directly.** Keeps
    routes testable and explicit about their deps.
19. **Loop events serialize explicitly, never `dataclasses.asdict`.** The
    plain-text `/chat` response carries no event stream, but two serializers now
    exist — the JSONL trace (`agent/tracing.py`) and the `/v1` SSE adapter — and
    each maps events field-by-field. `provider_metadata` holds `bytes` (Gemini's
    thought_signature) that won't survive a naive JSON encode and must not leak;
    the JSONL writer additionally base64-encodes any stray `bytes` defensively.
20. **`httpx.ASGITransport` does NOT run lifespan events.** Use
    `asgi-lifespan.LifespanManager` to drive lifespan in in-process
    tests. (Production via `uvicorn` runs lifespans natively.)
21. **`/chat` is plain text, not JSON.** The request body *is* the prompt and
    the response body *is* the answer; session continuation rides on the
    `X-Session-Id` header (echoed back, with `X-Done-Reason`). This replaced the
    bespoke `{prompt, commands, MCP}` request and rich JSON response — the
    serialization-free contract any client can drive. Per-call `system`,
    `max_iterations`, and `MCP` tool preferences left the wire; the orchestrator
    selects model/tools/system, the iteration cap is config-driven, and a
    per-call system prompt belongs on the planned OpenAI-compatible endpoint
    (the `ToolPreferences`/`_resolve_routing(preferences=...)` plumbing stays
    in-process for an easy re-add).
22. **`OrchestrationResult` is lenient (no `extra="forbid"`).** Gemini's
    `response_schema` dialect doesn't accept `additionalProperties: false`,
    which Pydantic emits when a model is strict. We sanitize the result
    in code instead. This is the reference pattern for any Pydantic
    class you intend to pass as a structured-output schema.
23. **Sanitization is not fallback.** The orchestrator's `_sanitize()`
    method drops unknown tool names and corrects unknown model_ids
    without raising the `fallback_used` flag. Fallback is reserved for
    cases where the orchestrator's LLM call itself failed.
24. **The orchestrator gets no tools.** It must decide, not act. Calling
    `complete()` with `tools=None` enforces this.
25. **Orchestration is optional and degradable.** Missing config files,
    failed LLM calls, or bad outputs all degrade to the legacy
    (pre-orchestrator) behavior. The route checks
    `app.state.orchestrator is None` and routes around it. This is what
    lets the orchestration subsystem be installed or removed without
    affecting anything else.
26. **`request.system` overrides `generated_system_prompt`.** The
    orchestrator still runs (to pick model + tools) but the user's
    explicit system prompt wins. Keeps pre-orchestrator clients working
    unchanged.
27. **One LLM provider's keys aren't all of them.** Each provider's builder
    pulls its own credentials via `Settings.api_key_for_provider(...)`, so one
    process can hold clients for any provider in `models.yaml`. The lookup
    falls back to `<PROVIDER>_API_KEY` for providers without a typed field, and
    `required_api_key()` is preserved as a thin wrapper for legacy callers.
28. **Thinking level is a routing dimension, not just a model choice.** The
    orchestrator emits `thinking_level` (`low`/`medium`/`high`) alongside
    the model, and the loop passes it to every `complete()` call. This makes
    deliberation a per-request lever independent of model selection — cheap
    requests can shed thinking for latency, hard ones can buy more. `None`
    leaves the model default, so legacy callers are unaffected. The field is
    coerced leniently (a bad value clamps to `medium`) so it can never sink
    an otherwise-valid decision — same spirit as decision #23.
29. **The orchestrator routes with conversation context.** The route resolves
    the session before routing and passes the prior history to `decide()`,
    which folds a clipped tail into a CONVERSATION SO FAR block. A follow-up
    turn is routed against the conversation, not a bare fragment, so it keeps
    the tools the thread depends on. The new prompt is appended only after
    routing, so it stays the orchestrator's separate USER MESSAGE input.
