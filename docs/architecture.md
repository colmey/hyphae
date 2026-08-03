# hyphae — Architecture

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
│              Native /chat* + /health; OpenAI-compatible /v1/*             │
│                                                                             │
│  ┌──────────── api/routes.py + api/openai_compatible.py ───────┐           │
│  │                                                                │         │
│  │  1. Parse the native or OpenAI-compatible request              │         │
│  │  2. Inject one typed ApplicationRuntime                        │         │
│  │  3. Derive one coherent TurnRunner and resolve the session     │         │
│  │  4. Run/stream the accepted turn                               │         │
│  │  5. Render native text/SSE or OpenAI JSON/SSE                  │         │
│  │                                                                │         │
│  └────────────────────────┬──────────────────────────────────────┘          │
│                           ▼                                                 │
│  ┌── api.dependencies.ApplicationRuntime / api.turn.TurnRunner ──┐         │
│  │ settings + routing + limits + MCP + store/guard + policy/trace │         │
│  └────────────┬─────────────────────────────────┬─────────────────┘         │
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
`run_agent()`, which is the provider-neutral reasoning loop and the bridge
between `LLMClient` and the turn-local `ToolRuntime`. MCP connection/catalog
ownership remains behind that neutral tool contract.

The FastAPI lifespan publishes exactly one
`api.dependencies.ApplicationRuntime` on `app.state`. It is a frozen process
composition containing settings, routing, limits, MCP ownership, session
store/guard, policy, and tracer. Every HTTP route receives that same value
through `get_application_runtime()` and derives its `TurnRunner` from it; there
are no independently injected stores or partially assembled routing fields.
Whole-runtime dependency overrides therefore replace one coherent graph.

`TurnRunner.open()` owns the accepted-turn envelope. After the route resolves or
creates the session identity, it claims that identity before consuming the
latest stored history, inventory, or routing inputs. It then creates one
`RunContext` with one run ID and absolute deadline, snapshots tools once,
resolves the actual model, and holds the claim until event iteration and cleanup
finish. `TurnMetadata` carries the resolved model and optional sanitized
`OrchestrationDecisionEvent` for both buffered and streaming renderers.

> **Caching lives outside the harness.** Native chat is a plain-text dumb pipe;
> the OpenAI-compatible surface accepts the standard `messages` history. A
> fronting cache can treat both contracts without harness-owned response state.

---

## Stack and Conventions

- **Python** 3.12
- **FastAPI** for the HTTP layer (async-native, fits MCP's async model).
- **`mcp`** official Python SDK for MCP client functionality.
- **`google-genai`** for the LLM (not the deprecated `google-generativeai`).
- **Pydantic + pydantic-settings** for config and request/response schemas.
- **PyYAML** for the MCP config file and the models registry.
- **`jsonschema`** to validate tool-call arguments against each tool's
  declared `input_schema` at the dispatch seam (Phase 1 bounded runs).
- **`pytest` + AnyIO** for hermetic regression discovery and async tests, and
  **`asgi-lifespan`** for FastAPI lifespan in in-process httpx checks.
- **Mypy strict mode** for all production modules plus the two reusable test
  support boundaries; target ownership lives in `pyproject.toml` and CI runs
  the same argument-free `uv run mypy` command.

### Runtime conventions

- **Pydantic Settings owns `.env`.** `SettingsConfigDict.env_file` is the only
  production reader. A private interpolation map retains raw, source-provided
  declared and undeclared values without exposing undeclared keys as Settings
  fields or mutating `os.environ`; real process values win.
  Direct fixtures use `Settings(_env_file=None)` as a hermetic boundary; the
  strict ASGI support uses `Settings.model_validate(...)` so it bypasses all
  settings sources without a third-party constructor typing workaround.
- **Settings are pulled lazily** via `config.get_settings()`. Nothing imports a
  module-level settings instance, so importing `main` is credential-free and
  leaves process environment state untouched.
- **`runscript.sh`** is the canonical launcher; it cd's to the project
  root, activates the venv, prepends `.` to `PYTHONPATH` (so scripts in
  subdirs like `tests/` can still import `config`, `main`,
  etc.), and runs Python.
- **LLM providers implemented: Gemini and OpenAI-compatible**, each isolated in
  `llm/providers/`. The OpenAI-compatible client
  (`llm/providers/openai_compatible/`) speaks the
  OpenAI wire protocol, so it also drives any OpenAI-compatible server (local
  Ollama/vLLM) via `base_url`. The **default runtime** is a local
  OpenAI-compatible Qwen (see `config/models.yaml`). Anthropic key fields exist
  in `Settings` but its client is still stubbed. Architecture is
  provider-agnostic; adding a provider is "a file in `llm/providers/` + one
  `_PROVIDERS` entry" (see the LLM Layer section).
- **All runtime config lives under `config/`.** Three files: MCP server
  definitions, model registry, orchestrator system prompt. See
  [configuration.md](configuration.md).

### Static typing boundaries

Strict typing is the production default, not an aspirational check. Concrete
domain values and narrow capability protocols carry data between packages;
`Any` remains only where the value is genuinely unvalidated or SDK-shaped,
not as a substitute for application ownership. The principal examples are tool
JSON dictionaries, provider payloads, Pydantic's pre-validation input, and
`LoggerAdapter.process()`'s standard-library-defined dynamic message mapping.

Settings-facing protocols expose read-only properties, so frozen Pydantic
settings satisfy consumers without falsely promising mutation. Per-run code
accepts either a plain `logging.Logger` or the run-ID
`logging.LoggerAdapter[logging.Logger]`; this is one logger capability at a
time, not a collection. Reusable test fakes are checked against the same
production protocols so ASGI/runtime tests cannot silently drift to a looser
object shape.

---

## Directory Layout

```
hyphae/
├── main.py                   # FastAPI app, lifespan, wires everything
├── pyproject.toml            # Direct dependencies and tool configuration
├── uv.lock                   # Exact tested dependency graph
├── .env.example              # Template for .env (copy and fill in)
├── setup.sh                  # Frozen uv sync + one-time .env seeding
├── runscript.sh              # Standard launcher (venv + PYTHONPATH + python)
├── chat_client.py            # Optional interactive REPL client (uses HTTP)
├── harness_client.py         # Reference async Python client (orchestration-aware)
│
├── config/                   # Config package: Settings, schemas, typed loaders
│   ├── mcp_config.yaml       #   ...plus the runtime YAML/text config it loads
│   ├── models.yaml           # Routable model registry for the orchestrator
│   └── orchestrator_prompt.md  # Orchestrator's own system prompt
│
├── tests/                    # Hermetic pytest suite + marked live checks
│   ├── conftest.py               # Shared pytest fixtures
│   ├── fakes.py                  # Strict reusable LLM/tool fakes
│   ├── _app_support.py           # Strict typed ASGI runtime wiring
│   ├── test_*.py                  # Hermetic regression tests
│   ├── test_*_live.py             # Explicit configured-backend integrations
│   └── eval_agent.py              # Separate YAML-driven evaluation runner
│
├── mcp_runtime/              # NOT `mcp/` - shadow-free name for the SDK
│   ├── __init__.py
│   ├── catalog.py            # Catalog snapshots, routes, and schema validation
│   ├── client.py             # MCPClient connection factory and SDK boundary
│   ├── lease.py              # Turn-local lazy server leases
│   └── manager.py            # MCPManager catalog discovery, refresh, and health
│
├── llm/
│   ├── __init__.py
│   ├── schemas.py            # Provider-agnostic Message / *Block / AssistantMessage
│   ├── client.py             # LLMClient ABC + _PROVIDERS registry + factories (SDK-free)
│   ├── tool_prompt_protocol.py  # JSON tool protocol for prose-only models
│   └── providers/            # Provider modules/packages; imported lazily by the registry
│       ├── __init__.py
│       ├── gemini/           # Gemini client lifecycle + provider-specific codec
│       └── openai_compatible/  # OpenAICompatibleLLMClient (OpenAI wire protocol)
│
├── agent/
│   ├── __init__.py
│   ├── session.py            # Session, SessionStore ABC, InMemorySessionStore
│   ├── events.py             # TextEvent / ToolCallEvent / ToolResultEvent /
│   │                         #   UsageEvent / OrchestrationDecisionEvent /
│   │                         #   DoneEvent / ErrorEvent
│   ├── context.py            # assemble_context() seam: token estimator, budget,
│   │                         #   naive / compaction strategies (view-only)
│   ├── generation.py         # Provider-neutral attempts, retries, stream cleanup
│   ├── tracing.py            # Async bounded JSONL tracing + run_id log adapter
│   └── loop.py               # run_agent() - event/transcript/tool composition
│
├── orchestrator/             # Routes requests to model + tool subset + system
│   ├── __init__.py
│   ├── schemas.py            # OrchestrationProposal / decision/preferences values
│   ├── registry.py           # LLMRegistry (lazy LLMClient cache per model_id)
│   └── orchestrator.py       # Orchestrator.decide() -> OrchestrationDecision
│
├── api/
│   ├── __init__.py           # Combines the routers into one
│   ├── schemas.py            # HealthResponse and HTTP TokenUsage (Pydantic)
│   ├── dependencies.py       # Typed ApplicationRuntime + whole-runtime Depends()
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

The `config` package exposes runtime `Settings` and typed file-config loading.
`Settings` is accessed via `get_settings()` in production and never instantiated
at module import.

- `Settings.api_key_for_provider(provider)` returns the key for an
  arbitrary provider — used by each provider's builder so the orchestrator
  can spin up clients for multiple providers in one process. It checks the
  typed key fields first, then falls back to the conventional
  `<PROVIDER>_API_KEY` from the Settings-owned `.env`/process mapping, so a newly
  registered credentialed provider needs no change here (and key-less
  providers never call it).
- `Settings.interpolation_environment()` exposes only raw source-provided
  values, with the process environment overlaid last. Declared defaults are not
  synthesized, and undeclared dotenv values remain private to this mapping.

`load_mcp_config_from_settings(settings)` composes that mapping with the
configured MCP path. Direct callers can instead use
`load_mcp_config(path, environment=...)`, which defaults to `os.environ`.
Both return `MCPConfig`, whose discriminated server-transport union and
`enabled_servers()` method define the typed file boundary.

### MCP Layer

`mcp_runtime/` (not `mcp/`, to avoid shadowing the official SDK package).

**`MCPClient`** wraps one server session and transport. It connects, lists tools,
filters `disabled_tools`, calls raw tool names, and flattens MCP content blocks
to text for v1. MCP-declared errors remain ordinary tool results; exceptions at
the SDK call boundary become provider-neutral transport/protocol failures.

**`MCPManager`** discovers enabled-server catalogs in parallel at startup and
indexes healthy tools as `{server}__{tool}` using provider-neutral `ToolSpec`
records. Startup discovery closes each connection after listing tools. It retains
a typed state record and catalog for every enabled server, bounds each catalog
discovery or turn-lease connection, and permits partial startup: failed servers
become `unhealthy` while healthy servers and the HTTP application remain
available.
Its immutable status snapshot drives `/health`; `connected_servers` remains the
healthy-only compatibility view.

Each accepted request refreshes catalogs that are due before routing, including
unhealthy catalogs whose retry backoff has elapsed. The accepted turn captures
an immutable tool snapshot. An actual call opens a server connection lazily as
a turn-local lease, reused within that turn and closed with it. A transport or
protocol failure removes the server's advertised tools and marks it unhealthy;
the failed call is never replayed. A successful refresh or lease discovery
replaces the current catalog. There is no periodic recovery task.

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
- **`AssistantMessage(content, stop_reason, model, reasoning, raw_stop_reason)`**
  with helpers `text_blocks()`, `tool_uses()`, `to_message()`. `stop_reason`
  uses the canonical vocabulary below; `raw_stop_reason` retains the optional
  provider-native value for diagnostics. `reasoning` is never replayed through
  `to_message()`; the `/v1` streaming adapter may render its sanitized text
  through `delta.reasoning_content`.
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
   as a `ReasoningEvent`; session replay stays clean, while `/v1` streaming can
   expose the sanitized text through its configured reasoning channel.

**Canonical provider outcomes:**

| Outcome | Meaning |
|---|---|
| `end_turn` | Ordinary natural completion |
| `tool_use` | One or more real tool-use blocks were returned |
| `max_tokens` | Provider output limit reached |
| `empty` | No candidate or usable content from an otherwise normal response |
| `content_filter` | Provider safety/content policy blocked output |
| `refusal` | Provider returned an explicit refusal |
| `provider_error` | Missing, unknown, or abnormal provider termination |
| `incomplete_stream` | Visible answer text arrived without a terminal provider message |

Real tool-use blocks are authoritative over missing or inconsistent provider
finish reasons. Only `empty` is response-retryable; blocked, refused, truncated,
incomplete, and provider-error outcomes are never retried merely for lacking
text.

`llm/client.py` — the abstraction + the provider registry, and **nothing
SDK-specific** (importing it never pulls in a provider SDK):

- **`GenerationRequest`** is the single shallow-immutable provider-generation
  input. Its message/tool containers are borrowed and must not be mutated;
  retries, timeouts, deadlines, cancellation, tracing, and persistence remain
  with their execution owners.
- **`LLMClient`** ABC with two call modes:
  ```python
  async complete(request: GenerationRequest) -> AssistantMessage

  async stream(
      request: GenerationRequest,
  ) -> AsyncIterator[StreamChunk]

  async aclose() -> None
  ```
  `complete()` is the canonical completed-turn API and remains the path for
  structured-output calls such as orchestration. `stream()` is an optional
  token-streaming call mode for ordinary agent turns; the ABC fallback calls
  `complete()`, emits optional reasoning as a coarse `ReasoningDelta`, then
  emits each final text block as a coarse `TextDelta` and finishes with
  `StreamEnd(AssistantMessage)`. Native streaming providers override it.
  Streaming rejects a non-`None` `response_schema` explicitly before making a
  provider call.
  `aclose()` is an idempotent no-op for resource-free clients. Provider
  implementations override it to release their long-lived async SDK client.
- **`StreamChunk`** is provider-agnostic and SDK-free:
  `TextDelta(text=...)` carries visible assistant text,
  `ReasoningDelta(text=...)` carries sanitized optional reasoning without
  provider wrapper syntax, and `StreamEnd(message=...)` carries the fully
  assembled `AssistantMessage`. Provider adapters own all vendor fields and
  tag parsing. Streamed reasoning remains provisional until answer text begins:
  if an eligible retry follows reasoning-only output, the generation layer
  inserts an explicit interruption marker before the next attempt. The failed
  attempt is not added to model history. The agent loop streams deltas to
  callers immediately, then reuses the normal assistant/session/usage/tool tail
  once `StreamEnd` arrives.
- **`_PROVIDERS`** — the single source of truth mapping a provider name onto a
  builder. Each builder imports its provider module *lazily* (inside the
  function), so the ABC can be imported without dragging in any SDK, and each
  builder pulls what it needs from the duck-typed `settings` itself (an API
  key, a `base_url`, nothing).
- **`supported_providers()`** exposes the registry's keys; config validators
  (`ModelEntry.provider`) key off it so no other place enumerates providers.
- **`build_llm_client(settings)`** factory dispatches on
  `settings.llm.provider`. Used by `main.py` to build the legacy/default
  client at startup.
- **`build_llm_client_from_entry(entry, settings)`** is the multi-model
  variant. Takes a `ModelEntry` (from `models.yaml`), resolves its
  `ModelProfile`, and pulls the API key by `entry.provider` (not by
  `settings.llm.provider`), so one process can hold clients for multiple
  providers simultaneously. Used by `LLMRegistry`. If the profile declares
  `supports_native_tools: false`, this factory wraps the provider client in
  `PromptedToolLLMClient`; omitted or `true` profiles are not wrapped.

`llm/tool_prompt_protocol.py` — the prompted-tool dialect adapter for weak/prose
models:

- Renders the already-filtered tool list into compact system-prompt text:
  tool name, one-line description, and compressed JSON schema.
- Calls the wrapped provider with `request.tools=None`, so endpoints without
  native tool calling see only ordinary text.
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
structured output (Gemini, OpenAI) honor a Pydantic class carried by the request
and return JSON conforming to its schema. The orchestrator uses this
for its routing decision; the agent loop does not. It is completion-only;
streaming rejects it rather than silently ignoring it.

**Thinking level (`thinking_level`).** `"low" | "medium" | "high"` (or
`None` to leave the model default). The agent loop passes the orchestrator's
chosen level through on every iteration. Providers consult the selected
model's `ModelProfile`: `hint-param` profiles may map it to a request field
such as `reasoning_effort`, `think-tags` profiles do not add a request knob,
and `none` profiles log once that the knob is inert.

**Gemini specifics** (isolated in `llm/providers/gemini/`):

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
  (see `OrchestrationProposal` for the reference pattern: lenient at the
  parse boundary, then sanitized in code).
- **Reasoning extraction is conservative**: Gemini returns `reasoning=None`
  unless the SDK exposes thought content in a form the provider can identify
  without guessing. Gemini `thought_signature` still uses `provider_metadata`
  for round-trip state.

Gemini preserves the provider enum name in `raw_stop_reason` and applies this
terminal map:

| Provider state | Canonical outcome |
|---|---|
| Actual function-call part | `tool_use`, regardless of finish reason |
| `STOP` with visible content | `end_turn` |
| `STOP` without usable content | `empty` |
| `MAX_TOKENS` | `max_tokens` |
| `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`, `IMAGE_RECITATION` | `content_filter` |
| `FINISH_REASON_UNSPECIFIED`, `LANGUAGE`, `OTHER`, `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, `NO_IMAGE`, `IMAGE_OTHER`, missing, or unknown | `provider_error` |
| No candidates | `empty` |

**OpenAI-compatible specifics** (isolated in `llm/providers/openai_compatible/`):

- Per-model sampling from `ModelProfile` is copied into the chat-completions
  request when present.
- For `thinking: hint-param`, `thinking_level` is passed as
  `reasoning_effort`. Structured compatible fields (`reasoning_content`,
  `reasoning`, or `thinking`) and leading `<think>...</think>` content are
  normalized inside the provider into wrapper-free reasoning values and
  removed from visible content. For `thinking: none`, the request knob is
  logged as inert once and omitted.
- Malformed tool-call argument JSON is surfaced as `ToolUseBlock.parse_error`
  instead of disappearing into an empty argument object.
- Missing OpenAI tool IDs are minted once as `call_<uuid>` when the complete
  tool call is finalized, then reused through dispatch, results, checkpoints,
  and replay.
- Real OpenAI requests use `max_completion_tokens`, omit `max_tokens`, and omit
  `top_k`. Configured compatible endpoints use `max_tokens` and merge `top_k`
  into `extra_body` without replacing other extension values.

OpenAI preserves the original finish-reason string in `raw_stop_reason` and
applies this terminal map:

| Provider state | Canonical outcome |
|---|---|
| Actual native or legacy function-call block | `tool_use`, regardless of finish reason |
| `refusal` field without tool content | `refusal` |
| `stop` with visible content | `end_turn` |
| `stop` without usable content | `empty` |
| `length` | `max_tokens` |
| `content_filter` | `content_filter` |
| `tool_calls` or `function_call` without a parsed call | `provider_error` |
| Missing or unknown finish reason | `provider_error` |
| No choices | `empty` |

### Agent Layer

**`agent/session.py`**

- `Session(session_id, messages, created_at, updated_at, metadata)`.
  Auto-generated `sess_<16-hex>` IDs.
- Mutation API: `append_user(text)`, `append_assistant(response)`,
  `append_tool_results(results)`. The agent loop uses these rather than
  poking `.messages` directly — one seam for future invariant checks.
- `staged_copy()` preserves identity/timestamps, copies the metadata bag and
  message list, and shares immutable canonical messages. Native turns route and
  execute against this copy, never the object currently owned by the store.
  Persistent turns resolve the latest checkpoint by session ID before staging,
  so an older caller-held `Session` remains a safe identity handle rather than
  overwriting newer stored history.
- `last_assistant_tool_uses()` — convenience for "what tools did the
  model just ask me to run?"
- `SessionStore` ABC: `create(metadata)`, `get(session_id)`,
  `save(session)`. Async throughout, even though `InMemorySessionStore`
  doesn't need to be — keeps call sites unchanged when a durable backend
  lands. The harness keeps no durable copy (LibreChat re-feeds context),
  so the ABC is retained purely as that future seam.
- `InMemorySessionStore` is **bounded**: it evicts on idle TTL
  (`SESSION_TTL_SECONDS`) and on a max-size cap (`SESSION_CAPACITY`,
  oldest-updated first) so it can't grow without limit under concurrent
  load. Eviction is lazy (swept on `create()`), not a background task.
  Active/in-flight sessions stay "young" because `save()` bumps
  `updated_at` every turn, so they aren't evicted out from under a request.
- `SessionNotFoundError(KeyError)` — subclassing `KeyError` means
  existing `except KeyError` catches still work; callers wanting
  specificity have it.

**Concurrency model.** FastAPI interleaves async handlers on one event loop.
Distinct sessions use distinct `Session` objects; the single shared
`app.state.runtime` composition carries no per-user state. The only crossover
vector is two concurrent requests on the same `session_id`.

- `SessionGuard` (`agent/session.py`) closes that vector: `claim(session_id)`
  is an async context manager that registers the id as in-flight; a second
  concurrent `claim` of the same id raises `SessionBusyError`, which the
  route maps to **HTTP 409**. New sessions get a fresh id and never contend;
  only repeated client-supplied `X-Session-Id` values can collide.
- It is **lock-free**: on the single-threaded event loop the membership
  check and the add happen with no `await` between them, so they're atomic
  relative to other tasks. To switch to wait-semantics (queue instead of
  reject), swap the in-flight `set` for a `dict[str, asyncio.Lock]`.

**`agent/events.py`** — dataclass event types with `type: Literal[...]`
discriminators for JSON serialization at the API boundary:

| Event                          | Fields                                       | Emitted when                              |
|--------------------------------|----------------------------------------------|-------------------------------------------|
| `ReasoningEvent`               | `text`                                       | Provider extracted sanitized reasoning from a model response |
| `TextEvent`                    | `text`                                       | Model produced a text block               |
| `ToolCallEvent`                | `id, name, input`                            | Model decided to call a tool (pre-call)   |
| `ToolResultEvent`              | `id, name, content, is_error, latency_ms`    | Tool call completed (`latency_ms` = `call_tool` duration; `None` if stall-skipped) |
| `UsageEvent`                   | `input_tokens, output_tokens, total_tokens, thinking_tokens, cached_tokens, iteration, latency_ms` | One LLM completion finished (`latency_ms` = `complete()` duration incl. retries) |
| `OrchestrationDecisionEvent`   | `model_id, tools, system_prompt, fallback_used, thinking_level` | Emitted first for orchestrated turns; traced and exposed by native SSE |
| `DoneEvent`                    | `reason, iterations, total_tokens, input_tokens, output_tokens, thinking_tokens` | Loop finished |
| `ErrorEvent`                   | `message`                                    | Unrecoverable internal failure            |

`DoneEvent.reason` ∈ `{"end_turn", "max_iterations", "llm_error", "empty",
"truncated", "budget_exceeded", "deadline_exceeded", "no_progress",
"content_filter", "refusal", "provider_error", "incomplete_stream"}`. Guard
exits carry any answer text already accrued.

Tool execution failures become `ToolResultEvent(is_error=True)` so the model can
recover. `ErrorEvent` is only for failures the model never sees.

**`agent/loop.py`**

```python
async def run_agent(
    session: Session,
    llm: LLMClient,
    mcp: ToolRuntime,
    *,
    store: SessionStore | None = None,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    thinking_level: str | None = None,
    limits: RunLimits | None = None,
    context: RunContext | None = None,
    policy: ToolPolicy | None = None,
    stream: bool = False,
) -> AsyncIterator[Event]:
    ...
```

`RunLimits` is the immutable policy object built once from Settings by
`TurnRunner`; `RunContext` carries the one live run ID, absolute deadline,
logger, and trace sequence. Direct callers may omit both to receive defaults.
`agent/generation.py` owns provider-neutral timeout and retry policy while
providers classify transient errors. Buffered and streaming generation share
one private attempt-policy controller for attempt counts, deadline-aware
timeout bounds, transient eligibility, jittered exponential backoff, and
cancellable retry sleep. Their completion mechanics remain separate: buffered
generation awaits one response, while streaming applies the configured LLM
timeout to every incremental read. Emitted answer text commits the attempt and
prevents retry. Emitted reasoning remains provisional; if a retry is still
eligible, an interruption marker separates it from the next attempt.

**The caller appends the user message before invoking `run_agent`.**
`TurnRunner` does so only on its staged copy after routing; the loop owns
assistant turns and tool round-trips.

The **`tools` parameter** is the orchestration seam. When `None`, the
loop pulls the full MCP inventory (`mcp.get_tools_for_llm()`) — legacy
behavior, used by code paths that aren't orchestration-aware. When
provided, the loop uses the list verbatim — this is how the route
hands the orchestrator's filtered tool subset to the model.

The **`thinking_level` parameter** is the deliberation seam. It is carried
straight through on every generation request of the run; the loop never
inspects it. `None` (the default) leaves the model's own default. The route
supplies the orchestrator's chosen level here; see *Thinking level* under
the Orchestration Layer.

**The bounded-run guard values** on `RunLimits` follow the default-disabled
`None`/`<=0` convention. `TurnRunner` starts the wall-clock deadline immediately
after the session claim, so routing, model resolution, generation, tool calls,
and backoff all consume the same budget. See *Bounded & safe runs* below.

Per-iteration algorithm:

0. Check bounded-run guards before another LLM call.
1. Assemble the outgoing context view, then consume the normalized generation
   stream from `agent.generation`. Exhausted LLM failures yield `ErrorEvent` +
   `DoneEvent("llm_error")`.
2. Append the complete assistant response only to the staged session. A
   non-tool response is now a safe terminal checkpoint; a tool-use response is
   not publishable yet.
3. Publish a complete non-tool assistant response once. Visible answer-text
   deltas followed by ordinary provider exhaustion are synthesized as an
   `incomplete_stream` assistant response, saved with identical text, and
   terminated abnormally.
4. Yield a `UsageEvent` for this iteration's tokens. Provider-reported usage
   is used as-is; absent/all-zero usage is filled by the local estimator
   (`estimate_usage_tokens`, chars/4 heuristic over the outgoing view +
   system + response) so the token cap works against local servers that
   report zero usage. Never double-counted.
5. Stream sanitized reasoning as `ReasoningEvent` values when the provider
   supplies it. A reasoning-only retry inserts an interruption marker before
   the next attempt. Reasoning is not appended to session content; `/v1` may
   render it through `delta.reasoning_content`.
6. Yield a `TextEvent` for each non-empty text block.
7. Collect tool calls first; if none were requested, finish with the explicit
   canonical provider outcome or the applicable run-limit reason.
8. For each tool call, sequentially: emit `ToolCallEvent`, run repeat-call
   detection, convert provider parse errors or schema validation failures into
   teaching `is_error` results, enforce `ToolPolicy`, call MCP with timeout,
   clip the result, update failure counters, emit `ToolResultEvent`, and build
   the matching `ToolResultBlock`.
9. Append the complete matching result batch and publish the balanced tool
   protocol. Continue from a fresh `staged_copy()` so later mutation cannot
   alias the object just saved by `InMemorySessionStore`. Cancellation keeps
   completed results, marks an in-flight call as outcome-unknown, marks
   unstarted calls cancelled, best-effort publishes the balanced batch, and
   re-raises the original cancellation or generator-close signal. Once the
   consumer has cancelled or closed, synthetic results are checkpointed for
   safe replay but cannot be delivered on that terminated event stream.
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
| Token budget | `max_run_tokens` | Before each iteration and immediately after each reported `CompletionUsage.total_tokens` update | `budget_exceeded` |
| Wall clock | `max_run_seconds` | Starts after the session claim; covers routing, retries, generation, tool calls, and backoff | `deadline_exceeded` |
| No-progress abort | `abort_after_consecutive_tool_failures` | After each tool result, against the consecutive-failure counter | `no_progress` |

All three default to `0` (disabled) in `Settings`, matching the repo's
established pattern for new safety knobs (e.g. `trace_enabled`) — installing
the harness doesn't change behavior until an operator opts in.

**Token cap and zero-usage providers:** when a provider reports absent or
all-zero `CompletionUsage` (common on local OpenAI-compatible servers), the loop fills
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
abort if one is configured. Concrete JSON Schema validators are selected,
schema-checked, and compiled once from the turn's immutable tool inventory,
then reused for every matching call in that run. A malformed advertised schema
warns once during setup and is permissive for that run; validator state never
mutates the snapshot or MCP inventory.

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
    default: bool = False  # exactly one default when multiple entries exist

    def to_profile(self) -> ModelProfile: ...

class ModelsConfig(BaseModel):
    models: Mapping[str, ModelEntry]  # immutable after loading

class OrchestrationProposal(BaseModel):
    selected_model_id: str
    selected_tools: list[str]              # namespaced tool names
    generated_system_prompt: str
    thinking_level: Literal["low","medium","high"] = "medium"

@dataclass
class OrchestrationDecision:
    result: OrchestrationProposal
    fallback_used: bool = False
    fallback_reason: str | None = None
```

`OrchestrationProposal` is the LLM's structured output. `OrchestrationDecision`
adds fallback metadata for in-process callers.

**`orchestrator/config.py`** — `load_models_config(path)` and
`load_orchestrator_prompt(path)`. Mirrors the loader pattern in
`config.load_mcp_config`.

**`orchestrator/registry.py`** — `LLMRegistry`:

- Lazy-builds an `LLMClient` per `model_id` on first use, caches it.
- `registry.get(model_id)` returns a built client (raises `KeyError`
  on unknown id).
- `registry.get_or_default(model_id)` returns `(resolved_id, client)`;
  silently falls back to default on unknown id.
- `registry.describe_for_prompt()` formats the model inventory as the
  orchestrator-prompt block ("AVAILABLE MODELS").
- `registry.aclose()` snapshots and clears the lazy cache, deduplicates concrete
  client identities, and closes them best-effort. The lifespan includes the
  legacy/default client in that identity set so an alias is never closed twice.
- Concurrency: clients are built on first use and stashed in a dict
  with no lock. Client construction is idempotent, so a rare double-build
  wastes a few cycles but cannot produce wrong behavior.

**`orchestrator/orchestrator.py`** —
`Orchestrator.decide(user_message, tools, preferences=None, history=None, ...)`:

1. Builds the prompt from the immutable per-turn tool snapshot: orchestrator system instruction (from the
   `orchestrator_prompt.md` file) + AVAILABLE MODELS block + AVAILABLE
   TOOLS block + optional PREFERRED TOOLS and CONVERSATION SO FAR
   block (see *Context-aware routing* below) + USER MESSAGE.
2. Calls the orchestrator's own LLM client with a `GenerationRequest` carrying
   `response_schema=OrchestrationProposal`. Tools are **not** exposed —
   the orchestrator must decide, not act.
3. Parses the JSON response into `OrchestrationProposal`. Strips any
   stray markdown fences defensively. `thinking_level` is coerced
   leniently (a stray/unknown value clamps to `"medium"`) so one odd
   field can't sink an otherwise-valid decision.
4. Sanitizes the result against that same snapshot: drops unknown tool names;
   rewrites an unknown `selected_model_id` to the registry default.
   Sanitization is **not** counted as fallback — the orchestrator made
   a real decision, we just trimmed it.
5. Returns `OrchestrationDecision(result=..., fallback_used=False)`.

**Thinking level.** The route passes `result.thinking_level` to `run_agent()`,
which forwards it on every `GenerationRequest`. The selected client's
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
    result=OrchestrationProposal(
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

**The system-prompt override.** `TurnRunner`, not the orchestrator, implements
precedence: an OpenAI `system` message override wins over
`result.generated_system_prompt`. Orchestration still runs to pick model and
tools; native `/chat` has no per-call system field.

**MCP tool preferences.** `ToolPreferences` remains an intentional direct
`Orchestrator.decide()` seam. Explicit in-process callers can supply priority
hints; `_sanitize()` unions valid preferred tools into the selection. HTTP
routes and `TurnRunner` carry no preference value because neither HTTP surface
accepts one.

**Disabled mode.** If orchestration is disabled or its config cannot load,
`TurnRunner` uses the default LLM, all MCP tools, and an optional `/v1` system
override directly.

### HTTP Layer

**`main.py`** — FastAPI entry point with a `lifespan` context manager.

Startup order:
1. `get_settings()` — Pydantic reads process variables and the project `.env`
   without mutating global environment state.
2. `load_mcp_config_from_settings(settings)` using the precedence-resolved,
   raw Settings mapping.
3. `MCPManager(mcp_config, connect_timeout_seconds=...).startup()` — discovers
   all enabled-server catalogs in parallel without retaining connections. Each
   transport-open + initialize + list operation, at startup, request-driven
   refresh, or turn-lease discovery, is bounded as one unit
   unless the setting is `<= 0`; startup failures degrade MCP health without
   aborting application startup.
4. Build the tool policy, `InMemorySessionStore`, and `SessionGuard`.
5. `_try_build_orchestration(settings)` — returns registry/orchestrator or
   `(None, None)` on optional-layer failure. The legacy default LLM is built
   only for that fallback path; successful orchestration owns its clients
   exclusively through the registry.
6. Start the optional tracer. Routine startup failure degrades to `None`.
7. Construct one explicit `OrchestratedRouting` or `UnorchestratedRouting`, then
   atomically publish `ApplicationRuntime` as `app.state.runtime`. No individual
   runtime component is published separately.
8. Emit one consolidated "harness ready" INFO log line.

Shutdown attempts LLM, MCP, and tracer cleanup independently. The default LLM
and every constructed registry client are deduplicated by object identity and
closed once; never-constructed lazy clients have no resources to release.
Provider generators own their SDK streams. The generation layer owns each
provider iterator, and the agent loop owns only the normalized generation
adapter. Cleanup failures are logged without replacing the request exception or
cancellation that initiated shutdown. Tracer cleanup first stops acceptance,
then drains healthy accepted records through the background writer; its final
counters are logged before active cancellation is re-raised.

**`api/dependencies.py`** — owns the typed process composition and the one
dynamic framework seam. `get_application_runtime()` reads and validates
`app.state.runtime`; routes and `require_api_key` derive everything else from
that value. `ApplicationMCP` is the consumer-owned structural boundary for MCP
turn ownership plus the health/catalog views HTTP actually needs.
`require_api_key` remains the optional route-layer auth gate for `/chat`,
`/chat/stream`, and `/v1/*`; `/health` stays open.

**`api/routes.py`** — request flow:

1. **Read the prompt** from the plain-text request body (empty → 400).
   **Resolve/create the session first** (the `X-Session-Id` header). The session
   is resolved *before* routing so the orchestrator can route a follow-up turn
   with the conversation in view. The new prompt is **not** appended yet — it's
   passed to the orchestrator separately, so `session.messages` at routing time
   is the prior history only.
2. **`runner.open(...)` / `runner.run(...)`** — every renderer reaches the
   shared core through `TurnRunner`. `open()` claims the session before consuming
   current history or inventory, creates the run context/deadline, stages the latest
   safe checkpoint, snapshots tools once, and resolves LLM, tools, system,
   thinking level, model ID, and optional orchestration metadata. `run()` is the
   buffered collector over the same owned event iterator.
3. **Per-request log line** at INFO level summarizing the routing decision
   (model, tool count, thinking level, fallback flag).
4. On the staged session, append the prompt and iterate `run_agent(...)`
   with the orchestrator's selections — including `thinking_level` — collecting
   `TextEvent` text into the answer and reading the final `DoneEvent` for the
   done-reason and token usage.
5. Return `PlainTextResponse(answer, headers={X-Session-Id, X-Done-Reason})`.

The full event stream is no longer serialized into the response (the body is
plain text); routing/usage detail lives in the logs and the JSONL trace.

The `RunContext` minted by `TurnRunner.open()` tags logs, native SSE, and trace
records with the same **`run_id`**. Tracing is best-effort and mirrors the event
stream. `emit()` serializes and submits synchronously to a 4096-record bounded
queue but performs no file I/O and never waits. One writer task preserves FIFO
order and performs append-plus-flush batches off the event-loop thread: at most
100 records, or a partial batch after 250 ms from its first record. A full queue
drops the newest submission, preserving older queued records. The first sink
write failure permanently disables tracing and accounts for the failed batch
and backlog as dropped; see `agent/tracing.py` and operations docs.

---

## Architectural Principles

1. **Separate the agent loop from provider and transport ownership.** The LLM
   client only sends canonical requests and returns canonical responses. The
   loop consumes the neutral `ToolRuntime`; it does not own MCP connections or
   provider SDK state.
2. **MCP catalogs are process-owned; transport leases are turn-owned.** Startup
   discovers immutable catalogs without retaining connections. Accepted turns
   refresh due catalogs, and actual dispatch lazily opens task-owned connections
   that close with that turn.
3. **One provider-neutral bridge between model and tools.** `agent/loop.py`
   coordinates `LLMClient` and `ToolRuntime`. Provider SDKs and MCP transports
   remain behind those contracts and do not know about each other.
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
   per request, receives tool descriptions but exposes no callable tools to its
   own LLM (it must decide, not act), and produces
   a structured decision that's fed into `run_agent()` as concrete
   arguments. The loop is unchanged from the pre-orchestrator design.
7. **A minimal, explicit boundary.** `/chat` is a plain-text dumb pipe — the
   body *is* the prompt, so there is nothing to over-accept; an empty body is
   the only rejection (400). Optional behavior travels as named headers
   (`X-Session-Id`), never as free-form fields. The `/v1` adapter accepts the
   standard OpenAI envelope, validates model identity and message ordering at
   its boundary, and tolerates unsupported standard fields without forwarding
   them. The surface is small and explicit.
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
2. **Settings are pulled lazily**, not snapshotted at import. Pydantic Settings
   is the sole production `.env` owner; imports remain credential-free and
   `Settings(_env_file=None)` keeps hermetic tests isolated.
3. **`mcp_runtime/` not `mcp/`** to avoid shadowing the SDK's `mcp` package.
4. **`__` is the tool-namespacing separator.** Config validation enforces
   that server names can't contain it.
5. **Graceful, bounded MCP lifecycle**: one bad server doesn't kill the harness.
   Every enabled server retains an explicit state and catalog, only healthy
   current tools are advertised, accepted requests refresh due catalogs before
   routing, and actual tool calls use bounded turn-local lazy leases. A failed
   call is never replayed.
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
13. **The loop publishes only safe checkpoints.** A complete non-tool response
    or a balanced assistant-tool-call/result batch may be saved; prompt-only and
    unmatched tool state never becomes visible. Cancellation preserves the last
    safe checkpoint and propagates.
14. **Tool execution is sequential, not parallel.** Some MCP tools have
    side effects; parallel makes failure modes harder to reason about.
    Wrap in `asyncio.gather` later if needed.
15. **Tool failures stay in-conversation; LLM failures kill the loop.**
    A `mcp.call_tool` error becomes `ToolResultBlock(is_error=True)` and
    goes back to the model. An `llm.complete` exception yields
    `ErrorEvent` + `DoneEvent("llm_error")` and stops.
16. **Caller appends the user message before invoking `run_agent`.**
    `TurnRunner` appends it to a staged session only after routing; the loop owns
    assistant turns and tool round-trips.
17. **System prompt is a per-call argument, not stored on the Session.**
    Session = conversation state; system prompt = call-time configuration.
18. **Routes inject one complete runtime, not individual state fields.** Only
    `get_application_runtime()` reads `app.state`; native/OpenAI/auth/health
    paths derive their owners from the same immutable composition. Tests
    override the whole runtime, preventing split stores or partial routing.
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
    selects model/tools/system, the iteration cap is config-driven, and `/v1`
    system messages provide the per-call override. `ToolPreferences` remains
    only on the intentional direct `Orchestrator.decide()` seam; route and
    runner contracts do not carry it.
22. **`OrchestrationProposal` is lenient (no `extra="forbid"`).** Gemini's
    `response_schema` dialect doesn't accept `additionalProperties: false`,
    which Pydantic emits when a model is strict. We sanitize the result
    in code instead. This is the reference pattern for any Pydantic
    class you intend to pass as a structured-output schema.
23. **Sanitization is not fallback.** The orchestrator's `_sanitize()`
    method drops unknown tool names and corrects unknown model_ids
    without raising the `fallback_used` flag. Fallback is reserved for
    cases where the orchestrator's LLM call itself failed.
24. **The orchestrator gets no tools.** It must decide, not act. Its
    `GenerationRequest` carries `tools=None` to enforce this.
25. **Orchestration is optional and degradable.** Missing config files,
    failed LLM calls, or bad outputs all degrade to the legacy
    (pre-orchestrator) behavior. Startup constructs one explicit routing union:
    `OrchestratedRouting` contains both orchestrator and registry;
    `UnorchestratedRouting` contains the fixed client/model and optional model
    inventory. `TurnRunner` never reconstructs that choice from loose optional
    state fields.
26. **`TurnRequest.system_override` overrides `generated_system_prompt`.** The
    orchestrator still runs to pick model and tools, but combined `/v1` system
    messages win for downstream generation. Native `/chat` supplies no override.
27. **One LLM provider's keys aren't all of them.** Each provider's builder
    pulls its own credentials via `Settings.api_key_for_provider(...)`, so one
    process can hold clients for any provider in `models.yaml`. The lookup
    falls back to `<PROVIDER>_API_KEY` from the Settings-owned environment
    mapping for providers without a typed field.
28. **Thinking level is a routing dimension, not just a model choice.** The
    orchestrator emits `thinking_level` (`low`/`medium`/`high`) alongside
    the model, and the loop carries it on every `GenerationRequest`. This makes
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
