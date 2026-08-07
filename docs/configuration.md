# hyphae — Configuration

Everything the harness reads at startup: environment variables and four files
under `hyphae/config/` (MCP servers, the model registry, the orchestrator control
prompt, and the trusted downstream agent prompt).

> See also: the [documentation index](README.md), [architecture.md](architecture.md)
> (how config flows through the subsystems), [operations.md](operations.md)
> (running, tuning, disabling orchestration).

## Environment Variables

Read lazily into `Settings` via `pydantic-settings`. `SettingsConfigDict.env_file`
is the sole production reader for the project-root `.env`; importing `main`
does not load credentials or mutate `os.environ`. Copy `.env.example` to `.env`
(the `scripts/setup.sh` script does this for you) and fill in your values. Real process
environment variables take precedence over `.env`, so the same file works for
local dev, containers, and CI. Tests that need hermetic defaults construct
`Settings(_env_file=None)`.

LLM values are grouped in application code under the typed `settings.llm`
object (`settings.llm.provider`, `settings.llm.model`,
`settings.llm.max_tokens`, and the retry/timeout controls). Environment
variables retain their existing flat `LLM_*` names. Sampling values such as
`temperature` remain per-model settings in `models.yaml`; there is no global
`settings.llm.temperature` override. The provider and model select the direct
client used when orchestration is disabled or unavailable; `max_tokens` also
supplies the default for model rows that omit their own output cap.

Relative application paths resolve from the repository root, not the process
working directory. This applies to MCP/models config, the orchestrator prompt,
the agent prompt, and trace output whether values come from `.env` or the
process environment.

<!-- BEGIN GENERATED SETTINGS -->

| Environment variable | Default | Description | Compatibility aliases |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | *unset* | Reserved for a future Anthropic provider; no Anthropic provider is implemented. | — |
| `GEMINI_API_KEY` | *unset* | Gemini API key | — |
| `OPENAI_API_KEY` | *unset* | OpenAI API key | — |
| `OPENAI_COMPAT_BASE_URL` | *unset* | Base URL for the OpenAI-compatible endpoint (e.g. a local Ollama server's /v1). Empty targets real OpenAI. | `OPENAI_PROVIDER_BASE_URL`, `OPENAI_BASE_URL` |
| `LLM_PROVIDER` | `gemini` | LLM provider name | — |
| `LLM_MODEL` | `gemini-3-flash-preview` | Provider-native model name to use | `LLM_MODEL_NAME` |
| `LLM_MAX_TOKENS` | `4096` | Default output-token cap for one LLM completion. | — |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-attempt cap on a single llm.complete() call. <=0 disables. | — |
| `LLM_MAX_RETRIES` | `3` | Retries (not attempts) on transient LLM failures / empty candidates. 0 disables retrying. | — |
| `LLM_RETRY_BASE_DELAY` | `0.5` | Base seconds for jittered exponential backoff between LLM retries. | — |
| `TOOL_TIMEOUT_SECONDS` | `60` | Cap on a single mcp.call_tool() call. <=0 disables. | — |
| `MCP_CONNECT_TIMEOUT_SECONDS` | `30` | Cap on complete MCP startup or recovery connection setup, including transport, initialization, and tool discovery. <=0 disables. | — |
| `MCP_CATALOG_TTL_SECONDS` | `300` | Age after which an accepted request refreshes an MCP catalog before routing. <=0 disables age-driven refresh; startup and lease discovery still run. | — |
| `TOOL_RESULT_MAX_CHARS` | `20000` | Clip threshold for a single flattened tool result before it enters session history. <=0 disables clipping. | — |
| `OPENAI_COMPAT_TOOL_ACTIVITY_MAX_CHARS` | `2000` | Presentation threshold for displayed arguments or results in one /v1 tool-activity payload. Distinct from tool_result_max_chars, which clips session history. | `OPENAI_TOOL_BLOCK_MAX_CHARS` |
| `OPENAI_COMPAT_TOOL_ACTIVITY_MODE` | `reasoning` | OpenAI-compatible streaming activity: 'reasoning' emits model reasoning and compact tool progress through delta.reasoning_content; 'reasoning_full' adds bounded tool arguments and results; 'hidden' omits that optional channel. | `OPENAI_COMPAT_TOOL_ACTIVITY` |
| `RUN_MAX_TOKENS` | `0` | Hard ceiling on cumulative total_tokens for one run; ends the run budget_exceeded with the partial answer. <=0 disables. When a provider reports absent/all-zero usage, the local token estimator (hyphae/agent/context.py) fills in, so the cap works against local OpenAI-compatible servers too. | `MAX_RUN_TOKENS` |
| `RUN_MAX_SECONDS` | `0` | Hard wall-clock ceiling on one accepted turn, measured immediately after the session claim and enforced across routing, retries, LLM/tool calls, and backoff; ends the run deadline_exceeded with the partial answer. <=0 disables. | `MAX_RUN_SECONDS` |
| `ABORT_AFTER_CONSECUTIVE_TOOL_FAILURES` | `0` | Abort the run no_progress after this many tool-call failures in a row (a success resets the count). Should be greater than the fixed at-3 consecutive-failure nudge so the model gets a chance to recover first. <=0 disables. | — |
| `CONTEXT_STRATEGY` | `naive` | Context assembly strategy: 'naive' rejects over-budget context; 'compaction' summarizes over-budget middle history before submission. | — |
| `CONTEXT_DEFAULT_WINDOW_TOKENS` | `32768` | Assumed context window (tokens) for models whose models.yaml entry has no context_window, and for unorchestrated mode. | — |
| `CONTEXT_SAFETY_MARGIN_TOKENS` | `1024` | Headroom subtracted from the context window (with max output tokens) when computing the input budget; absorbs estimator error. | — |
| `CONTEXT_RECENT_MESSAGES` | `6` | Recent protocol-safe units (a user turn, a no-tool assistant turn, or an assistant tool call plus its results) kept verbatim under compaction. Shrinks automatically if the tail alone overflows. | — |
| `CONTEXT_SUMMARY_MAX_TOKENS` | `512` | Output cap for the one-call compaction summarizer. | — |
| `SYSTEM_PROMPT_TIME_ENABLED` | `true` | Append one UTC turn-start timestamp to the downstream system prompt. | — |
| `MCP_CONFIG_PATH` | `hyphae/config/mcp_config.yaml` | Path to the MCP server and tool-policy YAML configuration. | — |
| `LOOP_MAX_ITERATIONS` | `10` | Maximum agent-loop iterations for one accepted run. | `MAX_LOOP_ITERATIONS` |
| `LOG_LEVEL` | `INFO` | Python logging level. | — |
| `HYPHAE_API_KEY` | *unset* | Optional API key protecting /chat, /chat/stream, /v1/*. Empty disables auth; set enforces it. Accepts X-API-Key or Bearer. | `HARNESS_API_KEY` |
| `TRACE_ENABLED` | `false` | Persist the loop's event stream as a JSONL trace. | — |
| `TRACE_JSONL_PATH` | `traces/harness.jsonl` | Active bounded JSONL trace file. Parent dirs are created. | `TRACE_PATH` |
| `SESSION_TTL_SECONDS` | `3600` | Idle TTL (seconds) for lazy in-memory eviction; active claims are protected. <=0 disables TTL eviction. | — |
| `SESSION_CAPACITY` | `1000` | Max in-memory sessions; admission evicts oldest-updated unclaimed sessions first. <=0 disables the count cap. | `SESSION_MAX_COUNT` |
| `SESSION_HISTORY_MAX_CHARS` | `256000` | Maximum canonical retained transcript characters per session. | — |
| `ORCHESTRATION_ENABLED` | `true` | Master toggle for the orchestration layer. | — |
| `MODELS_CONFIG_PATH` | `hyphae/config/models.yaml` | Path to the model registry YAML consumed by the orchestrator. | — |
| `ORCHESTRATOR_PROMPT_PATH` | `hyphae/config/orchestrator_prompt.md` | Path to the orchestrator's system prompt file (markdown or plain text). | — |
| `AGENT_PROMPT_PATH` | `hyphae/config/agent_prompt.md` | Path to the trusted downstream agent system prompt file. | — |
| `ORCHESTRATOR_MODEL_ID` | *unset* | Override which model_id the orchestrator itself uses to make routing decisions. Empty = use the default entry from models.yaml. | — |

<!-- END GENERATED SETTINGS -->

Compatibility aliases are listed with their canonical setting. When both forms
are set, the canonical name wins; new deployments should use it.

MCP YAML interpolation may also use source-provided environment variables that
are not declared application settings. The checked-in executable sample uses:

| Interpolation variable | Default | Purpose |
|---|---|---|
| `OPEN_WEBSEARCH_URL` | _(none)_ | Endpoint for the `open-websearch` MCP server, interpolated into `mcp_config.yaml` |

`.env.example` also includes `GOOGLE_TOOLBOX_URL` as an optional endpoint
example, but the checked-in `mcp_config.yaml` does not reference it.

Trace buffering is fixed internal policy, not environment configuration: the
queue holds 4096 records, batches contain at most 100 records, partial batches
flush after 250 ms, and overflow warnings repeat at most once per 60 seconds.
There are deliberately no queue-size, batch-size, flush-interval, warning-rate,
or overflow-policy settings.

## MCP Config YAML — `hyphae/config/mcp_config.yaml`

```yaml
mcpServers:
  open-websearch:
    transport: streamable-http
    url: ${OPEN_WEBSEARCH_URL}
    disabled: false
    disabled_tools:
      - fetchJuejinArticle
      - fetchCsdnArticle
      - fetchLinuxDoArticle

  # Example stdio server (supported but not currently used):
  # github:
  #   transport: stdio
  #   command: uvx
  #   args: ["mcp-server-github"]
  #   env:
  #     GITHUB_TOKEN: ${GITHUB_TOKEN}

# Optional top-level block: dispatch-time tool policy (what may EXECUTE).
tool_policy:
  mode: allow_all
  allow: []
```

**Schema notes:**

- `mcpServers` top-level key is camelCase for cross-tool compatibility
  (configs are pasteable to/from Claude Desktop, Cursor, Roo). Internal
  fields are snake_case.
- `transport` is `streamable-http`, `sse`, or `stdio`. Each transport
  validates its own required fields.
- `disabled: true` skips the server entirely at startup.
- `disabled_tools: [...]` hides specific tools from the LLM (defense in
  depth: filtered at list time and at call time). This is a **visibility**
  control (what the model sees) — separate from `tool_policy` below.
- `tool_policy` (optional, top-level) is the **enforcement** control. Denied
  tools are projected out before the router or downstream model sees the turn
  snapshot, and every requested call is checked again at dispatch after
  argument validation and before the MCP call. `mode: allow_all` (default; or
  omit the block) runs every tool.
  `mode: allow_list` runs only tools whose namespaced name matches an `allow`
  fnmatch pattern; anything else is denied with a teaching `is_error` result the
  model can adapt to (a denial is feedback, not a run failure, and repeated
  denials count toward no-progress handling). An `allow_list` with an empty
  `allow` **fails loudly at startup** — it would otherwise silently deny every
  tool.
  Server/tool visibility (`disabled_tools`), router narrowing, and execution
  enforcement (`tool_policy`) remain distinct layers; policy is the final
  authority even if another component presents a stale or forged tool call.
- String values support `${ENV_VAR}` interpolation from raw, source-provided
  Settings values plus the process environment. Undeclared `.env` names are
  retained privately for this purpose, real process values win, declared
  defaults are not synthesized, and missing names raise immediately rather
  than producing empty strings.
- Server names cannot contain `__` (reserved for tool namespacing) and
  must be alphanumeric (dashes/underscores allowed).
- HTTP/SSE URLs require an `http` or `https` scheme and a nonblank host;
  internal names such as `mcp.internal` or `service.local` remain valid.

## Model Registry — `hyphae/config/models.yaml`

```yaml
models:
  gpt-oss-20b:
    provider: openai_compatible
    model: gpt-oss-20b
    description: >
      General-purpose model for reasoning, code generation, and tool calling.
    supports_native_tools: true

  # The checked-in sample also includes glm-4.7-flash, qwen3.6-35b-a3b,
  # qwen3.5-9b, ornith-1.0-9b, and ornith-1.0-35b (the default).
```

The checked-in sample has six active OpenAI-compatible entries:
`gpt-oss-20b`, `glm-4.7-flash`, `qwen3.6-35b-a3b`, `qwen3.5-9b`,
`ornith-1.0-9b`, and default `ornith-1.0-35b`. The orchestrator routes among
them, so `selected_model_id` is a real decision, not a no-op.

**Schema notes:**

- `provider` must be registered in `hyphae/llm/client.py`'s `_PROVIDERS` —
  validated against `supported_providers()` at load time, so an unknown
  provider is rejected up front rather than at first request. Implemented
  today: `gemini` (`hyphae/llm/providers/gemini/`) and `openai_compatible`
  (`hyphae/llm/providers/openai_compatible/`). The compatibility `openai` provider ID
  remains accepted for existing configuration. Set `OPENAI_COMPAT_BASE_URL`
  to point the adapter at a local Ollama (or any compatible server); an empty
  value targets real OpenAI. For Ollama, `model` must match an `ollama list`
  tag and be tools-capable if you need tool calling.
- `model` is the provider-specific model identifier.
- `description` is what the **orchestrator LLM** sees when picking a
  model. Write for an LLM audience: terse, capability-focused, plain
  English.
- `max_tokens` is optional; falls back to `Settings.llm.max_tokens`.
- `context_window` is optional; the model's total context window in tokens,
  used by the agent loop's context budget (`window − max output − safety
  margin`). Falls back to `CONTEXT_DEFAULT_WINDOW_TOKENS`. Provider-agnostic
  (a plain size, no vendor branching) — set it accurately for small local
  models, where overflow is a hard failure.
- `supports_native_tools` is optional and defaults to `true`. It declares
  whether the served endpoint supports native tool/function calling. When set
  to `false`, `build_llm_client_from_entry()` wraps the provider client with
  the prompted-tool adapter: tools are rendered into the system prompt, the
  provider receives a request with `tools=None`, and one JSON action from prose is
  normalized back into a normal `ToolUseBlock`. Omitted or `true` keeps native
  tool behavior unchanged.
- `thinking` is optional and defaults to `none`. Values:
  `none` means the provider has no request-time thinking knob,
  `hint-param` means the provider can pass the orchestrator's
  `thinking_level` through as a request hint such as `reasoning_effort`,
  and `think-tags` means the model may self-emit a leading
  `<think>...</think>` block that the provider extracts as sanitized
  reasoning. It is not replayed into model history; `/v1` streaming can display
  it through `delta.reasoning_content`.
- `sampling` is optional. Supported fields are `temperature` (`0.0`-`2.0`),
  `top_p` (`0.0`-`1.0`), and `top_k` (`>= 1`). Omit the block, or omit an
  individual field, to leave the provider/server default in force. Sampling is
  per model entry, not an environment variable. Gemini receives its native
  `top_k`; a configured OpenAI-compatible endpoint receives `top_k` under
  `extra_body`; real OpenAI omits `top_k` because it is not a supported direct
  chat-completions parameter.
- `default: true` on **exactly one** entry when multiple models are configured.
  A single entry remains its own implicit default for compatibility. The default is used
  by the orchestrator itself (unless `ORCHESTRATOR_MODEL_ID` overrides)
  and is the safe fallback when orchestration fails.
  The orchestrator control model must not be prompted-only: if the resolved
  orchestrator row has `supports_native_tools: false`, startup logs a warning
  and publishes a fixed ready registry-default route. Prompted-tool models can
  still be selected for downstream agent turns when an eligible control model
  is active.
- Model IDs must be `[A-Za-z0-9_.\-]+` (clean keys for logging and
  routing).

Unknown fields, invalid enum values, and out-of-range sampling values fail
startup during `models.yaml` parsing, with the model id and field path in the
validation error.

## Orchestrator Prompt — `hyphae/config/orchestrator_prompt.md`

A markdown file (text is fine too — extension doesn't matter to the
loader) that's the system instruction for the orchestrator's own LLM
call. The current prompt teaches it to:

1. Read the AVAILABLE MODELS, AVAILABLE TOOLS, optional CONVERSATION SO FAR,
   and USER MESSAGE inputs (which the orchestrator injects at the start of
   the user turn).
2. Pick the cheapest/fastest model that fits.
3. Pick the smallest tool subset that fits.
4. Pick a `thinking_level` (`low`/`medium`/`high`) sizing deliberation to
   the request.
5. Return only `selected_model_id`, `selected_tools`, and `thinking_level`.

The orchestrator is selection-only. Its output cannot author downstream system
instructions. Modify this file to change routing guidance, not agent behavior.

## Agent Prompt — `hyphae/config/agent_prompt.md`

This file is the trusted application-owned system prompt for downstream agent
turns. An authorized `TurnRequest.system_override` (used by the `/v1` adapter
for caller-supplied system messages) replaces it for that ephemeral turn.
Router output, user text, conversation history, and tool metadata cannot modify
this trusted behavior prompt.

When `SYSTEM_PROMPT_TIME_ENABLED=true`, the application appends one
minute-precision UTC timestamp after the effective downstream prompt. It is
captured at turn start and stays unchanged across every model iteration in that
turn. The timestamp also applies to caller overrides and unorchestrated turns;
setting the toggle to `false` preserves the effective prompt verbatim.

Change this file when the default agent behavior itself should change. Prompt
changes can affect model behavior even when no Python interface changes, so
review and test them as behavior changes.
