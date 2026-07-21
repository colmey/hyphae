# hyphae — Configuration

Everything the harness reads at startup: environment variables and the
three files under `config/` (MCP servers, the model registry, and the
orchestrator's system prompt).

> See also: [README.md](README.md) (overview), [architecture.md](architecture.md)
> (how config flows through the subsystems), [operations.md](operations.md)
> (running, tuning, disabling orchestration).

## Environment Variables

Read lazily into `Settings` via `pydantic-settings`. `SettingsConfigDict.env_file`
is the sole production reader for the project-root `.env`; importing `main`
does not load credentials or mutate `os.environ`. Copy `.env.example` to `.env`
(the `setup.sh` script does this for you) and fill in your values. Real process
environment variables take precedence over `.env`, so the same file works for
local dev, containers, and CI. Tests that need hermetic defaults construct
`Settings(_env_file=None)`.

| Variable                       | Default                       | Purpose                                            |
|--------------------------------|-------------------------------|----------------------------------------------------|
| `LLM_PROVIDER`                 | `gemini`                      | Any provider registered in `llm/client.py`'s `_PROVIDERS` (currently `gemini`, `openai`); validated at build time |
| `LLM_MODEL`                    | `gemini-3-flash-preview`      | Model identifier (legacy default; orchestrator overrides per request) |
| `LLM_MAX_TOKENS`               | `4096`                        | Default max tokens per completion                  |
| `ANTHROPIC_API_KEY`            | `""`                          | Anthropic key (if used by any model in `models.yaml`) |
| `GEMINI_API_KEY`               | `""`                          | Gemini key (if used by any model in `models.yaml`)    |
| `OPENAI_API_KEY`               | `""`                          | OpenAI key (if used by any model in `models.yaml`); also required by the SDK for OpenAI-compatible servers (e.g. Ollama) even when the server ignores it |
| `OPENAI_BASE_URL`              | `""`                          | Base URL for the OpenAI-compatible endpoint (e.g. `http://localhost:11434/v1` for local Ollama). Empty targets real OpenAI |
| `MCP_CONFIG_PATH`              | `config/mcp_config.yaml`      | Path to MCP server config                          |
| `MCP_CONNECT_TIMEOUT_SECONDS`  | `30`                          | Cap on one complete MCP startup or lazy-recovery connection (transport open, initialize, and tool discovery); `<= 0` disables |
| `MAX_LOOP_ITERATIONS`          | `10`                          | Cap on agent loop iterations                       |
| `LLM_TIMEOUT_SECONDS`          | `120`                         | Per-attempt cap on a single `llm.complete()` call (`<= 0` disables) |
| `TOOL_TIMEOUT_SECONDS`         | `60`                          | Cap on a single `mcp.call_tool()`; on timeout the model gets an `is_error` tool result (`<= 0` disables) |
| `LLM_MAX_RETRIES`              | `3`                           | Retries on transient LLM failures (429/5xx/timeout/reset) and empty responses (`0` disables) |
| `LLM_RETRY_BASE_DELAY`         | `0.5`                         | Base seconds for jittered exponential backoff between LLM retries |
| `TOOL_RESULT_MAX_CHARS`        | `20000`                       | Clip threshold for a single flattened tool result before it enters session history (`<= 0` disables) |
| `MAX_RUN_TOKENS`               | `0`                           | Hard ceiling on cumulative `total_tokens` for one run; ends the run `budget_exceeded` (`<= 0` disables). When a provider reports absent/all-zero usage, the local token estimator (`agent/context.py`) fills in, so the cap works against local OpenAI-compatible servers too |
| `MAX_RUN_SECONDS`              | `0`                           | Hard wall-clock ceiling on one accepted turn, including routing, retries, LLM/tool calls, and backoff; ends the run `deadline_exceeded` (`<= 0` disables) |
| `ABORT_AFTER_CONSECUTIVE_TOOL_FAILURES` | `0`                   | Abort the run `no_progress` after this many tool-call failures in a row (a success resets the count); `<= 0` disables. Should exceed the fixed at-3 nudge so the model gets a chance to recover first |
| `CONTEXT_STRATEGY`             | `naive`                       | How the agent loop shapes the outgoing message view per LLM call: `naive` (pass-through; over budget only logs a warning) or `compaction` (summarize the over-budget middle of the history, keep the task header + recent tail verbatim). Unknown values degrade to `naive` with a warning. The view is per-call only — session history is never rewritten |
| `CONTEXT_DEFAULT_WINDOW_TOKENS` | `32768`                      | Assumed context window for models whose `models.yaml` entry has no `context_window`, and for legacy/no-orchestrator mode. Budget = window − max output tokens − safety margin |
| `CONTEXT_SAFETY_MARGIN_TOKENS` | `1024`                        | Headroom subtracted when computing the input budget; absorbs token-estimator error |
| `CONTEXT_RECENT_MESSAGES`      | `6`                           | Recent protocol-safe units (a user turn, a no-tool assistant turn, or an assistant tool call plus its results) kept verbatim under compaction; shrinks automatically if the tail alone overflows |
| `CONTEXT_SUMMARY_MAX_TOKENS`   | `512`                         | Output cap for the one-call compaction summarizer |
| `SESSION_TTL_SECONDS`          | `3600`                        | Idle TTL before an in-memory session is evicted (`<= 0` disables) |
| `SESSION_MAX_COUNT`            | `1000`                        | Max sessions retained in memory; oldest-updated evicted first (`<= 0` disables) |
| `LOG_LEVEL`                    | `INFO`                        | Python logging level (DEBUG opens per-request orchestrator detail) |
| `TRACE_ENABLED`                | `false`                       | Serialize the loop's event stream to a JSONL trace (one record per event, tagged with `run_id` + step + timestamp + latency). Off = `tracer=None`, zero hot-path cost |
| `TRACE_PATH`                   | `traces/harness.jsonl`        | Append-only JSONL trace file; parent dirs are created. Captures full prompts/args/results — treat as sensitive and guard at the filesystem level (the trace file is not covered by `HARNESS_API_KEY`) |
| `HARNESS_API_KEY`              | `""`                          | Optional API key gating `/chat`, `/chat/stream`, `/v1/*` (`/health` stays open). Empty = auth off (single-operator dev default); set = 401 without a valid key. Accepts `X-API-Key: <key>` or `Authorization: Bearer <key>`; compared constant-time |
| `ORCHESTRATION_ENABLED`        | `true`                        | Master toggle for the orchestration layer          |
| `MODELS_CONFIG_PATH`           | `config/models.yaml`          | Path to the model registry YAML                    |
| `ORCHESTRATOR_PROMPT_PATH`     | `config/orchestrator_prompt.md` | Path to orchestrator's system prompt             |
| `ORCHESTRATOR_MODEL_ID`        | `""`                          | Override which model the orchestrator itself uses (empty = default from models.yaml) |
| `GOOGLE_TOOLBOX_URL`           | _(none)_                      | Endpoint for the `google-toolbox` MCP server, interpolated into `mcp_config.yaml` |
| `OPEN_WEBSEARCH_URL`           | _(none)_                      | Endpoint for the `open-websearch` MCP server, interpolated into `mcp_config.yaml` |

## MCP Config YAML — `config/mcp_config.yaml`

```yaml
mcpServers:
  my-toolbox:
    transport: streamable-http
    url: ${GOOGLE_TOOLBOX_URL}      # endpoint lives in .env
    disabled: false
    disabled_tools: []

  web-search:
    transport: sse
    url: ${OPEN_WEBSEARCH_URL}      # endpoint lives in .env

  # Example stdio server (supported but not currently used):
  # github:
  #   transport: stdio
  #   command: uvx
  #   args: ["mcp-server-github"]
  #   env:
  #     GITHUB_TOKEN: ${GITHUB_TOKEN}

# Optional top-level block: dispatch-time tool policy (what may EXECUTE).
tool_policy:
  mode: allow_all        # allow_all (default) | allow_list
  allow: []              # fnmatch patterns over {server}__{tool}, e.g. ["web-search__*"]
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
- `tool_policy` (optional, top-level) is the **enforcement** control (what
  actually runs), checked at dispatch after argument validation and before the
  MCP call. `mode: allow_all` (default; or omit the block) runs every tool.
  `mode: allow_list` runs only tools whose namespaced name matches an `allow`
  fnmatch pattern; anything else is denied with a teaching `is_error` result the
  model can adapt to (a denial is feedback, not a run failure, and repeated
  denials trip the no-progress abort). An `allow_list` with an empty `allow`
  **fails loudly at startup** — it would otherwise silently deny every tool.
  Visibility (orchestrator selection + `disabled_tools`) and enforcement
  (`tool_policy`) are deliberately distinct layers.
- String values support `${ENV_VAR}` interpolation from raw, source-provided
  Settings values plus the process environment. Undeclared `.env` names are
  retained privately for this purpose, real process values win, declared
  defaults are not synthesized, and missing names raise immediately rather
  than producing empty strings.
- Server names cannot contain `__` (reserved for tool namespacing) and
  must be alphanumeric (dashes/underscores allowed).
- URLs are typed as `str`, not `HttpUrl`, so internal `.local` hostnames
  validate.

## Model Registry — `config/models.yaml`

```yaml
models:
  gemini-flash:
    provider: gemini
    model: gemini-3-flash-preview
    description: >
      Fast, cost-efficient. Best for straightforward queries, lookups,
      single-tool calls, and short summaries.
    default: true

  gemini-pro:
    provider: gemini
    model: gemini-3.1-pro-preview
    description: >
      Heavyweight advanced reasoning. Use ONLY for complex logic, zero-shot
      architectural design, advanced mathematics, deep analytical work, or
      highly ambiguous workflows demanding deliberate planning.

  qwen3-local:
    provider: openai
    model: qwen3.6-35b-a3b
    description: >
      Local Qwen3.6 model served over an OpenAI-compatible endpoint.
      Strong for agentic coding and deliberate tool-using workflows.
    context_window: 65536
    max_tokens: 16384
    supports_native_tools: true
    thinking: think-tags
    sampling:
      temperature: 0.6
      top_p: 0.95
      top_k: 20
```

Both models are **active** — the orchestrator routes between them (cheap
`gemini-flash` by default, `gemini-pro` for genuinely hard requests), so
`selected_model_id` is a real decision, not a no-op.

**Schema notes:**

- `provider` must be registered in `llm/client.py`'s `_PROVIDERS` —
  validated against `supported_providers()` at load time, so an unknown
  provider is rejected up front rather than at first request. Implemented
  today: `gemini` (`llm/providers/gemini.py`) and `openai`
  (`llm/providers/openai.py`). The `openai` provider is OpenAI-compatible:
  set `OPENAI_BASE_URL` to point it at a local Ollama (or any compatible)
  server, otherwise it targets real OpenAI. For Ollama, `model` must match
  an `ollama list` tag and be a tools-capable model if you need tool calling.
- `model` is the provider-specific model identifier.
- `description` is what the **orchestrator LLM** sees when picking a
  model. Write for an LLM audience: terse, capability-focused, plain
  English.
- `max_tokens` is optional; falls back to `Settings.llm_max_tokens`.
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
  `<think>...</think>` block that the provider extracts as trace-only
  reasoning.
- `sampling` is optional. Supported fields are `temperature` (`0.0`-`2.0`),
  `top_p` (`0.0`-`1.0`), and `top_k` (`>= 1`). Omit the block, or omit an
  individual field, to leave the provider/server default in force. Sampling is
  per model entry, not an environment variable. Gemini receives its native
  `top_k`; a configured OpenAI-compatible endpoint receives `top_k` under
  `extra_body`; real OpenAI omits `top_k` because it is not a supported direct
  chat-completions parameter.
- `default: true` on **exactly one** entry. The default model is used
  by the orchestrator itself (unless `ORCHESTRATOR_MODEL_ID` overrides)
  and is the safe fallback when orchestration fails.
  The orchestrator control model must not be prompted-only: if the resolved
  orchestrator row has `supports_native_tools: false`, startup logs a warning
  and runs in legacy no-orchestration mode. Prompted-tool models can still be
  selected for downstream agent turns.
- Model IDs must be `[A-Za-z0-9_.\-]+` (clean keys for logging and
  routing).

Unknown fields, invalid enum values, and out-of-range sampling values fail
startup during `models.yaml` parsing, with the model id and field path in the
validation error.

## Orchestrator Prompt — `config/orchestrator_prompt.md`

A markdown file (text is fine too — extension doesn't matter to the
loader) that's the system instruction for the orchestrator's own LLM
call. The current prompt teaches it to:

1. Read the AVAILABLE MODELS, AVAILABLE TOOLS, optional CONVERSATION SO FAR,
   and USER MESSAGE inputs (which the orchestrator injects at the start of
   the user turn).
2. Pick the cheapest/fastest model that fits.
3. Pick the smallest tool subset that fits.
4. Generate a 2–6 sentence system prompt for the downstream agent.
5. Pick a `thinking_level` (`low`/`medium`/`high`) sizing deliberation to
   the request.
6. Return ONLY a JSON object with four keys.

Modify this file to change orchestration behavior. No code changes
needed.
