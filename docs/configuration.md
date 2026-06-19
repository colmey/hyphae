# PyAiHarness — Configuration

Everything the harness reads at startup: environment variables and the
three files under `config/` (MCP servers, the model registry, and the
orchestrator's system prompt).

> See also: [README.md](README.md) (overview), [architecture.md](architecture.md)
> (how config flows through the subsystems), [operations.md](operations.md)
> (running, tuning, disabling orchestration).

## Environment Variables

Read into `Settings` via `pydantic-settings`. They live in a `.env` file at
the project root; `bootstrap.load_secrets()` loads it into `os.environ`
before settings are read. Copy `.env.example` to `.env` (the `setup.sh`
script does this for you) and fill in your values. Real environment
variables already set in the process take precedence over `.env`, so the
same file works for local dev, containers, and CI.

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
| `MAX_LOOP_ITERATIONS`          | `25`                          | Cap on agent loop iterations                       |
| `LLM_TIMEOUT_SECONDS`          | `120`                         | Per-attempt cap on a single `llm.complete()` call (`<= 0` disables) |
| `TOOL_TIMEOUT_SECONDS`         | `60`                          | Cap on a single `mcp.call_tool()`; on timeout the model gets an `is_error` tool result (`<= 0` disables) |
| `LLM_MAX_RETRIES`              | `3`                           | Retries on transient LLM failures (429/5xx/timeout/reset) and empty responses (`0` disables) |
| `LLM_RETRY_BASE_DELAY`         | `0.5`                         | Base seconds for jittered exponential backoff between LLM retries |
| `TOOL_RESULT_MAX_CHARS`        | `20000`                       | Clip threshold for a single flattened tool result before it enters session history (`<= 0` disables) |
| `SESSION_TTL_SECONDS`          | `3600`                        | Idle TTL before an in-memory session is evicted (`<= 0` disables) |
| `SESSION_MAX_COUNT`            | `1000`                        | Max sessions retained in memory; oldest-updated evicted first (`<= 0` disables) |
| `LOG_LEVEL`                    | `INFO`                        | Python logging level (DEBUG opens per-request orchestrator detail) |
| `TRACE_ENABLED`                | `false`                       | Serialize the loop's event stream to a JSONL trace (one record per event, tagged with `run_id` + step + timestamp + latency). Off = `tracer=None`, zero hot-path cost |
| `TRACE_PATH`                   | `traces/harness.jsonl`        | Append-only JSONL trace file; parent dirs are created. Captures full prompts/args/results — treat as sensitive (no auth yet) |
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
```

**Schema notes:**

- `mcpServers` top-level key is camelCase for cross-tool compatibility
  (configs are pasteable to/from Claude Desktop, Cursor, Roo). Internal
  fields are snake_case.
- `transport` is `streamable-http`, `sse`, or `stdio`. Each transport
  validates its own required fields.
- `disabled: true` skips the server entirely at startup.
- `disabled_tools: [...]` hides specific tools from the LLM (defense in
  depth: filtered at list time and at call time).
- String values support `${ENV_VAR}` interpolation; missing vars raise
  immediately rather than producing empty strings.
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
- `default: true` on **exactly one** entry. The default model is used
  by the orchestrator itself (unless `ORCHESTRATOR_MODEL_ID` overrides)
  and is the safe fallback when orchestration fails.
- Model IDs must be `[A-Za-z0-9_.\-]+` (clean keys for logging and
  routing).

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
