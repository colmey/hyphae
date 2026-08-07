# Operations

This guide covers installation, process operation, health, tracing, and common
failures. See the root [README](../README.md) for the only quick start,
[Configuration](configuration.md) for every input, and [Tests](../tests/README.md)
for verification commands.

## Install and run

Hyphae requires Python 3.12 and uses the committed `uv.lock`.

```bash
./scripts/setup.sh
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

`scripts/setup.sh` installs `uv` when absent, synchronizes the locked development
environment into `.venv`, and creates `.env` from `.env.example` only when
`.env` does not exist.

Run `uv` commands from the repository root. `uv run` selects the synchronized
project environment directly; application-relative configuration paths resolve
from the repository root.

For a network-accessible deployment, choose the bind address deliberately and
put normal TLS, proxy, and access controls in front of Uvicorn. Set
`HYPHAE_API_KEY` whenever untrusted clients can reach protected endpoints.

## Startup and shutdown

Application lifespan performs these operations before accepting traffic:

1. load and validate settings and config assets;
2. discover enabled MCP catalogs;
3. create the dispatch policy, in-memory session store, and run guard;
4. build direct or orchestrated model clients and preflight configured models;
5. start optional JSONL tracing and create the routing runtime and
   `ApplicationRuntime`;
6. log one readiness summary.

The readiness summary reports provider/default model, routing mode, MCP health,
and tool count; orchestrated mode also reports its ready-model count. It
deliberately omits trace state. When tracing is enabled, the tracer logs its
path separately; startup or writer failure is also logged separately. Disabled
tracing emits no readiness field.

Invalid present configuration is fatal. Missing optional orchestration assets,
an explicitly disabled orchestrator, or an unsuitable control model uses the
documented fixed/direct mode. A configured orchestration registry whose default
model is unavailable is fatal; other unavailable models are omitted from the
ready inventory.

On shutdown, Hyphae closes model clients, MCP leases/connections, and the trace
worker. Cleanup failures are logged without hiding another active failure or
cancellation.

## Health and smoke checks

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8000/chat \
  -H 'Content-Type: text/plain' \
  --data 'Reply with one short sentence.'
```

Add `-H 'X-API-Key: ...'` to protected calls when authentication is enabled.
The health endpoint is intentionally unauthenticated.

Interpret `/health` at the server level rather than only reading the top-level
status. A degraded MCP server retains its last sanitized error and catalog
metadata. `catalog_revision`, discovery/refresh times, and `active_leases` help
distinguish stale discovery, scheduled refresh, and in-flight turn ownership.

## Logging

`LOG_LEVEL` controls Python logging. Normal startup logs name the configured
provider/default model, routing state, ready model count, enabled MCP server
count, and tool count. Enabled tracing logs its path separately; disabled
tracing emits no state line. Provider and MCP failures keep private diagnostics
in logs while API responses use stable safe messages.

Do not use debug logs as a durable audit record. Do not assume they contain the
complete prompt or transcript; logging is owned by each subsystem and may omit
sensitive payloads.

## Optional JSONL tracing

Tracing is off by default. Enable it explicitly:

```dotenv
TRACE_ENABLED=true
TRACE_JSONL_PATH=traces/harness.jsonl
```

The tracer persists the loop's emitted event records with run ID, step,
timestamp, and latency metadata. Depending on the event, records can contain
assistant text or reasoning, orchestration selections, tool arguments/results,
usage, and terminal/error data. It does not independently serialize the full
incoming request, system prompt, or conversation history, but its event
payloads are still sensitive and should be protected at the filesystem level.

Retention is application-owned and fixed: one active file is limited to 10
MiB, with three backups (`.1` newest through `.3` oldest). Complete records are
never split. A single record larger than the file limit disables the optional
tracer through its nonfatal writer-failure path. The in-memory queue is also
bounded; overflow drops new trace records with rate-limited warnings rather
than blocking execution. There are no trace size/count/queue tuning settings.

## Sessions and process topology

Native `/chat` sessions live only in the process memory. They expire by idle
TTL, are bounded by capacity and transcript characters, and cannot be processed
concurrently within one process. Restarting the process loses them.

The `/v1/chat/completions` adapter creates a fresh ephemeral session per
request. The OpenAI client must resend prior messages.

These guarantees are process-local. Multiple workers do not share sessions or
the same-session guard. Use a single worker unless the client owns all history
or a separately designed shared session/claim backend is introduced.

## Common failures

### Startup rejects configuration

- Confirm `.env` contains the credential for every configured provider.
- Confirm every `${NAME}` referenced by MCP YAML is present.
- Check transport-specific MCP fields and URL schemes.
- Check `models.yaml` provider IDs, model IDs, exactly one default when needed,
  capability enums, sampling ranges, and context/output limits.
- Do not select a prompted-only model as the orchestration control model.

### Health is degraded or tools are absent

- Inspect the matching `mcp_servers[]` record and server logs.
- Confirm the endpoint is reachable from the Hyphae process.
- Confirm the configured transport and endpoint path match the server.
- Check `disabled`, `disabled_tools`, and the top-level `tool_policy`.
- An age-driven catalog refresh occurs only at an accepted request boundary;
  startup and a turn lease can also discover a catalog.

### Model requests fail

- Compare `/v1/models` with the requested ID.
- For OpenAI-compatible servers, ensure `OPENAI_COMPAT_BASE_URL` ends at the
  correct API root and `model` matches the server's identifier.
- Ensure a nonempty OpenAI API key is supplied even when a local server ignores
  it, because the SDK requires one.
- Distinguish `invalid_model` (not admitted) from `model_unavailable`
  (configured but not ready) and `provider_failure` (execution failed).

### A run stops early

Inspect `X-Done-Reason`, the stream terminal, or trace/log records. Iteration,
token, deadline, consecutive-failure, context, and retained-session limits are
separate controls. Their exact settings are documented in
[Configuration](configuration.md).

### Streaming appears buffered

Test directly against Uvicorn first. Reverse proxies can buffer SSE even when
Hyphae emits chunks correctly; disable proxy buffering for streaming routes.

## Known limitations

- Sessions and same-session exclusion are in-memory and single-process.
- There is no durable job queue, resume protocol, or cross-process run state.
- Each accepted turn's policy-visible tool snapshot is fixed; the underlying
  process-owned MCP catalogs have separate discovery/refresh ownership. There
  is no progressive tool discovery or mutable active tool view within a turn.
- Prompt composition has distinct local owners; there is no template framework,
  temporal context injection, or hot reload.
- The OpenAI adapter intentionally supports text `system`/`user`/`assistant`
  messages only, not the complete OpenAI API surface.
- Gemini uses the inherited coarse complete-based stream; the
  OpenAI-compatible provider implements native incremental streaming.
