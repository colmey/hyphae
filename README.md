# Hyphae

Hyphae is a compact Python agent harness. It accepts prompts over native or
OpenAI-compatible HTTP endpoints, optionally routes each request to a ready
Gemini or OpenAI-compatible model, and lets the selected model call tools from
configured MCP servers.

## Quick start

```bash
# Create .venv, install the locked development environment, and seed .env.
./scripts/setup.sh

# Edit .env and the files under hyphae/config/, then start the server.
uv run uvicorn hyphae.main:app --host 0.0.0.0 --port 8000
```

The four runtime configuration assets are:

- `hyphae/config/mcp_config.yaml` — MCP servers and dispatch policy;
- `hyphae/config/models.yaml` — routable model catalog and capabilities;
- `hyphae/config/orchestrator_prompt.md` — selection-only routing instructions; and
- `hyphae/config/agent_prompt.md` — trusted downstream agent instructions.

In another terminal:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: text/plain' \
  --data 'What tools do you have access to?'
```

`/chat` is plain text in and out. OpenAI clients can instead point their base
URL at `http://localhost:8000/v1`.

## Documentation

Start with the [documentation index](docs/README.md). The permanent references
cover [architecture](docs/architecture.md),
[execution](docs/execution.md), [providers](docs/providers.md),
[configuration](docs/configuration.md), the [HTTP API](docs/api.md), and
[operations](docs/operations.md). Test and evaluation commands live in the
[test guide](tests/README.md).

## License

_TBD_
