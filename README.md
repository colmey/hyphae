# Hyphae

Hyphae is a compact Python agent harness. It accepts prompts over native or
OpenAI-compatible HTTP endpoints, optionally routes each request to a ready
Gemini or OpenAI-compatible model, and lets the selected model call tools from
configured MCP servers.

## Quick start

```bash
# Create .venv, install the locked development environment, and seed .env.
./setup.sh

# Edit .env and the files under config/, then start the server.
./runscript.sh -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The four runtime configuration assets are:

- `config/mcp_config.yaml` — MCP servers and dispatch policy;
- `config/models.yaml` — routable model catalog and capabilities;
- `config/orchestrator_prompt.md` — selection-only routing instructions; and
- `config/agent_prompt.md` — trusted downstream agent instructions.

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
