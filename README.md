# hyphae

A small AI agent harness written in Python. It takes a prompt over HTTP, picks a
model and tool set for the request, runs an agent loop that can call tools from
external MCP (Model Context Protocol) servers, and returns the answer. It is built
on FastAPI and the official `mcp` SDK, and works with more than one LLM provider
(Gemini and OpenAI-compatible endpoints today).

## Quick start

```bash
# 1. Create the venv, install dependencies, and seed .env from .env.example
./setup.sh

# 2. Fill in your secrets and config:
#    - .env                          API keys, MCP server URLs
#    - config/mcp_config.yaml        MCP servers
#    - config/models.yaml            model registry
#    - config/orchestrator_prompt.md orchestrator behavior

# 3. Run the server
./runscript.sh -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Then, in another terminal:

```bash
# Check it's alive
curl http://localhost:8000/health

# Send a prompt (plain text in, plain text out)
curl -X POST http://localhost:8000/chat \
    -H 'Content-Type: text/plain' \
    --data 'What tools do you have access to?'
```

## Documentation

Everything else — architecture, configuration, the full HTTP API, and operations —
lives in [`docs/`](./docs/). Start at [`docs/README.md`](./docs/README.md), which
routes to [architecture](./docs/architecture.md), [configuration](./docs/configuration.md),
[api](./docs/api.md), and [operations](./docs/operations.md).

If you're handing this codebase to an AI assistant or a new contributor, point them
at `docs/README.md` — it indexes the complete context.

## License

_TBD_
