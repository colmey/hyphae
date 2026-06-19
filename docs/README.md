# PyAiHarness — Reference Docs

A minimal, extendable AI harness in Python. Receives a prompt over HTTP,
runs an **orchestrator** to pick a model + tool subset + system prompt,
runs an agent reasoning loop against the chosen LLM with the chosen
tools (exposed by external MCP / Model Context Protocol servers), and
returns the answer.

This is a working prototype. The design priority is a clean foundation that
can grow into more sophisticated loops (planning, sub-agents, parallel tool
use, streaming, persistence) without rewrites.

> This reference is split across several files. Use the router below to
> jump to the one you need — each is small enough to read end-to-end.

## Where to look

| If you're working on… | Read |
|---|---|
| The big picture, subsystems, design rationale | [architecture.md](architecture.md) |
| Env vars or the three `config/` files | [configuration.md](configuration.md) |
| The HTTP request/response contract (`/chat`, `/health`) | [api.md](api.md) |
| Running, extending, smoke tests, known limitations | [operations.md](operations.md) |

Each file also cross-links to the others at the top.

---

## Quick Start

```bash
# 1. Create the venv, install dependencies, and seed .env from .env.example
./setup.sh

# 2. Configure secrets: edit .env and set your API keys (e.g. GEMINI_API_KEY)
#    and MCP server URLs (GOOGLE_TOOLBOX_URL, OPEN_WEBSEARCH_URL).
#
#    Configure MCP servers in config/mcp_config.yaml.
#    Configure routable models in config/models.yaml.
#    Tune the orchestrator's behavior in config/orchestrator_prompt.md.

# 3. Run a smoke test in-process (no server needed)
./runscript.sh tests/smoke_test_http.py

# 4. Or run a real server
./runscript.sh -m uvicorn main:app --host 0.0.0.0 --port 8000

# 5. Hit it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "What is 2+2?"}'
```

For configuration details see [configuration.md](configuration.md); for
the full request/response shape see [api.md](api.md).
