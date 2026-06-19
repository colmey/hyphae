# PyAiHarness Claude Code Instructions

## Overview
PyAiHarness is an HTTP AI harness: it receives a prompt, an **orchestrator** picks a model + tool subset + system prompt per request, an **agent loop** runs that against the chosen LLM using tools from external **MCP** servers, and returns the answer.

Request flow: `POST /chat → orchestrator (pick model+tools+system) → run_agent loop (LLM ⇄ MCP tools) → response`.

Two invariants that shape most changes:
* `agent/loop.py` is the **only** bridge between the LLM client and the MCP manager — neither knows the other exists.
* Orchestration is optional and **degrades to a safe default; it must never break a request**.

Deep detail lives in `docs/` — see the Documentation Map below.

## Build and Execution
* First-time setup: `./setup.sh` creates `.venv`, installs `requirements.txt`, and seeds `.env` from `.env.example`
* Run the local server using: `./runscript.sh -m uvicorn main:app --host 0.0.0.0 --port 8000`
* Run tests using: `for t in tests/smoke_test_*.py; do ./runscript.sh "$t"; done`
* Most tests are not necessary to be ran, run what is directly impacting the changes
* The `runscript.sh` file is the canonical launcher that activates the virtual environment and sets `PYTHONPATH`

## Architecture and Coding Conventions
* Use Python 3.11+ and FastAPI for all HTTP layer modifications
* Environment variables live in a `.env` file at the project root; `bootstrap.load_secrets()` loads it into `os.environ` (via `python-dotenv`) before any settings are read. Real environment variables already set in the process take precedence over `.env`. Keep `.env.example` in sync when adding a setting
* Pull settings lazily via `harness_config.get_settings()`; never instantiate it directly
* Use the `google-genai` package for Gemini interactions; do not use the deprecated `google-generativeai` package
* Provider LLM clients live in `llm/providers/<name>.py` and register a builder in `llm/client.py`'s `_PROVIDERS` (the single source of truth for providers — `supported_providers()` keys off it); `llm/client.py` itself must never import a provider SDK at module level, so the `LLMClient` ABC stays importable without pulling in any SDK
* Do not write tests using `pytest`; all tests must be standalone runnable scripts placed in the `tests/` directory
* Do not name the MCP layer directory `mcp/` to avoid shadowing the official SDK; always use `mcp_layer/`
* Do not use `__` in MCP server names, as this is strictly reserved for tool namespacing
* Never strip `provider_metadata` from `TextBlock` or `ToolUseBlock`, as it holds opaque per-provider state
* Do not mutate session `.messages` directly; always use the `append_user`, `append_assistant`, or `append_tool_results` helpers
* Ensure `ChatRequest` schema validation remains strict by using `extra="forbid"`
* Do not implement parallel tool execution; tool execution must remain sequential
* Do not re-enable automatic function calling in the Gemini client; the agent loop acts as the orchestrator
* Whenever making changes, ensure that the documentation is updated to reflect it, if necessary

## Documentation Map
Full reference lives in `docs/` (read on demand, not always loaded). Open the smallest relevant file:
* `docs/architecture.md` — **read before** touching subsystems, the agent loop, or design decisions
* `docs/configuration.md` — **read before** changing env vars or `config/*.yaml`
* `docs/api.md` — **read before** changing the `/chat` or `/health` contract
* `docs/operations.md` — running, extending, smoke tests, known limitations
* `docs/README.md` — index/router + quick start