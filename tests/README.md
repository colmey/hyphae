# Tests

The default suite is hermetic and uses pytest with scripted model/MCP fakes and
temporary configuration. It must not require `.env`, credentials, configured MCP
servers, or a model endpoint.

```bash
./runscript.sh -m pytest
```

Configured-backend checks live in `test_*_live.py` modules and carry the `live`
marker plus one or more capability markers: `model`, `mcp`, and `http_server`.
They are excluded from the default command and run only when selected explicitly:

```bash
./runscript.sh -m pytest -m live
./runscript.sh -m pytest -m "live and mcp"
./runscript.sh -m pytest -m "live and model"
./runscript.sh -m pytest -m "live and http_server"
```

Live checks load developer configuration and may spend model tokens or contact
configured services. The `http_server` checks drive the configured FastAPI
application in-process; they do not require a separately launched Uvicorn process.

## Organization

- `test_*.py` contains pytest-discovered hermetic regression tests.
- `conftest.py` contains pytest fixtures.
- `fakes.py` contains strict reusable `LLMClient`/`ToolRuntime` fakes and the
  typed event collector.
- `_app_support.py` contains strict typed FastAPI wiring around one complete
  `ApplicationRuntime`; application tests replace that runtime or one named
  value through its focused helpers rather than mutating independent
  `app.state` fields.
- `test_*_live.py` contains explicitly marked configured MCP, model, agent,
  orchestrator, HTTP, concurrency, and OpenAI-provider checks.
- `eval_agent.py` remains a separate YAML-driven evaluation runner.

Run one configured integration directly by node id when diagnosing a layer, for example:

```bash
./runscript.sh -m pytest -m live tests/test_mcp_live.py
```

Run deterministic agent evaluations separately with
`./runscript.sh tests/eval_agent.py`; opt into its live tier with
`EVAL_LIVE=1 ./runscript.sh tests/eval_agent.py`.

The repository mypy configuration checks both shared support modules in strict
mode alongside production code. Run the exact local/CI gate with:

```bash
uv run mypy
```

Ordinary test modules remain excluded from strict checking; reusable support is
included because protocol drift there can invalidate many otherwise-hermetic
tests at once.
