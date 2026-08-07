# Test and evaluation guide

The default pytest suite is hermetic: it uses scripted model/MCP fakes and
temporary configuration, and must not require `.env`, credentials, network
access, configured MCP servers, or a model endpoint.

## Canonical commands

Run the default non-live suite:

```bash
uv run pytest -q
```

Run a focused module or node while changing code:

```bash
uv run pytest -q tests/test_turn_execution.py
uv run pytest -q tests/test_turn_execution.py::test_ephemeral_turn_does_not_publish_session_to_store
```

Run the repository gates after the change is stable:

```bash
uv lock --check
uv sync --frozen --group dev
uv run ruff check .
uv run mypy
uv run python -m compileall -q \
  agent api application config llm mcp_runtime orchestrator scripts tooling \
  main.py tests
uv run python scripts/render_settings_reference.py --check
uv run python scripts/check_markdown_links.py
uv run pytest --collect-only -q -m live
uv run pytest -q
git diff --check
```

The same Python test commands can be launched through `./runscript.sh`; `uv run`
is used above because it is also the CI form.

## Suite policy

`pyproject.toml` excludes `live` by default and gives hermetic tests a 30-second
deadlock timeout. Every `test_*_live.py` module must declare both the `live`
marker and `timeout(0)`: external dependencies do not have a repository-owned
completion bound. `test_suite_configuration.py` enforces these rules and also
forbids import-time global logging configuration in test modules.

Do not weaken a production protocol merely to make a fake convenient. Reusable
fakes should model lifecycle, task affinity, stream termination, and exact
calls closely enough to expose ownership bugs.

## Organization

- Focused `test_*.py` modules own detailed unit and contract behavior.
- `test_agent_parity.py` retains seven representative end-to-end public
  journeys plus one public import/signature contract; detailed branches live
  with their focused owners rather than being duplicated in a second broad
  harness.
- `fakes.py` contains strict reusable LLM/tool fakes and event collection.
- `_app_support.py` wires one complete typed `ApplicationRuntime` for HTTP
  tests; tests replace the runtime or one named value through focused helpers.
- `test_*_live.py` modules are explicit configured integration checks.
- `eval_data/agent_eval_v1.yaml` is the data set for the standalone evaluation
  runner.

Ordinary tests are excluded from strict mypy. `tests/fakes.py` and
`tests/_app_support.py` are included because drift in shared support can
invalidate many hermetic tests at once.

Verification-count record: the historical pre-later-session audit baseline was
693 selected hermetic cases. The final Session 20 inventory is 881 selected
hermetic cases plus seven explicitly live cases. Counts are evidence for this
handoff, not a target that future suites should preserve artificially.

## Live integrations

Live tests may contact configured services or spend model tokens. They never run
as part of the default suite.

```bash
uv run pytest -q -m live
uv run pytest -q -m "live and mcp"
uv run pytest -q -m "live and model"
uv run pytest -q -m "live and http_server"
```

The `http_server` tests drive the configured FastAPI lifespan in-process; a
separately launched Uvicorn process is not required. Prefer a single live module
or node while diagnosing an integration:

```bash
uv run pytest -q -m live tests/test_mcp_live.py
```

Collect live tests without executing them as part of every final verification:

```bash
uv run pytest --collect-only -q -m live
```

## Evaluation runner

`eval_agent.py` is a small YAML-driven runner, not a pytest plugin or general
evaluation framework. Its default tier is deterministic:

```bash
./runscript.sh tests/eval_agent.py
```

The live tier is separately opt-in and may use the configured model backend:

```bash
EVAL_LIVE=1 ./runscript.sh tests/eval_agent.py
```

Add hermetic evaluation scenarios as data in `eval_data/agent_eval_v1.yaml`.
Keep precise protocol and failure assertions in pytest rather than expanding
the evaluation runner into a second test framework.
