# tests/
 
Smoke test scripts for the harness. Each file is a **standalone runnable
script**, not a pytest test. Invoke via `runscript.sh` from the project
root:
 
```bash
./runscript.sh tests/smoke_test_orchestrator.py
./runscript.sh tests/smoke_test_http.py
# ...etc
```
 
`runscript.sh` adds the project root to `PYTHONPATH`, so each test's
top-level imports (`from bootstrap import load_secrets`,
`from harness_config import get_settings`, ...) work regardless of where
the script file lives.
 
## What each script verifies
 
| Script | Scope |
|---|---|
| `smoke_test_config.py` | Settings load, MCP config parse |
| `smoke_test_mcp.py` | MCP server connections + tool inventory |
| `smoke_test_llm.py` | LLM client direct call (no tools, no loop) |
| `smoke_test_session.py` | Session store CRUD |
| `smoke_test_agent.py` | End-to-end agent loop on a real prompt |
| `smoke_test_reliability.py` | Loop reliability hardening: retry, timeout, truncation, tool-result clip — scripted fakes, no network |
| `smoke_test_loop_intelligence.py` | Loop-intelligence scaffolding: final-iteration wrap-up, stall detection, consecutive-failure nudge — scripted fakes, no network |
| `smoke_test_concurrency.py` | Same-session concurrency guard (409) |
| `smoke_test_orchestrator.py` | Orchestrator decisions across model tiers |
| `smoke_test_http.py` | Live HTTP exercise of the running server |
 
## Convention
 
If you add a new script, prefix the filename `smoke_test_` so the purpose
is obvious from the file tree, and structure the output the way the
existing ones do: section banners (`===` lines), explicit `print()` of
what's being checked, hard `assert` for invariants, and `raise SystemExit(1)`
on failure so `./runscript.sh` exits non-zero (suitable for CI).
 
These scripts do not use a test framework on purpose — they're meant to
be readable end-to-end as documentation of the harness's expected
behavior. If we ever need fixtures, parameterization, or parallelism,
that's the point to move to pytest.