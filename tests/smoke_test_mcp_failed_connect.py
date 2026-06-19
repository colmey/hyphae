"""
Regression test: a failed MCP connection must clean up, not spin the loop.

When an MCP server is unreachable, the SDK's HTTP/SSE transport cancels its
internal task-group scope, which surfaces as CancelledError (a BaseException,
not Exception). If MCPClient.connect() guards its teardown with a bare
`except Exception`, that cleanup is skipped, the transport's background tasks
are never cancelled, and the event loop pins a CPU core on a dead connection
(observed as 100% CPU at idle). connect() uses a success-flag `finally` so the
exit stack is always unwound; this test pins that behavior.

Hermetic: points two clients at dead local ports (no server needed). For each,
it asserts connect() raises and that the process stays idle afterwards (CPU
time consumed over a short sleep is a tiny fraction of wall time).

Run from the project root:
    ./runscript.sh tests/smoke_test_mcp_failed_connect.py
"""

from __future__ import annotations

from bootstrap import load_secrets
load_secrets()

import asyncio
import os
import time

from harness_config import SSEServer, StreamableHTTPServer
from mcp_layer.client import MCPClient

# Dead ports: nothing should be listening here.
_STREAMABLE = StreamableHTTPServer(transport="streamable-http", url="http://127.0.0.1:5999/mcp")
_SSE = SSEServer(transport="sse", url="http://127.0.0.1:5998/sse")

# A failed connect that leaks a spinning task drives CPU≈wall (100%). A clean
# teardown leaves the loop idle. Anything under this margin proves no spin.
_MAX_IDLE_BUSY_FRACTION = 0.25
_IDLE_SECONDS = 2.0


def _cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


async def _check(label: str, cfg) -> None:
    client = MCPClient(label, cfg)

    raised = False
    try:
        await client.connect()
    except BaseException as e:  # CancelledError included by design
        raised = True
        print(f"  [{label}] connect failed as expected: {type(e).__name__}")
    assert raised, f"[{label}] connect() should have failed against a dead port"

    # The connection is gone; the loop must now be idle.
    c0, w0 = _cpu_seconds(), time.monotonic()
    await asyncio.sleep(_IDLE_SECONDS)
    busy = (_cpu_seconds() - c0) / (time.monotonic() - w0)
    print(f"  [{label}] idle busy = {busy * 100:.0f}% of one core")
    assert busy < _MAX_IDLE_BUSY_FRACTION, (
        f"[{label}] event loop is spinning after a failed connect "
        f"(busy={busy * 100:.0f}%); transport teardown leaked a task"
    )

    # close() on a never-connected client is a no-op and must not raise.
    await client.close()


async def main() -> None:
    print("Scenario: failed MCP connect cleans up and does not spin the loop")
    await _check("streamable-http", _STREAMABLE)
    await _check("sse", _SSE)
    print("mcp failed-connect smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
