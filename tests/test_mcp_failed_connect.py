"""Failed MCP transport entry must leave no owned resource or worker task."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import mcp_layer.client as client_module
from config import SSEServer, StreamableHTTPServer
from mcp_layer.client import MCPClient

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("patch_name", "name", "config"),
    [
        (
            "streamable_http_client",
            "streamable-http",
            StreamableHTTPServer(
                transport="streamable-http", url="http://example.invalid/mcp"
            ),
        ),
        (
            "sse_client",
            "sse",
            SSEServer(transport="sse", url="http://example.invalid/sse"),
        ),
    ],
)
async def test_failed_connection_cleans_owned_worker_and_context(
    patch_name: str,
    name: str,
    config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers: list[asyncio.Task[None]] = []
    active_contexts = 0

    class _FailingTransport:
        async def __aenter__(self) -> Any:
            nonlocal active_contexts
            active_contexts += 1
            worker = asyncio.create_task(asyncio.Event().wait())
            workers.append(worker)
            try:
                raise OSError("transport unavailable")
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                active_contexts -= 1

        async def __aexit__(self, exc_type, exc, tb) -> None:
            raise AssertionError("failed __aenter__ must not invoke __aexit__")

    monkeypatch.setattr(client_module, patch_name, lambda *args: _FailingTransport())
    client = MCPClient(name, config)

    with pytest.raises(OSError, match="transport unavailable"):
        async with client.open():
            raise AssertionError("failed transport must not yield a connection")

    assert active_contexts == 0
    assert len(workers) == 1
    assert workers[0].done()
    assert workers[0].cancelled()
