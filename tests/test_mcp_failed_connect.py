"""Failed MCP connections must unwind transport tasks instead of spinning."""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from config import SSEServer, StreamableHTTPServer
from mcp_layer.client import MCPClient


pytestmark = pytest.mark.anyio


def _cpu_seconds() -> float:
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


@pytest.mark.parametrize(
    ("name", "config"),
    [
        (
            "streamable-http",
            StreamableHTTPServer(
                transport="streamable-http", url="http://127.0.0.1:5999/mcp"
            ),
        ),
        ("sse", SSEServer(transport="sse", url="http://127.0.0.1:5998/sse")),
    ],
)
async def test_failed_connection_cleans_up_without_spinning(name, config) -> None:
    client = MCPClient(name, config)
    try:
        with pytest.raises(BaseException):
            await client.connect()

        cpu_before = _cpu_seconds()
        wall_before = time.monotonic()
        await asyncio.sleep(2.0)
        busy_fraction = (_cpu_seconds() - cpu_before) / (time.monotonic() - wall_before)
        assert busy_fraction < 0.25
    finally:
        await client.aclose()
