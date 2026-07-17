# mcp_layer/manager.py

"""Aggregate MCP clients and expose namespaced tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import MCPConfig
from .client import MCPClient, Tool, ToolCallResult
from .contracts import NAMESPACE_SEP

logger = logging.getLogger(__name__)

class MCPManager:
    """Aggregates MCPClient instances and presents a unified tool registry."""

    def __init__(self, mcp_config: MCPConfig) -> None:
        self._config = mcp_config
        self._clients: dict[str, MCPClient] = {}
        self._tool_index: dict[str, tuple[str, Tool]] = {}

    async def startup(self) -> None:
        """Connect to every enabled server in parallel.

        Individual server failures are logged but do not abort startup.
        """
        enabled = self._config.enabled_servers()
        if not enabled:
            logger.warning("no enabled MCP servers in config")
            return

        clients = {name: MCPClient(name, cfg) for name, cfg in enabled.items()}
        results = await asyncio.gather(
            *(c.connect() for c in clients.values()),
            return_exceptions=True,
        )

        for (name, client), result in zip(clients.items(), results):
            if isinstance(result, Exception):
                logger.error("failed to connect to %r: %s", name, result)
                try:
                    await client.close()
                except Exception:
                    pass
                continue
            self._clients[name] = client
            for tool in client.tools:
                namespaced = f"{name}{NAMESPACE_SEP}{tool.name}"
                self._tool_index[namespaced] = (name, tool)

        logger.info(
            "MCP startup complete: %d/%d servers connected, %d tools available",
            len(self._clients), len(enabled), len(self._tool_index),
        )

    async def shutdown(self) -> None:
        """Close all client connections. Safe to call multiple times."""
        if not self._clients:
            return
        await asyncio.gather(
            *(c.close() for c in self._clients.values()),
            return_exceptions=True,
        )
        self._clients.clear()
        self._tool_index.clear()

    @property
    def connected_servers(self) -> list[str]:
        return list(self._clients.keys())

    def list_tools(self) -> list[tuple[str, Tool]]:
        """Return [(namespaced_name, Tool), ...] for every available tool."""
        return [
            (namespaced, tool)
            for namespaced, (_server_name, tool) in self._tool_index.items()
        ]

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        """Return tools in the provider-agnostic schema shape."""
        return [
            {
                "name": namespaced,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for namespaced, tool in self.list_tools()
        ]

    async def call_tool(self, namespaced_name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """Route a namespaced tool call to the appropriate server."""
        if namespaced_name not in self._tool_index:
            return ToolCallResult(
                content=f"unknown tool: {namespaced_name!r}",
                is_error=True,
            )
        server_name, tool = self._tool_index[namespaced_name]
        client = self._clients.get(server_name)
        if client is None:
            return ToolCallResult(
                content=f"server {server_name!r} is not connected",
                is_error=True,
            )
        return await client.call_tool(tool.name, arguments)
