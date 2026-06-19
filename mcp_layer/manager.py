# mcp_layer/manager.py

"""
MCP manager: holds all configured server clients and exposes a unified tool API.

Lifecycle (driven by FastAPI's lifespan in main.py later):
    manager = MCPManager(mcp_config)
    await manager.startup()    # connects to all enabled servers in parallel
    ...
    await manager.shutdown()   # closes all sessions

Tool namespacing:
    Each tool is exposed externally (to the LLM and to call_tool callers) as
    "{server_name}__{tool_name}". This prevents collisions when two servers
    define a tool with the same name. The separator is "__" -- server names
    containing "__" are rejected in config validation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from harness_config import MCPConfig
from .client import MCPClient, Tool, ToolCallResult

logger = logging.getLogger(__name__)

NAMESPACE_SEP = "__"


class MCPManager:
    """Aggregates MCPClient instances and presents a unified tool registry."""

    def __init__(self, mcp_config: MCPConfig) -> None:
        self._config = mcp_config
        self._clients: dict[str, MCPClient] = {}
        # Map of namespaced tool name -> (server_name, raw_tool_name)
        self._tool_index: dict[str, tuple[str, str]] = {}

    # ----- lifecycle -----

    async def startup(self) -> None:
        """Connect to every enabled server in parallel.

        Failures on individual servers are logged but do not abort startup;
        the harness can still operate with a degraded tool set. Adjust later
        if you want strict-mode startup.
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
                # Best-effort cleanup in case partial state was left behind.
                try:
                    await client.close()
                except Exception:
                    pass
                continue
            self._clients[name] = client
            for tool in client.tools:
                namespaced = f"{name}{NAMESPACE_SEP}{tool.name}"
                self._tool_index[namespaced] = (name, tool.name)

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

    # ----- tool inspection -----

    @property
    def connected_servers(self) -> list[str]:
        return list(self._clients.keys())

    def list_tools(self) -> list[tuple[str, Tool]]:
        """Return [(namespaced_name, Tool), ...] for every available tool."""
        out: list[tuple[str, Tool]] = []
        for namespaced, (server_name, raw_name) in self._tool_index.items():
            client = self._clients[server_name]
            tool = next((t for t in client.tools if t.name == raw_name), None)
            if tool is not None:
                out.append((namespaced, tool))
        return out

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        """Return tools in a generic schema shape.

        Adapting to a specific LLM provider's tool format (Anthropic, Gemini,
        OpenAI) happens in the LLM client. We keep this provider-agnostic.
        """
        return [
            {
                "name": namespaced,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for namespaced, tool in self.list_tools()
        ]

    # ----- tool execution -----

    async def call_tool(self, namespaced_name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """Route a namespaced tool call to the appropriate server."""
        if namespaced_name not in self._tool_index:
            return ToolCallResult(
                content=f"unknown tool: {namespaced_name!r}",
                is_error=True,
            )
        server_name, raw_name = self._tool_index[namespaced_name]
        client = self._clients.get(server_name)
        if client is None:
            return ToolCallResult(
                content=f"server {server_name!r} is not connected",
                is_error=True,
            )
        return await client.call_tool(raw_name, arguments)