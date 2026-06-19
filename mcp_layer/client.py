#mcp_layer/client.py

"""
Per-server MCP client wrapper.

Each MCPClient owns one ClientSession to one configured MCP server. The session
is established via the appropriate transport (streamable-http / sse / stdio)
and kept alive for the lifetime of the harness process.

This module knows nothing about the LLM or about tool namespacing. It exposes a
single server's tools by their raw names; the MCPManager handles namespacing.

Lifecycle:
  client = MCPClient(name, config)
  await client.connect()        # opens transport, initializes session, lists tools
  await client.call_tool(...)   # any number of times
  await client.close()          # closes session and transport
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from harness_config import MCPServerConfig, SSEServer, StdioServer, StreamableHTTPServer

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """A tool exposed by an MCP server (un-namespaced)."""
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCallResult:
    """Result of a tool call. `content` is the concatenated text output."""
    content: str
    is_error: bool


class MCPClient:
    """Manages the connection and session for one MCP server."""

    def __init__(self, name: str, server_config: MCPServerConfig) -> None:
        self.name = name
        self.config = server_config
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list[Tool] = []

    @property
    def tools(self) -> list[Tool]:
        """Tools advertised by this server (filtered by disabled_tools)."""
        return self._tools

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        """Open the transport, initialize the MCP session, and fetch tool list.

        On any failure, ensures partial resources are cleaned up before raising.
        """
        if self._session is not None:
            raise RuntimeError(f"MCPClient {self.name!r} is already connected")

        stack = AsyncExitStack()
        try:
            # Open the right transport. Each transport context manager returns
            # (read_stream, write_stream); streamable-http also returns a third
            # value (session_id callback) which we don't need.
            # streamable-http
            if isinstance(self.config, StreamableHTTPServer):
                logger.info("connecting to %r via streamable-http: %s", self.name, self.config.url)
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(self.config.url)
                )
            # sse
            elif isinstance(self.config, SSEServer):
                logger.info("connecting to %r via sse: %s", self.name, self.config.url)
                read, write = await stack.enter_async_context(
                    sse_client(self.config.url)
                )
            # stdio
            elif isinstance(self.config, StdioServer):
                logger.info("connecting to %r via stdio: %s", self.name, self.config.command)
                params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                    env=self.config.env or None,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            else:  # pragma: no cover -- exhaustive over the discriminated union
                raise TypeError(f"unknown transport: {type(self.config).__name__}")

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            # Fetch and filter the tool list.
            tool_response = await session.list_tools()
            disabled = set(self.config.disabled_tools)
            self._tools = [
                Tool(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema or {"type": "object", "properties": {}},
                )
                for t in tool_response.tools
                if t.name not in disabled
            ]

            self._session = session
            self._exit_stack = stack
            logger.info("connected to %r: %d tool(s) available", self.name, len(self._tools))
        except Exception:
            # Clean up anything we managed to open before the failure.
            await stack.aclose()
            raise

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """Call a tool by its un-namespaced name."""
        if self._session is None:
            raise RuntimeError(f"MCPClient {self.name!r} is not connected")

        if tool_name in self.config.disabled_tools:
            return ToolCallResult(
                content=f"tool {tool_name!r} is disabled on server {self.name!r}",
                is_error=True,
            )

        try:
            result = await self._session.call_tool(tool_name, arguments)
        except Exception as e:
            logger.exception("tool call %s.%s failed", self.name, tool_name)
            return ToolCallResult(content=f"tool call failed: {e}", is_error=True)

        # MCP tool results are a list of content blocks. For v1 we flatten
        # everything to text; richer handling (images, embedded resources) can
        # come later.
        parts: list[str] = []
        for block in result.content:
            # text blocks have .text; other types we stringify their repr.
            text = getattr(block, "text", None)
            parts.append(text if text is not None else repr(block))

        return ToolCallResult(
            content="\n".join(parts) if parts else "",
            is_error=bool(result.isError),
        )

    async def close(self) -> None:
        """Tear down the session and transport."""
        if self._exit_stack is None:
            return
        try:
            await self._exit_stack.aclose()
        finally:
            self._session = None
            self._exit_stack = None
            self._tools = []
            logger.info("closed connection to %r", self.name)