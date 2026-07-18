# mcp_layer/client.py

"""Per-server MCP client wrapper; MCPManager handles aggregation/namespacing."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from config import MCPServerConfig, SSEServer, StdioServer, StreamableHTTPServer

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
        """Open transport, initialize the session, and fetch tools."""
        if self._session is not None:
            raise RuntimeError(f"MCPClient {self.name!r} is already connected")

        stack = AsyncExitStack()
        connected = False
        primary_error: BaseException | None = None
        try:
            if isinstance(self.config, StreamableHTTPServer):
                logger.info(
                    "connecting to %r via streamable-http: %s",
                    self.name,
                    self.config.url,
                )
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(self.config.url)
                )
            elif isinstance(self.config, SSEServer):
                logger.info("connecting to %r via sse: %s", self.name, self.config.url)
                read, write = await stack.enter_async_context(
                    sse_client(self.config.url)
                )
            elif isinstance(self.config, StdioServer):
                logger.info(
                    "connecting to %r via stdio: %s", self.name, self.config.command
                )
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
            connected = True
            logger.info(
                "connected to %r: %d tool(s) available", self.name, len(self._tools)
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            # CancelledError from failed transports is BaseException, so cleanup
            # must live in finally rather than `except Exception`.
            if not connected:
                try:
                    await stack.aclose()
                except asyncio.CancelledError:
                    # A new application cancellation takes precedence over an
                    # ordinary connection error. If cancellation was already the
                    # primary outcome, preserve that original cancellation.
                    if not isinstance(primary_error, asyncio.CancelledError):
                        raise
                    logger.warning(
                        "cleanup after cancelled connection to %r was cancelled",
                        self.name,
                    )
                except BaseException:
                    if primary_error is None:
                        raise
                    # Cleanup failure must not replace a timeout, connection
                    # exception, or application cancellation already in flight.
                    logger.warning(
                        "cleanup after failed connection to %r failed",
                        self.name,
                        exc_info=True,
                    )

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
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

        parts: list[str] = []
        for block in result.content:
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
