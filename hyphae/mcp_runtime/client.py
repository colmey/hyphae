"""Task-local MCP connections; :mod:`hyphae.mcp_runtime.manager` owns state."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncContextManager, Protocol

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from hyphae.config import MCPServerConfig, SSEServer, StdioServer, StreamableHTTPServer
from hyphae.tooling import ToolCallResult

logger = logging.getLogger(__name__)

_NON_TEXT_MAX_CHARS = 2_000
_NON_TEXT_RESULT_MAX_CHARS = 20_000
_OMITTED = "[omitted]"
_ADDITIONAL_NON_TEXT_OMITTED = "[additional non-text content omitted]"


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool exposed by an MCP server (un-namespaced)."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


class MCPTransportError(RuntimeError):
    """Provider-neutral wrapper for an SDK call-boundary failure."""

    def __init__(self, cause: Exception) -> None:
        super().__init__("MCP transport/protocol failure during tool call")
        self.cause = cause


def _bounded(value: str, limit: int = _NON_TEXT_MAX_CHARS) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 14]}...[truncated]"


def _render_non_text(block: Any) -> str:
    """Render metadata only; never copy binary/blob resource payloads."""
    block_type = getattr(block, "type", None) or type(block).__name__
    metadata: dict[str, Any] = {"type": str(block_type)}
    for source, target in (
        ("name", "name"),
        ("title", "title"),
        ("uri", "uri"),
        ("mimeType", "mime_type"),
        ("mime_type", "mime_type"),
        ("size", "size"),
    ):
        value = getattr(block, source, None)
        if value is not None and target not in metadata:
            metadata[target] = _bounded(str(value), 512)
    for payload_name in ("data", "blob", "resource"):
        if getattr(block, payload_name, None) is not None:
            metadata[payload_name] = _OMITTED
    try:
        rendered = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):  # pragma: no cover - values are normalized above.
        rendered = json.dumps({"type": type(block).__name__}, sort_keys=True)
    return _bounded(rendered)


class MCPConnection:
    """One initialized task-local SDK session."""

    def __init__(
        self,
        name: str,
        config: MCPServerConfig,
        session: ClientSession,
    ) -> None:
        self._name = name
        self._config = config
        self._session = session

    async def list_tools(self) -> list[Tool]:
        """Fetch the raw live tool list for manager-owned validation."""
        response = await self._session.list_tools()
        return [
            Tool(
                name=tool.name,
                description=(tool.description if tool.description is not None else ""),
                input_schema=(
                    tool.inputSchema
                    if tool.inputSchema is not None
                    else {"type": "object", "properties": {}}
                ),
            )
            for tool in response.tools
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult:
        """Call one live, un-namespaced tool without replay."""
        if tool_name in self._config.disabled_tools:
            return ToolCallResult(
                content=(f"tool {tool_name!r} is disabled on server {self._name!r}"),
                is_error=True,
            )
        try:
            result = await self._session.call_tool(tool_name, arguments)
        except Exception as exc:
            logger.warning(
                "tool call %s.%s failed (%s)",
                self._name,
                tool_name,
                type(exc).__name__,
            )
            raise MCPTransportError(exc) from exc

        try:
            parts: list[str] = []
            non_text_chars = 0
            for block in result.content:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                rendered = _render_non_text(block)
                separator_chars = 1 if parts else 0
                if (
                    non_text_chars + separator_chars + len(rendered)
                    > _NON_TEXT_RESULT_MAX_CHARS
                ):
                    parts.append(_ADDITIONAL_NON_TEXT_OMITTED)
                    break
                parts.append(rendered)
                non_text_chars += separator_chars + len(rendered)
            return ToolCallResult(
                content="\n".join(parts) if parts else "",
                is_error=bool(result.isError),
            )
        except Exception as exc:
            logger.warning(
                "tool result %s.%s could not be decoded (%s)",
                self._name,
                tool_name,
                type(exc).__name__,
            )
            raise MCPTransportError(exc) from exc


class ManagedConnection(Protocol):
    """Connection operations consumed by discovery and turn leases."""

    async def list_tools(self) -> list[Tool]: ...

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult: ...


class ManagedClient(Protocol):
    """Factory for independently owned task-local MCP connections."""

    def open(self) -> AsyncContextManager[ManagedConnection]: ...


ClientFactory = Callable[[str, MCPServerConfig], ManagedClient]


class MCPClient:
    """Create independent task-local connections for one configured server."""

    def __init__(self, name: str, server_config: MCPServerConfig) -> None:
        self.name = name
        self.config = server_config

    @asynccontextmanager
    async def open(self) -> AsyncIterator[MCPConnection]:
        """Enter transport and session contexts, and exit them in this task."""
        transport = self.config.transport
        async with AsyncExitStack() as stack:
            if isinstance(self.config, StreamableHTTPServer):
                logger.info("connecting to MCP server %r via %s", self.name, transport)
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(self.config.url)
                )
            elif isinstance(self.config, SSEServer):
                logger.info("connecting to MCP server %r via %s", self.name, transport)
                read, write = await stack.enter_async_context(
                    sse_client(self.config.url)
                )
            elif isinstance(self.config, StdioServer):
                logger.info("connecting to MCP server %r via %s", self.name, transport)
                params = StdioServerParameters(
                    command=self.config.command,
                    args=list(self.config.args),
                    env=dict(self.config.env) or None,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            else:  # pragma: no cover - exhaustive discriminated union.
                raise TypeError(f"unknown transport: {type(self.config).__name__}")

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            logger.info("initialized MCP server %r via %s", self.name, transport)
            yield MCPConnection(self.name, self.config, session)
        logger.info("closed MCP server %r via %s", self.name, transport)
