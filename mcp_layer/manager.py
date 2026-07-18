"""Aggregate MCP clients and expose namespaced tools."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from config import MCPConfig, MCPServerConfig

from .client import MCPClient, Tool, ToolCallResult
from .contracts import NAMESPACE_SEP, ToolSpec

logger = logging.getLogger(__name__)

_ERROR_MAX_CHARS = 300
_URL_PATTERN = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s'\"]+", re.IGNORECASE)
_BEARER_PATTERN = re.compile(r"\bBearer\s+[^\s,;]+", re.IGNORECASE)
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"\b(authorization|cookie|api[-_ ]?key|token|password|secret|request[-_ ]?body|body)"
    r"\b\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)


class MCPServerState(str, Enum):
    """Lifecycle state for one configured, enabled MCP server."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass(frozen=True)
class MCPServerStatus:
    """Sanitized immutable status rendered by the health endpoint."""

    name: str
    state: MCPServerState
    last_error: str | None
    tool_count: int


@dataclass
class _ServerRecord:
    name: str
    config: MCPServerConfig
    client: "_ManagedClient"
    state: MCPServerState = MCPServerState.DISCONNECTED
    last_error: str | None = None
    advertised_tool_count: int = 0


@dataclass(frozen=True)
class _ToolRoute:
    server_name: str
    local_name: str
    tool: Tool
    spec: ToolSpec


class _ManagedClient(Protocol):
    """Narrow manager/client seam used by production clients and test fakes."""

    @property
    def tools(self) -> list[Tool]: ...

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallResult: ...


_ClientFactory = Callable[[str, MCPServerConfig], _ManagedClient]


def _sanitize_connection_error(
    error: BaseException,
    *,
    timeout_seconds: float,
) -> str:
    """Return a useful health-safe summary without transport or secret detail."""
    if isinstance(error, TimeoutError):
        return f"connection timed out after {timeout_seconds:g} seconds"
    if isinstance(error, asyncio.CancelledError):
        return "connection was cancelled"

    # Only the first line is retained: SDK exceptions often append request bodies,
    # transport representations, or trace-like detail on following lines.
    message = str(error).splitlines()[0].strip()
    message = _URL_PATTERN.sub("[redacted-url]", message)
    message = _BEARER_PATTERN.sub("Bearer [redacted]", message)
    message = _SENSITIVE_VALUE_PATTERN.sub(
        lambda match: f"{match.group(1)}=[redacted]", message
    )
    message = "".join(ch for ch in message if ch.isprintable())
    if len(message) > _ERROR_MAX_CHARS:
        message = f"{message[: _ERROR_MAX_CHARS - 3]}..."
    error_type = type(error).__name__
    return f"{error_type}: {message}" if message else f"{error_type}: connection failed"


class MCPManager:
    """Aggregates MCPClient instances and presents a unified tool registry."""

    def __init__(
        self,
        mcp_config: MCPConfig,
        *,
        connect_timeout_seconds: float = 30,
        client_factory: _ClientFactory | None = None,
    ) -> None:
        self._connect_timeout_seconds = connect_timeout_seconds
        build_client = client_factory or MCPClient
        self._records: dict[str, _ServerRecord] = {
            name: _ServerRecord(
                name=name,
                config=server_config,
                client=build_client(name, server_config),
            )
            for name, server_config in mcp_config.enabled_servers().items()
        }
        self._tool_index: dict[str, _ToolRoute] = {}

    async def _connect(self, record: _ServerRecord) -> None:
        if self._connect_timeout_seconds <= 0:
            await record.client.connect()
            return
        async with asyncio.timeout(self._connect_timeout_seconds):
            await record.client.connect()

    @staticmethod
    async def _cancel_and_wait(tasks: Sequence[asyncio.Task[Any]]) -> None:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _close_records(
        self,
        records: Sequence[_ServerRecord],
        *,
        context: str,
    ) -> None:
        if not records:
            return
        tasks = [asyncio.create_task(record.client.close()) for record in records]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            await self._cancel_and_wait(tasks)
            raise
        for record, result in zip(records, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "%s for MCP server %r: %s",
                    context,
                    record.name,
                    type(result).__name__,
                )

    def _clear_inventory(self) -> None:
        self._tool_index.clear()
        for record in self._records.values():
            record.advertised_tool_count = 0

    @staticmethod
    def _mark_unhealthy(record: _ServerRecord, last_error: str) -> None:
        record.state = MCPServerState.UNHEALTHY
        record.last_error = last_error
        record.advertised_tool_count = 0

    def _publish_inventory(self, record: _ServerRecord) -> None:
        published_names: set[str] = set()
        for tool in record.client.tools:
            namespaced = f"{record.name}{NAMESPACE_SEP}{tool.name}"
            spec = ToolSpec.from_mapping(
                {
                    "name": namespaced,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            )
            self._tool_index[namespaced] = _ToolRoute(
                server_name=record.name,
                local_name=tool.name,
                tool=tool,
                spec=spec,
            )
            published_names.add(namespaced)
        record.advertised_tool_count = len(published_names)

    async def _cancel_startup(
        self,
        tasks: list[asyncio.Task[None]],
    ) -> None:
        await self._cancel_and_wait(tasks)

        self._clear_inventory()
        records = list(self._records.values())
        for record in records:
            self._mark_unhealthy(record, "application startup was cancelled")

        await self._close_records(
            records,
            context="cleanup after cancelled MCP startup failed",
        )

    async def startup(self) -> None:
        """Connect enabled servers concurrently while retaining every outcome."""
        if not self._records:
            logger.warning("no enabled MCP servers in config")
            return
        if any(
            record.state is not MCPServerState.DISCONNECTED
            for record in self._records.values()
        ):
            raise RuntimeError("MCPManager.startup() may only be called once")

        records = list(self._records.values())
        for record in records:
            record.state = MCPServerState.CONNECTING
            record.last_error = None
            record.advertised_tool_count = 0

        tasks = [asyncio.create_task(self._connect(record)) for record in records]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            await self._cancel_startup(tasks)
            raise

        failed: list[_ServerRecord] = []
        for record, result in zip(records, results):
            if isinstance(result, BaseException):
                self._mark_unhealthy(
                    record,
                    _sanitize_connection_error(
                        result,
                        timeout_seconds=self._connect_timeout_seconds,
                    ),
                )
                failed.append(record)
                logger.error("failed to connect to %r: %s", record.name, record.last_error)
                continue

            record.state = MCPServerState.HEALTHY
            record.last_error = None
            self._publish_inventory(record)

        try:
            await self._close_records(
                failed,
                context="cleanup after failed MCP startup failed",
            )
        except asyncio.CancelledError:
            await self._cancel_startup(tasks)
            raise

        logger.info(
            "MCP startup complete: %d/%d servers healthy, %d tools available",
            len(self.connected_servers),
            len(records),
            len(self._tool_index),
        )

    async def shutdown(self) -> None:
        """Close every retained client and clear inventory; safe to repeat."""
        records = [
            record
            for record in self._records.values()
            if record.state is not MCPServerState.CLOSED
        ]
        if not records:
            return

        self._clear_inventory()
        for record in records:
            record.state = MCPServerState.CLOSED

        await self._close_records(records, context="MCP shutdown cleanup failed")

    @property
    def connected_servers(self) -> list[str]:
        """Compatibility view containing healthy server names only."""
        return [
            record.name
            for record in self._records.values()
            if record.state is MCPServerState.HEALTHY
        ]

    def status_snapshot(self) -> tuple[MCPServerStatus, ...]:
        """Return configuration-ordered, health-safe server status values."""
        return tuple(
            MCPServerStatus(
                name=record.name,
                state=record.state,
                last_error=record.last_error,
                tool_count=record.advertised_tool_count,
            )
            for record in self._records.values()
        )

    def list_tools(self) -> list[tuple[str, Tool]]:
        """Return the existing namespaced-name/Tool compatibility shape."""
        return [(namespaced, route.tool) for namespaced, route in self._tool_index.items()]

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        """Return only healthy advertised tools in provider-neutral shape."""
        return [route.spec.as_llm_dict() for route in self._tool_index.values()]

    async def call_tool(
        self, namespaced_name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Route a namespaced tool call to the healthy owning server."""
        route = self._tool_index.get(namespaced_name)
        if route is None:
            return ToolCallResult(
                content=f"unknown tool: {namespaced_name!r}",
                is_error=True,
            )
        record = self._records[route.server_name]
        if record.state is not MCPServerState.HEALTHY:
            return ToolCallResult(
                content=f"server {route.server_name!r} is not connected",
                is_error=True,
            )
        return await record.client.call_tool(route.local_name, arguments)
