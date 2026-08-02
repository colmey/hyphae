"""MCP catalog discovery, refresh, health, and turn-runtime orchestration."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections.abc import AsyncIterator, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from config import MCPConfig
from tooling import ToolRuntime

from .catalog import CatalogSnapshot, ToolRoute, normalize_tools
from .client import ClientFactory, MCPClient, Tool
from .lease import (
    LeaseCoordinator,
    LeaseServer,
    TurnToolRuntime,
)

logger = logging.getLogger(__name__)

_ERROR_MAX_CHARS = 300
_BACKOFF_INITIAL_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 30.0
_BACKOFF_MAX_EXPONENT = 5
_URL_PATTERN = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s'\"]+", re.IGNORECASE)
_BEARER_PATTERN = re.compile(r"\bBearer\s+[^\s,;]+", re.IGNORECASE)
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"\b(authorization|cookie|api[-_ ]?key|token|password|secret|request[-_ ]?body|body)"
    r"\b\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)


class MCPServerState(str, Enum):
    """Catalog lifecycle state for one configured, enabled MCP server."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class MCPServerStatus:
    """Sanitized immutable status rendered by the health endpoint."""

    name: str
    state: MCPServerState
    last_error: str | None
    tool_count: int
    catalog_revision: int
    last_discovered_at: datetime | None
    next_refresh_at: datetime | None
    active_leases: int


@dataclass
class _ServerRecord:
    server: LeaseServer
    state: MCPServerState = MCPServerState.DISCONNECTED
    last_error: str | None = None
    catalog: CatalogSnapshot = field(default_factory=CatalogSnapshot)
    next_refresh_at: datetime | None = None
    next_refresh_monotonic: float | None = None
    refresh_failures: int = 0
    active_leases: int = 0
    next_attempt: int = 0
    latest_started_attempt: int = 0
    refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _DiscoveryTimeout(TimeoutError):
    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(f"discovery exceeded {timeout_seconds:g} seconds")
        self.timeout_seconds = timeout_seconds


def _sanitize_mcp_error(
    error: BaseException,
    *,
    timeout_seconds: float,
) -> str:
    """Return a useful health-safe summary without transport or secret detail."""
    if isinstance(error, _DiscoveryTimeout):
        return f"connection timed out after {error.timeout_seconds:g} seconds"
    if isinstance(error, TimeoutError):
        return f"connection timed out after {timeout_seconds:g} seconds"
    if isinstance(error, asyncio.CancelledError):
        return "connection was cancelled"

    message = str(error).splitlines()[0].strip()
    message = _URL_PATTERN.sub("[redacted-url]", message)
    message = _BEARER_PATTERN.sub("Bearer [redacted]", message)
    message = _SENSITIVE_VALUE_PATTERN.sub(
        lambda match: f"{match.group(1)}=[redacted]",
        message,
    )
    message = "".join(ch for ch in message if ch.isprintable())
    if len(message) > _ERROR_MAX_CHARS:
        message = f"{message[: _ERROR_MAX_CHARS - 3]}..."
    error_type = type(error).__name__
    return f"{error_type}: {message}" if message else f"{error_type}: connection failed"


class MCPManager:
    """Own immutable catalogs and create isolated turn-local tool runtimes."""

    def __init__(
        self,
        mcp_config: MCPConfig,
        *,
        connect_timeout_seconds: float = 30,
        catalog_ttl_seconds: float = 300,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._connect_timeout_seconds = connect_timeout_seconds
        self._catalog_ttl_seconds = catalog_ttl_seconds
        build_client = client_factory or MCPClient
        self._servers = {
            name: LeaseServer(
                name=name,
                config=server_config,
                client=build_client(name, server_config),
            )
            for name, server_config in mcp_config.enabled_servers().items()
        }
        self._records = {
            name: _ServerRecord(server) for name, server in self._servers.items()
        }
        self._startup_started = False
        self._shutdown_started = False
        self._active_turns: set[TurnToolRuntime] = set()
        self._inflight_tasks: set[asyncio.Task[Any]] = set()
        self._lease_coordinator = LeaseCoordinator(
            connect_timeout_seconds=connect_timeout_seconds,
            shutdown_started=lambda: self._shutdown_started,
            start_attempt=self._start_attempt,
            publish_success=self._publish_success,
            publish_failure=self._publish_failure,
            invalidate_revision=self._invalidate_revision,
            lease_entered=self._lease_entered,
            lease_exited=self._lease_exited,
            sanitize_error=self._sanitize_error,
        )

    def _sanitize_error(self, error: BaseException) -> str:
        return _sanitize_mcp_error(
            error,
            timeout_seconds=self._connect_timeout_seconds,
        )

    def _create_task(
        self,
        operation: Coroutine[Any, Any, Any],
        *,
        name: str,
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(operation, name=name)
        self._inflight_tasks.add(task)
        task.add_done_callback(self._inflight_tasks.discard)
        return task

    def _start_attempt(self, server_name: str) -> int:
        record = self._records[server_name]
        record.next_attempt += 1
        record.latest_started_attempt = record.next_attempt
        return record.next_attempt

    def _publish_success(
        self,
        server_name: str,
        attempt: int,
        routes: tuple[ToolRoute, ...],
    ) -> int | None:
        record = self._records[server_name]
        if (
            self._shutdown_started
            or record.state is MCPServerState.CLOSED
            or attempt != record.latest_started_attempt
        ):
            return None
        now = datetime.now(UTC)
        record.catalog = CatalogSnapshot(
            routes=routes,
            revision=record.catalog.revision + 1,
            discovered_at=now,
            discovered_monotonic=time.monotonic(),
        )
        record.state = MCPServerState.HEALTHY
        record.last_error = None
        record.next_refresh_at = None
        record.next_refresh_monotonic = None
        record.refresh_failures = 0
        return record.catalog.revision

    def _publish_failure(
        self,
        server_name: str,
        attempt: int,
        error: BaseException,
    ) -> None:
        record = self._records[server_name]
        if (
            self._shutdown_started
            or record.state is MCPServerState.CLOSED
            or attempt != record.latest_started_attempt
        ):
            return
        record.catalog = replace(record.catalog, routes=())
        record.state = MCPServerState.UNHEALTHY
        record.last_error = self._sanitize_error(error)
        record.refresh_failures += 1
        exponent = min(record.refresh_failures - 1, _BACKOFF_MAX_EXPONENT)
        base = min(
            _BACKOFF_INITIAL_SECONDS * (2**exponent),
            _BACKOFF_MAX_SECONDS,
        )
        delay = base * random.uniform(0.8, 1.2)
        record.next_refresh_monotonic = time.monotonic() + delay
        record.next_refresh_at = datetime.now(UTC) + timedelta(seconds=delay)

    def _invalidate_revision(
        self,
        server_name: str,
        revision: int | None,
        error: BaseException,
    ) -> None:
        record = self._records[server_name]
        if revision is None or revision != record.catalog.revision:
            return
        attempt = self._start_attempt(server_name)
        self._publish_failure(server_name, attempt, error)

    def _lease_entered(self, server_name: str) -> None:
        self._records[server_name].active_leases += 1

    def _lease_exited(self, server_name: str) -> None:
        record = self._records[server_name]
        if record.active_leases <= 0:
            raise RuntimeError(f"MCP lease count underflow for {server_name!r}")
        record.active_leases -= 1

    async def _discover_once(
        self,
        record: _ServerRecord,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[ToolRoute, ...]:
        server = record.server

        async def operation() -> tuple[ToolRoute, ...]:
            async with server.client.open() as connection:
                tools = await connection.list_tools()
            return normalize_tools(
                server.name,
                server.config.disabled_tools,
                tools,
            )

        if timeout_seconds is not None and timeout_seconds <= 0:
            raise _DiscoveryTimeout(timeout_seconds)
        enabled_timeouts = [
            value
            for value in (self._connect_timeout_seconds, timeout_seconds)
            if value is not None and value > 0
        ]
        if not enabled_timeouts:
            return await operation()
        timeout = min(enabled_timeouts)
        try:
            async with asyncio.timeout(timeout):
                return await operation()
        except TimeoutError as exc:
            raise _DiscoveryTimeout(timeout) from exc

    async def _wait_for_tasks(
        self,
        tasks: Sequence[asyncio.Task[Any]],
        *,
        cancel: bool,
    ) -> bool:
        if cancel:
            for task in tasks:
                if not task.done() and task.cancelling() == 0:
                    task.cancel()
        if not tasks:
            return False
        join = asyncio.gather(*tasks, return_exceptions=True)
        current = asyncio.current_task()
        initial_cancellations = current.cancelling() if current is not None else 0
        interrupted = False
        while not join.done():
            try:
                await asyncio.shield(join)
            except asyncio.CancelledError:
                if join.done():
                    break
                if current is None or current.cancelling() <= initial_cancellations:
                    continue
                current.uncancel()
                interrupted = True
        results = join.result()
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                continue
            if isinstance(result, Exception):
                if initial_cancellations or interrupted:
                    logger.warning(
                        "MCP cleanup task failed while cancellation was pending: %s",
                        self._sanitize_error(result),
                    )
                    continue
                raise result
            if isinstance(result, BaseException):
                raise result
        return interrupted

    async def _cancel_and_wait(
        self,
        tasks: Sequence[asyncio.Task[Any]],
    ) -> bool:
        return await self._wait_for_tasks(tasks, cancel=True)

    async def startup(self) -> None:
        """Discover every enabled server without retaining any connection."""
        if self._startup_started:
            raise RuntimeError("MCPManager.startup() may only be called once")
        self._startup_started = True
        if not self._records:
            logger.warning("no enabled MCP servers in config")
            return

        records = tuple(self._records.values())
        attempts: dict[str, int] = {}
        for record in records:
            server_name = record.server.name
            record.state = MCPServerState.CONNECTING
            record.last_error = None
            attempts[server_name] = self._start_attempt(server_name)

        async def discover(record: _ServerRecord) -> tuple[ToolRoute, ...]:
            return await self._discover_once(record)

        tasks = [
            self._create_task(
                discover(record),
                name=f"mcp-discovery-{record.server.name}",
            )
            for record in records
        ]
        discovery = asyncio.gather(*tasks, return_exceptions=True)
        try:
            results = await asyncio.shield(discovery)
        except asyncio.CancelledError:
            await self._cancel_and_wait(tasks)
            for record in records:
                server_name = record.server.name
                attempt = self._start_attempt(server_name)
                self._publish_failure(
                    server_name,
                    attempt,
                    asyncio.CancelledError("application startup was cancelled"),
                )
                record.last_error = "application startup was cancelled"
            raise

        for record, result in zip(records, results):
            server_name = record.server.name
            attempt = attempts[server_name]
            if isinstance(result, asyncio.CancelledError):
                self._publish_failure(server_name, attempt, result)
                logger.error(
                    "failed to discover %r: %s",
                    server_name,
                    record.last_error,
                )
            elif isinstance(result, Exception):
                self._publish_failure(server_name, attempt, result)
                logger.error(
                    "failed to discover %r: %s",
                    server_name,
                    record.last_error,
                )
            elif isinstance(result, BaseException):
                raise result
            else:
                self._publish_success(server_name, attempt, result)

        logger.info(
            "MCP discovery complete: %d/%d catalogs ready, %d tools available",
            len(self.connected_servers),
            len(records),
            len(self.list_tools()),
        )

    def _refresh_due(self, record: _ServerRecord) -> bool:
        now = time.monotonic()
        if record.state is MCPServerState.CONNECTING:
            return record.refresh_lock.locked()
        if record.state is MCPServerState.UNHEALTHY:
            return (
                record.next_refresh_monotonic is None
                or now >= record.next_refresh_monotonic
            )
        return bool(
            record.state is MCPServerState.HEALTHY
            and self._catalog_ttl_seconds > 0
            and record.catalog.discovered_monotonic is not None
            and now - record.catalog.discovered_monotonic >= self._catalog_ttl_seconds
        )

    async def _refresh_record_if_due(
        self,
        record: _ServerRecord,
        *,
        timeout_seconds: float | None,
    ) -> None:
        async with record.refresh_lock:
            if self._shutdown_started or not self._refresh_due(record):
                return
            record.state = MCPServerState.CONNECTING
            server_name = record.server.name
            attempt = self._start_attempt(server_name)
            try:
                routes = await self._discover_once(
                    record,
                    timeout_seconds=timeout_seconds,
                )
            except asyncio.CancelledError as exc:
                self._publish_failure(server_name, attempt, exc)
                raise
            except Exception as exc:
                self._publish_failure(server_name, attempt, exc)
                logger.warning(
                    "request-driven MCP refresh failed for %r: %s",
                    record.server.name,
                    record.last_error,
                )
            else:
                self._publish_success(server_name, attempt, routes)

    async def _refresh_due_catalogs(
        self,
        *,
        timeout_seconds: float | None,
    ) -> None:
        records = tuple(
            record for record in self._records.values() if self._refresh_due(record)
        )
        tasks = [
            self._create_task(
                self._refresh_record_if_due(
                    record,
                    timeout_seconds=timeout_seconds,
                ),
                name=f"mcp-refresh-{record.server.name}",
            )
            for record in records
        ]
        refresh = asyncio.gather(*tasks) if tasks else None
        try:
            if refresh is not None:
                await asyncio.shield(refresh)
        except asyncio.CancelledError:
            await self._cancel_and_wait(tasks)
            if refresh is not None and refresh.done():
                try:
                    refresh.exception()
                except asyncio.CancelledError:
                    pass
            raise

    @asynccontextmanager
    async def open_turn(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ToolRuntime]:
        """Capture a catalog and own every lazy server lease for one turn."""
        if not self._startup_started:
            raise RuntimeError("MCPManager.startup() must run before open_turn()")
        if self._shutdown_started:
            raise RuntimeError("MCPManager is closed")
        await self._refresh_due_catalogs(timeout_seconds=timeout_seconds)
        if self._shutdown_started:
            raise RuntimeError("MCPManager is closed")
        routes = tuple(
            route
            for record in self._records.values()
            for route in record.catalog.routes
        )
        runtime = TurnToolRuntime(
            self._lease_coordinator,
            routes,
            self._servers,
        )
        self._active_turns.add(runtime)
        primary_error: BaseException | None = None
        try:
            yield runtime
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                try:
                    await runtime.aclose()
                except asyncio.CancelledError:
                    if primary_error is None:
                        raise
                    logger.warning("MCP turn cleanup was cancelled")
                except Exception as cleanup_error:
                    if primary_error is None:
                        raise
                    logger.warning(
                        "MCP turn cleanup failed: %s",
                        self._sanitize_error(cleanup_error),
                    )
            finally:
                self._active_turns.discard(runtime)

    async def shutdown(self) -> None:
        """Close active turn workers and clear the in-memory catalog."""
        self._shutdown_started = True
        refreshes = tuple(self._inflight_tasks)
        interrupted = await self._cancel_and_wait(refreshes)
        turns = tuple(self._active_turns)
        if turns:
            turn_cleanup = [
                asyncio.create_task(
                    runtime._abort(),
                    name="mcp-turn-shutdown",
                )
                for runtime in turns
            ]
            interrupted = (
                await self._wait_for_tasks(turn_cleanup, cancel=False) or interrupted
            )
        self._active_turns.clear()
        for record in self._records.values():
            record.catalog = replace(record.catalog, routes=())
            record.state = MCPServerState.CLOSED
            record.next_refresh_at = None
            record.next_refresh_monotonic = None
        if interrupted:
            raise asyncio.CancelledError

    @property
    def connected_servers(self) -> list[str]:
        """Compatibility view of catalog-ready server names."""
        return [
            record.server.name
            for record in self._records.values()
            if record.state is MCPServerState.HEALTHY
        ]

    def status_snapshot(self) -> tuple[MCPServerStatus, ...]:
        return tuple(
            MCPServerStatus(
                name=record.server.name,
                state=record.state,
                last_error=record.last_error,
                tool_count=len(record.catalog.routes),
                catalog_revision=record.catalog.revision,
                last_discovered_at=record.catalog.discovered_at,
                next_refresh_at=record.next_refresh_at,
                active_leases=record.active_leases,
            )
            for record in self._records.values()
        )

    def list_tools(self) -> list[tuple[str, Tool]]:
        return [
            (
                route.spec.name,
                Tool(
                    route.local_name,
                    route.spec.description,
                    route.spec.input_schema,
                ),
            )
            for record in self._records.values()
            for route in record.catalog.routes
        ]

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [
            route.spec.as_llm_dict()
            for record in self._records.values()
            for route in record.catalog.routes
        ]
