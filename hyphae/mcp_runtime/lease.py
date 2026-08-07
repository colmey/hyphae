"""Lazy turn-local MCP leases with task-affine connection ownership."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from enum import Enum
from typing import Any

from hyphae.config import MCPServerConfig
from hyphae.tooling import ToolCallResult

from .catalog import ToolRoute, normalize_tools, validation_error
from .client import (
    MCPTransportError,
    ManagedClient,
    ManagedConnection,
)

logger = logging.getLogger(__name__)

_OUTCOME_UNKNOWN = (
    "tool call outcome is unknown after a transport/protocol failure; "
    "the call was not replayed"
)


@dataclass(frozen=True, slots=True)
class LeaseServer:
    """Immutable server dependencies required by one lease worker."""

    name: str
    config: MCPServerConfig
    client: ManagedClient


@dataclass(frozen=True, slots=True)
class LeaseCoordinator:
    """Narrow catalog and health callbacks consumed by lease workers."""

    connect_timeout_seconds: float
    shutdown_started: Callable[[], bool]
    start_attempt: Callable[[str], int]
    publish_success: Callable[[str, int, tuple[ToolRoute, ...]], int | None]
    publish_failure: Callable[[str, int, BaseException], None]
    invalidate_revision: Callable[[str, int | None, BaseException], None]
    lease_entered: Callable[[str], None]
    lease_exited: Callable[[str], None]
    sanitize_error: Callable[[BaseException], str]


@dataclass(slots=True)
class _LeaseRequest:
    namespaced_name: str
    arguments: dict[str, Any]
    future: asyncio.Future[ToolCallResult]


class _LeaseCommand(Enum):
    CLOSE = "close"


class _ServerLeaseWorker:
    """One server connection whose SDK lifetime belongs to its worker task."""

    def __init__(
        self,
        coordinator: LeaseCoordinator,
        server: LeaseServer,
    ) -> None:
        self._coordinator = coordinator
        self._server = server
        self._queue: asyncio.Queue[_LeaseRequest | _LeaseCommand] = asyncio.Queue()
        self._task = asyncio.create_task(
            self._run(),
            name=f"mcp-lease-{server.name}",
        )
        self._closed = False
        self._abort_requested = False
        self._cleanup_started = False
        self._lease_entered = False
        self._terminal_result: ToolCallResult | None = None

    async def call(
        self,
        namespaced_name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult:
        if self._terminal_result is not None:
            return self._terminal_result
        if self._closed:
            return ToolCallResult(
                content=f"server {self._server.name!r} turn lease is closed",
                is_error=True,
            )
        future: asyncio.Future[ToolCallResult] = (
            asyncio.get_running_loop().create_future()
        )
        self._queue.put_nowait(_LeaseRequest(namespaced_name, arguments, future))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            try:
                await self.abort()
            except Exception as cleanup_error:
                logger.warning(
                    "MCP call cleanup failed for %s (%s)",
                    type(self._server.client).__name__,
                    type(cleanup_error).__name__,
                )
            raise

    def _settle_pending(self, result: ToolCallResult) -> None:
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if isinstance(item, _LeaseRequest) and not item.future.done():
                item.future.set_result(result)

    async def _open_connection(
        self,
        stack: AsyncExitStack,
    ) -> tuple[ManagedConnection, tuple[ToolRoute, ...]]:
        connection = await stack.enter_async_context(self._server.client.open())
        self._lease_entered = True
        self._coordinator.lease_entered(self._server.name)
        tools = await connection.list_tools()
        return connection, normalize_tools(
            self._server.name,
            self._server.config.disabled_tools,
            tools,
        )

    async def _dispatch_request(
        self,
        connection: ManagedConnection,
        live_routes: Mapping[str, ToolRoute],
        request: _LeaseRequest,
        revision: int | None,
    ) -> ToolCallResult:
        route = live_routes.get(request.namespaced_name)
        if route is None:
            return ToolCallResult(
                content=(
                    "tool_catalog_changed: tool "
                    f"{request.namespaced_name!r} is no longer available"
                ),
                is_error=True,
            )
        changed_error = validation_error(route, request.arguments)
        if changed_error is not None:
            return ToolCallResult(
                content=f"tool_catalog_changed: {changed_error}",
                is_error=True,
            )
        try:
            return await connection.call_tool(route.local_name, request.arguments)
        except asyncio.CancelledError:
            self._coordinator.invalidate_revision(
                self._server.name,
                revision,
                RuntimeError("active tool call was cancelled"),
            )
            self._terminal_result = ToolCallResult(_OUTCOME_UNKNOWN, True)
            raise
        except MCPTransportError as exc:
            self._coordinator.invalidate_revision(
                self._server.name,
                revision,
                exc.cause,
            )
            self._terminal_result = ToolCallResult(_OUTCOME_UNKNOWN, True)
            return self._terminal_result

    async def _run(self) -> None:
        server_name = self._server.name
        attempt = self._coordinator.start_attempt(server_name)
        stack = AsyncExitStack()
        revision: int | None = None
        current: _LeaseRequest | None = None
        primary_error: BaseException | None = None
        cleanup_cancelled = False
        try:
            timeout = self._coordinator.connect_timeout_seconds
            if timeout > 0:
                async with asyncio.timeout(timeout):
                    connection, routes = await self._open_connection(stack)
            else:
                connection, routes = await self._open_connection(stack)

            revision = self._coordinator.publish_success(
                server_name,
                attempt,
                routes,
            )
            if self._coordinator.shutdown_started():
                self._terminal_result = ToolCallResult(
                    content=f"server {server_name!r} is closed",
                    is_error=True,
                )
                return
            live_routes = {route.spec.name: route for route in routes}

            while True:
                item = await self._queue.get()
                if item is _LeaseCommand.CLOSE:
                    return
                assert isinstance(item, _LeaseRequest)
                current = item
                result = await self._dispatch_request(
                    connection,
                    live_routes,
                    item,
                    revision,
                )
                if not item.future.done():
                    item.future.set_result(result)
                current = None
                if self._terminal_result is not None:
                    return
        except asyncio.CancelledError as exc:
            primary_error = exc
            if self._terminal_result is None:
                self._terminal_result = ToolCallResult(
                    content=f"server {server_name!r} turn lease was cancelled",
                    is_error=True,
                )
            if current is not None and not current.future.done():
                current.future.set_result(self._terminal_result)
            raise
        except Exception as exc:
            primary_error = exc
            self._coordinator.publish_failure(server_name, attempt, exc)
            self._terminal_result = ToolCallResult(
                content=f"server {server_name!r} is unavailable; lease discovery failed",
                is_error=True,
            )
            if current is not None and not current.future.done():
                current.future.set_result(self._terminal_result)
        finally:
            self._cleanup_started = True
            try:
                await stack.aclose()
            except asyncio.CancelledError as cleanup_error:
                cleanup_cancelled = True
                if primary_error is None:
                    raise
                logger.warning(
                    "MCP lease cleanup was cancelled for %s (%s)",
                    type(self._server.client).__name__,
                    type(cleanup_error).__name__,
                )
            except Exception as cleanup_error:
                if primary_error is None:
                    self._coordinator.invalidate_revision(
                        server_name,
                        revision,
                        cleanup_error,
                    )
                    self._terminal_result = ToolCallResult(
                        content=f"server {server_name!r} lease cleanup failed",
                        is_error=True,
                    )
                else:
                    logger.warning(
                        "MCP lease cleanup failed for %s (%s)",
                        type(self._server.client).__name__,
                        type(cleanup_error).__name__,
                    )
            finally:
                if self._lease_entered and not cleanup_cancelled:
                    self._coordinator.lease_exited(server_name)
            terminal = self._terminal_result or ToolCallResult(
                content=f"server {server_name!r} turn lease is closed",
                is_error=True,
            )
            self._settle_pending(terminal)

    async def _join(self) -> None:
        """Wait for the owner task without forwarding a second cancellation."""
        interrupted = False
        current = asyncio.current_task()
        while not self._task.done():
            try:
                await asyncio.shield(self._task)
            except asyncio.CancelledError:
                if self._task.done():
                    break
                if current is None or current.cancelling() == 0:
                    continue
                current.uncancel()
                interrupted = True
        if not self._task.cancelled():
            error = self._task.exception()
            if error is not None:
                raise error
        if interrupted:
            raise asyncio.CancelledError

    async def close(self) -> None:
        if self._closed:
            await self._join()
            return
        self._closed = True
        if not self._task.done():
            self._queue.put_nowait(_LeaseCommand.CLOSE)
        await self._join()

    async def abort(self) -> None:
        self._closed = True
        if (
            not self._abort_requested
            and not self._cleanup_started
            and not self._task.done()
        ):
            self._abort_requested = True
            self._task.cancel()
        await self._join()


class TurnToolRuntime:
    """Immutable turn catalog plus lazy, isolated server leases."""

    def __init__(
        self,
        coordinator: LeaseCoordinator,
        routes: tuple[ToolRoute, ...],
        servers: Mapping[str, LeaseServer],
    ) -> None:
        self._coordinator = coordinator
        self._routes = routes
        self._servers = dict(servers)
        self._route_index = {route.spec.name: route for route in routes}
        self._workers: dict[str, _ServerLeaseWorker] = {}
        self._worker_lock = asyncio.Lock()
        self._closed = False

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [route.spec.as_llm_dict() for route in self._routes]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult:
        async with self._worker_lock:
            if self._closed:
                return ToolCallResult(
                    content="turn tool runtime is closed",
                    is_error=True,
                )
            route = self._route_index.get(name)
            if route is None:
                return ToolCallResult(content=f"unknown tool: {name!r}", is_error=True)
            worker = self._workers.get(route.server_name)
            if worker is None:
                worker = _ServerLeaseWorker(
                    self._coordinator,
                    self._servers[route.server_name],
                )
                self._workers[route.server_name] = worker
        return await worker.call(name, arguments)

    async def aclose(self) -> None:
        async with self._worker_lock:
            if self._closed:
                return
            self._closed = True
            workers = tuple(self._workers.values())
        if workers:
            await asyncio.gather(*(worker.close() for worker in workers))

    async def _abort(self) -> None:
        async with self._worker_lock:
            self._closed = True
            workers = tuple(self._workers.values())
        if workers:
            await asyncio.gather(*(worker.abort() for worker in workers))
