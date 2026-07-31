"""Plan 06 regression coverage for bounded MCP startup and truthful health."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import mcp_layer.client as client_module
from config import MCPConfig
from mcp_layer import MCPManager, MCPServerState, Tool, ToolSpec
from tests._app_support import wired_app

pytestmark = pytest.mark.anyio


def _config(*names: str, disabled: tuple[str, ...] = ()) -> MCPConfig:
    return MCPConfig.model_validate(
        {
            "mcpServers": {
                name: {
                    "transport": "streamable-http",
                    "url": f"http://{name}.invalid/mcp",
                    "disabled": name in disabled,
                }
                for name in names
            }
        }
    )


class _FakeClient:
    def __init__(
        self,
        name: str,
        *,
        tools: list[Tool] | None = None,
        connect_error: BaseException | None = None,
        blocker: asyncio.Event | None = None,
        close_error: BaseException | None = None,
        close_blocker: asyncio.Event | None = None,
    ) -> None:
        self.name = name
        self.tools = tools or []
        self.connect_error = connect_error
        self.blocker = blocker
        self.close_error = close_error
        self.close_blocker = close_blocker
        self.connect_started = asyncio.Event()
        self.connect_cancelled = False
        self.close_started = asyncio.Event()
        self.close_cancelled = False
        self.close_calls = 0

    async def connect(self) -> None:
        self.connect_started.set()
        try:
            if self.blocker is not None:
                await self.blocker.wait()
            if self.connect_error is not None:
                raise self.connect_error
        except asyncio.CancelledError:
            self.connect_cancelled = True
            raise

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        try:
            if self.close_blocker is not None:
                await self.close_blocker.wait()
            if self.close_error is not None:
                raise self.close_error
        except asyncio.CancelledError:
            self.close_cancelled = True
            raise

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("tool execution is outside Plan 06 startup tests")


def _manager(
    config: MCPConfig,
    clients: dict[str, _FakeClient],
) -> MCPManager:
    return MCPManager(
        config,
        client_factory=lambda name, server_config: clients[name],
    )


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"{name} description",
        input_schema={"type": "object", "properties": {}},
    )


@pytest.mark.parametrize(
    ("phase", "cleanup_raises"),
    [
        ("transport", False),
        ("initialize", False),
        ("list_tools", False),
        ("initialize", True),
    ],
    ids=["transport", "initialize", "list-tools", "cleanup-failure"],
)
async def test_one_timeout_bounds_every_real_client_startup_phase(
    phase: str,
    cleanup_raises: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    blocker = asyncio.Event()
    cleanup = {"transport": False, "session": False}

    class _TransportContext:
        async def __aenter__(self):
            if phase == "transport":
                entered.set()
                try:
                    await blocker.wait()
                finally:
                    cleanup["transport"] = True
            return object(), object(), None

        async def __aexit__(self, exc_type, exc, tb):
            cleanup["transport"] = True

    class _Session:
        def __init__(self, read: Any, write: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            cleanup["session"] = True
            if cleanup_raises:
                raise RuntimeError("session cleanup failed")

        async def initialize(self) -> None:
            if phase == "initialize":
                entered.set()
                await blocker.wait()

        async def list_tools(self):
            if phase == "list_tools":
                entered.set()
                await blocker.wait()
            return SimpleNamespace(tools=[])

    monkeypatch.setattr(client_module, "streamablehttp_client", lambda url: _TransportContext())
    monkeypatch.setattr(client_module, "ClientSession", _Session)

    manager = MCPManager(_config("phase-test"), connect_timeout_seconds=0.05)
    await manager.startup()

    assert entered.is_set()
    assert manager.connected_servers == []
    assert manager.list_tools() == []
    status = manager.status_snapshot()[0]
    assert status.state is MCPServerState.UNHEALTHY
    assert status.last_error == "connection timed out after 0.05 seconds"
    assert status.tool_count == 0
    assert cleanup["transport"] is True
    assert cleanup["session"] is (phase != "transport")


async def test_partial_startup_retains_truthful_order_and_sanitizes_errors(
) -> None:
    secret_error = RuntimeError(
        "connect failed Authorization: Bearer topsecret "
        "password=hunter2 at https://user:pass@example.invalid/mcp?token=hidden\n"
        "request_body={'private': 'payload'}"
    )
    clients = {
        "healthy": _FakeClient("healthy", tools=[_tool("one"), _tool("two")]),
        "broken": _FakeClient(
            "broken",
            connect_error=secret_error,
            close_error=RuntimeError("cleanup also failed"),
        ),
    }
    manager = _manager(_config("healthy", "broken"), clients)

    assert [status.state for status in manager.status_snapshot()] == [
        MCPServerState.DISCONNECTED,
        MCPServerState.DISCONNECTED,
    ]
    await manager.startup()

    assert manager.connected_servers == ["healthy"]
    assert [name for name, _tool_value in manager.list_tools()] == [
        "healthy__one",
        "healthy__two",
    ]
    assert all(isinstance(route.spec, ToolSpec) for route in manager._tool_index.values())
    statuses = manager.status_snapshot()
    assert [status.name for status in statuses] == ["healthy", "broken"]
    assert statuses[0].state is MCPServerState.HEALTHY
    assert statuses[0].tool_count == 2
    assert statuses[1].state is MCPServerState.UNHEALTHY
    assert statuses[1].tool_count == 0
    assert statuses[1].last_error is not None
    assert "RuntimeError: connect failed" in statuses[1].last_error
    for secret in ("topsecret", "hunter2", "user:pass", "hidden", "payload"):
        assert secret not in statuses[1].last_error
    assert clients["broken"].close_calls == 1


async def test_independent_child_cancellation_is_unhealthy_not_registered(
) -> None:
    clients = {
        "healthy": _FakeClient("healthy", tools=[_tool("available")]),
        "cancelled": _FakeClient(
            "cancelled",
            tools=[_tool("must-not-leak")],
            connect_error=asyncio.CancelledError("child only"),
        ),
    }
    manager = _manager(_config("healthy", "cancelled"), clients)

    await manager.startup()

    assert manager.connected_servers == ["healthy"]
    assert [name for name, _ in manager.list_tools()] == ["healthy__available"]
    cancelled = manager.status_snapshot()[1]
    assert cancelled.state is MCPServerState.UNHEALTHY
    assert cancelled.last_error == "connection was cancelled"
    assert cancelled.tool_count == 0


async def test_application_cancellation_propagates_and_cleans_every_sibling(
) -> None:
    blocker = asyncio.Event()
    clients = {
        "completed": _FakeClient("completed", tools=[_tool("not-published")]),
        "blocked": _FakeClient("blocked", blocker=blocker),
    }
    manager = _manager(_config("completed", "blocked"), clients)
    startup = asyncio.create_task(manager.startup())
    await clients["blocked"].connect_started.wait()

    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup

    assert clients["blocked"].connect_cancelled is True
    assert clients["completed"].close_calls == 1
    assert clients["blocked"].close_calls == 1
    assert manager.connected_servers == []
    assert manager.list_tools() == []
    assert all(
        status.state is MCPServerState.UNHEALTHY
        and status.last_error == "application startup was cancelled"
        and status.tool_count == 0
        for status in manager.status_snapshot()
    )


@pytest.mark.parametrize("timeout", [0, -1])
async def test_nonpositive_timeout_is_disabled(
    timeout: float,
) -> None:
    blocker = asyncio.Event()
    client = _FakeClient("waiting", blocker=blocker)
    manager = MCPManager(
        _config("waiting"),
        connect_timeout_seconds=timeout,
        client_factory=lambda name, server_config: client,
    )
    startup = asyncio.create_task(manager.startup())
    await client.connect_started.wait()
    await asyncio.sleep(0.02)

    assert not startup.done()
    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup
    assert client.close_calls == 1


async def test_shutdown_attempts_all_clients_clears_inventory_and_is_idempotent(
) -> None:
    clients = {
        "close-fails": _FakeClient(
            "close-fails",
            tools=[_tool("one")],
            close_error=RuntimeError("close failed"),
        ),
        "closes": _FakeClient("closes", tools=[_tool("two")]),
    }
    manager = _manager(_config("close-fails", "closes"), clients)
    await manager.startup()

    await manager.shutdown()
    await manager.shutdown()

    assert clients["close-fails"].close_calls == 1
    assert clients["closes"].close_calls == 1
    assert manager.connected_servers == []
    assert manager.list_tools() == []
    assert all(
        status.state is MCPServerState.CLOSED and status.tool_count == 0
        for status in manager.status_snapshot()
    )


async def test_shutdown_cancellation_propagates_after_cancelling_close_siblings() -> None:
    blocker = asyncio.Event()
    clients = {
        "first": _FakeClient("first", tools=[_tool("one")], close_blocker=blocker),
        "second": _FakeClient("second", tools=[_tool("two")], close_blocker=blocker),
    }
    manager = _manager(_config("first", "second"), clients)
    await manager.startup()
    shutdown = asyncio.create_task(manager.shutdown())
    await asyncio.gather(*(client.close_started.wait() for client in clients.values()))

    shutdown.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown

    assert all(client.close_calls == 1 for client in clients.values())
    assert all(client.close_cancelled for client in clients.values())
    assert manager.connected_servers == []
    assert manager.list_tools() == []
    assert all(
        status.state is MCPServerState.CLOSED and status.tool_count == 0
        for status in manager.status_snapshot()
    )


async def test_repeated_shutdown_retries_cleanup_not_started_before_cancellation(
) -> None:
    client = _FakeClient("server", tools=[_tool("one")])
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()
    record = manager._records["server"]
    await record.lock.acquire()
    try:
        shutdown = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)
        assert manager.status_snapshot()[0].state is MCPServerState.CLOSED
        assert client.close_started.is_set() is False

        shutdown.cancel()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
    finally:
        record.lock.release()

    await manager.shutdown()

    assert client.close_calls == 1
    assert client.close_started.is_set() is True
    assert manager.status_snapshot()[0].state is MCPServerState.CLOSED


async def test_no_enabled_servers_has_empty_ok_health(
    asgi_client,
) -> None:
    manager = MCPManager(_config("disabled", disabled=("disabled",)))
    await manager.startup()

    with wired_app(SimpleNamespace(), mcp=manager) as (app, _settings):
        response = await asgi_client(app).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mcp_servers"] == []
    assert body["connected_servers"] == []
    assert body["tool_count"] == 0


async def test_health_reports_mixed_startup_and_preserves_compatibility_fields(
    asgi_client,
) -> None:
    clients = {
        "healthy": _FakeClient("healthy", tools=[_tool("one")]),
        "broken": _FakeClient("broken", connect_error=RuntimeError("offline")),
    }
    manager = _manager(_config("healthy", "broken"), clients)
    await manager.startup()

    with wired_app(SimpleNamespace(), mcp=manager) as (app, settings):
        response = await asgi_client(app).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["provider"] == settings.llm.provider
    assert body["model"] == settings.llm.model_name
    assert body["connected_servers"] == ["healthy"]
    assert body["tool_count"] == 1
    assert body["orchestration_enabled"] is False
    assert body["available_model_ids"] == []
    assert body["mcp_servers"] == [
        {
            "name": "healthy",
            "state": "healthy",
            "last_error": None,
            "tool_count": 1,
        },
        {
            "name": "broken",
            "state": "unhealthy",
            "last_error": "RuntimeError: offline",
            "tool_count": 0,
        },
    ]


async def test_health_is_ok_when_every_enabled_server_is_healthy(
    asgi_client,
) -> None:
    client = _FakeClient("healthy", tools=[_tool("one")])
    manager = _manager(_config("healthy"), {"healthy": client})
    await manager.startup()

    with wired_app(SimpleNamespace(), mcp=manager) as (app, _settings):
        response = await asgi_client(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
