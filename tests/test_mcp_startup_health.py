"""Bounded MCP discovery, same-task cleanup, and idle-ready health."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

import mcp_runtime.client as client_module
from config import MCPConfig
from mcp_runtime import MCPManager, MCPServerState, Tool
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


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"{name} description",
        input_schema={"type": "object", "properties": {}},
    )


@dataclass
class _OpenStep:
    tools: list[Tool] = field(default_factory=list)
    enter_error: BaseException | None = None
    list_error: BaseException | None = None
    enter_blocker: asyncio.Event | None = None
    list_blocker: asyncio.Event | None = None
    cleanup_error: BaseException | None = None
    cleanup_blocker: asyncio.Event | None = None
    enter_started: asyncio.Event = field(default_factory=asyncio.Event)
    list_started: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_started: asyncio.Event = field(default_factory=asyncio.Event)


class _FakeProvider:
    def __init__(self, name: str, steps: list[_OpenStep]) -> None:
        self.name = name
        self.steps = deque(steps)
        self.open_calls = 0
        self.exit_calls = 0
        self.active_contexts = 0
        self.enter_tasks: list[asyncio.Task[Any]] = []
        self.exit_tasks: list[asyncio.Task[Any]] = []

    def open(self):
        provider = self
        if not self.steps:
            raise AssertionError(f"unexpected open for {self.name}")
        step = self.steps.popleft()

        class _Connection:
            async def list_tools(self) -> list[Tool]:
                step.list_started.set()
                if step.list_blocker is not None:
                    await step.list_blocker.wait()
                if step.list_error is not None:
                    raise step.list_error
                return step.tools

            async def call_tool(self, name: str, arguments: dict[str, Any]):
                raise AssertionError("startup discovery must not dispatch tools")

        class _Context:
            async def __aenter__(self) -> _Connection:
                provider.open_calls += 1
                task = asyncio.current_task()
                assert task is not None
                provider.enter_tasks.append(task)
                step.enter_started.set()
                if step.enter_blocker is not None:
                    await step.enter_blocker.wait()
                if step.enter_error is not None:
                    raise step.enter_error
                provider.active_contexts += 1
                return _Connection()

            async def __aexit__(self, exc_type, exc, tb) -> None:
                provider.exit_calls += 1
                task = asyncio.current_task()
                assert task is not None
                provider.exit_tasks.append(task)
                step.cleanup_started.set()
                if step.cleanup_blocker is not None:
                    await step.cleanup_blocker.wait()
                provider.active_contexts -= 1
                if step.cleanup_error is not None:
                    raise step.cleanup_error

        return _Context()


def _manager(
    config: MCPConfig,
    clients: dict[str, _FakeProvider],
    *,
    timeout: float = 30,
) -> MCPManager:
    return MCPManager(
        config,
        connect_timeout_seconds=timeout,
        client_factory=lambda name, server_config: clients[name],
    )


@pytest.mark.parametrize(
    ("transport", "server"),
    [
        (
            "streamable_http_client",
            {
                "transport": "streamable-http",
                "url": "http://user:password@example.invalid/mcp?token=secret",
            },
        ),
        (
            "sse_client",
            {
                "transport": "sse",
                "url": "https://user:password@example.invalid/sse?api_key=secret",
            },
        ),
        ("stdio_client", {"transport": "stdio", "command": "example"}),
    ],
)
async def test_startup_discovery_enters_and_exits_resources_in_the_same_task(
    transport: str,
    server: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    active_resources: set[str] = set()
    affinity_errors: list[str] = []
    enter_tasks: dict[str, asyncio.Task[Any]] = {}

    class _TaskAffineContext:
        def __init__(self, label: str, value: Any) -> None:
            self.label = label
            self.value = value

        async def __aenter__(self) -> Any:
            task = asyncio.current_task()
            assert task is not None
            enter_tasks[self.label] = task
            active_resources.add(self.label)
            return self.value

        async def __aexit__(self, exc_type, exc, tb) -> None:
            task = asyncio.current_task()
            assert task is not None
            if task is not enter_tasks[self.label]:
                affinity_errors.append(self.label)
                raise RuntimeError(f"{self.label} exited from the wrong task")
            active_resources.remove(self.label)

    class _Session:
        def __init__(self, read: Any, write: Any) -> None:
            self._context = _TaskAffineContext("session", self)

        async def __aenter__(self) -> Any:
            return await self._context.__aenter__()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            await self._context.__aexit__(exc_type, exc, tb)

        async def initialize(self) -> None:
            pass

        async def list_tools(self) -> Any:
            return SimpleNamespace(tools=[])

    streams = (
        (object(), object(), lambda: None)
        if transport == "streamable_http_client"
        else (object(), object())
    )
    monkeypatch.setattr(
        client_module,
        transport,
        lambda *args: _TaskAffineContext("transport", streams),
    )
    monkeypatch.setattr(client_module, "ClientSession", _Session)
    config = MCPConfig.model_validate({"mcpServers": {"task-affine": server}})

    manager = MCPManager(config)
    await manager.startup()
    await manager.shutdown()

    assert affinity_errors == []
    assert active_resources == set()
    assert enter_tasks["transport"] is enter_tasks["session"]
    for secret in ("user:password", "token=secret", "api_key=secret"):
        assert secret not in caplog.text


@pytest.mark.parametrize("phase", ["transport", "initialize", "list_tools"])
async def test_one_timeout_bounds_every_discovery_phase_and_cleans(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocker = asyncio.Event()
    active = {"transport": 0, "session": 0}

    class _Transport:
        async def __aenter__(self):
            if phase == "transport":
                await blocker.wait()
            active["transport"] += 1
            return object(), object(), lambda: None

        async def __aexit__(self, exc_type, exc, tb):
            active["transport"] -= 1

    class _Session:
        def __init__(self, read: Any, write: Any) -> None:
            pass

        async def __aenter__(self):
            active["session"] += 1
            return self

        async def __aexit__(self, exc_type, exc, tb):
            active["session"] -= 1

        async def initialize(self):
            if phase == "initialize":
                await blocker.wait()

        async def list_tools(self):
            if phase == "list_tools":
                await blocker.wait()
            return SimpleNamespace(tools=[])

    monkeypatch.setattr(
        client_module, "streamable_http_client", lambda url: _Transport()
    )
    monkeypatch.setattr(client_module, "ClientSession", _Session)
    manager = MCPManager(_config("phase"), connect_timeout_seconds=0.02)

    await manager.startup()

    status = manager.status_snapshot()[0]
    assert status.state is MCPServerState.UNHEALTHY
    assert status.last_error == "connection timed out after 0.02 seconds"
    assert manager.list_tools() == []
    assert active == {"transport": 0, "session": 0}


async def test_partial_startup_is_ordered_sanitized_and_resource_free() -> None:
    secret_error = RuntimeError(
        "connect failed Authorization: Bearer topsecret password=hunter2 "
        "at https://user:pass@example.invalid/mcp?token=hidden"
    )
    healthy = _FakeProvider("healthy", [_OpenStep([_tool("one"), _tool("two")])])
    broken = _FakeProvider("broken", [_OpenStep(enter_error=secret_error)])
    manager = _manager(
        _config("healthy", "broken"),
        {"healthy": healthy, "broken": broken},
    )

    await manager.startup()

    assert manager.connected_servers == ["healthy"]
    assert [name for name, _ in manager.list_tools()] == [
        "healthy__one",
        "healthy__two",
    ]
    statuses = manager.status_snapshot()
    assert [status.state for status in statuses] == [
        MCPServerState.HEALTHY,
        MCPServerState.UNHEALTHY,
    ]
    assert statuses[0].catalog_revision == 1
    assert statuses[0].last_discovered_at is not None
    assert statuses[0].active_leases == 0
    assert statuses[1].next_refresh_at is not None
    assert statuses[1].last_error is not None
    for secret in ("topsecret", "hunter2", "user:pass", "hidden"):
        assert secret not in statuses[1].last_error
    assert healthy.active_contexts == broken.active_contexts == 0
    assert healthy.enter_tasks == healthy.exit_tasks


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("line one password=hunter2\nline two token=topsecret"),
        RuntimeError({"authorization": "Bearer hidden", "cookie": "secret"}),
        RuntimeError("https://user:pass@example.invalid/mcp?token=hidden"),
    ],
)
async def test_arbitrary_mcp_exception_text_is_never_exposed(
    failure: RuntimeError,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    provider = _FakeProvider("broken", [_OpenStep(enter_error=failure)])
    manager = _manager(_config("broken"), {"broken": provider})

    await manager.startup()

    status = manager.status_snapshot()[0]
    assert status.last_error == "connection failed"
    combined = f"{status.last_error}\n{caplog.text}"
    for secret in (
        "hunter2",
        "topsecret",
        "Bearer hidden",
        "cookie",
        "user:pass",
        "token=hidden",
        "line two",
    ):
        assert secret not in combined


async def test_cleanup_failure_prevents_catalog_publication() -> None:
    provider = _FakeProvider(
        "server",
        [
            _OpenStep(
                [_tool("must-not-publish")],
                cleanup_error=RuntimeError("cleanup failed"),
            )
        ],
    )
    manager = _manager(_config("server"), {"server": provider})

    await manager.startup()

    assert manager.get_tools_for_llm() == []
    assert provider.active_contexts == 0
    status = manager.status_snapshot()[0]
    assert status.state is MCPServerState.UNHEALTHY
    assert status.last_error == "connection failed"


async def test_child_cancellation_degrades_only_that_server() -> None:
    healthy = _FakeProvider("healthy", [_OpenStep([_tool("available")])])
    cancelled = _FakeProvider(
        "cancelled",
        [_OpenStep(enter_error=asyncio.CancelledError("child only"))],
    )
    manager = _manager(
        _config("healthy", "cancelled"),
        {"healthy": healthy, "cancelled": cancelled},
    )

    await manager.startup()

    assert manager.connected_servers == ["healthy"]
    assert [name for name, _ in manager.list_tools()] == ["healthy__available"]
    assert manager.status_snapshot()[1].last_error == "connection was cancelled"


async def test_application_cancellation_cleans_every_discovery_task() -> None:
    blocker = asyncio.Event()
    completed = _FakeProvider("completed", [_OpenStep([_tool("hidden")])])
    blocked_step = _OpenStep(enter_blocker=blocker)
    blocked = _FakeProvider("blocked", [blocked_step])
    manager = _manager(
        _config("completed", "blocked"),
        {"completed": completed, "blocked": blocked},
    )
    startup = asyncio.create_task(manager.startup())
    await blocked_step.enter_started.wait()

    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup

    assert completed.active_contexts == blocked.active_contexts == 0
    assert manager.list_tools() == []
    assert all(
        status.state is MCPServerState.UNHEALTHY
        and status.last_error == "application startup was cancelled"
        for status in manager.status_snapshot()
    )


async def test_repeated_application_cancellation_waits_for_discovery_cleanup() -> None:
    list_blocker = asyncio.Event()
    cleanup_release = asyncio.Event()
    step = _OpenStep(
        [_tool("hidden")],
        list_blocker=list_blocker,
        cleanup_blocker=cleanup_release,
    )
    provider = _FakeProvider("server", [step])
    manager = _manager(_config("server"), {"server": provider})
    startup = asyncio.create_task(manager.startup())
    await step.list_started.wait()

    startup.cancel()
    await step.cleanup_started.wait()
    startup.cancel()
    await asyncio.sleep(0)
    assert not startup.done()
    assert provider.active_contexts == 1

    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await startup

    assert provider.active_contexts == 0
    assert provider.enter_tasks[0] is provider.exit_tasks[0]


@pytest.mark.parametrize("timeout", [0, -1])
async def test_nonpositive_timeout_is_disabled(timeout: float) -> None:
    blocker = asyncio.Event()
    step = _OpenStep(list_blocker=blocker)
    provider = _FakeProvider("waiting", [step])
    manager = _manager(_config("waiting"), {"waiting": provider}, timeout=timeout)
    startup = asyncio.create_task(manager.startup())
    await step.list_started.wait()
    await asyncio.sleep(0)
    assert not startup.done()

    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup
    assert provider.active_contexts == 0


async def test_shutdown_is_idempotent_and_opens_no_cleanup_connection() -> None:
    provider = _FakeProvider("server", [_OpenStep([_tool("one")])])
    manager = _manager(_config("server"), {"server": provider})
    await manager.startup()
    assert provider.open_calls == provider.exit_calls == 1

    await manager.shutdown()
    await manager.shutdown()

    assert provider.open_calls == provider.exit_calls == 1
    assert manager.list_tools() == []
    assert manager.status_snapshot()[0].state is MCPServerState.CLOSED


async def test_no_enabled_servers_has_empty_ok_health(asgi_client) -> None:
    manager = MCPManager(_config("disabled", disabled=("disabled",)))
    await manager.startup()
    with wired_app(SimpleNamespace(), mcp=manager) as (app, _settings):
        response = await asgi_client(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["mcp_servers"] == []


async def test_health_distinguishes_idle_ready_and_degraded(asgi_client) -> None:
    healthy = _FakeProvider("healthy", [_OpenStep([_tool("one")])])
    broken = _FakeProvider("broken", [_OpenStep(enter_error=RuntimeError("offline"))])
    manager = _manager(
        _config("healthy", "broken"),
        {"healthy": healthy, "broken": broken},
    )
    await manager.startup()

    with wired_app(SimpleNamespace(), mcp=manager) as (app, _settings):
        body = (await asgi_client(app).get("/health")).json()

    assert body["status"] == "degraded"
    assert body["connected_servers"] == ["healthy"]
    assert body["tool_count"] == 1
    assert body["mcp_servers"][0]["state"] == "healthy"
    assert body["mcp_servers"][0]["active_leases"] == 0
    assert body["mcp_servers"][0]["catalog_revision"] == 1
    assert body["mcp_servers"][0]["last_discovered_at"] is not None
    assert body["mcp_servers"][0]["next_refresh_at"] is None
    assert body["mcp_servers"][1]["state"] == "unhealthy"
    assert body["mcp_servers"][1]["next_refresh_at"] is not None
