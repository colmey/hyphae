"""Plan 07 regression coverage for lazy MCP recovery and mutable inventory."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from agent import InMemorySessionStore, RunLimits, SessionGuard
from api.turn import (
    PersistencePolicy,
    TurnRequest,
    TurnRunner,
    UnorchestratedRouting,
)
from config import MCPConfig, Settings
from llm.client import GenerationRequest, LLMClient
from llm.schemas import AssistantMessage, TextBlock, Usage
from mcp_layer import MCPManager, MCPServerState, Tool, ToolCallResult, ToolSnapshot
from mcp_layer.client import MCPClient, MCPTransportError
from tests._app_support import wired_app

pytestmark = pytest.mark.anyio


def _config(*names: str) -> MCPConfig:
    return MCPConfig.model_validate(
        {
            "mcpServers": {
                name: {
                    "transport": "streamable-http",
                    "url": f"http://{name}.invalid/mcp",
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
class _ConnectStep:
    tools: list[Tool]
    error: BaseException | None = None
    blocker: asyncio.Event | None = None
    swallow_cancellation: bool = False
    started: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _CallStep:
    outcome: ToolCallResult | BaseException
    blocker: asyncio.Event | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)


class _RecoveryClient:
    def __init__(
        self,
        name: str,
        *,
        connect_steps: list[_ConnectStep],
        call_steps: list[_CallStep] | None = None,
        close_errors: list[BaseException] | None = None,
    ) -> None:
        self.name = name
        self._tools: list[Tool] = []
        self.connect_steps = deque(connect_steps)
        self.call_steps = deque(call_steps or [])
        self.close_errors = deque(close_errors or [])
        self.connect_calls = 0
        self.connect_cancellations = 0
        self.close_calls = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    async def connect(self) -> None:
        self.connect_calls += 1
        if not self.connect_steps:
            raise AssertionError("unexpected connection attempt")
        step = self.connect_steps.popleft()
        step.started.set()
        if step.blocker is not None:
            try:
                await step.blocker.wait()
            except asyncio.CancelledError:
                self.connect_cancellations += 1
                if not step.swallow_cancellation:
                    raise
        # Deliberately expose tools before raising so failed reconnect tests prove
        # that the manager never publishes a client's partial/stale inventory.
        self._tools = step.tools
        if step.error is not None:
            raise step.error

    async def close(self) -> None:
        self.close_calls += 1
        try:
            if self.close_errors:
                raise self.close_errors.popleft()
        finally:
            self._tools = []

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        self.calls.append((tool_name, arguments))
        if not self.call_steps:
            return ToolCallResult(content=f"{tool_name} ok", is_error=False)
        step = self.call_steps.popleft()
        step.started.set()
        if step.blocker is not None:
            await step.blocker.wait()
        if isinstance(step.outcome, BaseException):
            raise step.outcome
        return step.outcome


def _manager(
    config: MCPConfig,
    clients: dict[str, _RecoveryClient],
    *,
    timeout: float = 30,
) -> MCPManager:
    return MCPManager(
        config,
        connect_timeout_seconds=timeout,
        client_factory=lambda name, server_config: clients[name],
    )


async def test_client_preserves_declared_errors_and_types_sdk_exceptions() -> None:
    config = _config("server").enabled_servers()["server"]
    client = MCPClient("server", config)

    class _DeclaredErrorSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]):
            return SimpleNamespace(
                content=[SimpleNamespace(text="declared failure")],
                isError=True,
            )

    client._session = _DeclaredErrorSession()  # type: ignore[assignment]
    result = await client.call_tool("known", {})
    assert result == ToolCallResult(content="declared failure", is_error=True)

    cause = RuntimeError("protocol broke")

    class _RaisingSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]):
            raise cause

    client._session = _RaisingSession()  # type: ignore[assignment]
    with pytest.raises(MCPTransportError) as raised:
        await client.call_tool("known", {})
    assert raised.value.cause is cause


async def test_transport_failure_invalidates_without_replay_and_sanitizes_health(
    asgi_client,
) -> None:
    failure = RuntimeError(
        "reset Authorization: Bearer topsecret at "
        "https://user:pass@example.invalid/mcp?token=hidden"
    )
    client = _RecoveryClient(
        "server",
        connect_steps=[_ConnectStep([_tool("write")])],
        call_steps=[_CallStep(MCPTransportError(failure))],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()

    result = await manager.call_tool("server__write", {"value": 1})

    assert result.is_error is True
    assert "outcome is unknown" in result.content
    assert "was not replayed" in result.content
    assert client.calls == [("write", {"value": 1})]
    assert client.close_calls == 1
    assert manager.connected_servers == []
    assert manager.list_tools() == []
    assert manager.get_tools_for_llm() == []
    assert "server__write" in manager._last_known_routes
    status = manager.status_snapshot()[0]
    assert status.state is MCPServerState.UNHEALTHY
    assert status.tool_count == 0
    assert status.last_error is not None
    for secret in ("topsecret", "user:pass", "hidden"):
        assert secret not in status.last_error

    with wired_app(SimpleNamespace(), mcp=manager) as (app, _settings):
        health = (await asgi_client(app).get("/health")).json()
    assert health["status"] == "degraded"
    assert health["connected_servers"] == []
    assert health["tool_count"] == 0
    assert health["mcp_servers"][0]["tool_count"] == 0


async def test_declared_tool_error_keeps_server_healthy() -> None:
    client = _RecoveryClient(
        "server",
        connect_steps=[_ConnectStep([_tool("read")])],
        call_steps=[
            _CallStep(ToolCallResult(content="ordinary MCP error", is_error=True))
        ],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()

    result = await manager.call_tool("server__read", {})

    assert result == ToolCallResult(content="ordinary MCP error", is_error=True)
    assert client.connect_calls == 1
    assert client.close_calls == 0
    assert manager.connected_servers == ["server"]
    assert [name for name, _ in manager.list_tools()] == ["server__read"]
    assert manager.status_snapshot()[0].state is MCPServerState.HEALTHY


async def test_later_call_recovers_and_atomically_adds_inventory(asgi_client) -> None:
    release_reconnect = asyncio.Event()
    reconnect = _ConnectStep(
        [_tool("write"), _tool("new")], blocker=release_reconnect
    )
    client = _RecoveryClient(
        "server",
        connect_steps=[_ConnectStep([_tool("write")]), reconnect],
        call_steps=[
            _CallStep(MCPTransportError(ConnectionResetError("lost transport"))),
            _CallStep(ToolCallResult(content="later invocation", is_error=False)),
        ],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()
    old_snapshot = ToolSnapshot.from_llm_tools(manager.get_tools_for_llm())
    await manager.call_tool("server__write", {"value": 1})

    later = asyncio.create_task(
        manager.call_tool("server__write", {"value": 2})
    )
    await reconnect.started.wait()
    connecting = manager.status_snapshot()[0]
    assert connecting.state is MCPServerState.CONNECTING
    assert connecting.last_error is None
    assert connecting.tool_count == 0
    assert manager.list_tools() == []
    assert manager.get_tools_for_llm() == []

    release_reconnect.set()
    result = await later

    assert result == ToolCallResult(content="later invocation", is_error=False)
    assert client.connect_calls == 2
    assert client.calls == [("write", {"value": 1}), ("write", {"value": 2})]
    assert [name for name, _ in manager.list_tools()] == [
        "server__write",
        "server__new",
    ]
    assert old_snapshot.names == frozenset({"server__write"})
    assert ToolSnapshot.from_llm_tools(manager.get_tools_for_llm()).names == frozenset(
        {"server__write", "server__new"}
    )
    status = manager.status_snapshot()[0]
    assert status.state is MCPServerState.HEALTHY
    assert status.last_error is None
    assert status.tool_count == 2

    with wired_app(SimpleNamespace(), mcp=manager) as (app, _settings):
        health = (await asgi_client(app).get("/health")).json()
    assert health["status"] == "ok"
    assert health["connected_servers"] == ["server"]
    assert health["tool_count"] == 2


async def test_in_flight_turn_keeps_snapshot_while_next_turn_sees_refresh() -> None:
    class _SnapshotLLM(LLMClient):
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def complete(self, request: GenerationRequest) -> AssistantMessage:
            self.requests.append(request)
            if len(self.requests) == 1:
                self.first_started.set()
                await self.release_first.wait()
            return AssistantMessage(
                content=[TextBlock("done")],
                stop_reason="end_turn",
                usage=Usage(total_tokens=1),
            )

    client = _RecoveryClient(
        "server",
        connect_steps=[
            _ConnectStep([_tool("known")]),
            _ConnectStep([_tool("known"), _tool("added")]),
        ],
        call_steps=[
            _CallStep(MCPTransportError(ConnectionResetError("lost transport"))),
            _CallStep(ToolCallResult(content="recovered", is_error=False)),
        ],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()
    llm = _SnapshotLLM()
    store = InMemorySessionStore()
    settings = Settings(
        _env_file=None,
        orchestration_enabled=False,
        llm_max_retries=0,
    )
    runner = TurnRunner(
        routing=UnorchestratedRouting(
            llm=llm,
            model_id=settings.llm_model,
        ),
        limits=RunLimits.from_settings(settings),
        mcp=manager,
        store=store,
        guard=SessionGuard(),
        policy=None,
        tracer=None,
    )
    first_session = await store.create()
    first_turn = asyncio.create_task(
        runner.run(
            TurnRequest("first", first_session, PersistencePolicy.PERSISTENT)
        )
    )
    await llm.first_started.wait()

    await manager.call_tool("server__known", {"attempt": 1})
    await manager.call_tool("server__known", {"attempt": 2})
    llm.release_first.set()
    await first_turn

    second_session = await store.create()
    await runner.run(
        TurnRequest("second", second_session, PersistencePolicy.PERSISTENT)
    )

    assert [tool["name"] for tool in llm.requests[0].tools or []] == [
        "server__known"
    ]
    assert [tool["name"] for tool in llm.requests[1].tools or []] == [
        "server__known",
        "server__added",
    ]


async def test_concurrent_later_calls_share_one_reconnect() -> None:
    release_reconnect = asyncio.Event()
    reconnect = _ConnectStep([_tool("read")], blocker=release_reconnect)
    client = _RecoveryClient(
        "server",
        connect_steps=[_ConnectStep([_tool("read")]), reconnect],
        call_steps=[
            _CallStep(MCPTransportError(ConnectionResetError("lost transport")))
        ],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()
    await manager.call_tool("server__read", {"attempt": "failed"})

    calls = [
        asyncio.create_task(manager.call_tool("server__read", {"caller": index}))
        for index in range(10)
    ]
    await reconnect.started.wait()
    release_reconnect.set()
    results = await asyncio.gather(*calls)

    assert client.connect_calls == 2
    assert len(client.calls) == 11
    assert all(result.is_error is False for result in results)
    assert manager.status_snapshot()[0].state is MCPServerState.HEALTHY


async def test_cancelled_waiter_does_not_cancel_shared_reconnect() -> None:
    release_reconnect = asyncio.Event()
    reconnect = _ConnectStep([_tool("read")], blocker=release_reconnect)
    client = _RecoveryClient(
        "server",
        connect_steps=[_ConnectStep([_tool("read")]), reconnect],
        call_steps=[
            _CallStep(MCPTransportError(ConnectionResetError("lost transport")))
        ],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()
    await manager.call_tool("server__read", {"attempt": "failed"})

    cancelled = asyncio.create_task(manager.call_tool("server__read", {"caller": 1}))
    await reconnect.started.wait()
    survivor = asyncio.create_task(manager.call_tool("server__read", {"caller": 2}))
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release_reconnect.set()
    result = await survivor

    assert result.is_error is False
    assert client.connect_calls == 2
    assert client.calls == [
        ("read", {"attempt": "failed"}),
        ("read", {"caller": 2}),
    ]


async def test_removed_tool_is_not_dispatched_after_refresh() -> None:
    client = _RecoveryClient(
        "server",
        connect_steps=[
            _ConnectStep([_tool("removed")]),
            _ConnectStep([_tool("replacement")]),
        ],
        call_steps=[
            _CallStep(MCPTransportError(ConnectionResetError("lost transport")))
        ],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()
    await manager.call_tool("server__removed", {})

    result = await manager.call_tool("server__removed", {})

    assert result.is_error is True
    assert "no longer available after reconnect" in result.content
    assert client.calls == [("removed", {})]
    assert [name for name, _ in manager.list_tools()] == ["server__replacement"]
    assert "server__removed" not in manager._last_known_routes
    assert "server__replacement" in manager._last_known_routes


async def test_failed_reconnect_leaks_no_tools_and_allows_a_later_attempt() -> None:
    client = _RecoveryClient(
        "server",
        connect_steps=[
            _ConnectStep([_tool("known")]),
            _ConnectStep(
                [_tool("must-not-leak")],
                error=RuntimeError("still offline"),
            ),
            _ConnectStep([_tool("known"), _tool("added")]),
        ],
        call_steps=[
            _CallStep(MCPTransportError(ConnectionResetError("lost transport")))
        ],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()
    await manager.call_tool("server__known", {"attempt": 1})

    failed = await manager.call_tool("server__known", {"attempt": 2})

    assert failed.is_error is True
    assert "reconnect failed" in failed.content
    assert client.connect_calls == 2
    assert manager.list_tools() == []
    assert manager.get_tools_for_llm() == []
    assert set(manager._last_known_routes) == {"server__known"}
    failed_status = manager.status_snapshot()[0]
    assert failed_status.state is MCPServerState.UNHEALTHY
    assert failed_status.tool_count == 0

    recovered = await manager.call_tool("server__known", {"attempt": 3})

    assert recovered.is_error is False
    assert client.connect_calls == 3
    assert client.calls == [
        ("known", {"attempt": 1}),
        ("known", {"attempt": 3}),
    ]
    assert [name for name, _ in manager.list_tools()] == [
        "server__known",
        "server__added",
    ]


async def test_never_seen_name_does_not_probe_any_server() -> None:
    clients = {
        "one": _RecoveryClient("one", connect_steps=[_ConnectStep([_tool("a")])]),
        "two": _RecoveryClient("two", connect_steps=[_ConnectStep([_tool("b")])]),
    }
    manager = _manager(_config("one", "two"), clients)
    await manager.startup()

    result = await manager.call_tool("missing__tool", {})

    assert result == ToolCallResult(
        content="unknown tool: 'missing__tool'", is_error=True
    )
    assert [client.connect_calls for client in clients.values()] == [1, 1]
    assert all(client.calls == [] for client in clients.values())


async def test_reconnect_uses_the_same_bounded_connect_contract() -> None:
    reconnect = _ConnectStep([_tool("known")], blocker=asyncio.Event())
    client = _RecoveryClient(
        "server",
        connect_steps=[_ConnectStep([_tool("known")]), reconnect],
        call_steps=[
            _CallStep(MCPTransportError(ConnectionResetError("lost transport")))
        ],
    )
    manager = _manager(_config("server"), {"server": client}, timeout=0.02)
    await manager.startup()
    await manager.call_tool("server__known", {})

    result = await manager.call_tool("server__known", {})

    assert result.is_error is True
    assert "reconnect failed" in result.content
    assert reconnect.started.is_set()
    assert client.connect_cancellations == 1
    status = manager.status_snapshot()[0]
    assert status.state is MCPServerState.UNHEALTHY
    assert status.last_error == "connection timed out after 0.02 seconds"


async def test_shutdown_during_reconnect_cannot_republish_tools() -> None:
    reconnect = _ConnectStep(
        [_tool("known"), _tool("late")],
        blocker=asyncio.Event(),
        swallow_cancellation=True,
    )
    client = _RecoveryClient(
        "server",
        connect_steps=[_ConnectStep([_tool("known")]), reconnect],
        call_steps=[
            _CallStep(MCPTransportError(ConnectionResetError("lost transport")))
        ],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()
    await manager.call_tool("server__known", {})
    recovering = asyncio.create_task(manager.call_tool("server__known", {}))
    await reconnect.started.wait()

    await manager.shutdown()
    result = await recovering

    assert result.is_error is True
    assert result.content == "server 'server' is closed"
    assert client.connect_cancellations == 1
    assert manager.connected_servers == []
    assert manager.list_tools() == []
    assert manager.get_tools_for_llm() == []
    assert manager._last_known_routes == {}
    status = manager.status_snapshot()[0]
    assert status.state is MCPServerState.CLOSED
    assert status.tool_count == 0


async def test_cleanup_failure_does_not_undo_invalidation() -> None:
    client = _RecoveryClient(
        "server",
        connect_steps=[_ConnectStep([_tool("known")])],
        call_steps=[
            _CallStep(MCPTransportError(ConnectionResetError("lost transport")))
        ],
        close_errors=[RuntimeError("cleanup failed")],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()

    result = await manager.call_tool("server__known", {})

    assert result.is_error is True
    assert "was not replayed" in result.content
    assert manager.list_tools() == []
    assert manager.status_snapshot()[0].state is MCPServerState.UNHEALTHY
    assert manager.status_snapshot()[0].tool_count == 0


async def test_active_call_cancellation_propagates_without_invalidation() -> None:
    blocker = asyncio.Event()
    call = _CallStep(
        ToolCallResult(content="unreachable", is_error=False), blocker=blocker
    )
    client = _RecoveryClient(
        "server",
        connect_steps=[_ConnectStep([_tool("known")])],
        call_steps=[call],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()
    task = asyncio.create_task(manager.call_tool("server__known", {}))
    await call.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert manager.status_snapshot()[0].state is MCPServerState.HEALTHY
    assert [name for name, _ in manager.list_tools()] == ["server__known"]
    assert client.close_calls == 0


async def test_stale_generation_failure_cannot_poison_recovered_connection() -> None:
    release_stale = asyncio.Event()
    stale = _CallStep(
        MCPTransportError(ConnectionResetError("late stale failure")),
        blocker=release_stale,
    )
    client = _RecoveryClient(
        "server",
        connect_steps=[
            _ConnectStep([_tool("known")]),
            _ConnectStep([_tool("known"), _tool("new")]),
        ],
        call_steps=[
            stale,
            _CallStep(MCPTransportError(ConnectionResetError("first failure"))),
            _CallStep(ToolCallResult(content="recovered", is_error=False)),
        ],
    )
    manager = _manager(_config("server"), {"server": client})
    await manager.startup()
    stale_call = asyncio.create_task(manager.call_tool("server__known", {"id": 1}))
    await stale.started.wait()

    first_failure = await manager.call_tool("server__known", {"id": 2})
    recovered = await manager.call_tool("server__known", {"id": 3})
    release_stale.set()
    stale_result = await stale_call

    assert first_failure.is_error is True
    assert recovered == ToolCallResult(content="recovered", is_error=False)
    assert stale_result.is_error is True
    assert "was not replayed" in stale_result.content
    assert client.connect_calls == 2
    assert client.close_calls == 1
    assert manager.status_snapshot()[0].state is MCPServerState.HEALTHY
    assert [name for name, _ in manager.list_tools()] == [
        "server__known",
        "server__new",
    ]
