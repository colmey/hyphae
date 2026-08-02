"""Catalog refresh, lazy lease, drift, isolation, and no-replay coverage."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import pytest

import mcp_layer.manager as manager_module
from config import MCPConfig
from mcp_layer import MCPManager, MCPServerState, Tool, ToolCallResult, ToolSnapshot
from mcp_layer.client import MCPConnection, MCPTransportError

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


def _tool(name: str, schema: Any = None, description: str | None = None) -> Tool:
    return Tool(
        name=name,
        description=description if description is not None else f"{name} description",
        input_schema=(
            schema if schema is not None else {"type": "object", "properties": {}}
        ),
    )


@dataclass
class _CallStep:
    outcome: ToolCallResult | BaseException = field(
        default_factory=lambda: ToolCallResult("ok", False)
    )
    blocker: asyncio.Event | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _ConnectionStep:
    tools: list[Tool]
    enter_error: BaseException | None = None
    list_error: BaseException | None = None
    list_blocker: asyncio.Event | None = None
    cleanup_error: BaseException | None = None
    cleanup_blocker: asyncio.Event | None = None
    calls: list[_CallStep] = field(default_factory=list)
    connection_id: str = "connection"
    list_started: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_started: asyncio.Event = field(default_factory=asyncio.Event)


class _LeaseProvider:
    def __init__(self, name: str, steps: list[_ConnectionStep]) -> None:
        self.name = name
        self.steps = deque(steps)
        self.open_calls = 0
        self.exit_calls = 0
        self.active_contexts = 0
        self.enter_tasks: list[asyncio.Task[Any]] = []
        self.exit_tasks: list[asyncio.Task[Any]] = []
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def open(self):
        provider = self
        if not self.steps:
            raise AssertionError(f"unexpected connection to {self.name}")
        step = self.steps.popleft()
        call_steps = deque(step.calls)

        class _Connection:
            async def list_tools(self) -> list[Tool]:
                step.list_started.set()
                if step.list_blocker is not None:
                    await step.list_blocker.wait()
                if step.list_error is not None:
                    raise step.list_error
                return step.tools

            async def call_tool(
                self, tool_name: str, arguments: dict[str, Any]
            ) -> ToolCallResult:
                provider.calls.append((step.connection_id, tool_name, arguments))
                call = call_steps.popleft() if call_steps else _CallStep()
                call.started.set()
                if call.blocker is not None:
                    await call.blocker.wait()
                if isinstance(call.outcome, BaseException):
                    raise call.outcome
                return call.outcome

        class _Context:
            async def __aenter__(self) -> _Connection:
                provider.open_calls += 1
                task = asyncio.current_task()
                assert task is not None
                provider.enter_tasks.append(task)
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
    clients: dict[str, _LeaseProvider],
    *,
    ttl: float = 300,
    timeout: float = 30,
) -> MCPManager:
    return MCPManager(
        _config(*clients),
        connect_timeout_seconds=timeout,
        catalog_ttl_seconds=ttl,
        client_factory=lambda name, server_config: clients[name],
    )


async def test_client_preserves_declared_errors_and_bounds_non_text_content() -> None:
    class _Session:
        async def call_tool(self, name: str, arguments: dict[str, Any]):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(text="declared failure"),
                    SimpleNamespace(
                        type="image",
                        mimeType="image/png",
                        data="secret-binary" * 100_000,
                    ),
                ],
                isError=True,
            )

    config = _config("server").enabled_servers()["server"]
    connection = MCPConnection("server", config, _Session())  # type: ignore[arg-type]
    result = await connection.call_tool("known", {})

    assert result.is_error is True
    assert result.content.startswith("declared failure\n")
    assert len(result.content) < 2_100
    assert "secret-binary" not in result.content
    assert '"data":"[omitted]"' in result.content

    cause = RuntimeError("protocol broke")

    class _RaisingSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]):
            raise cause

    raising = MCPConnection("server", config, _RaisingSession())  # type: ignore[arg-type]
    with pytest.raises(MCPTransportError) as raised:
        await raising.call_tool("known", {})
    assert raised.value.cause is cause


async def test_client_bounds_aggregate_non_text_content_and_classifies_decode_failure() -> (
    None
):
    class _ManyBlocksSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="image", name="x" * 1_000, data=b"secret")
                    for _ in range(1_000)
                ],
                isError=False,
            )

    config = _config("server").enabled_servers()["server"]
    connection = MCPConnection(  # type: ignore[arg-type]
        "server", config, _ManyBlocksSession()
    )
    result = await connection.call_tool("known", {})
    assert len(result.content) <= 20_100
    assert "additional non-text content omitted" in result.content
    assert "secret" not in result.content

    class _BrokenContent:
        def __iter__(self):
            raise RuntimeError("response decoding failed")

    class _BrokenResultSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]):
            return SimpleNamespace(content=_BrokenContent(), isError=False)

    broken = MCPConnection(  # type: ignore[arg-type]
        "server", config, _BrokenResultSession()
    )
    with pytest.raises(MCPTransportError) as raised:
        await broken.call_tool("known", {})
    assert str(raised.value.cause) == "response decoding failed"


async def test_disabled_tools_are_validated_before_they_are_filtered() -> None:
    config = MCPConfig.model_validate(
        {
            "mcpServers": {
                "server": {
                    "transport": "streamable-http",
                    "url": "http://server.invalid/mcp",
                    "disabled_tools": ["hidden"],
                }
            }
        }
    )
    server_config = config.enabled_servers()["server"]

    class _Session:
        async def list_tools(self):
            hidden = SimpleNamespace(
                name="hidden",
                description="hidden",
                inputSchema=[],
            )
            return SimpleNamespace(tools=[hidden, hidden])

    connection = MCPConnection(  # type: ignore[arg-type]
        "server",
        server_config,
        _Session(),
    )
    raw_tools = await connection.list_tools()
    assert [tool.name for tool in raw_tools] == [
        "hidden",
        "hidden",
    ]
    assert raw_tools[0].input_schema == []

    provider = _LeaseProvider(
        "server",
        [_ConnectionStep([_tool("hidden"), _tool("hidden")])],
    )
    manager = MCPManager(
        config,
        client_factory=lambda name, server_config: provider,
    )

    await manager.startup()

    status = manager.status_snapshot()[0]
    assert status.state is MCPServerState.UNHEALTHY
    assert status.last_error is not None and "duplicate tool name" in status.last_error


async def test_selecting_without_dispatch_opens_no_lease() -> None:
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([_tool("read")], connection_id="discovery"),
            _ConnectionStep([_tool("read")], connection_id="must-not-open"),
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()

    async with manager.open_turn() as runtime:
        assert [tool["name"] for tool in runtime.get_tools_for_llm()] == [
            "server__read"
        ]
        assert provider.open_calls == 1

    assert provider.open_calls == 1
    assert manager.status_snapshot()[0].active_leases == 0


async def test_first_dispatch_opens_once_and_same_turn_reuses_lease() -> None:
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([_tool("read")], connection_id="discovery"),
            _ConnectionStep(
                [_tool("read")],
                calls=[
                    _CallStep(ToolCallResult("first", False)),
                    _CallStep(ToolCallResult("second", False)),
                ],
                connection_id="lease",
            ),
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()

    async with manager.open_turn() as runtime:
        first = await runtime.call_tool("server__read", {"id": 1})
        second = await runtime.call_tool("server__read", {"id": 2})
        status = manager.status_snapshot()[0]
        assert status.active_leases == 1
        assert status.catalog_revision == 2

    assert first.content == "first"
    assert second.content == "second"
    assert provider.open_calls == provider.exit_calls == 2
    assert provider.calls == [
        ("lease", "read", {"id": 1}),
        ("lease", "read", {"id": 2}),
    ]
    assert provider.enter_tasks[1] is provider.exit_tasks[1]
    assert manager.status_snapshot()[0].active_leases == 0


async def test_concurrent_turns_have_independent_sessions() -> None:
    release = asyncio.Event()
    first_call = _CallStep(blocker=release)
    second_call = _CallStep(blocker=release)
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([_tool("read")], connection_id="discovery"),
            _ConnectionStep(
                [_tool("read")], calls=[first_call], connection_id="turn-1"
            ),
            _ConnectionStep(
                [_tool("read")], calls=[second_call], connection_id="turn-2"
            ),
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()

    async with manager.open_turn() as first_runtime:
        async with manager.open_turn() as second_runtime:
            first = asyncio.create_task(
                first_runtime.call_tool("server__read", {"turn": 1})
            )
            second = asyncio.create_task(
                second_runtime.call_tool("server__read", {"turn": 2})
            )
            await asyncio.gather(first_call.started.wait(), second_call.started.wait())
            assert manager.status_snapshot()[0].active_leases == 2
            release.set()
            await asyncio.gather(first, second)

    assert {call[0] for call in provider.calls} == {"turn-1", "turn-2"}
    assert provider.enter_tasks[1] is not provider.enter_tasks[2]
    assert provider.active_contexts == 0


async def test_one_turn_opens_one_worker_for_each_used_server() -> None:
    first = _LeaseProvider(
        "first",
        [
            _ConnectionStep([_tool("read")], connection_id="first-discovery"),
            _ConnectionStep(
                [_tool("read")],
                calls=[_CallStep(ToolCallResult("first", False))],
                connection_id="first-lease",
            ),
        ],
    )
    second = _LeaseProvider(
        "second",
        [
            _ConnectionStep([_tool("read")], connection_id="second-discovery"),
            _ConnectionStep(
                [_tool("read")],
                calls=[_CallStep(ToolCallResult("second", False))],
                connection_id="second-lease",
            ),
        ],
    )
    manager = _manager({"first": first, "second": second})
    await manager.startup()

    async with manager.open_turn() as runtime:
        first_result = await runtime.call_tool("first__read", {})
        second_result = await runtime.call_tool("second__read", {})
        assert [status.active_leases for status in manager.status_snapshot()] == [
            1,
            1,
        ]

    assert first_result.content == "first"
    assert second_result.content == "second"
    assert first.open_calls == first.exit_calls == 2
    assert second.open_calls == second.exit_calls == 2


async def test_removed_added_and_changed_schema_are_atomic_and_turn_local() -> None:
    old_schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    new_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([_tool("removed"), _tool("changed", old_schema)]),
            _ConnectionStep([_tool("added"), _tool("changed", new_schema)]),
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()
    before = ToolSnapshot.from_llm_tools(manager.get_tools_for_llm())

    async with manager.open_turn() as runtime:
        removed = await runtime.call_tool("server__removed", {})
        changed = await runtime.call_tool("server__changed", {"value": 7})
        assert (
            ToolSnapshot.from_llm_tools(runtime.get_tools_for_llm()).names
            == before.names
        )

    assert removed.is_error and removed.content.startswith("tool_catalog_changed:")
    assert changed.is_error and "refreshed schema" in changed.content
    assert provider.calls == []
    assert ToolSnapshot.from_llm_tools(manager.get_tools_for_llm()).names == frozenset(
        {"server__added", "server__changed"}
    )


@pytest.mark.parametrize(
    "tools",
    [
        [_tool("")],
        [_tool("same"), _tool("same")],
        [_tool("oversized", description="x" * 20_001)],
        [_tool("bad-schema", {"type": "not-a-json-schema-type"})],
        [_tool(f"tool-{index}") for index in range(257)],
    ],
    ids=["blank", "duplicate", "description", "schema", "count"],
)
async def test_malformed_startup_catalog_is_rejected_as_one_unit(
    tools: list[Tool],
) -> None:
    provider = _LeaseProvider("server", [_ConnectionStep(tools)])
    manager = _manager({"server": provider})

    await manager.startup()

    assert manager.get_tools_for_llm() == []
    status = manager.status_snapshot()[0]
    assert status.state is MCPServerState.UNHEALTHY
    assert status.tool_count == 0
    assert status.last_error is not None
    assert provider.active_contexts == 0


async def test_malformed_lease_catalog_clears_future_inventory_without_dispatch() -> (
    None
):
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([_tool("known")]),
            _ConnectionStep([_tool("same"), _tool("same")]),
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()

    async with manager.open_turn() as runtime:
        result = await runtime.call_tool("server__known", {})

    assert result.is_error and "lease discovery failed" in result.content
    assert provider.calls == []
    assert manager.get_tools_for_llm() == []
    assert manager.status_snapshot()[0].state is MCPServerState.UNHEALTHY


async def test_failed_startup_recovers_by_request_after_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(manager_module.random, "uniform", lambda low, high: 1.0)
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([], enter_error=RuntimeError("offline")),
            _ConnectionStep([_tool("recovered")]),
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()
    assert provider.open_calls == 1

    async with manager.open_turn() as runtime:
        assert runtime.get_tools_for_llm() == []
    assert provider.open_calls == 1

    now = 101.0
    async with manager.open_turn() as runtime:
        assert [tool["name"] for tool in runtime.get_tools_for_llm()] == [
            "server__recovered"
        ]
    assert provider.open_calls == 2
    assert manager.status_snapshot()[0].state is MCPServerState.HEALTHY


async def test_backoff_cap_does_not_overflow_after_many_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(manager_module.random, "uniform", lambda low, high: 1.0)
    provider = _LeaseProvider(
        "server",
        [_ConnectionStep([], enter_error=RuntimeError("offline"))],
    )
    manager = _manager({"server": provider})
    record = manager._records["server"]
    record.refresh_failures = 10_000
    attempt = manager._start_attempt("server")

    manager._publish_failure("server", attempt, RuntimeError("still offline"))

    assert record.next_refresh_monotonic == now + 30.0
    assert record.refresh_failures == 10_001


async def test_stale_catalog_refreshes_on_request_but_not_when_ttl_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: now)
    refreshing = _LeaseProvider(
        "refreshing",
        [
            _ConnectionStep([_tool("old")]),
            _ConnectionStep([_tool("new")]),
        ],
    )
    disabled = _LeaseProvider(
        "disabled",
        [
            _ConnectionStep([_tool("stable")]),
            _ConnectionStep([_tool("must-not-open")]),
        ],
    )
    refresh_manager = _manager({"refreshing": refreshing}, ttl=5)
    disabled_manager = _manager({"disabled": disabled}, ttl=0)
    await refresh_manager.startup()
    await disabled_manager.startup()
    now = 6.0

    async with refresh_manager.open_turn() as runtime:
        assert [tool["name"] for tool in runtime.get_tools_for_llm()] == [
            "refreshing__new"
        ]
    async with disabled_manager.open_turn() as runtime:
        assert [tool["name"] for tool in runtime.get_tools_for_llm()] == [
            "disabled__stable"
        ]

    assert refreshing.open_calls == 2
    assert disabled.open_calls == 1


async def test_concurrent_request_refreshes_coalesce_per_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(manager_module.random, "uniform", lambda low, high: 1.0)
    release = asyncio.Event()
    refresh = _ConnectionStep([_tool("ready")], list_blocker=release)
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([], enter_error=RuntimeError("offline")),
            refresh,
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()
    now = 1.0

    async def snapshot() -> list[str]:
        async with manager.open_turn() as runtime:
            return [tool["name"] for tool in runtime.get_tools_for_llm()]

    first = asyncio.create_task(snapshot())
    await refresh.list_started.wait()
    second = asyncio.create_task(snapshot())
    await asyncio.sleep(0)
    assert provider.open_calls == 2
    release.set()

    assert await asyncio.gather(first, second) == [
        ["server__ready"],
        ["server__ready"],
    ]
    assert provider.open_calls == 2


async def test_request_refresh_cancellation_closes_in_the_refresh_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(manager_module.random, "uniform", lambda low, high: 1.0)
    refresh = _ConnectionStep([_tool("ready")], list_blocker=asyncio.Event())
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([], enter_error=RuntimeError("offline")),
            refresh,
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()
    now = 1.0

    async def open_cancelled_turn() -> None:
        async with manager.open_turn():
            raise AssertionError("cancelled refresh must not yield a runtime")

    task = asyncio.create_task(open_cancelled_turn())
    await refresh.list_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.active_contexts == 0
    assert provider.enter_tasks[1] is provider.exit_tasks[0]
    assert manager.status_snapshot()[0].state is MCPServerState.UNHEALTHY


async def test_shutdown_cancels_and_joins_request_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(manager_module.random, "uniform", lambda low, high: 1.0)
    refresh = _ConnectionStep([_tool("ready")], list_blocker=asyncio.Event())
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([], enter_error=RuntimeError("offline")),
            refresh,
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()
    now = 1.0

    async def open_turn() -> None:
        async with manager.open_turn():
            raise AssertionError("shutdown must prevent the turn from opening")

    opening = asyncio.create_task(open_turn())
    await refresh.list_started.wait()
    await manager.shutdown()

    assert provider.active_contexts == 0
    assert refresh.cleanup_started.is_set()
    assert opening.done()
    with pytest.raises((asyncio.CancelledError, RuntimeError)):
        await opening


async def test_request_refresh_uses_callers_timeout_budget() -> None:
    stale_refresh = _ConnectionStep([_tool("new")], list_blocker=asyncio.Event())
    provider = _LeaseProvider(
        "server",
        [_ConnectionStep([_tool("old")]), stale_refresh],
    )
    manager = _manager({"server": provider}, ttl=1, timeout=30)
    await manager.startup()
    record = manager._records["server"]
    record.catalog = replace(
        record.catalog,
        discovered_monotonic=time.monotonic() - 2,
    )

    async with manager.open_turn(timeout_seconds=0.01) as runtime:
        assert runtime.get_tools_for_llm() == []

    assert provider.active_contexts == 0
    status = manager.status_snapshot()[0]
    assert status.state is MCPServerState.UNHEALTHY
    assert status.last_error == "connection timed out after 0.01 seconds"


async def test_older_lease_failure_cannot_poison_newer_catalog_revision() -> None:
    release_old = asyncio.Event()
    old_call = _CallStep(
        MCPTransportError(ConnectionResetError("stale failure")),
    )
    old_lease = _ConnectionStep(
        [_tool("read")],
        list_blocker=release_old,
        calls=[old_call],
        connection_id="old",
    )
    new_lease = _ConnectionStep(
        [_tool("read"), _tool("new")],
        calls=[_CallStep(ToolCallResult("new result", False))],
        connection_id="new",
    )
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([_tool("read")]),
            old_lease,
            new_lease,
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()

    async with manager.open_turn() as old_runtime:
        async with manager.open_turn() as new_runtime:
            old_task = asyncio.create_task(old_runtime.call_tool("server__read", {}))
            await old_lease.list_started.wait()
            new_result = await new_runtime.call_tool("server__read", {})
            release_old.set()
            stale_result = await old_task

    assert new_result == ToolCallResult("new result", False)
    assert stale_result.is_error and "outcome is unknown" in stale_result.content
    assert manager.status_snapshot()[0].state is MCPServerState.HEALTHY
    assert ToolSnapshot.from_llm_tools(manager.get_tools_for_llm()).names == frozenset(
        {"server__read", "server__new"}
    )


async def test_tool_timeout_cancels_and_closes_the_poisoned_lease() -> None:
    call = _CallStep(blocker=asyncio.Event())
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([_tool("read")]),
            _ConnectionStep([_tool("read")], calls=[call]),
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()

    async with manager.open_turn() as runtime:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                await runtime.call_tool("server__read", {})

    assert provider.active_contexts == 0
    assert provider.enter_tasks[1] is provider.exit_tasks[1]
    assert manager.status_snapshot()[0].active_leases == 0


async def test_transport_failure_reports_unknown_once_and_never_replays() -> None:
    failure = RuntimeError("reset at https://secret.invalid/mcp?token=hidden")
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([_tool("write")]),
            _ConnectionStep(
                [_tool("write")],
                calls=[_CallStep(MCPTransportError(failure))],
                connection_id="lease",
            ),
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()

    async with manager.open_turn() as runtime:
        result = await runtime.call_tool("server__write", {"value": 1})
        later = await runtime.call_tool("server__write", {"value": 2})

    assert result.is_error and "outcome is unknown" in result.content
    assert "was not replayed" in result.content
    assert later == result
    assert provider.calls == [("lease", "write", {"value": 1})]
    assert manager.get_tools_for_llm() == []
    status = manager.status_snapshot()[0]
    assert status.state is MCPServerState.UNHEALTHY
    assert status.last_error is not None and "hidden" not in status.last_error


async def test_active_call_cancellation_closes_worker_in_its_own_task() -> None:
    blocker = asyncio.Event()
    call = _CallStep(blocker=blocker)
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([_tool("read")]),
            _ConnectionStep([_tool("read")], calls=[call]),
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()

    async with manager.open_turn() as runtime:
        task = asyncio.create_task(runtime.call_tool("server__read", {}))
        await call.started.wait()
        assert manager.status_snapshot()[0].active_leases == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert provider.active_contexts == 0
    assert provider.enter_tasks[1] is provider.exit_tasks[1]
    assert manager.status_snapshot()[0].active_leases == 0


async def test_caller_cancellation_and_shutdown_do_not_recancel_lease_cleanup() -> None:
    call = _CallStep(blocker=asyncio.Event())
    cleanup_release = asyncio.Event()
    lease = _ConnectionStep(
        [_tool("read")],
        calls=[call],
        cleanup_blocker=cleanup_release,
    )
    provider = _LeaseProvider(
        "server",
        [_ConnectionStep([_tool("read")]), lease],
    )
    manager = _manager({"server": provider})
    await manager.startup()
    turn = manager.open_turn()
    runtime = await turn.__aenter__()
    dispatch = asyncio.create_task(runtime.call_tool("server__read", {}))
    await call.started.wait()

    dispatch.cancel()
    await lease.cleanup_started.wait()
    shutdown = asyncio.create_task(manager.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    assert provider.active_contexts == 1

    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    await shutdown
    await turn.__aexit__(None, None, None)

    assert provider.active_contexts == 0
    assert provider.enter_tasks[1] is provider.exit_tasks[1]
    assert manager.status_snapshot()[0].active_leases == 0


async def test_cleanup_failure_log_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "https://user:pass@example.invalid/mcp?token=hidden"
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([_tool("read")]),
            _ConnectionStep(
                [_tool("read")],
                calls=[_CallStep(RuntimeError("unexpected dispatch defect"))],
                cleanup_error=RuntimeError(f"cleanup failed at {secret}"),
            ),
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()

    with caplog.at_level(logging.WARNING, logger="mcp_layer.lease"):
        async with manager.open_turn() as runtime:
            result = await runtime.call_tool("server__read", {})

    assert result.is_error
    rendered = caplog.text
    assert "hidden" not in rendered
    assert "user:pass" not in rendered
    assert "[redacted-url]" in rendered


async def test_shutdown_aborts_active_leases_without_cross_task_context_exit() -> None:
    blocker = asyncio.Event()
    call = _CallStep(blocker=blocker)
    provider = _LeaseProvider(
        "server",
        [
            _ConnectionStep([_tool("read")]),
            _ConnectionStep([_tool("read")], calls=[call]),
        ],
    )
    manager = _manager({"server": provider})
    await manager.startup()
    turn = manager.open_turn()
    runtime = await turn.__aenter__()
    task = asyncio.create_task(runtime.call_tool("server__read", {}))
    await call.started.wait()

    await manager.shutdown()
    await asyncio.gather(task, return_exceptions=True)
    await turn.__aexit__(None, None, None)

    assert provider.active_contexts == 0
    assert provider.enter_tasks[1] is provider.exit_tasks[1]
    assert manager.status_snapshot()[0].state is MCPServerState.CLOSED
    assert manager.status_snapshot()[0].active_leases == 0


async def test_cancelled_shutdown_still_joins_refreshes_and_active_leases() -> None:
    active_call = _CallStep(blocker=asyncio.Event())
    lease_provider = _LeaseProvider(
        "lease",
        [
            _ConnectionStep([_tool("read")]),
            _ConnectionStep([_tool("read")], calls=[active_call]),
        ],
    )
    refresh_release = asyncio.Event()
    refresh_step = _ConnectionStep(
        [_tool("ready")],
        list_blocker=asyncio.Event(),
        cleanup_blocker=refresh_release,
    )
    refresh_provider = _LeaseProvider(
        "refresh",
        [
            _ConnectionStep([], enter_error=RuntimeError("offline")),
            refresh_step,
        ],
    )
    manager = _manager({"lease": lease_provider, "refresh": refresh_provider})
    await manager.startup()

    turn = manager.open_turn()
    runtime = await asyncio.wait_for(turn.__aenter__(), 1)
    dispatch = asyncio.create_task(runtime.call_tool("lease__read", {}))
    await asyncio.wait_for(active_call.started.wait(), 1)
    manager._records["refresh"].next_refresh_monotonic = 0

    async def open_refreshing_turn() -> None:
        async with manager.open_turn():
            raise AssertionError("shutdown must prevent the refreshing turn")

    opening = asyncio.create_task(open_refreshing_turn())
    await asyncio.wait_for(refresh_step.list_started.wait(), 1)
    shutdown = asyncio.create_task(manager.shutdown())
    await asyncio.wait_for(refresh_step.cleanup_started.wait(), 1)
    shutdown.cancel()
    await asyncio.sleep(0)
    assert not shutdown.done()

    refresh_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(shutdown), 1)
    await asyncio.wait_for(
        asyncio.gather(opening, dispatch, return_exceptions=True),
        1,
    )
    await asyncio.wait_for(turn.__aexit__(None, None, None), 1)

    assert lease_provider.active_contexts == 0
    assert refresh_provider.active_contexts == 0
    assert all(status.active_leases == 0 for status in manager.status_snapshot())
    assert all(
        status.state is MCPServerState.CLOSED for status in manager.status_snapshot()
    )
