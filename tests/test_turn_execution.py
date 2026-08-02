"""Hermetic coverage for the guarded turn-execution envelope."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
from fastapi import HTTPException

import api.schemas as api_schemas
import api.dependencies as api_dependencies
import main as main_module
from agent import (
    DoneEvent,
    OrchestrationDecisionEvent,
    RunLimits,
    Session,
    SessionGuard,
    Tracer,
)
from agent.runtime import ModelLimits
from api.schemas import TokenUsage
from api.turn import (
    OrchestratedRouting,
    PersistencePolicy,
    TurnRequest,
    TurnRunner,
    UnorchestratedRouting,
)
from config import Settings
from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    CompletionUsage,
)
from tooling import ToolCallResult
from orchestrator import Orchestrator
from orchestrator.contracts import ModelRegistry, RoutingService
from orchestrator.schemas import OrchestrationDecision, OrchestrationProposal
from tests._app_support import wired_app

pytestmark = pytest.mark.anyio


class CountingMCP:
    connected_servers: list[str] = []

    def __init__(self, tools: list[dict[str, Any]] | None = None) -> None:
        self.tools = tools or [
            {
                "name": "srv__one",
                "description": "the first tool",
                "input_schema": {"type": "object"},
            }
        ]
        self.inventory_reads = 0
        self.fail_next_inventory = False
        self.turn_entries = 0
        self.turn_exits = 0
        self.active_turns = 0
        self.turn_timeouts: list[float | None] = []

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        self.inventory_reads += 1
        if self.fail_next_inventory:
            self.fail_next_inventory = False
            raise RuntimeError("inventory unavailable")
        return deepcopy(self.tools)

    def list_tools(self) -> list[Any]:
        raise AssertionError("routing must use the turn snapshot, not live inventory")

    @asynccontextmanager
    async def open_turn(self, *, timeout_seconds: float | None = None):
        self.turn_timeouts.append(timeout_seconds)
        self.turn_entries += 1
        self.active_turns += 1
        try:
            yield self
        finally:
            self.active_turns -= 1
            self.turn_exits += 1

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        raise AssertionError(f"unexpected tool dispatch: {name} {arguments!r}")


class AnswerLLM(LLMClient):
    def __init__(self, answer: str = "done") -> None:
        self.answer = answer
        self.calls = 0
        self.requests_seen: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        self.requests_seen.append(request)
        return AssistantMessage(
            content=[TextBlock(self.answer)],
            stop_reason="end_turn",
            model="executing-provider-model",
            usage=CompletionUsage(total_tokens=2),
        )


class RoutingLLM(LLMClient):
    def __init__(self, payload: dict[str, Any], delay: float | None = None) -> None:
        self.payload = payload
        self.delay = delay
        self.calls = 0
        self.prompt = ""

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        self.prompt = "".join(
            block.text
            for block in request.messages[-1].content
            if isinstance(block, TextBlock)
        )
        if self.delay is None:
            await asyncio.Event().wait()
        elif self.delay:
            await asyncio.sleep(self.delay)
        return AssistantMessage(
            content=[TextBlock(json.dumps(self.payload))],
            stop_reason="end_turn",
            usage=CompletionUsage(total_tokens=1),
        )


class RegistryStub:
    model_ids = ["agent", "router"]

    def __init__(self, agent: LLMClient, router: LLMClient | None = None) -> None:
        self.agent = agent
        self.router = router or agent

    def default_id(self) -> str:
        return "agent"

    def describe_for_prompt(self) -> str:
        return "- agent\n    default agent"

    def get(self, model_id: str) -> LLMClient:
        return self.router if model_id == "router" else self.agent

    def get_or_default(self, model_id: str | None) -> tuple[str, LLMClient]:
        resolved = model_id if model_id in self.model_ids else "agent"
        return resolved, self.get(resolved)

    def get_entry(self, model_id: str) -> ModelLimits:
        return SimpleNamespace(max_tokens=256, context_window=4096)


class FakeOrchestrator:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, prompt, tools, history=None, timeout=None, log=None):
        self.calls += 1
        return OrchestrationDecision(
            result=OrchestrationProposal(
                selected_model_id="agent",
                selected_tools=[tool.name for tool in tools.tools],
                generated_system_prompt="routed system",
            )
        )


class RecordingTracer(Tracer):
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def start(self) -> None:
        pass

    def emit(self, record: dict) -> None:
        self.records.append(record)

    async def aclose(self) -> None:
        pass

    @property
    def accepted(self) -> int:
        return len(self.records)

    @property
    def written(self) -> int:
        return len(self.records)

    @property
    def dropped(self) -> int:
        return 0

    @property
    def writer_failures(self) -> int:
        return 0


def _settings(**overrides: Any) -> Settings:
    values = {
        "orchestration_enabled": True,
        "llm": {
            "model_name": "unorchestrated-executing-model",
            "max_retries": 0,
        },
    }
    llm_overrides = overrides.pop("llm", {})
    values["llm"].update(llm_overrides)
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _runner(
    *,
    agent: LLMClient,
    mcp: CountingMCP,
    settings: Settings | None = None,
    orchestrator: RoutingService | None = None,
    registry: ModelRegistry | None = None,
    tracer: Tracer | None = None,
) -> TurnRunner:
    from agent import InMemorySessionStore

    store = InMemorySessionStore()
    resolved_settings = settings or _settings()
    if orchestrator is not None and registry is not None:
        routing = OrchestratedRouting(
            orchestrator=orchestrator,
            registry=registry,
        )
    else:
        routing = UnorchestratedRouting(
            llm=agent,
            model_id=resolved_settings.llm.model_name,
        )
    return TurnRunner(
        routing=routing,
        limits=RunLimits.from_settings(resolved_settings),
        mcp=mcp,
        store=store,
        guard=SessionGuard(),
        policy=None,
        tracer=tracer,
    )


async def test_busy_session_rejects_before_inventory_or_routing() -> None:
    agent = AnswerLLM()
    mcp = CountingMCP()
    orchestrator = FakeOrchestrator()
    registry = RegistryStub(agent)
    runner = _runner(agent=agent, mcp=mcp, orchestrator=orchestrator, registry=registry)
    session = await runner.store.create()

    async with runner.open(TurnRequest("first", session, PersistencePolicy.PERSISTENT)):
        with pytest.raises(HTTPException) as exc_info:
            await runner.run(
                TurnRequest("second", session, PersistencePolicy.PERSISTENT)
            )
        assert exc_info.value.status_code == 409

    assert orchestrator.calls == 1
    assert mcp.inventory_reads == 1


async def test_distinct_sessions_execute_concurrently() -> None:
    class BarrierLLM(AnswerLLM):
        def __init__(self) -> None:
            super().__init__()
            self.both_started = asyncio.Event()

        async def complete(self, request: GenerationRequest) -> AssistantMessage:
            self.calls += 1
            if self.calls == 2:
                self.both_started.set()
            await self.both_started.wait()
            return AssistantMessage(
                content=[TextBlock("done")],
                stop_reason="end_turn",
                usage=CompletionUsage(total_tokens=1),
            )

    agent = BarrierLLM()
    runner = _runner(
        agent=agent, mcp=CountingMCP(), settings=_settings(orchestration_enabled=False)
    )
    first = await runner.store.create()
    second = await runner.store.create()

    results = await asyncio.wait_for(
        asyncio.gather(
            runner.run(TurnRequest("one", first, PersistencePolicy.PERSISTENT)),
            runner.run(TurnRequest("two", second, PersistencePolicy.PERSISTENT)),
        ),
        timeout=0.5,
    )

    assert [result.answer for result in results] == ["done", "done"]
    assert agent.calls == 2


async def test_routing_timeout_falls_back_while_turn_budget_remains() -> None:
    agent = AnswerLLM()
    router = RoutingLLM({})
    registry = RegistryStub(agent, router)
    orchestrator = Orchestrator(
        registry=registry,
        system_prompt="route",
        model_id="router",
    )
    mcp = CountingMCP()
    runner = _runner(
        agent=agent,
        mcp=mcp,
        settings=_settings(llm={"timeout_seconds": 0.02}, run_max_seconds=0.3),
        orchestrator=orchestrator,
        registry=registry,
    )
    session = await runner.store.create()

    started = time.perf_counter()
    result = await runner.run(
        TurnRequest("hello", session, PersistencePolicy.PERSISTENT)
    )
    elapsed = time.perf_counter() - started

    assert result.answer == "done"
    assert result.metadata.orchestration is not None
    assert result.metadata.orchestration.fallback_used is True
    assert result.metadata.orchestration.tools == ["srv__one"]
    assert router.calls == 1 and agent.calls == 1
    assert mcp.inventory_reads == 1
    assert elapsed < 0.15


async def test_routing_consumes_absolute_deadline_and_skips_agent_model() -> None:
    agent = AnswerLLM()
    router = RoutingLLM({})
    registry = RegistryStub(agent, router)
    orchestrator = Orchestrator(
        registry=registry,
        system_prompt="route",
        model_id="router",
    )
    runner = _runner(
        agent=agent,
        mcp=CountingMCP(),
        settings=_settings(llm={"timeout_seconds": 1}, run_max_seconds=0.025),
        orchestrator=orchestrator,
        registry=registry,
    )
    session = await runner.store.create()

    result = await runner.run(
        TurnRequest("hello", session, PersistencePolicy.PERSISTENT)
    )

    assert result.done_reason == "deadline_exceeded"
    assert agent.calls == 0
    assert result.metadata.orchestration is not None
    assert result.metadata.orchestration.fallback_used is True


async def test_one_inventory_snapshot_drives_prompt_sanitize_and_filtering() -> None:
    agent = AnswerLLM()
    router = RoutingLLM(
        {
            "selected_model_id": "unknown-model",
            "selected_tools": ["srv__one", "srv__unknown"],
            "generated_system_prompt": "use the selected tool",
        },
        delay=0,
    )
    registry = RegistryStub(agent, router)
    orchestrator = Orchestrator(
        registry=registry,
        system_prompt="route",
        model_id="router",
    )
    mcp = CountingMCP()
    runner = _runner(agent=agent, mcp=mcp, orchestrator=orchestrator, registry=registry)
    session = await runner.store.create()

    result = await runner.run(
        TurnRequest("use a tool", session, PersistencePolicy.PERSISTENT)
    )

    assert mcp.inventory_reads == 1
    assert "srv__one" in router.prompt
    assert result.metadata.model_id == "agent"
    assert result.metadata.orchestration is not None
    assert result.metadata.orchestration.tools == ["srv__one"]
    assert [tool["name"] for tool in agent.requests_seen[0].tools or []] == ["srv__one"]
    assert agent.requests_seen[0].max_tokens == 256
    model_limits = runner.limits.for_model(registry.get_entry("agent"))
    assert model_limits.max_tokens == 256
    assert model_limits.context_window == 4096


async def test_guard_releases_after_normal_completion_and_exception() -> None:
    agent = AnswerLLM()
    mcp = CountingMCP()
    runner = _runner(
        agent=agent, mcp=mcp, settings=_settings(orchestration_enabled=False)
    )
    session = await runner.store.create()

    await runner.run(TurnRequest("normal", session, PersistencePolicy.PERSISTENT))
    assert runner.guard.in_flight() == set()
    assert mcp.turn_entries == mcp.turn_exits == 1
    assert mcp.active_turns == 0

    mcp.fail_next_inventory = True
    with pytest.raises(RuntimeError, match="inventory unavailable"):
        await runner.run(TurnRequest("raises", session, PersistencePolicy.PERSISTENT))
    assert runner.guard.in_flight() == set()
    assert mcp.active_turns == 0

    await runner.run(
        TurnRequest("after exception", session, PersistencePolicy.PERSISTENT)
    )
    assert agent.calls == 2


async def test_guard_releases_when_stream_is_closed_early() -> None:
    class ClosingStreamLLM(AnswerLLM):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        async def stream(
            self, request: GenerationRequest
        ) -> AsyncIterator[StreamChunk]:
            try:
                yield TextDelta("partial")
                await asyncio.Event().wait()
            finally:
                self.closed = True

    agent = ClosingStreamLLM()
    runner = _runner(
        agent=agent, mcp=CountingMCP(), settings=_settings(orchestration_enabled=False)
    )
    session = await runner.store.create()

    async with runner.open(
        TurnRequest("stream", session, PersistencePolicy.PERSISTENT, stream=True)
    ) as execution:
        event = await anext(execution.events)
        assert event.type == "text"

    assert runner.guard.in_flight() == set()
    assert agent.closed is True
    assert runner.mcp.active_turns == 0  # type: ignore[attr-defined]
    assert runner.mcp.turn_entries == runner.mcp.turn_exits == 1  # type: ignore[attr-defined]


async def test_early_generator_close_after_tool_dispatch_closes_turn_lease() -> None:
    class LeaseMCP(CountingMCP):
        def __init__(self) -> None:
            super().__init__()
            self.lease_active = False
            self.calls = 0

        @asynccontextmanager
        async def open_turn(self, *, timeout_seconds: float | None = None):
            self.turn_entries += 1
            self.active_turns += 1
            try:
                yield self
            finally:
                self.lease_active = False
                self.active_turns -= 1
                self.turn_exits += 1

        async def call_tool(
            self, name: str, arguments: dict[str, Any]
        ) -> ToolCallResult:
            self.calls += 1
            self.lease_active = True
            return ToolCallResult("tool result", False)

    class ToolThenBlockingStream(AnswerLLM):
        def __init__(self) -> None:
            super().__init__()
            self.stream_calls = 0
            self.closed = False

        async def stream(
            self, request: GenerationRequest
        ) -> AsyncIterator[StreamChunk]:
            self.stream_calls += 1
            if self.stream_calls == 1:
                yield StreamEnd(
                    AssistantMessage(
                        content=[ToolUseBlock("call-1", "srv__one", {"value": 1})],
                        stop_reason="tool_use",
                        usage=CompletionUsage(total_tokens=1),
                    )
                )
                return
            try:
                yield TextDelta("partial")
                await asyncio.Event().wait()
            finally:
                self.closed = True

    agent = ToolThenBlockingStream()
    mcp = LeaseMCP()
    runner = _runner(
        agent=agent,
        mcp=mcp,
        settings=_settings(orchestration_enabled=False),
    )
    session = await runner.store.create()

    async with runner.open(
        TurnRequest("stream", session, PersistencePolicy.PERSISTENT, stream=True)
    ) as execution:
        async for event in execution.events:
            if event.type == "text":
                break
        assert mcp.lease_active is True

    assert mcp.calls == 1
    assert mcp.lease_active is False
    assert mcp.active_turns == 0
    assert mcp.turn_entries == mcp.turn_exits == 1
    assert agent.closed is True


async def test_turn_passes_remaining_deadline_to_catalog_refresh() -> None:
    mcp = CountingMCP()
    runner = _runner(
        agent=AnswerLLM(),
        mcp=mcp,
        settings=_settings(orchestration_enabled=False, run_max_seconds=1),
    )
    session = await runner.store.create()

    await runner.run(TurnRequest("go", session, PersistencePolicy.PERSISTENT))

    assert len(mcp.turn_timeouts) == 1
    timeout = mcp.turn_timeouts[0]
    assert timeout is not None
    assert 0 < timeout <= 1


async def test_active_task_cancellation_releases_guard() -> None:
    class BlockingOrchestrator(FakeOrchestrator):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def decide(self, prompt, tools, history=None, timeout=None, log=None):
            self.calls += 1
            self.started.set()
            await asyncio.Event().wait()

    agent = AnswerLLM()
    orchestrator = BlockingOrchestrator()
    runner = _runner(
        agent=agent,
        mcp=CountingMCP(),
        settings=_settings(llm={"timeout_seconds": 0}, run_max_seconds=0),
        orchestrator=orchestrator,
        registry=RegistryStub(agent),
    )
    session = await runner.store.create()
    task = asyncio.create_task(
        runner.run(TurnRequest("cancel", session, PersistencePolicy.PERSISTENT))
    )
    await orchestrator.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runner.guard.in_flight() == set()
    assert runner.mcp.active_turns == 0  # type: ignore[attr-defined]


async def test_native_sse_trace_and_logs_share_one_run_id(
    asgi_client, parse_sse, caplog: pytest.LogCaptureFixture
) -> None:
    agent = AnswerLLM()
    mcp = CountingMCP()
    orchestrator = FakeOrchestrator()
    registry = RegistryStub(agent)
    tracer = RecordingTracer()

    caplog.set_level(logging.INFO)
    with wired_app(agent, mcp=mcp, registry=registry) as (app, _settings_obj):
        app.state.orchestrator = orchestrator
        app.state.tracer = tracer
        response = await asgi_client(app).post("/chat/stream", content="hello")

    events = parse_sse(response.text)
    run_ids = {event["run_id"] for event in events}
    assert len(run_ids) == 1
    run_id = run_ids.pop()
    assert events[0]["type"] == "orchestration"
    assert tracer.records[0]["type"] == "orchestration"
    assert {record["run_id"] for record in tracer.records} == {run_id}
    assert any(f"[run {run_id}]" in record.getMessage() for record in caplog.records)


async def test_unorchestrated_metadata_reports_executing_model_without_decision() -> (
    None
):
    agent = AnswerLLM()
    runner = _runner(
        agent=agent, mcp=CountingMCP(), settings=_settings(orchestration_enabled=False)
    )
    session = await runner.store.create()

    request = TurnRequest(
        "hello",
        session,
        PersistencePolicy.PERSISTENT,
        model_id="unorchestrated-executing-model",
    )
    async with runner.open(request) as execution:
        events = [event async for event in execution.events]

    result = await runner.run(
        TurnRequest(
            "again",
            session,
            PersistencePolicy.PERSISTENT,
            model_id="unorchestrated-executing-model",
        )
    )

    assert not any(isinstance(event, OrchestrationDecisionEvent) for event in events)
    assert result.metadata.model_id == "unorchestrated-executing-model"
    assert result.metadata.orchestration is None


async def test_model_inventory_uses_registry_or_unorchestrated_setting() -> None:
    agent = AnswerLLM()
    unorchestrated = _runner(
        agent=agent,
        mcp=CountingMCP(),
        settings=_settings(orchestration_enabled=False),
    )
    orchestrated = _runner(
        agent=agent,
        mcp=CountingMCP(),
        orchestrator=FakeOrchestrator(),
        registry=RegistryStub(agent),
    )

    assert unorchestrated.available_model_ids() == ["unorchestrated-executing-model"]
    assert orchestrated.available_model_ids() == ["agent", "router"]


async def test_explicit_unknown_model_is_rejected_before_turn_setup() -> None:
    agent = AnswerLLM()
    mcp = CountingMCP()
    orchestrator = FakeOrchestrator()
    runner = _runner(
        agent=agent,
        mcp=mcp,
        orchestrator=orchestrator,
        registry=RegistryStub(agent),
    )

    with pytest.raises(HTTPException) as exc_info:
        await runner.run(
            TurnRequest(
                "hello",
                Session(),
                PersistencePolicy.EPHEMERAL,
                model_id="unknown",
            )
        )

    assert exc_info.value.status_code == 400
    assert mcp.inventory_reads == 0
    assert orchestrator.calls == 0
    assert agent.calls == 0


async def test_ephemeral_turn_does_not_publish_session_to_store() -> None:
    agent = AnswerLLM()
    runner = _runner(
        agent=agent,
        mcp=CountingMCP(),
        settings=_settings(orchestration_enabled=False),
    )
    original_ids = runner.store.ids()  # type: ignore[attr-defined]

    result = await runner.run(
        TurnRequest(
            "hello",
            Session(),
            PersistencePolicy.EPHEMERAL,
        )
    )

    assert result.answer == "done"
    assert runner.store.ids() == original_ids  # type: ignore[attr-defined]


async def test_turn_request_rejects_untyped_persistence_policy() -> None:
    with pytest.raises(TypeError, match="persistence must be a PersistencePolicy"):
        TurnRequest(
            "hello",
            Session(),
            "persistent",  # type: ignore[arg-type]
        )


def test_route_facing_turn_contract_has_no_preference_plumbing() -> None:
    assert "preferences" not in TurnRequest.__dataclass_fields__
    for method in (
        TurnRunner.open,
        TurnRunner.run,
        TurnRunner._resolve_routing,
        TurnRunner._events,
    ):
        assert "preferences" not in inspect.signature(method).parameters


def test_runtime_wiring_names_optional_unorchestrated_client_and_removes_dead_helpers() -> (
    None
):
    assert list(inspect.signature(TurnRunner).parameters) == [
        "routing",
        "limits",
        "mcp",
        "store",
        "guard",
        "policy",
        "tracer",
    ]
    assert "routing" in TurnRunner.__dataclass_fields__
    assert "unorchestrated_llm" not in TurnRunner.__dataclass_fields__
    assert "unorchestrated_model_id" not in TurnRunner.__dataclass_fields__
    assert "orchestrator" not in TurnRunner.__dataclass_fields__
    assert "registry" not in TurnRunner.__dataclass_fields__
    assert "limits" in TurnRunner.__dataclass_fields__
    assert "settings" not in TurnRunner.__dataclass_fields__
    assert "llm" not in TurnRunner.__dataclass_fields__
    assert not hasattr(api_dependencies, "get_llm")
    assert list(inspect.signature(main_module._try_build_orchestration).parameters) == [
        "settings"
    ]


def test_token_usage_conversion_preserves_every_done_field() -> None:
    event = DoneEvent(
        reason="end_turn",
        iterations=3,
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
        thinking_tokens=5,
    )

    assert TokenUsage.from_done_event(event).model_dump() == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "thinking_tokens": 5,
    }


def test_obsolete_orchestration_schema_is_removed() -> None:
    assert not hasattr(api_schemas, "OrchestrationInfo")
