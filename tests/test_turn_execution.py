"""Hermetic coverage for the guarded turn-execution envelope."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
from fastapi import HTTPException

import api.schemas as api_schemas
import api.dependencies as api_dependencies
import main as main_module
from agent import DoneEvent, OrchestrationDecisionEvent, Session, SessionGuard, Tracer
from api.schemas import TokenUsage
from api.turn import PersistencePolicy, TurnRequest, TurnRunner
from config import Settings
from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    StreamChunk,
    TextBlock,
    TextDelta,
    Usage,
)
from orchestrator import Orchestrator
from orchestrator.schemas import OrchestrationDecision, OrchestrationResult
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

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        self.inventory_reads += 1
        if self.fail_next_inventory:
            self.fail_next_inventory = False
            raise RuntimeError("inventory unavailable")
        return deepcopy(self.tools)

    def list_tools(self) -> list[Any]:
        raise AssertionError("routing must use the turn snapshot, not live inventory")


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
            usage=Usage(total_tokens=2),
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
            usage=Usage(total_tokens=1),
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

    def get_entry(self, model_id: str) -> Any:
        return SimpleNamespace(max_tokens=256, context_window=4096)


class FakeOrchestrator:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(
        self, prompt, tools, history=None, timeout=None, log=None
    ):
        self.calls += 1
        return OrchestrationDecision(
            result=OrchestrationResult(
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
        "llm_model": "legacy-executing-model",
        "llm_max_retries": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _runner(
    *,
    agent: LLMClient,
    mcp: CountingMCP,
    settings: Settings | None = None,
    orchestrator: Any | None = None,
    registry: Any | None = None,
    tracer: Tracer | None = None,
) -> TurnRunner:
    from agent import InMemorySessionStore

    store = InMemorySessionStore()
    return TurnRunner(
        legacy_llm=agent,
        mcp=mcp,  # type: ignore[arg-type]
        store=store,
        guard=SessionGuard(),
        settings=settings or _settings(),
        orchestrator=orchestrator,
        registry=registry,
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
                usage=Usage(total_tokens=1),
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
        registry=registry,  # type: ignore[arg-type]
        system_prompt="route",
        model_id="router",
    )
    mcp = CountingMCP()
    runner = _runner(
        agent=agent,
        mcp=mcp,
        settings=_settings(llm_timeout_seconds=0.02, max_run_seconds=0.3),
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
        registry=registry,  # type: ignore[arg-type]
        system_prompt="route",
        model_id="router",
    )
    runner = _runner(
        agent=agent,
        mcp=CountingMCP(),
        settings=_settings(llm_timeout_seconds=1, max_run_seconds=0.025),
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
        registry=registry,  # type: ignore[arg-type]
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
    assert [tool["name"] for tool in agent.requests_seen[0].tools or []] == [
        "srv__one"
    ]


async def test_guard_releases_after_normal_completion_and_exception() -> None:
    agent = AnswerLLM()
    mcp = CountingMCP()
    runner = _runner(
        agent=agent, mcp=mcp, settings=_settings(orchestration_enabled=False)
    )
    session = await runner.store.create()

    await runner.run(TurnRequest("normal", session, PersistencePolicy.PERSISTENT))
    assert runner.guard.in_flight() == set()

    mcp.fail_next_inventory = True
    with pytest.raises(RuntimeError, match="inventory unavailable"):
        await runner.run(TurnRequest("raises", session, PersistencePolicy.PERSISTENT))
    assert runner.guard.in_flight() == set()

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


async def test_active_task_cancellation_releases_guard() -> None:
    class BlockingOrchestrator(FakeOrchestrator):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def decide(
            self, prompt, tools, history=None, timeout=None, log=None
        ):
            self.calls += 1
            self.started.set()
            await asyncio.Event().wait()

    agent = AnswerLLM()
    orchestrator = BlockingOrchestrator()
    runner = _runner(
        agent=agent,
        mcp=CountingMCP(),
        settings=_settings(llm_timeout_seconds=0, max_run_seconds=0),
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


async def test_legacy_metadata_reports_executing_model_without_decision() -> None:
    agent = AnswerLLM()
    runner = _runner(
        agent=agent, mcp=CountingMCP(), settings=_settings(orchestration_enabled=False)
    )
    session = await runner.store.create()

    request = TurnRequest(
        "hello",
        session,
        PersistencePolicy.PERSISTENT,
        model_id="legacy-executing-model",
    )
    async with runner.open(request) as execution:
        events = [event async for event in execution.events]

    result = await runner.run(
        TurnRequest(
            "again",
            session,
            PersistencePolicy.PERSISTENT,
            model_id="legacy-executing-model",
        )
    )

    assert not any(isinstance(event, OrchestrationDecisionEvent) for event in events)
    assert result.metadata.model_id == "legacy-executing-model"
    assert result.metadata.orchestration is None


async def test_model_inventory_uses_registry_or_legacy_setting() -> None:
    agent = AnswerLLM()
    legacy = _runner(
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

    assert legacy.available_model_ids() == ["legacy-executing-model"]
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


def test_runtime_wiring_names_optional_legacy_client_and_removes_dead_helpers() -> None:
    assert "legacy_llm" in TurnRunner.__dataclass_fields__
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
