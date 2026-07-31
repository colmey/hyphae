"""Narrow ownership contracts for the private agent-run composition object."""

from __future__ import annotations

from typing import Any

import pytest

from agent import RunLimits, Session, TextEvent, ToolPolicy, UsageEvent
from agent.loop import _AgentRun
from agent.tool_execution import ToolDispatcher
from llm.client import GenerationRequest, LLMClient
from llm.schemas import AssistantMessage, CompletionUsage, TextBlock
from mcp_layer import ToolCallResult


class _BufferedLLM(LLMClient):
    def __init__(self, response: AssistantMessage) -> None:
        self.response = response

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        return self.response


class _ToolRuntime:
    def __init__(self) -> None:
        self.tools: list[dict[str, Any]] = [
            {
                "name": "server__tool",
                "description": "test tool",
                "input_schema": {},
            }
        ]

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return self.tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult:
        return ToolCallResult(content="ok", is_error=False)


def _response() -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock("done")],
        stop_reason="end_turn",
        usage=CompletionUsage(),
    )


def test_from_inputs_owns_resolved_defaults_and_one_dispatcher() -> None:
    runtime = _ToolRuntime()
    run = _AgentRun.from_inputs(
        Session(),
        _BufferedLLM(_response()),
        runtime,
    )

    assert run.limits == RunLimits()
    assert run.tool_runtime is runtime
    assert run.visible_tools is runtime.tools
    assert isinstance(run.tool_policy, ToolPolicy)
    assert run.context.deadline is None
    assert isinstance(run.tool_dispatcher, ToolDispatcher)
    assert run.iteration == 0
    assert run.cumulative_usage == CompletionUsage()
    assert run.active_tool_batch is None


@pytest.mark.anyio
async def test_record_response_clears_short_lived_generation_state() -> None:
    session = Session()
    session.append_user("go")
    response = _response()
    run = _AgentRun.from_inputs(
        session,
        _BufferedLLM(response),
        _ToolRuntime(),
    )
    dispatcher = run.tool_dispatcher
    run.iteration = 1

    prepared = await run._prepare_generation()
    generation_events = [
        event async for event in run._consume_generation(prepared)
    ]

    assert generation_events == []
    assert run._pending_response is response
    assert run.tool_dispatcher is dispatcher

    record_events = [
        event async for event in run._record_response(prepared)
    ]

    assert [type(event) for event in record_events] == [UsageEvent, TextEvent]
    assert run._pending_response is None
    assert run._pending_streamed_reasoning is False
    assert run._pending_latency_ms is None
    assert run.active_tool_batch is None
    assert run._take_terminal_reason() == "end_turn"
    assert run._take_terminal_reason() is None
    assert run.tool_dispatcher is dispatcher
