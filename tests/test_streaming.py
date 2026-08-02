"""Hermetic pytest coverage for token streaming.

Scripted LLM clients drive ``run_agent`` directly, with no network or provider
backend.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from agent import (
    DoneEvent,
    ErrorEvent,
    InMemorySessionStore,
    ReasoningEvent,
    RunLimits,
    TextEvent,
    ToolCallEvent,
    run_agent,
)
from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    ReasoningDelta,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    CompletionUsage,
)
from tooling import ToolCallResult

pytestmark = pytest.mark.anyio


class FakeMCP:
    def __init__(self) -> None:
        self.call_count = 0

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "srv__echo",
                "description": "echo",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            }
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.call_count += 1
        return ToolCallResult(content=f"echo: {arguments.get('value')}", is_error=False)


class NativeStreamingLLM(LLMClient):
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("complete() should not be used in native stream test")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        yield ReasoningDelta("considering")
        yield TextDelta("Hel")
        yield TextDelta("lo")
        yield StreamEnd(
            AssistantMessage(
                content=[TextBlock("Hello")],
                stop_reason="end_turn",
                usage=CompletionUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        )


class CompleteOnlyLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        self.requests.append(request)
        return AssistantMessage(
            content=[TextBlock("fallback text")],
            stop_reason="end_turn",
            usage=CompletionUsage(total_tokens=3),
            reasoning="fallback thought",
        )


class ToolStreamingLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("complete() should not be used in stream mode")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        if self.calls == 1:
            yield StreamEnd(
                AssistantMessage(
                    content=[
                        ToolUseBlock(
                            id="call_1", name="srv__echo", input={"value": "hi"}
                        )
                    ],
                    stop_reason="tool_use",
                    usage=CompletionUsage(total_tokens=2),
                )
            )
            return
        yield TextDelta("done")
        yield StreamEnd(
            AssistantMessage(
                content=[TextBlock("done")],
                stop_reason="end_turn",
                usage=CompletionUsage(total_tokens=2),
            )
        )


class ErrorAfterDeltaLLM(LLMClient):
    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("complete() should not be used in stream mode")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        yield TextDelta("partial")
        raise RuntimeError("stream broke")


async def collect(llm: LLMClient, mcp: FakeMCP | None = None) -> list[Any]:
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    events: list[Any] = []
    async for event in run_agent(
        session=session,
        llm=llm,
        mcp=mcp or FakeMCP(),
        store=store,
        stream=True,
        limits=RunLimits(max_retries=1, retry_base_delay=0.0),
    ):
        events.append(event)
    return events


def text_events(events: list[Any]) -> list[str]:
    return [event.text for event in events if isinstance(event, TextEvent)]


def done_reason(events: list[Any]) -> str | None:
    done = [event for event in events if isinstance(event, DoneEvent)]
    return done[-1].reason if done else None


async def test_native_streaming_emits_incremental_text() -> None:
    events = await collect(NativeStreamingLLM())
    reasoning = [event for event in events if isinstance(event, ReasoningEvent)]
    assert [event.text for event in reasoning] == ["considering"]
    assert events.index(reasoning[0]) < next(
        index for index, event in enumerate(events) if isinstance(event, TextEvent)
    )
    assert text_events(events) == ["Hel", "lo"]
    assert "".join(text_events(events)) == "Hello"
    assert done_reason(events) == "end_turn"


async def test_complete_fallback_streams_coarse_text() -> None:
    fallback = CompleteOnlyLLM()
    events = await collect(fallback)
    reasoning = [event for event in events if isinstance(event, ReasoningEvent)]
    assert [event.text for event in reasoning] == ["fallback thought"]
    assert events.index(reasoning[0]) < next(
        index for index, event in enumerate(events) if isinstance(event, TextEvent)
    )
    assert fallback.calls == 1
    assert text_events(events) == ["fallback text"]
    assert done_reason(events) == "end_turn"


async def test_agent_buffered_and_streaming_generation_fields_match() -> None:
    native = NativeStreamingLLM()
    buffered = CompleteOnlyLLM()

    await collect(native)
    await collect(buffered)

    native_request = native.requests[0]
    buffered_request = buffered.requests[0]
    assert native_request.messages == buffered_request.messages
    assert native_request.tools == buffered_request.tools
    assert native_request.system == buffered_request.system
    assert native_request.max_tokens == buffered_request.max_tokens
    assert native_request.response_schema == buffered_request.response_schema
    assert native_request.thinking_level == buffered_request.thinking_level


async def test_streamed_tool_call_reaches_dispatch() -> None:
    mcp = FakeMCP()
    events = await collect(ToolStreamingLLM(), mcp)
    assert mcp.call_count == 1
    assert any(isinstance(event, ToolCallEvent) for event in events)
    assert text_events(events) == ["done"]
    assert done_reason(events) == "end_turn"


async def test_error_after_first_delta_surfaces_partial_text() -> None:
    events = await collect(ErrorAfterDeltaLLM())
    assert text_events(events) == ["partial"]
    assert any(isinstance(event, ErrorEvent) for event in events)
    assert done_reason(events) == "llm_error"
