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
    RunLimits,
    TextEvent,
    ToolCallEvent,
    run_agent,
)
from llm.client import LLMClient
from llm.providers.openai import _ReasoningStreamStripper
from llm.schemas import (
    AssistantMessage,
    Message,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    Usage,
)
from mcp_layer.client import ToolCallResult

pytestmark = pytest.mark.anyio


class FakeMCP:
    def __init__(self) -> None:
        self.call_count = 0

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [{
            "name": "srv__echo",
            "description": "echo",
            "input_schema": {"type": "object", "properties": {"value": {"type": "string"}}},
        }]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.call_count += 1
        return ToolCallResult(content=f"echo: {arguments.get('value')}", is_error=False)


class NativeStreamingLLM(LLMClient):
    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        raise AssertionError("complete() should not be used in native stream test")

    async def stream(self, messages: list[Message], tools=None, system=None,
                     max_tokens=None, thinking_level=None) -> AsyncIterator[StreamChunk]:
        yield TextDelta("Hel")
        yield TextDelta("lo")
        yield StreamEnd(AssistantMessage(
            content=[TextBlock("Hello")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        ))


class CompleteOnlyLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        self.calls += 1
        return AssistantMessage(
            content=[TextBlock("fallback text")],
            stop_reason="end_turn",
            usage=Usage(total_tokens=3),
        )


class ToolStreamingLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        raise AssertionError("complete() should not be used in stream mode")

    async def stream(self, messages: list[Message], tools=None, system=None,
                     max_tokens=None, thinking_level=None) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        if self.calls == 1:
            yield StreamEnd(AssistantMessage(
                content=[ToolUseBlock(id="call_1", name="srv__echo", input={"value": "hi"})],
                stop_reason="tool_use",
                usage=Usage(total_tokens=2),
            ))
            return
        yield TextDelta("done")
        yield StreamEnd(AssistantMessage(
            content=[TextBlock("done")],
            stop_reason="end_turn",
            usage=Usage(total_tokens=2),
        ))


class ErrorAfterDeltaLLM(LLMClient):
    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        raise AssertionError("complete() should not be used in stream mode")

    async def stream(self, messages: list[Message], tools=None, system=None,
                     max_tokens=None, thinking_level=None) -> AsyncIterator[StreamChunk]:
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


def run_stripper(pieces: list[str]) -> tuple[str, str | None]:
    stripper = _ReasoningStreamStripper()
    visible = "".join(stripper.feed(piece) for piece in pieces)
    visible += stripper.finish()
    return visible, stripper.reasoning


async def test_native_streaming_emits_incremental_text() -> None:
    events = await collect(NativeStreamingLLM())
    assert text_events(events) == ["Hel", "lo"]
    assert "".join(text_events(events)) == "Hello"
    assert done_reason(events) == "end_turn"

async def test_complete_fallback_streams_coarse_text() -> None:
    fallback = CompleteOnlyLLM()
    events = await collect(fallback)
    assert fallback.calls == 1
    assert text_events(events) == ["fallback text"]
    assert done_reason(events) == "end_turn"

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

@pytest.mark.parametrize(
    ("pieces", "expected_visible", "expected_reasoning", "assert_no_tags"),
    [
        (["<think>secret</think>Hello"], "Hello", "secret", True),
        (["<thi", "nk>secret</think>Hello"], "Hello", "secret", True),
        (["<think>sec", "ret</thi", "nk>Hello"], "Hello", "secret", True),
        (["Hello <think>not leading</think>"], "Hello <think>not leading</think>", None, False),
        (["   <think>secret</think>  Hello"], "Hello", "secret", True),
        (["plain"], "plain", None, True),
        (["<think>unfinished"], "<think>unfinished", None, False),
    ],
)
def test_reasoning_stripper_boundaries(
    pieces: list[str], expected_visible: str, expected_reasoning: str | None, assert_no_tags: bool
) -> None:
    visible, reasoning = run_stripper(pieces)
    assert visible == expected_visible
    assert reasoning == expected_reasoning
    if assert_no_tags:
        assert "<think>" not in visible and "</think>" not in visible
