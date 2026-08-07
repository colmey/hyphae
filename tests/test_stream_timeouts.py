from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from hyphae.agent import DoneEvent, ErrorEvent, TextEvent
from hyphae.llm.client import GenerationRequest, LLMClient
from hyphae.llm.schemas import (
    AssistantMessage,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    CompletionUsage,
)


class StallingLLM(LLMClient):
    def __init__(self, position: str, *, recover_on_retry: bool = False) -> None:
        self.position = position
        self.recover_on_retry = recover_on_retry
        self.calls = 0
        self.closed = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("stream() must be used")

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        call = self.calls
        try:
            if self.recover_on_retry and call > 1:
                yield TextDelta("recovered")
                yield StreamEnd(_answer("recovered"))
                return
            if self.position == "before_first":
                await asyncio.Event().wait()
            yield TextDelta("first")
            if self.position == "between_chunks":
                await asyncio.Event().wait()
            yield TextDelta("second")
            if self.position == "before_end":
                await asyncio.Event().wait()
            yield StreamEnd(_answer("firstsecond"))
        finally:
            self.closed += 1


class PacedLLM(LLMClient):
    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("stream() must be used")

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        for text in ("a", "b", "c"):
            await asyncio.sleep(0.02)
            yield TextDelta(text)
        yield StreamEnd(_answer("abc"))


class BlockingCompleteLLM(LLMClient):
    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _answer(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text)],
        stop_reason="end_turn",
        usage=CompletionUsage(total_tokens=1),
    )


def _done_reason(events: list[Any]) -> str:
    return next(
        event.reason for event in reversed(events) if isinstance(event, DoneEvent)
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("position", "expected_text", "expected_calls"),
    [
        ("before_first", [], 2),
        ("between_chunks", ["first"], 1),
        ("before_end", ["first", "second"], 1),
    ],
)
async def test_each_blocked_stream_read_is_timed_and_closed(
    position: str,
    expected_text: list[str],
    expected_calls: int,
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    llm = StallingLLM(position)

    events = await agent_event_collector(
        llm=llm,
        mcp=scripted_mcp_factory(tools=[]),
        stream=True,
        retry_base_delay=0,
        llm_timeout_seconds=0.02,
        max_retries=1,
    )

    assert [
        event.text for event in events if isinstance(event, TextEvent)
    ] == expected_text
    assert _done_reason(events) == "llm_error"
    assert sum(isinstance(event, ErrorEvent) for event in events) == 1
    assert llm.calls == expected_calls
    assert llm.closed == expected_calls


@pytest.mark.anyio
@pytest.mark.parametrize("llm_timeout", [None, 1.0])
async def test_absolute_deadline_interrupts_a_blocked_stream_read(
    llm_timeout: float | None,
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    llm = StallingLLM("before_first")

    events = await agent_event_collector(
        llm=llm,
        mcp=scripted_mcp_factory(tools=[]),
        stream=True,
        retry_base_delay=0,
        llm_timeout_seconds=llm_timeout,
        max_run_seconds=0.03,
        max_retries=3,
    )

    assert _done_reason(events) == "deadline_exceeded"
    assert not any(isinstance(event, ErrorEvent) for event in events)
    assert llm.calls == 1
    assert llm.closed == 1


@pytest.mark.anyio
async def test_pre_visible_timeout_can_retry_after_closing_first_stream(
    scripted_mcp_factory, agent_event_collector
) -> None:
    llm = StallingLLM("before_first", recover_on_retry=True)

    events = await agent_event_collector(
        llm=llm,
        mcp=scripted_mcp_factory(tools=[]),
        stream=True,
        retry_base_delay=0,
        llm_timeout_seconds=0.02,
        max_retries=1,
    )

    assert [event.text for event in events if isinstance(event, TextEvent)] == [
        "recovered"
    ]
    assert _done_reason(events) == "end_turn"
    assert not any(isinstance(event, ErrorEvent) for event in events)
    assert llm.calls == 2
    assert llm.closed == 2


@pytest.mark.anyio
async def test_per_read_timeout_resets_after_each_chunk(
    scripted_mcp_factory, agent_event_collector
) -> None:
    events = await agent_event_collector(
        llm=PacedLLM(),
        mcp=scripted_mcp_factory(tools=[]),
        stream=True,
        retry_base_delay=0,
        llm_timeout_seconds=0.05,
        max_retries=0,
    )

    assert [event.text for event in events if isinstance(event, TextEvent)] == [
        "a",
        "b",
        "c",
    ]
    assert _done_reason(events) == "end_turn"


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True], ids=["complete", "stream-fallback"])
async def test_timeout_terminal_semantics_match_complete_and_stream_paths(
    stream: bool,
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    events = await agent_event_collector(
        llm=BlockingCompleteLLM(),
        mcp=scripted_mcp_factory(tools=[]),
        stream=stream,
        llm_timeout_seconds=0.02,
        max_retries=0,
    )

    assert [type(event) for event in events] == [ErrorEvent, DoneEvent]
    assert _done_reason(events) == "llm_error"
