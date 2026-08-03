"""Plan 08 parity coverage for shared buffered/streaming attempt policy."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

import agent.generation as generation_module
from agent import (
    DoneEvent,
    ErrorEvent,
    InMemorySessionStore,
    ReasoningEvent,
    RunContext,
    RunLimits,
    TextEvent,
    run_agent,
)
from agent.runtime import RunDeadlineExceeded
from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    ReasoningDelta,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    CompletionUsage,
)

pytestmark = pytest.mark.anyio


class TransientFailure(RuntimeError):
    pass


def _answer(text: str = "ok") -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text)], stop_reason="end_turn", usage=CompletionUsage(total_tokens=1)
    )


def _empty() -> AssistantMessage:
    return AssistantMessage(content=[], stop_reason="empty", usage=CompletionUsage())


def _done(events: list[Any]) -> str:
    return next(event.reason for event in reversed(events) if isinstance(event, DoneEvent))


def _capture_retry_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []

    async def record(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(generation_module.asyncio, "sleep", record)
    return sleeps


class ParityLLM(LLMClient):
    def __init__(self, script: list[AssistantMessage | BaseException]) -> None:
        self.script = list(script)
        self.complete_calls = 0
        self.stream_calls = 0
        self.stream_closes = 0

    def _next(self) -> AssistantMessage | BaseException:
        if not self.script:
            raise AssertionError("LLM script exhausted")
        return self.script.pop(0)

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.complete_calls += 1
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        return item

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        self.stream_calls += 1
        try:
            item = self._next()
            if isinstance(item, BaseException):
                raise item
            for block in item.content:
                if isinstance(block, TextBlock) and block.text:
                    yield TextDelta(block.text)
            yield StreamEnd(item)
        finally:
            self.stream_closes += 1

    def is_transient_error(self, exc: BaseException) -> bool:
        return isinstance(exc, TransientFailure)


def _generation_context() -> RunContext:
    return RunContext.start(
        max_run_seconds=None,
        base_logger=logging.getLogger("generation-test"),
    )


async def _generation_chunks(
    llm: LLMClient,
    *,
    stream: bool,
    limits: RunLimits | None = None,
) -> list[StreamChunk]:
    return [
        chunk
        async for chunk in generation_module.generate_with_retry(
            llm,
            GenerationRequest(messages=[]),
            limits=limits or RunLimits(),
            context=_generation_context(),
            stream=stream,
            log=logging.getLogger("generation-test"),
        )
    ]


async def test_buffered_generation_normalizes_to_one_stream_end() -> None:
    llm = ParityLLM([_answer("buffered")])

    chunks = await _generation_chunks(llm, stream=False)

    assert len(chunks) == 1
    assert isinstance(chunks[0], StreamEnd)
    assert chunks[0].message == _answer("buffered")
    assert not any(isinstance(chunk, TextDelta) for chunk in chunks)


class NativeSignalLLM(LLMClient):
    def __init__(self) -> None:
        self.closes = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("stream expected")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        try:
            yield ReasoningDelta("thinking")
            yield TextDelta("streamed")
            yield StreamEnd(_answer("streamed"))
        finally:
            self.closes += 1


async def test_streaming_generation_forwards_signals_and_one_end() -> None:
    llm = NativeSignalLLM()

    chunks = await _generation_chunks(llm, stream=True)

    assert chunks == [
        ReasoningDelta("thinking"),
        TextDelta("streamed"),
        StreamEnd(_answer("streamed")),
    ]
    assert llm.closes == 1


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_first_attempt_success_has_no_retry_sleep(
    stream: bool,
    monkeypatch: pytest.MonkeyPatch,
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    sleeps = _capture_retry_sleeps(monkeypatch)
    llm = ParityLLM([_answer()])
    events = await agent_event_collector(
        llm=llm, mcp=scripted_mcp_factory(tools=[]), stream=stream, max_retries=2
    )

    assert _done(events) == "end_turn"
    assert llm.complete_calls + llm.stream_calls == 1
    assert sleeps == []
    assert llm.stream_closes == (1 if stream else 0)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_transient_failure_then_success_has_exact_calls_and_sleep(
    stream: bool,
    monkeypatch: pytest.MonkeyPatch,
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    sleeps = _capture_retry_sleeps(monkeypatch)
    monkeypatch.setattr(generation_module.random, "uniform", lambda low, high: 0.0)
    llm = ParityLLM([TransientFailure("retry me"), _answer("recovered")])
    events = await agent_event_collector(
        llm=llm,
        mcp=scripted_mcp_factory(tools=[]),
        stream=stream,
        max_retries=2,
        retry_base_delay=0.25,
    )

    assert _done(events) == "end_turn"
    assert llm.complete_calls + llm.stream_calls == 2
    assert sleeps == [0.25]
    assert llm.stream_closes == (2 if stream else 0)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_non_transient_failure_is_not_retried(
    stream: bool,
    monkeypatch: pytest.MonkeyPatch,
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    sleeps = _capture_retry_sleeps(monkeypatch)
    llm = ParityLLM([ValueError("stop")])
    events = await agent_event_collector(
        llm=llm, mcp=scripted_mcp_factory(tools=[]), stream=stream, max_retries=2
    )

    assert _done(events) == "llm_error"
    assert llm.complete_calls + llm.stream_calls == 1
    assert sleeps == []
    assert llm.stream_closes == (1 if stream else 0)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_empty_response_recovers_and_exhaustion_returns_empty(
    stream: bool,
    monkeypatch: pytest.MonkeyPatch,
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    sleeps = _capture_retry_sleeps(monkeypatch)
    monkeypatch.setattr(generation_module.random, "uniform", lambda low, high: 0.0)

    recovered = ParityLLM([_empty(), _answer("recovered")])
    recovered_events = await agent_event_collector(
        llm=recovered,
        mcp=scripted_mcp_factory(tools=[]),
        stream=stream,
        max_retries=1,
        retry_base_delay=0.1,
    )
    exhausted = ParityLLM([_empty(), _empty()])
    exhausted_events = await agent_event_collector(
        llm=exhausted,
        mcp=scripted_mcp_factory(tools=[]),
        stream=stream,
        max_retries=1,
        retry_base_delay=0.1,
    )

    assert _done(recovered_events) == "end_turn"
    assert _done(exhausted_events) == "empty"
    assert recovered.complete_calls + recovered.stream_calls == 2
    assert exhausted.complete_calls + exhausted.stream_calls == 2
    assert sleeps == [0.1, 0.1]
    if stream:
        assert recovered.stream_closes == exhausted.stream_closes == 2


class IncompleteStreamLLM(LLMClient):
    def __init__(self, visible: bool) -> None:
        self.visible = visible
        self.calls = 0
        self.closes = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("stream expected")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        try:
            if self.calls > 1:
                yield TextDelta("recovered")
                yield StreamEnd(_answer("recovered"))
            elif self.visible:
                yield TextDelta("partial")
        finally:
            self.closes += 1


@pytest.mark.parametrize(
    ("visible", "reason", "calls", "text"),
    [(False, "end_turn", 2, ["recovered"]), (True, "incomplete_stream", 1, ["partial"])],
)
async def test_missing_stream_end_retries_only_before_visible_output(
    visible: bool,
    reason: str,
    calls: int,
    text: list[str],
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    llm = IncompleteStreamLLM(visible)
    events = await agent_event_collector(
        llm=llm,
        mcp=scripted_mcp_factory(tools=[]),
        stream=True,
        max_retries=1,
        retry_base_delay=0,
    )

    assert _done(events) == reason
    assert [event.text for event in events if isinstance(event, TextEvent)] == text
    assert llm.calls == calls
    assert llm.closes == calls


@pytest.mark.parametrize(
    ("visible", "expected_text", "expected_reason", "expected_calls"),
    [
        (False, ["recovered"], "end_turn", 2),
        (True, ["partial"], "incomplete_stream", 1),
    ],
)
async def test_generation_normalizes_streams_without_terminal_chunks(
    visible: bool,
    expected_text: list[str],
    expected_reason: str,
    expected_calls: int,
) -> None:
    llm = IncompleteStreamLLM(visible)

    chunks = await _generation_chunks(
        llm,
        stream=True,
        limits=RunLimits(max_retries=1, retry_base_delay=0),
    )

    assert [
        chunk.text for chunk in chunks if isinstance(chunk, TextDelta)
    ] == expected_text
    end = next(chunk for chunk in chunks if isinstance(chunk, StreamEnd))
    assert end.message.stop_reason == expected_reason
    assert llm.calls == expected_calls
    assert llm.closes == expected_calls


class ReasoningThenTransientLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.closes = 0
        self.requests: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("stream expected")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        self.requests.append(request)
        try:
            if self.calls == 1:
                yield ReasoningDelta("first thought")
                raise TransientFailure("retry after reasoning")
            yield TextDelta("recovered")
            yield StreamEnd(_answer("recovered"))
        finally:
            self.closes += 1

    def is_transient_error(self, exc: BaseException) -> bool:
        return isinstance(exc, TransientFailure)


async def test_reasoning_only_retry_marks_the_interrupted_attempt(
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    llm = ReasoningThenTransientLLM()

    events = await agent_event_collector(
        llm=llm,
        mcp=scripted_mcp_factory(tools=[]),
        stream=True,
        max_retries=1,
        retry_base_delay=0,
    )

    assert [event.text for event in events if isinstance(event, ReasoningEvent)] == [
        "first thought",
        "\n\n[Generation was interrupted; retrying...]\n\n",
    ]
    assert [event.text for event in events if isinstance(event, TextEvent)] == [
        "recovered"
    ]
    assert _done(events) == "end_turn"
    assert llm.calls == 2
    assert llm.closes == 2
    assert llm.requests[0] is llm.requests[1]


class ReasoningThenIncompleteLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("stream expected")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        if self.calls == 1:
            yield ReasoningDelta("unfinished thought")
            return
        yield TextDelta("recovered")
        yield StreamEnd(_answer("recovered"))


async def test_reasoning_only_incomplete_stream_marks_retry_boundary() -> None:
    llm = ReasoningThenIncompleteLLM()

    chunks = await _generation_chunks(
        llm,
        stream=True,
        limits=RunLimits(max_retries=1, retry_base_delay=0),
    )

    assert chunks == [
        ReasoningDelta("unfinished thought"),
        ReasoningDelta(
            "\n\n[Generation was interrupted; retrying...]\n\n"
        ),
        TextDelta("recovered"),
        StreamEnd(_answer("recovered")),
    ]
    assert llm.calls == 2


class BlockingLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.closes = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        try:
            self.started.set()
            await asyncio.Event().wait()
            yield StreamEnd(_answer())
        finally:
            self.closes += 1


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_per_attempt_timeout_retries_in_both_modes(
    stream: bool,
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    llm = BlockingLLM()
    events = await agent_event_collector(
        llm=llm,
        mcp=scripted_mcp_factory(tools=[]),
        stream=stream,
        llm_timeout_seconds=0.01,
        max_retries=1,
        retry_base_delay=0,
    )

    assert _done(events) == "llm_error"
    assert llm.calls == 2
    assert llm.closes == (2 if stream else 0)


async def test_deadline_exhausted_before_attempt_makes_no_provider_call(
    scripted_mcp_factory,
) -> None:
    llm = ParityLLM([_answer()])
    context = RunContext.start(
        max_run_seconds=10,
        base_logger=logging.getLogger("retry-policy-test"),
    )
    context.deadline = context.started_at - 1

    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    events = [
        event
        async for event in run_agent(
            session,
            llm,
            scripted_mcp_factory(tools=[]),
            store=store,
            context=context,
            limits=RunLimits(max_run_seconds=10),
        )
    ]

    assert _done(events) == "deadline_exceeded"
    assert llm.complete_calls == 0


class FakeClockContext:
    def __init__(self, remaining: float) -> None:
        self.now = 0.0
        self.deadline = remaining

    def remaining_seconds(self) -> float:
        return self.deadline - self.now

    def deadline_exceeded(self) -> bool:
        return self.remaining_seconds() <= 0


async def test_backoff_cap_clamp_and_deadline_during_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation_module.random, "uniform", lambda low, high: 0.25)
    assert generation_module._backoff_delay_seconds(2.0, 0) == 2.25
    assert generation_module._backoff_delay_seconds(2.0, 3) == 16.25
    assert generation_module._backoff_delay_seconds(2.0, 10) == 30.25

    clock = FakeClockContext(0.2)
    slept: list[float] = []

    async def advance(delay: float) -> None:
        slept.append(delay)
        clock.now += delay

    monkeypatch.setattr(generation_module.asyncio, "sleep", advance)
    attempts = generation_module._AttemptController(
        limits=RunLimits(max_retries=1, retry_base_delay=2.0),
        context=clock,  # type: ignore[arg-type]
        log=logging.getLogger("retry-policy-test"),
    )

    with pytest.raises(RunDeadlineExceeded):
        await attempts.retry(
            0,
            eligible=True,
            mode="buffered",
            reason="test",
        )
    assert slept == [0.2]


@pytest.mark.parametrize("stream", [False, True], ids=["provider-call", "stream-read"])
async def test_cancellation_during_provider_await_remains_cancellation(
    stream: bool,
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    llm = BlockingLLM()
    task = asyncio.create_task(
        agent_event_collector(
            llm=llm,
            mcp=scripted_mcp_factory(tools=[]),
            stream=stream,
            max_retries=1,
        )
    )
    await llm.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert llm.calls == 1
    assert llm.closes == (1 if stream else 0)


async def test_cancellation_during_retry_sleep_occurs_after_stream_close(
    monkeypatch: pytest.MonkeyPatch,
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    sleep_started = asyncio.Event()

    async def blocking_sleep(delay: float) -> None:
        sleep_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(generation_module.asyncio, "sleep", blocking_sleep)
    llm = ParityLLM([TransientFailure("retry"), _answer()])
    task = asyncio.create_task(
        agent_event_collector(
            llm=llm,
            mcp=scripted_mcp_factory(tools=[]),
            stream=True,
            max_retries=1,
        )
    )
    await sleep_started.wait()
    assert llm.stream_closes == 1
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert llm.stream_calls == 1


class CleanupStream:
    def __init__(
        self,
        outcome: AssistantMessage | BaseException,
        *,
        block_close: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.outcome = outcome
        self.block_close = block_close
        self.fail_close = fail_close
        self.delivered = False
        self.close_calls = 0
        self.close_started = asyncio.Event()

    def __aiter__(self) -> CleanupStream:
        return self

    async def __anext__(self) -> StreamChunk:
        if self.delivered:
            raise StopAsyncIteration
        self.delivered = True
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return StreamEnd(self.outcome)

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        if self.block_close:
            await asyncio.Event().wait()
        if self.fail_close:
            raise RuntimeError("cleanup failed")


class CleanupLLM(LLMClient):
    def __init__(self, stream: CleanupStream) -> None:
        self.owned_stream = stream

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("stream expected")

    def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:  # type: ignore[override]
        return self.owned_stream  # type: ignore[return-value]


async def test_cancellation_during_stream_cleanup_remains_cancellation(
    scripted_mcp_factory,
    agent_event_collector,
) -> None:
    stream = CleanupStream(_answer(), block_close=True)
    task = asyncio.create_task(
        agent_event_collector(
            llm=CleanupLLM(stream),
            mcp=scripted_mcp_factory(tools=[]),
            stream=True,
        )
    )
    await stream.close_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.close_calls == 1


class DeltaThenBlockLLM(LLMClient):
    def __init__(self) -> None:
        self.closes = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("stream expected")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        try:
            yield TextDelta("visible")
            await asyncio.Event().wait()
        finally:
            self.closes += 1


async def test_agent_generator_close_closes_active_stream_exactly_once(
    scripted_mcp_factory,
) -> None:
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    llm = DeltaThenBlockLLM()
    events = run_agent(
        session,
        llm,
        scripted_mcp_factory(tools=[]),
        store=store,
        stream=True,
    )

    first = await anext(events)
    assert isinstance(first, TextEvent)
    await events.aclose()

    assert llm.closes == 1


async def test_generation_consumer_close_closes_active_stream_exactly_once() -> None:
    llm = DeltaThenBlockLLM()
    generation = generation_module.generate_with_retry(
        llm,
        GenerationRequest(messages=[]),
        limits=RunLimits(),
        context=_generation_context(),
        stream=True,
        log=logging.getLogger("generation-test"),
    )

    first = await anext(generation)
    assert first == TextDelta("visible")
    await generation.aclose()

    assert llm.closes == 1


@pytest.mark.parametrize(
    ("outcome", "reason", "has_error"),
    [(_answer(), "end_turn", False), (ValueError("provider failed"), "llm_error", True)],
)
async def test_cleanup_failure_does_not_replace_primary_outcome(
    outcome: AssistantMessage | BaseException,
    reason: str,
    has_error: bool,
    scripted_mcp_factory,
    agent_event_collector,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = CleanupStream(outcome, fail_close=True)
    events = await agent_event_collector(
        llm=CleanupLLM(stream),
        mcp=scripted_mcp_factory(tools=[]),
        stream=True,
    )

    assert _done(events) == reason
    assert any(isinstance(event, ErrorEvent) for event in events) is has_error
    assert stream.close_calls == 1
    assert "cleanup failed" not in caplog.text
    assert "CleanupStream" in caplog.text
    assert "RuntimeError" in caplog.text
