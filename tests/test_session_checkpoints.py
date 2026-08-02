"""Regression coverage for copy-on-write transcript checkpoints."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import pytest

from agent import (
    DoneEvent,
    ErrorEvent,
    InMemorySessionStore,
    RunLimits,
    Session,
    SessionGuard,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from api.turn import (
    PersistencePolicy,
    TurnRequest,
    TurnRunner,
    UnorchestratedRouting,
)
from config import Settings
from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    Message,
    Role,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    CompletionUsage,
)
from tooling import ToolCallResult

pytestmark = pytest.mark.anyio


UNKNOWN_OUTCOME = "tool call outcome is unknown because execution was cancelled while the call was in flight"
CANCELLED_BEFORE_START = "tool call was not executed because execution was cancelled"


class CountingStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.save_count = 0

    async def save(self, session: Session) -> None:
        self.save_count += 1
        await super().save(session)


class FailingSaveStore(CountingStore):
    async def save(self, session: Session) -> None:
        self.save_count += 1
        raise RuntimeError("checkpoint unavailable")


class ToolMCP:
    connected_servers: list[str] = []

    def __init__(self, *, block_index: int | None = None) -> None:
        self.block_index = block_index
        self.calls: list[str] = []
        self.started = [asyncio.Event() for _ in range(3)]

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [
            {"name": f"srv__tool_{i}", "description": "test", "input_schema": {}}
            for i in range(3)
        ]

    @asynccontextmanager
    async def open_turn(self, *, timeout_seconds: float | None = None):
        yield self

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        index = int(name.rsplit("_", 1)[1])
        self.calls.append(name)
        self.started[index].set()
        if index == self.block_index:
            await asyncio.Event().wait()
        return ToolCallResult(content=f"result-{index}", is_error=False)


class AnswerLLM(LLMClient):
    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        return AssistantMessage(
            content=[TextBlock("follow-up")],
            stop_reason="end_turn",
            usage=CompletionUsage(total_tokens=1),
        )


class BlockingLLM(LLMClient):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.started.set()
        await asyncio.Event().wait()


class BlockingAfterDeltaLLM(LLMClient):
    def __init__(self) -> None:
        self.waiting = asyncio.Event()

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("stream() expected")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        yield TextDelta("visible")
        self.waiting.set()
        await asyncio.Event().wait()


class ScriptedStreamLLM(LLMClient):
    def __init__(self, attempts: list[list[StreamChunk]]) -> None:
        self.attempts = attempts
        self.calls = 0
        self.messages_seen: list[list[Message]] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("stream() expected")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        self.messages_seen.append(list(request.messages))
        chunks = self.attempts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk


class ToolBatchLLM(LLMClient):
    def __init__(self, *, second_tool_batch: bool = False) -> None:
        self.calls = 0
        self.second_tool_batch = second_tool_batch

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        if self.calls == 1 or self.second_tool_batch:
            batch = self.calls - 1
            return AssistantMessage(
                content=[
                    ToolUseBlock(
                        id=f"batch-{batch}-call-{i}",
                        name=f"srv__tool_{i}",
                        input={"index": i},
                    )
                    for i in range(3)
                ],
                stop_reason="tool_use",
                usage=CompletionUsage(total_tokens=1),
            )
        await asyncio.Event().wait()


class ToolThenBlockingStreamLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.waiting = asyncio.Event()

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("stream() expected")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        if self.calls == 1:
            yield StreamEnd(
                AssistantMessage(
                    content=[
                        ToolUseBlock(
                            id="first-tool",
                            name="srv__tool_0",
                            input={"index": 0},
                        )
                    ],
                    stop_reason="tool_use",
                    usage=CompletionUsage(total_tokens=1),
                )
            )
            return
        yield TextDelta("later partial")
        self.waiting.set()
        await asyncio.Event().wait()


def _runner(
    llm: LLMClient,
    mcp: ToolMCP,
    store: CountingStore,
    *,
    retries: int = 0,
) -> TurnRunner:
    settings = Settings(
        _env_file=None,
        orchestration_enabled=False,
        llm={
            "model_name": "test-model",
            "max_retries": retries,
            "retry_base_delay": 0,
            "timeout_seconds": 0,
        },
        tool_timeout_seconds=0,
        run_max_seconds=0,
        loop_max_iterations=5,
    )
    return TurnRunner(
        routing=UnorchestratedRouting(
            llm=llm,
            model_id=settings.llm.model_name,
        ),
        limits=RunLimits.from_settings(settings),
        mcp=mcp,
        store=store,
        guard=SessionGuard(),
        policy=None,
        tracer=None,
    )


def _tool_results(session: Session) -> list[ToolResultBlock]:
    return [
        block
        for message in session.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]


async def _assert_reusable(store: CountingStore, session: Session) -> None:
    runner = _runner(AnswerLLM(), ToolMCP(), store)
    result = await runner.run(
        TurnRequest(
            "continue",
            session,
            PersistencePolicy.PERSISTENT,
        )
    )
    assert result.answer == "follow-up"


async def test_staged_copy_preserves_identity_without_mutable_aliases() -> None:
    session = Session(metadata={"tag": "original"})
    session.append_user("prior")

    staged = session.staged_copy()

    assert staged.session_id == session.session_id
    assert staged.created_at == session.created_at
    assert staged.updated_at == session.updated_at
    assert staged.messages == session.messages
    assert staged.messages is not session.messages
    assert staged.messages[0] is session.messages[0]
    assert staged.metadata == session.metadata
    assert staged.metadata is not session.metadata
    staged.metadata["tag"] = "changed"
    staged.append_user("new")
    assert session.metadata == {"tag": "original"}
    assert len(session.messages) == 1


async def test_cancellation_before_first_model_output_publishes_nothing() -> None:
    store = CountingStore()
    original = await store.create(metadata={"stable": True})
    original.append_user("prior")
    llm = BlockingLLM()
    runner = _runner(llm, ToolMCP(), store)
    before = list(original.messages)

    task = asyncio.create_task(
        runner.run(TurnRequest("new prompt", original, PersistencePolicy.PERSISTENT))
    )
    await llm.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    fetched = await store.get(original.session_id)
    assert fetched is original
    assert fetched.messages == before
    assert store.save_count == 0
    await _assert_reusable(store, original)


async def test_cancellation_after_deltas_before_stream_end_publishes_nothing() -> None:
    store = CountingStore()
    original = await store.create()
    llm = BlockingAfterDeltaLLM()
    runner = _runner(llm, ToolMCP(), store)

    task = asyncio.create_task(
        runner.run(
            TurnRequest(
                "new prompt", original, PersistencePolicy.PERSISTENT, stream=True
            )
        )
    )
    await llm.waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await store.get(original.session_id)).messages == []
    assert store.save_count == 0
    await _assert_reusable(store, original)


async def test_visible_silent_exhaustion_is_abnormal_and_persisted_identically() -> (
    None
):
    store = CountingStore()
    original = await store.create()
    llm = ScriptedStreamLLM([[TextDelta("part "), TextDelta("one")]])
    runner = _runner(llm, ToolMCP(), store)

    async with runner.open(
        TurnRequest("prompt", original, PersistencePolicy.PERSISTENT, stream=True)
    ) as execution:
        events = [event async for event in execution.events]

    assert [event.text for event in events if isinstance(event, TextEvent)] == [
        "part ",
        "one",
    ]
    assert any(isinstance(event, ErrorEvent) for event in events)
    assert [event.reason for event in events if isinstance(event, DoneEvent)] == [
        "incomplete_stream"
    ]
    fetched = await store.get(original.session_id)
    assert fetched.messages[-1].role is Role.ASSISTANT
    assert fetched.messages[-1].content[0] == TextBlock("part one")
    assert store.save_count == 1


async def test_empty_silent_exhaustion_retries_without_prompt_duplication() -> None:
    store = CountingStore()
    original = await store.create()
    answer = AssistantMessage(
        content=[TextBlock("answer")],
        stop_reason="end_turn",
        usage=CompletionUsage(total_tokens=1),
    )
    llm = ScriptedStreamLLM([[], [TextDelta("answer"), StreamEnd(answer)]])
    runner = _runner(llm, ToolMCP(), store, retries=1)

    result = await runner.run(
        TurnRequest("prompt", original, PersistencePolicy.PERSISTENT, stream=True)
    )

    assert result.answer == "answer"
    assert llm.calls == 2
    assert all(
        sum(message.role is Role.USER for message in messages) == 1
        for messages in llm.messages_seen
    )
    assert [
        message.role for message in (await store.get(original.session_id)).messages
    ] == [
        Role.USER,
        Role.ASSISTANT,
    ]


async def test_normal_stream_end_is_authoritative_without_delta_duplication() -> None:
    store = CountingStore()
    original = await store.create()
    answer = AssistantMessage(
        content=[TextBlock("joined")],
        stop_reason="end_turn",
        usage=CompletionUsage(total_tokens=1),
    )
    llm = ScriptedStreamLLM([[TextDelta("join"), TextDelta("ed"), StreamEnd(answer)]])
    runner = _runner(llm, ToolMCP(), store)

    result = await runner.run(
        TurnRequest("prompt", original, PersistencePolicy.PERSISTENT, stream=True)
    )

    assert result.answer == "joined"
    fetched = await store.get(original.session_id)
    assert fetched.messages[-1].content == [TextBlock("joined")]


@pytest.mark.parametrize("tool_index", [0, 1, 2])
async def test_active_cancellation_during_each_tool_balances_checkpoint(
    tool_index: int,
) -> None:
    store = CountingStore()
    original = await store.create()
    mcp = ToolMCP(block_index=tool_index)
    runner = _runner(ToolBatchLLM(), mcp, store)

    task = asyncio.create_task(
        runner.run(TurnRequest("tools", original, PersistencePolicy.PERSISTENT))
    )
    await mcp.started[tool_index].wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    fetched = await store.get(original.session_id)
    results = _tool_results(fetched)
    assert len(results) == 3
    assert [result.tool_use_id for result in results] == [
        f"batch-0-call-{i}" for i in range(3)
    ]
    assert [result.content for result in results[:tool_index]] == [
        f"result-{i}" for i in range(tool_index)
    ]
    assert results[tool_index].content == UNKNOWN_OUTCOME
    assert all(
        result.content == CANCELLED_BEFORE_START for result in results[tool_index + 1 :]
    )
    assert fetched.messages[-2].role is Role.ASSISTANT
    assert fetched.messages[-1].role is Role.TOOL
    assert store.save_count == 1
    await _assert_reusable(store, original)


@pytest.mark.parametrize(
    ("boundary", "tool_index"),
    [
        ("before", 0),
        ("before", 1),
        ("before", 2),
        ("after", 0),
        ("after", 1),
        ("after", 2),
    ],
)
async def test_generator_close_before_and_after_each_tool_balances_checkpoint(
    boundary: str,
    tool_index: int,
) -> None:
    store = CountingStore()
    original = await store.create()
    mcp = ToolMCP()
    runner = _runner(ToolBatchLLM(), mcp, store)

    async with runner.open(
        TurnRequest("tools", original, PersistencePolicy.PERSISTENT)
    ) as execution:
        async for event in execution.events:
            if boundary == "before" and isinstance(event, ToolCallEvent):
                if event.id == f"batch-0-call-{tool_index}":
                    break
            if boundary == "after" and isinstance(event, ToolResultEvent):
                if event.id == f"batch-0-call-{tool_index}":
                    break

    fetched = await store.get(original.session_id)
    results = _tool_results(fetched)
    completed = tool_index if boundary == "before" else tool_index + 1
    assert [result.content for result in results[:completed]] == [
        f"result-{i}" for i in range(completed)
    ]
    assert all(
        result.content == CANCELLED_BEFORE_START for result in results[completed:]
    )
    assert len(results) == 3
    assert store.save_count == 1
    await _assert_reusable(store, original)


async def test_intermediate_checkpoint_is_detached_from_later_tool_batch() -> None:
    store = CountingStore()
    original = await store.create()
    runner = _runner(ToolBatchLLM(second_tool_batch=True), ToolMCP(), store)

    async with runner.open(
        TurnRequest("tools", original, PersistencePolicy.PERSISTENT)
    ) as execution:
        seen_second_batch = False
        async for event in execution.events:
            if (
                isinstance(event, ToolCallEvent)
                and event.id == "batch-1-call-0"
            ):
                seen_second_batch = True
                published = await store.get(original.session_id)
                assert len(published.messages) == 3
                assert [result.tool_use_id for result in _tool_results(published)] == [
                    f"batch-0-call-{i}" for i in range(3)
                ]
                break
        assert seen_second_batch

    balanced = await store.get(original.session_id)
    assert len(balanced.messages) == 5
    assert len(_tool_results(balanced)) == 6


async def test_later_incomplete_generation_retains_earlier_tool_checkpoint() -> None:
    store = CountingStore()
    original = await store.create()
    llm = ToolThenBlockingStreamLLM()
    runner = _runner(llm, ToolMCP(), store)

    task = asyncio.create_task(
        runner.run(
            TurnRequest("tools", original, PersistencePolicy.PERSISTENT, stream=True)
        )
    )
    await llm.waiting.wait()
    published_before_cancel = await store.get(original.session_id)
    assert len(published_before_cancel.messages) == 3
    assert [
        result.tool_use_id for result in _tool_results(published_before_cancel)
    ] == ["first-tool"]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    published_after_cancel = await store.get(original.session_id)
    assert published_after_cancel is published_before_cancel
    assert len(published_after_cancel.messages) == 3
    assert store.save_count == 1
    await _assert_reusable(store, original)


async def test_checkpoint_failure_does_not_replace_active_cancellation() -> None:
    store = FailingSaveStore()
    original = await store.create()
    mcp = ToolMCP(block_index=0)
    runner = _runner(ToolBatchLLM(), mcp, store)

    task = asyncio.create_task(
        runner.run(TurnRequest("tools", original, PersistencePolicy.PERSISTENT))
    )
    await mcp.started[0].wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.save_count == 1
    assert (await store.get(original.session_id)).messages == []


async def test_persistent_turn_refreshes_a_stale_session_handle() -> None:
    store = CountingStore()
    original = await store.create()
    runner = _runner(AnswerLLM(), ToolMCP(), store)

    await runner.run(TurnRequest("first", original, PersistencePolicy.PERSISTENT))
    await runner.run(TurnRequest("second", original, PersistencePolicy.PERSISTENT))

    fetched = await store.get(original.session_id)
    assert [message.role for message in fetched.messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
        Role.ASSISTANT,
    ]
    assert [
        block.text
        for message in fetched.messages
        for block in message.content
        if isinstance(block, TextBlock)
    ] == ["first", "follow-up", "second", "follow-up"]
    assert original.messages == []


@pytest.mark.parametrize("stream", [False, True])
async def test_normal_native_turn_commits_once(stream: bool) -> None:
    store = CountingStore()
    original = await store.create()
    if stream:
        answer = AssistantMessage(
            content=[TextBlock("answer")],
            stop_reason="end_turn",
            usage=CompletionUsage(total_tokens=1),
        )
        llm: LLMClient = ScriptedStreamLLM([[TextDelta("answer"), StreamEnd(answer)]])
    else:
        llm = AnswerLLM()
    runner = _runner(llm, ToolMCP(), store)

    await runner.run(
        TurnRequest("prompt", original, PersistencePolicy.PERSISTENT, stream=stream)
    )

    assert store.save_count == 1


async def test_ephemeral_turn_never_saves_shared_store() -> None:
    store = CountingStore()
    original_ids = store.ids()
    runner = _runner(AnswerLLM(), ToolMCP(), store)

    result = await runner.run(
        TurnRequest("prompt", Session(), PersistencePolicy.EPHEMERAL)
    )

    assert result.answer == "follow-up"
    assert store.save_count == 0
    assert store.ids() == original_ids
