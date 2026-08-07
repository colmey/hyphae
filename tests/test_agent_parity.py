"""Representative public event and transcript journeys for ``run_agent``.

Focused suites own retry, timeout, tool-policy, budget, and checkpoint matrices.
This module keeps only complete provider-neutral runs that freeze the public
composition of those behaviors.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from dataclasses import fields
from typing import Any

import pytest

from hyphae.agent import (
    DoneEvent,
    ErrorEvent,
    Event,
    InMemorySessionStore,
    RunLimits,
    Session,
    ToolCallEvent,
    ToolResultEvent,
    run_agent,
)
from hyphae.llm.client import GenerationRequest, LLMClient
from hyphae.llm.providers.gemini import GeminiLLMClient
from hyphae.llm.providers.openai_compatible import OpenAICompatibleLLMClient
from hyphae.llm.schemas import (
    AssistantMessage,
    CompletionUsage,
    Message,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
)
from tests.fakes import ScriptedLLM, ScriptedMCP
from hyphae.tooling import ToolCallResult, ToolRuntime


pytestmark = pytest.mark.anyio

_NUMERIC = "<numeric>"
_NOT_STARTED = "tool call was not executed because execution was cancelled"
_USAGE = CompletionUsage(
    input_tokens=2,
    output_tokens=3,
    total_tokens=5,
    thinking_tokens=1,
    cached_tokens=1,
)
_TOOL = {
    "name": "srv__tool",
    "description": "test tool",
    "input_schema": {},
}


class TransientFailure(RuntimeError):
    """Retryable provider failure for the representative retry journey."""


class RecordingStore(InMemorySessionStore):
    """Record each public checkpoint before the native store detaches it."""

    def __init__(self) -> None:
        super().__init__()
        self.save_count = 0
        self.saved_messages: list[list[dict[str, object]]] = []

    async def save(
        self,
        session: Session,
        *,
        protected_session_ids: Callable[[], frozenset[str]] | None = None,
    ) -> None:
        self.save_count += 1
        self.saved_messages.append(normalize_messages(session.messages))
        await super().save(
            session,
            protected_session_ids=protected_session_ids,
        )


class NativeStreamingLLM(LLMClient):
    """Emit two native deltas and record exact stream ownership."""

    def __init__(self) -> None:
        self.requests_seen: list[GenerationRequest] = []
        self.closes = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("native streaming journey must not call complete()")

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        self.requests_seen.append(request)
        try:
            yield TextDelta("Hel")
            yield TextDelta("lo")
            yield StreamEnd(_answer("Hello"))
        finally:
            self.closes += 1


class BlockingMCP:
    """Hold one real dispatch so call-event visibility is observable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [dict(_TOOL)]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult:
        self.calls.append((name, arguments))
        self.started.set()
        await self.release.wait()
        return ToolCallResult("abcdefghij", False)


def normalize_event(event: Event) -> dict[str, object]:
    """Snapshot every public event field, normalizing measured latency only."""
    normalized: dict[str, object] = {}
    for field in fields(event):
        value = getattr(event, field.name)
        if field.name == "latency_ms" and value is not None:
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            value = _NUMERIC
        normalized[field.name] = value
    return normalized


def normalize_events(events: list[Event]) -> list[dict[str, object]]:
    return [normalize_event(event) for event in events]


def _normalize_block(block: object) -> dict[str, object]:
    if isinstance(block, TextBlock):
        return {
            "text": block.text,
            "provider_metadata": block.provider_metadata,
            "type": block.type,
        }
    if isinstance(block, ToolUseBlock):
        return {
            "id": block.id,
            "name": block.name,
            "input": block.input,
            "provider_metadata": block.provider_metadata,
            "parse_error": block.parse_error,
            "type": block.type,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "tool_use_id": block.tool_use_id,
            "name": block.name,
            "content": block.content,
            "is_error": block.is_error,
            "type": block.type,
        }
    raise AssertionError(f"unexpected public content block: {type(block)!r}")


def normalize_messages(messages: list[Message]) -> list[dict[str, object]]:
    return [
        {
            "role": message.role.value,
            "content": [_normalize_block(block) for block in message.content],
        }
        for message in messages
    ]


def _text_block(text: str) -> dict[str, object]:
    return {"text": text, "provider_metadata": {}, "type": "text"}


def _tool_use_block(
    call_id: str,
    arguments: dict[str, Any],
) -> dict[str, object]:
    return {
        "id": call_id,
        "name": "srv__tool",
        "input": arguments,
        "provider_metadata": {},
        "parse_error": None,
        "type": "tool_use",
    }


def _tool_result_block(
    call_id: str,
    content: str,
    *,
    is_error: bool,
) -> dict[str, object]:
    return {
        "tool_use_id": call_id,
        "name": "srv__tool",
        "content": content,
        "is_error": is_error,
        "type": "tool_result",
    }


def _message(role: str, *content: dict[str, object]) -> dict[str, object]:
    return {"role": role, "content": list(content)}


def _answer(
    text: str,
    *,
    stop_reason: str = "end_turn",
) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text)],
        stop_reason=stop_reason,
        raw_stop_reason=("raw-provider-reason" if stop_reason != "end_turn" else None),
        usage=_USAGE,
    )


def _tool_call(
    *calls: tuple[str, dict[str, Any]],
) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolUseBlock(id=call_id, name="srv__tool", input=arguments)
            for call_id, arguments in calls
        ],
        stop_reason="tool_use",
        usage=_USAGE,
    )


def _usage_event(iteration: int = 1) -> dict[str, object]:
    return {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
        "thinking_tokens": 1,
        "cached_tokens": 1,
        "iteration": iteration,
        "latency_ms": _NUMERIC,
        "type": "usage",
    }


def _done_event(reason: str, *, iterations: int = 1) -> dict[str, object]:
    return {
        "reason": reason,
        "iterations": iterations,
        "total_tokens": 5 * iterations,
        "input_tokens": 2 * iterations,
        "output_tokens": 3 * iterations,
        "thinking_tokens": iterations,
        "type": "done",
    }


async def _persistent_run(
    llm: LLMClient,
    mcp: ToolRuntime,
    *,
    stream: bool = False,
    limits: RunLimits | None = None,
) -> tuple[list[Event], RecordingStore, Session]:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    events = [
        event
        async for event in run_agent(
            session,
            llm,
            mcp,
            store=store,
            stream=stream,
            limits=limits,
        )
    ]
    return events, store, session


def test_public_imports_and_run_agent_calling_contract() -> None:
    assert run_agent.__module__ == "hyphae.agent.loop"
    assert OpenAICompatibleLLMClient.__name__ == "OpenAICompatibleLLMClient"
    assert GeminiLLMClient.__name__ == "GeminiLLMClient"

    parameters = inspect.signature(run_agent).parameters
    assert [
        (name, parameter.kind, parameter.default)
        for name, parameter in parameters.items()
    ] == [
        ("session", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("llm", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("mcp", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("store", inspect.Parameter.KEYWORD_ONLY, None),
        ("protected_session_ids", inspect.Parameter.KEYWORD_ONLY, None),
        ("system", inspect.Parameter.KEYWORD_ONLY, None),
        ("tools", inspect.Parameter.KEYWORD_ONLY, None),
        ("thinking_level", inspect.Parameter.KEYWORD_ONLY, None),
        ("limits", inspect.Parameter.KEYWORD_ONLY, None),
        ("context", inspect.Parameter.KEYWORD_ONLY, None),
        ("policy", inspect.Parameter.KEYWORD_ONLY, None),
        ("stream", inspect.Parameter.KEYWORD_ONLY, False),
        ("initial_usage", inspect.Parameter.KEYWORD_ONLY, None),
    ]


async def test_buffered_final_answer_public_journey() -> None:
    events, store, session = await _persistent_run(
        ScriptedLLM([_answer("Hello")]),
        ScriptedMCP(tools=[]),
    )
    expected_messages = [
        _message("user", _text_block("go")),
        _message("assistant", _text_block("Hello")),
    ]

    assert normalize_events(events) == [
        _usage_event(),
        {"text": "Hello", "type": "text"},
        _done_event("end_turn"),
    ]
    assert normalize_messages(session.messages) == expected_messages
    assert store.saved_messages == [expected_messages]


async def test_native_streaming_final_answer_public_journey() -> None:
    llm = NativeStreamingLLM()
    events, store, session = await _persistent_run(
        llm,
        ScriptedMCP(tools=[]),
        stream=True,
    )
    expected_messages = [
        _message("user", _text_block("go")),
        _message("assistant", _text_block("Hello")),
    ]

    assert normalize_events(events) == [
        {"text": "Hel", "type": "text"},
        {"text": "lo", "type": "text"},
        _usage_event(),
        _done_event("end_turn"),
    ]
    assert normalize_messages(session.messages) == expected_messages
    assert store.saved_messages == [expected_messages]
    assert len(llm.requests_seen) == 1
    assert llm.closes == 1


async def test_tool_round_trip_public_journey() -> None:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    llm = ScriptedLLM(
        [_tool_call(("call-slow", {"q": "wait"})), _answer("done")]
    )
    mcp = BlockingMCP()
    generator = run_agent(
        session,
        llm,
        mcp,
        store=store,
        limits=RunLimits(tool_result_max_chars=4),
    )

    usage = await anext(generator)
    call = await anext(generator)
    assert isinstance(call, ToolCallEvent)
    assert normalize_event(call) == {
        "id": "call-slow",
        "name": "srv__tool",
        "input": {"q": "wait"},
        "type": "tool_call",
    }
    assert not mcp.started.is_set()

    result_task = asyncio.create_task(anext(generator))
    await mcp.started.wait()
    assert not result_task.done()
    mcp.release.set()
    result = await result_task
    remaining = [event async for event in generator]
    events = [usage, call, result, *remaining]
    clipped = "abcd\n…[truncated, 6 chars omitted]"

    assert isinstance(result, ToolResultEvent)
    assert normalize_event(result) == {
        "id": "call-slow",
        "name": "srv__tool",
        "content": clipped,
        "is_error": False,
        "latency_ms": _NUMERIC,
        "type": "tool_result",
    }
    assert [event.type for event in events] == [
        "usage",
        "tool_call",
        "tool_result",
        "usage",
        "text",
        "done",
    ]
    assert normalize_event(events[-1]) == _done_event("end_turn", iterations=2)
    assert mcp.calls == [("srv__tool", {"q": "wait"})]
    persisted = await store.get(session.session_id)
    assert normalize_messages(persisted.messages) == [
        _message("user", _text_block("go")),
        _message("assistant", _tool_use_block("call-slow", {"q": "wait"})),
        _message("tool", _tool_result_block("call-slow", clipped, is_error=False)),
        _message("assistant", _text_block("done")),
    ]
    assert store.save_count == 2


async def test_retry_then_success_public_journey() -> None:
    llm = ScriptedLLM(
        [TransientFailure("retry"), _answer("recovered")],
        transient_types=(TransientFailure,),
    )
    events, store, session = await _persistent_run(
        llm,
        ScriptedMCP(tools=[]),
        limits=RunLimits(max_retries=1, retry_base_delay=0),
    )

    assert llm.calls == 2
    assert not any(isinstance(event, ErrorEvent) for event in events)
    assert normalize_events(events) == [
        _usage_event(),
        {"text": "recovered", "type": "text"},
        _done_event("end_turn"),
    ]
    assert normalize_messages(session.messages)[-1] == _message(
        "assistant", _text_block("recovered")
    )
    assert store.save_count == 1


async def test_abnormal_terminal_public_journey() -> None:
    events, store, session = await _persistent_run(
        ScriptedLLM([_answer("provider text", stop_reason="provider_error")]),
        ScriptedMCP(tools=[]),
    )

    assert [event.type for event in events] == ["usage", "text", "error", "done"]
    assert [event.message for event in events if isinstance(event, ErrorEvent)] == [
        "LLM provider terminated abnormally"
    ]
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.reason == "provider_error"
    assert normalize_messages(session.messages) == [
        _message("user", _text_block("go")),
        _message("assistant", _text_block("provider text")),
    ]
    assert store.save_count == 1


async def test_interrupted_tool_batch_public_journey() -> None:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    llm = ScriptedLLM(
        [_tool_call(("call-first", {"n": 1}), ("call-second", {"n": 2}))]
    )
    mcp = ScriptedMCP(content="first")
    observed: list[Event] = []
    paused = asyncio.Event()

    async def consume() -> None:
        generator = run_agent(session, llm, mcp, store=store)
        try:
            async for event in generator:
                observed.append(event)
                if isinstance(event, ToolResultEvent):
                    paused.set()
                    await asyncio.Event().wait()
        finally:
            await generator.aclose()

    task = asyncio.create_task(consume())
    await paused.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [event.type for event in observed] == ["usage", "tool_call", "tool_result"]
    assert mcp.calls == [("srv__tool", {"n": 1})]
    persisted = await store.get(session.session_id)
    assert normalize_messages(persisted.messages)[-1] == _message(
        "tool",
        _tool_result_block("call-first", "first", is_error=False),
        _tool_result_block("call-second", _NOT_STARTED, is_error=True),
    )
    assert store.save_count == 1


async def test_ephemeral_composition_public_journey() -> None:
    native = RecordingStore()
    persistent = await native.create(metadata={"owner": "native"})
    persistent.append_user("existing")
    await native.save(persistent)
    native.save_count = 0
    before = normalize_messages(persistent.messages)

    ephemeral = Session(session_id="ephemeral-fixed")
    ephemeral.append_user("go")
    events = [
        event
        async for event in run_agent(
            ephemeral,
            ScriptedLLM([_answer("private")]),
            ScriptedMCP(tools=[]),
            store=None,
        )
    ]

    assert isinstance(events[-1], DoneEvent)
    assert events[-1].reason == "end_turn"
    assert normalize_messages(ephemeral.messages) == [
        _message("user", _text_block("go")),
        _message("assistant", _text_block("private")),
    ]
    assert native.ids() == [persistent.session_id]
    assert normalize_messages((await native.get(persistent.session_id)).messages) == before
    assert native.save_count == 0
