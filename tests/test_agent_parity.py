"""Consolidated behavioral and public-contract parity for ``run_agent``.

The focused loop suites remain the best diagnostics for individual helpers.  This
module instead freezes the public event stream and replayable transcript produced
by representative complete runs.  All fakes are provider-neutral and hermetic.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, fields
from typing import Any

import pytest

from agent import (
    DoneEvent,
    ErrorEvent,
    Event,
    InMemorySessionStore,
    RunContext,
    RunLimits,
    Session,
    TextEvent,
    ToolCallEvent,
    ToolPolicy,
    ToolResultEvent,
    UsageEvent,
    run_agent,
)
from agent.loop import _FAILURE_NUDGE
from agent.tool_execution import _STALL_MESSAGE
from llm.client import GenerationRequest, LLMClient
from llm.providers.gemini import GeminiLLMClient
from llm.providers.openai_compatible import OpenAICompatibleLLMClient
from llm.schemas import (
    AssistantMessage,
    Message,
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

_NUMERIC = "<numeric>"
_UNKNOWN_OUTCOME = (
    "tool call outcome is unknown because execution was cancelled while the call "
    "was in flight"
)
_NOT_STARTED = "tool call was not executed because execution was cancelled"
_DEFAULT_USAGE = CompletionUsage(
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
    """Provider-neutral retryable failure used by the scripted LLM."""


@dataclass(frozen=True)
class StreamAttempt:
    """One explicit stream attempt, including incomplete and failing streams."""

    chunks: tuple[StreamChunk | BaseException, ...]


@dataclass(frozen=True)
class ParityCase:
    """A complete public event/transcript expectation for one agent run."""

    name: str
    stream: bool
    llm_script: tuple[AssistantMessage | BaseException | StreamAttempt, ...]
    tool_script: tuple[ToolCallResult | BaseException, ...] = ()
    limits: RunLimits = RunLimits()
    expected_events: tuple[dict[str, object], ...] = ()
    expected_messages: tuple[dict[str, object], ...] = ()


class _ScriptedStream:
    def __init__(
        self,
        chunks: tuple[StreamChunk | BaseException, ...],
        owner: ParityLLM,
    ) -> None:
        self._chunks = iter(chunks)
        self._owner = owner
        self._closed = False

    def __aiter__(self) -> _ScriptedStream:
        return self

    async def __anext__(self) -> StreamChunk:
        try:
            item = next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        assert not self._closed, "a stream attempt was closed more than once"
        self._closed = True
        self._owner.stream_closes += 1


class ParityLLM(LLMClient):
    """Replay buffered responses or explicit stream attempts and record calls."""

    def __init__(
        self,
        script: tuple[AssistantMessage | BaseException | StreamAttempt, ...],
    ) -> None:
        self._script = list(script)
        self.complete_calls = 0
        self.stream_calls = 0
        self.stream_closes = 0
        self.requests_seen: list[GenerationRequest] = []

    def _next(self) -> AssistantMessage | BaseException | StreamAttempt:
        if not self._script:
            raise AssertionError("ParityLLM ran out of scripted attempts")
        return self._script.pop(0)

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.complete_calls += 1
        self.requests_seen.append(request)
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, StreamAttempt):
            raise AssertionError("explicit stream attempt used in buffered mode")
        return item

    def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        self.stream_calls += 1
        self.requests_seen.append(request)
        item = self._next()
        if isinstance(item, StreamAttempt):
            chunks = item.chunks
        elif isinstance(item, BaseException):
            chunks = (item,)
        else:
            chunks = tuple(
                TextDelta(block.text)
                for block in item.content
                if isinstance(block, TextBlock) and block.text
            ) + (StreamEnd(item),)
        return _ScriptedStream(chunks, self)

    def is_transient_error(self, exc: BaseException) -> bool:
        return isinstance(exc, TransientFailure)


class ScriptedMCP:
    """Replay tool results or exceptions while recording every dispatch."""

    def __init__(
        self,
        script: tuple[ToolCallResult | BaseException, ...] = (),
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self._script = list(script)
        self._tools = list(tools) if tools is not None else [dict(_TOOL)]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.calls.append((name, arguments))
        if not self._script:
            raise AssertionError("ScriptedMCP ran out of scripted results")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class RecordingStore(InMemorySessionStore):
    """Record each public checkpoint and the object published for detachment tests."""

    def __init__(self) -> None:
        super().__init__()
        self.save_count = 0
        self.saved_sessions: list[Session] = []
        self.saved_messages: list[list[dict[str, object]]] = []

    async def save(self, session: Session) -> None:
        self.save_count += 1
        self.saved_sessions.append(session)
        self.saved_messages.append(normalize_messages(session.messages))
        await super().save(session)


def normalize_event(event: Event) -> dict[str, object]:
    """Retain every public event field, normalizing only measured latency.

    ``None`` is semantically meaningful for short-circuited tool calls and stays
    ``None``.  A real latency must still prove it is numeric before replacement.
    """

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
    """Snapshot replayable public message/block fields without object identity."""

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
    *,
    parse_error: str | None = None,
) -> dict[str, object]:
    return {
        "id": call_id,
        "name": "srv__tool",
        "input": arguments,
        "provider_metadata": {},
        "parse_error": parse_error,
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
    text: str = "answer",
    *,
    stop_reason: str | None = "end_turn",
    usage: CompletionUsage | None = None,
    raw_stop_reason: str | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text)],
        stop_reason=stop_reason,  # type: ignore[arg-type]
        raw_stop_reason=raw_stop_reason,
        usage=usage or _DEFAULT_USAGE,
    )


def _empty() -> AssistantMessage:
    return AssistantMessage(content=[], stop_reason="empty", usage=_DEFAULT_USAGE)


def _tool_call(
    call_id: str,
    arguments: dict[str, Any],
    *,
    parse_error: str | None = None,
    usage: CompletionUsage | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolUseBlock(
                id=call_id,
                name="srv__tool",
                input=arguments,
                parse_error=parse_error,
            )
        ],
        stop_reason="tool_use",
        usage=usage or _DEFAULT_USAGE,
    )


def _multi_tool_call(
    calls: tuple[tuple[str, dict[str, Any]], ...],
) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolUseBlock(id=call_id, name="srv__tool", input=arguments)
            for call_id, arguments in calls
        ],
        stop_reason="tool_use",
        usage=_DEFAULT_USAGE,
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


def _done_event(
    reason: str,
    *,
    iterations: int = 1,
    multiplier: int = 1,
) -> dict[str, object]:
    return {
        "reason": reason,
        "iterations": iterations,
        "total_tokens": 5 * multiplier,
        "input_tokens": 2 * multiplier,
        "output_tokens": 3 * multiplier,
        "thinking_tokens": multiplier,
        "type": "done",
    }


async def _run_case(
    case: ParityCase,
) -> tuple[list[Event], RecordingStore, Session, ParityLLM, ScriptedMCP]:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    llm = ParityLLM(case.llm_script)
    mcp = ScriptedMCP(case.tool_script, tools=[])
    events = [
        event
        async for event in run_agent(
            session,
            llm,
            mcp,  # type: ignore[arg-type]
            store=store,
            limits=case.limits,
            stream=case.stream,
        )
    ]
    return events, store, session, llm, mcp


def _assert_terminal(events: list[Event], reason: str) -> DoneEvent:
    done = [event for event in events if isinstance(event, DoneEvent)]
    assert len(done) == 1
    assert events[-1] is done[0]
    assert done[0].reason == reason
    return done[0]


def test_public_imports_and_run_agent_calling_contract() -> None:
    assert run_agent.__module__ == "agent.loop"
    assert (
        OpenAICompatibleLLMClient.__name__
        == "OpenAICompatibleLLMClient"
    )
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
        ("system", inspect.Parameter.KEYWORD_ONLY, None),
        ("tools", inspect.Parameter.KEYWORD_ONLY, None),
        ("thinking_level", inspect.Parameter.KEYWORD_ONLY, None),
        ("limits", inspect.Parameter.KEYWORD_ONLY, None),
        ("context", inspect.Parameter.KEYWORD_ONLY, None),
        ("policy", inspect.Parameter.KEYWORD_ONLY, None),
        ("stream", inspect.Parameter.KEYWORD_ONLY, False),
    ]


@pytest.mark.parametrize(
    "case",
    [
        ParityCase(
            name="buffered",
            stream=False,
            llm_script=(_answer("Hello"),),
            expected_events=(
                _usage_event(),
                {"text": "Hello", "type": "text"},
                _done_event("end_turn"),
            ),
            expected_messages=(
                _message("user", _text_block("go")),
                _message("assistant", _text_block("Hello")),
            ),
        ),
        ParityCase(
            name="streaming",
            stream=True,
            llm_script=(
                StreamAttempt(
                    (
                        TextDelta("Hel"),
                        TextDelta("lo"),
                        StreamEnd(_answer("Hello")),
                    )
                ),
            ),
            expected_events=(
                {"text": "Hel", "type": "text"},
                {"text": "lo", "type": "text"},
                _usage_event(),
                _done_event("end_turn"),
            ),
            expected_messages=(
                _message("user", _text_block("go")),
                _message("assistant", _text_block("Hello")),
            ),
        ),
    ],
    ids=lambda case: case.name,
)
async def test_happy_path_public_artifacts(case: ParityCase) -> None:
    events, store, session, llm, _ = await _run_case(case)

    assert normalize_events(events) == list(case.expected_events)
    assert normalize_messages(session.messages) == list(case.expected_messages)
    assert store.saved_messages == [list(case.expected_messages)]
    assert store.save_count == 1
    assert llm.complete_calls == (0 if case.stream else 1)
    assert llm.stream_calls == (1 if case.stream else 0)
    assert llm.stream_closes == (1 if case.stream else 0)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
@pytest.mark.parametrize(
    ("script", "reason", "calls", "error_message"),
    [
        ((TransientFailure("retry"), _answer("recovered")), "end_turn", 2, None),
        ((_empty(), _answer("recovered")), "end_turn", 2, None),
        (
            (TransientFailure("retry one"), TransientFailure("retry two")),
            "llm_error",
            2,
            "LLM call failed: retry two",
        ),
        ((ValueError("stop"),), "llm_error", 1, "LLM call failed: stop"),
    ],
    ids=["transient-recovery", "empty-recovery", "exhausted", "non-transient"],
)
async def test_retry_parity(
    stream: bool,
    script: tuple[AssistantMessage | BaseException, ...],
    reason: str,
    calls: int,
    error_message: str | None,
) -> None:
    case = ParityCase(
        name="retry",
        stream=stream,
        llm_script=script,
        limits=RunLimits(max_retries=1, retry_base_delay=0),
    )
    events, _, session, llm, _ = await _run_case(case)

    done = _assert_terminal(events, reason)
    assert llm.complete_calls + llm.stream_calls == calls
    assert llm.stream_closes == (calls if stream else 0)
    errors = [event.message for event in events if isinstance(event, ErrorEvent)]
    assert errors == ([error_message] if error_message is not None else [])
    if reason == "end_turn":
        assert done.total_tokens == 5
        assert normalize_messages(session.messages)[-1] == _message(
            "assistant", _text_block("recovered")
        )
    else:
        assert done.total_tokens == 0
        assert normalize_messages(session.messages) == [
            _message("user", _text_block("go"))
        ]


class TimeoutLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.closes = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        try:
            await asyncio.Event().wait()
            yield StreamEnd(_answer())
        finally:
            self.closes += 1


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_per_attempt_timeout_exhaustion_is_exact(stream: bool) -> None:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    llm = TimeoutLLM()
    events = [
        event
        async for event in run_agent(
            session,
            llm,
            ScriptedMCP(tools=[]),  # type: ignore[arg-type]
            store=store,
            stream=stream,
            limits=RunLimits(
                llm_timeout_seconds=0.01,
                max_retries=1,
                retry_base_delay=0,
            ),
        )
    ]

    assert normalize_events(events) == [
        {"message": "LLM call failed: ", "type": "error"},
        _done_event("llm_error", multiplier=0),
    ]
    assert llm.calls == 2
    assert llm.closes == (2 if stream else 0)
    assert store.save_count == 0


@pytest.mark.parametrize(
    ("attempts", "reason", "text", "calls"),
    [
        (
            (
                StreamAttempt(()),
                StreamAttempt((TextDelta("recovered"), StreamEnd(_answer("recovered")))),
            ),
            "end_turn",
            ["recovered"],
            2,
        ),
        (
            (StreamAttempt((TextDelta("partial"),)),),
            "incomplete_stream",
            ["partial"],
            1,
        ),
    ],
    ids=["silent-retry", "visible-incomplete"],
)
async def test_missing_stream_end_contract(
    attempts: tuple[StreamAttempt, ...],
    reason: str,
    text: list[str],
    calls: int,
) -> None:
    case = ParityCase(
        name="missing-end",
        stream=True,
        llm_script=attempts,
        limits=RunLimits(max_retries=1, retry_base_delay=0),
    )
    events, store, session, llm, _ = await _run_case(case)

    _assert_terminal(events, reason)
    assert [event.text for event in events if isinstance(event, TextEvent)] == text
    assert llm.stream_calls == llm.stream_closes == calls
    assert store.save_count == 1
    assert normalize_messages(session.messages)[-1] == _message(
        "assistant", _text_block("".join(text))
    )
    if reason == "incomplete_stream":
        assert [type(event) for event in events] == [
            TextEvent,
            UsageEvent,
            ErrorEvent,
            DoneEvent,
        ]
        assert normalize_event(events[-2]) == {
            "message": "LLM stream ended without a terminal provider message",
            "type": "error",
        }


@pytest.mark.parametrize(
    ("stop_reason", "raw_stop_reason", "done_reason", "has_error"),
    [
        ("end_turn", None, "end_turn", False),
        ("max_tokens", None, "truncated", False),
        ("content_filter", None, "content_filter", False),
        ("refusal", None, "refusal", False),
        ("empty", None, "empty", False),
        ("provider_error", "provider-failed", "provider_error", True),
        (None, "future-reason", "provider_error", True),
    ],
)
async def test_canonical_terminal_matrix(
    stop_reason: str | None,
    raw_stop_reason: str | None,
    done_reason: str,
    has_error: bool,
) -> None:
    response = _answer(
        "terminal text",
        stop_reason=stop_reason,
        raw_stop_reason=raw_stop_reason,
    )
    case = ParityCase(name="terminal", stream=False, llm_script=(response,))
    events, store, session, _, _ = await _run_case(case)

    done = _assert_terminal(events, done_reason)
    assert done.total_tokens == 5
    assert [isinstance(event, ErrorEvent) for event in events].count(True) == int(
        has_error
    )
    assert [event.type for event in events] == [
        "usage",
        "text",
        *(["error"] if has_error else []),
        "done",
    ]
    assert normalize_messages(session.messages) == [
        _message("user", _text_block("go")),
        _message("assistant", _text_block("terminal text")),
    ]
    assert store.save_count == 1


async def test_max_iterations_and_token_budget_terminals() -> None:
    max_iteration_case = ParityCase(
        name="max-iterations",
        stream=False,
        llm_script=(_answer("best effort"),),
        limits=RunLimits(max_iterations=1),
    )
    max_events, _, _, _, _ = await _run_case(max_iteration_case)
    assert normalize_events(max_events) == [
        _usage_event(),
        {"text": "best effort", "type": "text"},
        _done_event("max_iterations"),
    ]

    budget_response = _answer(
        "partial",
        usage=CompletionUsage(input_tokens=20, output_tokens=20, total_tokens=40),
    )
    budget_case = ParityCase(
        name="budget",
        stream=False,
        llm_script=(budget_response,),
        limits=RunLimits(max_run_tokens=30),
    )
    budget_events, _, _, _, _ = await _run_case(budget_case)
    assert [event.type for event in budget_events] == ["usage", "text", "done"]
    budget_done = _assert_terminal(budget_events, "budget_exceeded")
    assert (
        budget_done.input_tokens,
        budget_done.output_tokens,
        budget_done.total_tokens,
    ) == (20, 20, 40)


async def test_preexpired_wall_clock_deadline_is_terminal_without_work() -> None:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    llm = ParityLLM((_answer(),))
    limits = RunLimits(max_run_seconds=10)
    context = RunContext.start(
        max_run_seconds=10,
        base_logger=logging.getLogger("agent-parity"),
        run_id="run-fixed",
    )
    context.deadline = context.started_at - 1
    events = [
        event
        async for event in run_agent(
            session,
            llm,
            ScriptedMCP(tools=[]),  # type: ignore[arg-type]
            store=store,
            limits=limits,
            context=context,
        )
    ]

    assert normalize_events(events) == [
        {
            "reason": "deadline_exceeded",
            "iterations": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "type": "done",
        }
    ]
    assert llm.complete_calls == 0
    assert store.save_count == 0


@pytest.mark.parametrize(
    ("label", "tool_response", "tool_result", "expected_content", "is_error"),
    [
        (
            "success",
            _tool_call("call-success", {"q": "ok"}),
            ToolCallResult("tool ok", False),
            "tool ok",
            False,
        ),
        (
            "declared-error",
            _tool_call("call-declared", {"q": "bad"}),
            ToolCallResult("provider said no", True),
            "provider said no",
            True,
        ),
        (
            "exception",
            _tool_call("call-raised", {"q": "boom"}),
            RuntimeError("transport broke"),
            "tool execution raised: transport broke",
            True,
        ),
    ],
)
async def test_dispatched_tool_outcomes(
    label: str,
    tool_response: AssistantMessage,
    tool_result: ToolCallResult | BaseException,
    expected_content: str,
    is_error: bool,
) -> None:
    del label
    case = ParityCase(
        name="tool",
        stream=False,
        llm_script=(tool_response, _answer("done")),
        tool_script=(tool_result,),
    )
    events, store, session, _, mcp = await _run_case(case)

    _assert_terminal(events, "end_turn")
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert normalize_event(result) == {
        "id": tool_response.tool_uses()[0].id,
        "name": "srv__tool",
        "content": expected_content,
        "is_error": is_error,
        "latency_ms": _NUMERIC,
        "type": "tool_result",
    }
    assert mcp.calls == [("srv__tool", tool_response.tool_uses()[0].input)]
    assert normalize_messages(session.messages)[2] == _message(
        "tool",
        _tool_result_block(
            tool_response.tool_uses()[0].id,
            expected_content,
            is_error=is_error,
        ),
    )
    assert store.save_count == 2


@pytest.mark.parametrize("guard", ["invalid-json", "schema", "policy"])
async def test_tool_dispatch_guards_never_reach_mcp(guard: str) -> None:
    tools = [dict(_TOOL)]
    policy: ToolPolicy | None = None
    if guard == "invalid-json":
        response = _tool_call(
            "call-json",
            {},
            parse_error="arguments were not valid JSON: '{bad'",
        )
    elif guard == "schema":
        response = _tool_call("call-schema", {"q": 7})
        tools[0]["input_schema"] = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
    else:
        response = _tool_call("call-policy", {"q": "blocked"})
        policy = ToolPolicy(mode="allow_list", allow=[])

    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    llm = ParityLLM((response, _answer("done")))
    mcp = ScriptedMCP(tools=tools)
    events = [
        event
        async for event in run_agent(
            session,
            llm,
            mcp,  # type: ignore[arg-type]
            store=store,
            policy=policy,
        )
    ]

    _assert_terminal(events, "end_turn")
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert result.is_error is True
    assert result.latency_ms is None
    assert mcp.calls == []
    assert normalize_messages(session.messages)[2] == _message(
        "tool",
        _tool_result_block(
            response.tool_uses()[0].id,
            result.content,
            is_error=True,
        ),
    )
    if guard == "invalid-json":
        assert "not valid JSON" in result.content
    elif guard == "schema":
        assert "field 'q'" in result.content
    else:
        assert "blocked by policy" in result.content


async def test_exact_repeat_is_not_dispatched_twice() -> None:
    calls = (
        _tool_call("call-first", {"q": "same"}),
        _tool_call("call-repeat", {"q": "same"}),
        _answer("done"),
    )
    case = ParityCase(
        name="repeat",
        stream=False,
        llm_script=calls,
        tool_script=(ToolCallResult("first result", False),),
    )
    events, store, session, _, mcp = await _run_case(case)

    _assert_terminal(events, "end_turn")
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert [
        (
            result.id,
            result.content,
            result.is_error,
            result.latency_ms,
        )
        for result in results
    ] == [
        ("call-first", "first result", False, results[0].latency_ms),
        ("call-repeat", _STALL_MESSAGE, True, None),
    ]
    assert isinstance(results[0].latency_ms, (int, float))
    assert mcp.calls == [("srv__tool", {"q": "same"})]
    persisted = await store.get(session.session_id)
    assert normalize_messages(persisted.messages)[4] == _message(
        "tool", _tool_result_block("call-repeat", _STALL_MESSAGE, is_error=True)
    )


class BlockingToolMCP:
    def __init__(self, *, results: tuple[ToolCallResult, ...] = ()) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [dict(_TOOL)]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.calls.append((name, arguments))
        self.started.set()
        await self.release.wait()
        if self._results:
            return self._results.pop(0)
        return ToolCallResult("released", False)


async def test_tool_call_event_is_visible_before_blocked_dispatch_completes() -> None:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    mcp = BlockingToolMCP(results=(ToolCallResult("released", False),))
    stream = run_agent(
        session,
        ParityLLM((_tool_call("call-slow", {"q": "wait"}), _answer("done"))),
        mcp,  # type: ignore[arg-type]
        store=store,
    )

    assert isinstance(await anext(stream), UsageEvent)
    call_event = await anext(stream)
    assert normalize_event(call_event) == {
        "id": "call-slow",
        "name": "srv__tool",
        "input": {"q": "wait"},
        "type": "tool_call",
    }
    assert not mcp.started.is_set()

    result_task = asyncio.create_task(anext(stream))
    await mcp.started.wait()
    assert not result_task.done()
    mcp.release.set()
    result = await result_task
    assert isinstance(result, ToolResultEvent) and result.content == "released"
    remaining = [event async for event in stream]
    _assert_terminal([call_event, result, *remaining], "end_turn")


async def test_per_tool_timeout_and_result_clipping() -> None:
    timeout_store = RecordingStore()
    timeout_session = await timeout_store.create()
    timeout_session.append_user("go")
    timeout_mcp = BlockingToolMCP()
    timeout_events = [
        event
        async for event in run_agent(
            timeout_session,
            ParityLLM((_tool_call("call-timeout", {}), _answer("done"))),
            timeout_mcp,  # type: ignore[arg-type]
            store=timeout_store,
            limits=RunLimits(tool_timeout_seconds=0.01),
        )
    ]
    timeout_result = next(
        event for event in timeout_events if isinstance(event, ToolResultEvent)
    )
    assert normalize_event(timeout_result) == {
        "id": "call-timeout",
        "name": "srv__tool",
        "content": "tool 'srv__tool' timed out after 0.01s",
        "is_error": True,
        "latency_ms": _NUMERIC,
        "type": "tool_result",
    }
    _assert_terminal(timeout_events, "end_turn")

    clipped_case = ParityCase(
        name="clipped",
        stream=False,
        llm_script=(_tool_call("call-clipped", {}), _answer("done")),
        tool_script=(ToolCallResult("abcdefghij", False),),
        limits=RunLimits(tool_result_max_chars=4),
    )
    clipped_events, _, clipped_session, _, _ = await _run_case(clipped_case)
    clipped_result = next(
        event for event in clipped_events if isinstance(event, ToolResultEvent)
    )
    assert clipped_result.content == "abcd\n…[truncated, 6 chars omitted]"
    assert normalize_messages(clipped_session.messages)[2] == _message(
        "tool",
        _tool_result_block(
            "call-clipped",
            "abcd\n…[truncated, 6 chars omitted]",
            is_error=False,
        ),
    )


async def test_third_error_nudge_and_threshold_abort_are_exact() -> None:
    calls = tuple(
        _tool_call(f"call-{index}", {"index": index}) for index in range(1, 4)
    )
    nudge_case = ParityCase(
        name="nudge",
        stream=False,
        llm_script=(*calls, _answer("done")),
        tool_script=tuple(ToolCallResult("failed", True) for _ in range(3)),
    )
    nudge_events, _, _, _, _ = await _run_case(nudge_case)
    nudge_results = [
        event for event in nudge_events if isinstance(event, ToolResultEvent)
    ]
    assert [result.content for result in nudge_results[:2]] == ["failed", "failed"]
    assert nudge_results[2].content == f"failed\n\n{_FAILURE_NUDGE}"
    _assert_terminal(nudge_events, "end_turn")

    abort_case = ParityCase(
        name="abort",
        stream=False,
        llm_script=(
            _multi_tool_call(
                (
                    ("abort-1", {"index": 1}),
                    ("abort-2", {"index": 2}),
                    ("abort-3", {"index": 3}),
                )
            ),
        ),
        tool_script=(ToolCallResult("failed", True), ToolCallResult("failed", True)),
        limits=RunLimits(abort_after_consecutive_tool_failures=2),
    )
    abort_events, store, session, _, abort_mcp = await _run_case(abort_case)
    _assert_terminal(abort_events, "no_progress")
    abort_results = [
        event for event in abort_events if isinstance(event, ToolResultEvent)
    ]
    assert [result.id for result in abort_results] == [
        "abort-1",
        "abort-2",
        "abort-3",
    ]
    assert abort_results[-1].content == (
        "tool call skipped because the run aborted after consecutive tool failures"
    )
    assert abort_results[-1].latency_ms is None
    assert len(abort_mcp.calls) == 2
    assert store.save_count == 1
    assert len(normalize_messages(session.messages)[-1]["content"]) == 3  # type: ignore[arg-type]


async def test_complete_multi_tool_checkpoint_is_balanced_and_ordered() -> None:
    case = ParityCase(
        name="batch",
        stream=False,
        llm_script=(
            _multi_tool_call(
                (("batch-1", {"n": 1}), ("batch-2", {"n": 2}))
            ),
            _answer("done"),
        ),
        tool_script=(ToolCallResult("one", False), ToolCallResult("two", False)),
    )
    events, store, session, _, mcp = await _run_case(case)

    _assert_terminal(events, "end_turn")
    assert [event.type for event in events] == [
        "usage",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "usage",
        "text",
        "done",
    ]
    assert mcp.calls == [("srv__tool", {"n": 1}), ("srv__tool", {"n": 2})]
    expected_checkpoint = [
        _message("user", _text_block("go")),
        _message(
            "assistant",
            _tool_use_block("batch-1", {"n": 1}),
            _tool_use_block("batch-2", {"n": 2}),
        ),
        _message(
            "tool",
            _tool_result_block("batch-1", "one", is_error=False),
            _tool_result_block("batch-2", "two", is_error=False),
        ),
    ]
    assert store.saved_messages[0] == expected_checkpoint
    persisted = await store.get(session.session_id)
    assert normalize_messages(persisted.messages) == [
        *expected_checkpoint,
        _message("assistant", _text_block("done")),
    ]


class DetachingLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.second_started = asyncio.Event()
        self.release_second = asyncio.Event()

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        if self.calls == 1:
            return _tool_call("first-batch", {"n": 1})
        self.second_started.set()
        await self.release_second.wait()
        return _tool_call("second-batch", {"n": 2})


async def test_published_checkpoint_detaches_before_continuation() -> None:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    llm = DetachingLLM()
    mcp = ScriptedMCP(
        (ToolCallResult("first", False), ToolCallResult("second", False))
    )
    events: list[Event] = []
    second_call_visible = asyncio.Event()

    async def consume() -> None:
        generator = run_agent(
            session,
            llm,
            mcp,  # type: ignore[arg-type]
            store=store,
        )
        try:
            async for event in generator:
                events.append(event)
                if isinstance(event, ToolCallEvent) and event.id == "second-batch":
                    second_call_visible.set()
                    await asyncio.Event().wait()
        finally:
            await generator.aclose()

    task = asyncio.create_task(consume())
    await llm.second_started.wait()
    first_published = store.saved_sessions[0]
    first_snapshot = list(store.saved_messages[0])
    assert len(first_snapshot) == 3

    llm.release_second.set()
    await second_call_visible.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert normalize_messages(first_published.messages) == first_snapshot
    assert store.saved_sessions[-1] is not first_published
    assert normalize_messages((await store.get(session.session_id)).messages)[-1] == _message(
        "tool",
        _tool_result_block("second-batch", _NOT_STARTED, is_error=True),
    )


async def test_ephemeral_run_leaves_native_store_unchanged() -> None:
    native = RecordingStore()
    persisted = await native.create(metadata={"owner": "native"})
    persisted.append_user("existing")
    before = normalize_messages(persisted.messages)
    ephemeral = Session(session_id="ephemeral-fixed")
    ephemeral.append_user("go")

    events = [
        event
        async for event in run_agent(
            ephemeral,
            ParityLLM((_answer("private"),)),
            ScriptedMCP(tools=[]),  # type: ignore[arg-type]
            store=None,
        )
    ]

    _assert_terminal(events, "end_turn")
    assert native.ids() == [persisted.session_id]
    assert normalize_messages((await native.get(persisted.session_id)).messages) == before
    assert native.save_count == 0


class BlockingGenerationLLM(LLMClient):
    def __init__(self, *, stream: bool, visible: bool = False) -> None:
        self.use_stream = stream
        self.visible = visible
        self.started = asyncio.Event()
        self.waiting = asyncio.Event()
        self.closes = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        assert not self.use_stream
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        assert self.use_stream
        try:
            self.started.set()
            if self.visible:
                yield TextDelta("visible")
            self.waiting.set()
            await asyncio.Event().wait()
        finally:
            self.closes += 1


async def _cancel_collection(
    session: Session,
    store: RecordingStore,
    llm: LLMClient,
    mcp: Any,
    *,
    stream: bool = False,
    pause_after: Callable[[Event], bool] | None = None,
) -> list[Event]:
    observed: list[Event] = []
    paused = asyncio.Event()

    async def consume() -> None:
        generator = run_agent(
            session,
            llm,
            mcp,
            store=store,
            stream=stream,
        )
        try:
            async for event in generator:
                observed.append(event)
                if pause_after is not None and pause_after(event):
                    paused.set()
                    await asyncio.Event().wait()
        finally:
            await generator.aclose()

    task = asyncio.create_task(consume())
    if pause_after is not None:
        await paused.wait()
    elif isinstance(llm, BlockingGenerationLLM):
        await (llm.waiting.wait() if llm.visible else llm.started.wait())
    elif isinstance(mcp, BlockingToolMCP):
        await mcp.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not any(isinstance(event, (ErrorEvent, DoneEvent)) for event in observed)
    return observed


@pytest.mark.parametrize(
    ("stream", "visible"),
    [(False, False), (True, False), (True, True)],
    ids=["before-buffered-generation", "before-stream-delta", "after-stream-delta"],
)
async def test_cancellation_during_generation_preserves_unpublished_transcript(
    stream: bool,
    visible: bool,
) -> None:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    llm = BlockingGenerationLLM(stream=stream, visible=visible)
    observed = await _cancel_collection(
        session,
        store,
        llm,
        ScriptedMCP(tools=[]),
        stream=stream,
    )

    assert [event.text for event in observed if isinstance(event, TextEvent)] == (
        ["visible"] if visible else []
    )
    assert normalize_messages(session.messages) == [_message("user", _text_block("go"))]
    assert store.save_count == 0
    assert llm.closes == (1 if stream else 0)


async def test_cancellation_after_tool_call_before_dispatch_marks_not_started() -> None:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    mcp = BlockingToolMCP()
    observed = await _cancel_collection(
        session,
        store,
        ParityLLM((_tool_call("call-before", {}),)),
        mcp,
        pause_after=lambda event: isinstance(event, ToolCallEvent),
    )

    assert [event.type for event in observed] == ["usage", "tool_call"]
    assert mcp.calls == []
    assert store.save_count == 1
    assert normalize_messages((await store.get(session.session_id)).messages)[-1] == _message(
        "tool", _tool_result_block("call-before", _NOT_STARTED, is_error=True)
    )


async def test_cancellation_during_in_flight_tool_marks_outcome_unknown() -> None:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    mcp = BlockingToolMCP()
    observed = await _cancel_collection(
        session,
        store,
        ParityLLM((_tool_call("call-active", {}),)),
        mcp,
    )

    assert [event.type for event in observed] == ["usage", "tool_call"]
    assert mcp.calls == [("srv__tool", {})]
    assert store.save_count == 1
    assert normalize_messages((await store.get(session.session_id)).messages)[-1] == _message(
        "tool", _tool_result_block("call-active", _UNKNOWN_OUTCOME, is_error=True)
    )


async def test_cancellation_after_one_result_repairs_remaining_call() -> None:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    mcp = ScriptedMCP((ToolCallResult("first", False),))
    llm = ParityLLM(
        (
            _multi_tool_call(
                (("call-first", {"n": 1}), ("call-second", {"n": 2}))
            ),
        )
    )
    observed = await _cancel_collection(
        session,
        store,
        llm,
        mcp,
        pause_after=lambda event: (
            isinstance(event, ToolResultEvent) and event.id == "call-first"
        ),
    )

    assert [event.type for event in observed] == ["usage", "tool_call", "tool_result"]
    assert mcp.calls == [("srv__tool", {"n": 1})]
    persisted = normalize_messages((await store.get(session.session_id)).messages)
    assert persisted[-1] == _message(
        "tool",
        _tool_result_block("call-first", "first", is_error=False),
        _tool_result_block("call-second", _NOT_STARTED, is_error=True),
    )


async def test_generator_close_after_visible_delta_closes_stream_once() -> None:
    store = RecordingStore()
    session = await store.create()
    session.append_user("go")
    llm = BlockingGenerationLLM(stream=True, visible=True)
    generator = run_agent(
        session,
        llm,
        ScriptedMCP(tools=[]),  # type: ignore[arg-type]
        store=store,
        stream=True,
    )

    first = await anext(generator)
    assert normalize_event(first) == {"text": "visible", "type": "text"}
    await generator.aclose()

    assert llm.closes == 1
    assert store.save_count == 0
    assert normalize_messages(session.messages) == [_message("user", _text_block("go"))]
