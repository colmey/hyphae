"""Focused tests for run-scoped tool dispatch and active batch state."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
import logging
from typing import Any

import pytest

from agent import RunContext, RunLimits, Session, ToolPolicy
from agent.runtime import RunDeadlineExceeded
from agent.tool_execution import (
    ActiveToolBatch,
    ToolDispatchResult,
    ToolDispatcher,
    _STALL_MESSAGE,
    make_skipped_tool_result,
)
from agent.tool_policy import PolicyDecision, PolicyVerdict
from llm.schemas import Role, ToolResultBlock, ToolUseBlock
from tooling import ToolCallResult


pytestmark = pytest.mark.anyio

_TOOL_NAME = "srv__tool"
_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {"q": {"type": "string"}},
    "required": ["q"],
}


def _tool_use(
    call_id: str,
    args: dict[str, Any] | None = None,
    *,
    name: str = _TOOL_NAME,
    parse_error: str | None = None,
) -> ToolUseBlock:
    return ToolUseBlock(
        id=call_id,
        name=name,
        input={} if args is None else args,
        parse_error=parse_error,
    )


def _tools(schema: Any = None) -> list[dict[str, Any]]:
    return [
        {
            "name": _TOOL_NAME,
            "description": "test tool",
            "input_schema": {} if schema is None else schema,
        }
    ]


def _result(
    tool_use: ToolUseBlock,
    dispatch: ToolDispatchResult,
) -> ToolResultBlock:
    return ToolResultBlock(
        tool_use_id=tool_use.id,
        name=tool_use.name,
        content=dispatch.content,
        is_error=dispatch.is_error,
    )


class ScriptedRuntime:
    def __init__(self, script: list[ToolCallResult | BaseException]) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.active_calls = 0
        self.max_active_calls = 0

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return _tools()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult:
        self.calls.append((name, arguments))
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(0)
            if not self.script:
                raise AssertionError("tool script exhausted")
            outcome = self.script.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        finally:
            self.active_calls -= 1


class BlockingRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return _tools()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult:
        self.calls.append((name, arguments))
        self.started.set()
        await self.release.wait()
        return ToolCallResult("released", False)


class RecordingPolicy(ToolPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, Any]] = []

    def check(self, tool_name: str, args: Any = None) -> PolicyDecision:
        self.calls.append((tool_name, args))
        return PolicyDecision(PolicyVerdict.ALLOW)


def _dispatcher(
    runtime: Any,
    *,
    tools: list[dict[str, Any]] | None = None,
    policy: ToolPolicy | None = None,
    limits: RunLimits | None = None,
    context: RunContext | None = None,
) -> ToolDispatcher:
    resolved_limits = limits or RunLimits()
    return ToolDispatcher(
        runtime=runtime,
        policy=policy or ToolPolicy(),
        tools=tools or _tools(),
        limits=resolved_limits,
        context=context
        or RunContext.start(
            max_run_seconds=resolved_limits.max_run_seconds,
            base_logger=logging.getLogger("tests.tool_execution"),
        ),
        log=logging.getLogger("tests.tool_execution"),
    )


async def test_dispatch_result_is_frozen_and_slotted() -> None:
    result = ToolDispatchResult("ok", False, 1.25)

    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.content = "changed"  # type: ignore[misc]


async def test_parse_and_validation_failures_skip_policy_and_runtime() -> None:
    runtime = ScriptedRuntime([])
    policy = RecordingPolicy()
    dispatcher = _dispatcher(runtime, tools=_tools(_OBJECT_SCHEMA), policy=policy)
    invalid_json = _tool_use("json", parse_error="unexpected token")
    invalid_schema = _tool_use("schema", {"q": 3})

    json_result = await dispatcher.dispatch(
        ActiveToolBatch((invalid_json,)),
        invalid_json,
    )
    schema_result = await dispatcher.dispatch(
        ActiveToolBatch((invalid_schema,)),
        invalid_schema,
    )

    assert json_result == ToolDispatchResult(
        "tool call arguments were not valid JSON (unexpected token); "
        "return the arguments as a JSON object matching the tool schema.",
        True,
        None,
    )
    assert schema_result.is_error is True
    assert "field 'q'" in schema_result.content
    assert policy.calls == []
    assert runtime.calls == []


async def test_policy_denial_happens_after_validation_without_runtime_call() -> None:
    runtime = ScriptedRuntime([])
    policy = ToolPolicy(mode="allow_list", allow=[])
    dispatcher = _dispatcher(runtime, tools=_tools(_OBJECT_SCHEMA), policy=policy)
    tool_use = _tool_use("denied", {"q": "valid"})

    result = await dispatcher.dispatch(ActiveToolBatch((tool_use,)), tool_use)

    assert result.is_error is True
    assert "blocked by policy" in result.content
    assert result.latency_ms is None
    assert runtime.calls == []


async def test_unadvertised_tool_is_permissive_before_policy() -> None:
    runtime = ScriptedRuntime([ToolCallResult("ok", False)])
    policy = RecordingPolicy()
    dispatcher = _dispatcher(runtime, tools=_tools(_OBJECT_SCHEMA), policy=policy)
    tool_use = _tool_use("unknown", {"anything": 1}, name="srv__unknown")
    batch = ActiveToolBatch((tool_use,))

    dispatch = await dispatcher.dispatch(batch, tool_use)
    batch.complete(_result(tool_use, dispatch))

    assert dispatch.content == "ok"
    assert policy.calls == [("srv__unknown", {"anything": 1})]
    assert runtime.calls == [("srv__unknown", {"anything": 1})]


async def test_exact_repeat_across_batches_uses_canonical_arguments() -> None:
    runtime = ScriptedRuntime([ToolCallResult("first", False)])
    dispatcher = _dispatcher(runtime)
    first = _tool_use("first", {"a": 1, "b": 2})
    first_batch = ActiveToolBatch((first,))
    first_dispatch = await dispatcher.dispatch(first_batch, first)
    first_batch.complete(_result(first, first_dispatch))

    repeated = _tool_use("repeat", {"b": 2, "a": 1})
    repeated_batch = ActiveToolBatch((repeated,))
    repeated_dispatch = await dispatcher.dispatch(repeated_batch, repeated)
    repeated_batch.complete(_result(repeated, repeated_dispatch))

    assert repeated_dispatch == ToolDispatchResult(_STALL_MESSAGE, True, None)
    assert runtime.calls == [(_TOOL_NAME, {"a": 1, "b": 2})]


@pytest.mark.parametrize(
    ("outcome", "content", "is_error"),
    [
        (ToolCallResult("ok", False), "ok", False),
        (ToolCallResult("declared failure", True), "declared failure", True),
        (RuntimeError("transport broke"), "tool execution raised: transport broke", True),
    ],
)
async def test_real_dispatch_outcomes_have_latency(
    outcome: ToolCallResult | BaseException,
    content: str,
    is_error: bool,
) -> None:
    runtime = ScriptedRuntime([outcome])
    dispatcher = _dispatcher(runtime)
    tool_use = _tool_use("call")
    batch = ActiveToolBatch((tool_use,))

    dispatch = await dispatcher.dispatch(batch, tool_use)
    batch.complete(_result(tool_use, dispatch))

    assert dispatch.content == content
    assert dispatch.is_error is is_error
    assert isinstance(dispatch.latency_ms, float)
    assert runtime.calls == [(_TOOL_NAME, {})]


async def test_caller_dispatches_multi_call_batch_sequentially() -> None:
    runtime = ScriptedRuntime(
        [ToolCallResult("one", False), ToolCallResult("two", False)]
    )
    dispatcher = _dispatcher(runtime)
    tool_uses = (_tool_use("one", {"n": 1}), _tool_use("two", {"n": 2}))
    batch = ActiveToolBatch(tool_uses)

    for tool_use in tool_uses:
        dispatch = await dispatcher.dispatch(batch, tool_use)
        batch.complete(_result(tool_use, dispatch))

    assert runtime.calls == [
        (_TOOL_NAME, {"n": 1}),
        (_TOOL_NAME, {"n": 2}),
    ]
    assert runtime.max_active_calls == 1


async def test_ordinary_tool_timeout_is_a_recoverable_result() -> None:
    runtime = BlockingRuntime()
    limits = RunLimits(tool_timeout_seconds=0.01, max_run_seconds=1.0)
    dispatcher = _dispatcher(runtime, limits=limits)
    tool_use = _tool_use("timeout")
    batch = ActiveToolBatch((tool_use,))

    dispatch = await dispatcher.dispatch(batch, tool_use)
    batch.complete(_result(tool_use, dispatch))

    assert dispatch.content == "tool 'srv__tool' timed out after 0.01s"
    assert dispatch.is_error is True
    assert isinstance(dispatch.latency_ms, float)
    assert runtime.calls == [(_TOOL_NAME, {})]


async def test_absolute_deadline_raises_and_balances_in_flight_call() -> None:
    runtime = BlockingRuntime()
    limits = RunLimits(tool_timeout_seconds=1.0, max_run_seconds=0.01)
    context = RunContext.start(
        max_run_seconds=limits.max_run_seconds,
        base_logger=logging.getLogger("tests.tool_execution.deadline"),
    )
    dispatcher = _dispatcher(runtime, limits=limits, context=context)
    tool_uses = (_tool_use("active"), _tool_use("remaining"))
    batch = ActiveToolBatch(tool_uses)

    with pytest.raises(RunDeadlineExceeded):
        await dispatcher.dispatch(batch, tool_uses[0])

    batch.balance_after_interruption()
    session = Session()
    batch.append_to(session)
    results = session.messages[-1].content

    assert runtime.calls == [(_TOOL_NAME, {})]
    assert [result.content for result in results] == [
        "tool call outcome is unknown because execution was cancelled while "
        "the call was in flight",
        "tool call was not executed because execution was cancelled",
    ]


async def test_batch_rejects_out_of_order_dispatch_and_mismatched_completion() -> None:
    first = _tool_use("first")
    second = _tool_use("second")
    batch = ActiveToolBatch((first, second))

    with pytest.raises(ValueError, match="batch order"):
        batch.start_dispatch(second)
    with pytest.raises(ValueError, match="does not match"):
        batch.complete(
            ToolResultBlock(
                tool_use_id="wrong",
                name=first.name,
                content="bad",
            )
        )

    batch.start_dispatch(first)
    with pytest.raises(RuntimeError, match="already in flight"):
        batch.start_dispatch(first)


async def test_complete_remaining_validates_exact_order() -> None:
    first = _tool_use("first")
    second = _tool_use("second")
    batch = ActiveToolBatch((first, second))

    with pytest.raises(ValueError, match="every remaining"):
        batch.complete_remaining([make_skipped_tool_result(second, "skipped")])

    batch.complete_remaining(
        [
            make_skipped_tool_result(first, "skipped"),
            make_skipped_tool_result(second, "skipped"),
        ]
    )
    session = Session()
    batch.append_to(session)
    assert [block.tool_use_id for block in session.messages[-1].content] == [
        "first",
        "second",
    ]


async def test_interruption_balancing_and_append_are_idempotent() -> None:
    first = _tool_use("first")
    active = _tool_use("active")
    remaining = _tool_use("remaining")
    batch = ActiveToolBatch((first, active, remaining))
    batch.complete(
        ToolResultBlock(
            tool_use_id=first.id,
            name=first.name,
            content="completed",
        )
    )
    batch.start_dispatch(active)

    batch.balance_after_interruption()
    session = Session()
    batch.append_to(session)
    batch.balance_after_interruption()
    batch.append_to(session)

    assert len(session.messages) == 1
    assert session.messages[0].role is Role.TOOL
    assert [block.content for block in session.messages[0].content] == [
        "completed",
        "tool call outcome is unknown because execution was cancelled while "
        "the call was in flight",
        "tool call was not executed because execution was cancelled",
    ]


async def test_incomplete_batch_cannot_be_appended() -> None:
    batch = ActiveToolBatch((_tool_use("pending"),))

    with pytest.raises(ValueError, match="incomplete"):
        batch.append_to(Session())


async def test_cancellation_escapes_while_real_call_remains_in_flight() -> None:
    runtime = BlockingRuntime()
    dispatcher = _dispatcher(runtime)
    tool_uses = (_tool_use("active"), _tool_use("remaining"))
    batch = ActiveToolBatch(tool_uses)

    task = asyncio.create_task(dispatcher.dispatch(batch, tool_uses[0]))
    await runtime.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    batch.balance_after_interruption()
    session = Session()
    batch.append_to(session)
    results = session.messages[-1].content

    assert runtime.calls == [(_TOOL_NAME, {})]
    assert [result.content for result in results] == [
        "tool call outcome is unknown because execution was cancelled while "
        "the call was in flight",
        "tool call was not executed because execution was cancelled",
    ]
