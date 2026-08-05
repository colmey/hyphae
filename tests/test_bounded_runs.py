"""Pytest coverage for bounded and safe agent runs: token cap, wall-clock cap,
tool-argument validation, and the no-progress abort threshold.

Scripted LLM and MCP fakes drive ``run_agent`` without network access.

  1. Token cap            - cumulative usage crosses max_run_tokens -> done
                             "budget_exceeded" with the partial text already
                             streamed; no further LLM call is made.
  2. Wall-clock cap        - a slow fake LLM call pushes elapsed time past
                             max_run_seconds -> done "deadline_exceeded"
                             before a second LLM call.
  3. Zero usage now trips the token cap via the local estimator (Phase 3):
                             with max_run_tokens set, an all-zero CompletionUsage is
                             estimated from the outgoing messages + response
                             and can reach budget_exceeded; without a cap the
                             estimate changes nothing.
  4. Tool-argument validation - an invalid call (wrong type for a required
                             field) never reaches mcp.call_tool and comes
                             back as an is_error result naming the field; a
                             subsequent valid call executes normally.
  5. No-progress abort     - N consecutive tool failures end the run
                             "no_progress" without asking the LLM again;
                             N-1 failures followed by a success reset the
                             counter and the run completes normally.
  6. /v1 finish_reason mapping - the three new done_reasons map to a sane,
                             non-None OpenAI finish_reason.

"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from agent import (
    DoneEvent,
    InMemorySessionStore,
    RunLimits,
    Session,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    run_agent,
)
from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    CompletionUsage,
)
from tooling import ToolCallResult

logging.basicConfig(
    level=logging.WARNING, format="%(levelname)-5s %(name)s: %(message)s"
)
pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Scripted fakes
# ---------------------------------------------------------------------------


class ScriptedLLM(LLMClient):
    """Replays a fixed script of responses, with an optional artificial delay
    before each return (for exercising the wall-clock cap)."""

    def __init__(self, script: list[Any], delay: float = 0.0) -> None:
        self._script = list(script)
        self._delay = delay
        self.calls = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._script.pop(0)

    def is_transient_error(self, exc: BaseException) -> bool:
        return False


_QTOOL_SCHEMA = {
    "type": "object",
    "properties": {"q": {"type": "string"}},
    "required": ["q"],
}


class ScriptedMCP:
    """Minimal MCPManager stand-in: replays (content, is_error) per call_tool()."""

    def __init__(
        self,
        script: list[tuple[str, bool]] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self._script = list(script or [])
        self._schema = schema if schema is not None else {}
        self.call_count = 0

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "srv__tool",
                "description": "test tool",
                "input_schema": self._schema,
            }
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.call_count += 1
        content, is_error = self._script.pop(0)
        return ToolCallResult(content=content, is_error=is_error)


# ----- AssistantMessage builders -----


def text_response(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)], stop_reason="end_turn", usage=CompletionUsage()
    )


def text_and_tool_response(
    text: str,
    args: dict[str, Any],
    *,
    call_id: str = "call_1",
    usage: CompletionUsage | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=[
            TextBlock(text=text),
            ToolUseBlock(
                id=call_id,
                name="srv__tool",
                input=args,
            ),
        ],
        stop_reason="tool_use",
        usage=usage or CompletionUsage(),
    )


def tool_call_response(
    args: dict[str, Any], *, call_id: str = "call_1"
) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolUseBlock(
                id=call_id,
                name="srv__tool",
                input=args,
            )
        ],
        stop_reason="tool_use",
        usage=CompletionUsage(),
    )


def multi_tool_response(args: list[dict[str, Any]]) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolUseBlock(
                id=f"call_{index}",
                name="srv__tool",
                input=tool_arguments,
            )
            for index, tool_arguments in enumerate(args, start=1)
        ],
        stop_reason="tool_use",
        usage=CompletionUsage(),
    )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def check(cond: bool, msg: str) -> None:
    assert cond, msg


async def collect(label: str, llm: LLMClient, mcp: Any, **kwargs: Any) -> list[Any]:
    print(f"--- {label} ---")
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    events: list[Any] = []
    async for event in run_agent(
        session=session, llm=llm, mcp=mcp, store=store, limits=RunLimits(**kwargs)
    ):
        events.append(event)
    return events


async def collect_with_session(
    label: str,
    llm: LLMClient,
    mcp: Any,
    **kwargs: Any,
) -> tuple[list[Any], Session]:
    print(f"--- {label} ---")
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    events: list[Any] = []
    async for event in run_agent(
        session=session, llm=llm, mcp=mcp, store=store, limits=RunLimits(**kwargs)
    ):
        events.append(event)
    return events, session


def done_reason(events: list[Any]) -> str | None:
    for e in events:
        if isinstance(e, DoneEvent):
            return e.reason
    return None


def all_text(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextEvent))


def tool_results(events: list[Any]) -> list[ToolResultEvent]:
    return [e for e in events if isinstance(e, ToolResultEvent)]


def tool_calls(events: list[Any]) -> list[ToolCallEvent]:
    return [e for e in events if isinstance(e, ToolCallEvent)]


async def test_token_cap_preserves_partial_text() -> None:
    # 1. Token cap
    llm = ScriptedLLM(
        [
            text_and_tool_response(
                "partial answer", {"q": "x"}, usage=CompletionUsage(total_tokens=40)
            ),
        ]
    )
    mcp = ScriptedMCP([("tool ok", False)], schema=_QTOOL_SCHEMA)
    events = await collect("token cap", llm, mcp, max_run_tokens=30, max_iterations=10)
    check(done_reason(events) == "budget_exceeded", "done_reason == budget_exceeded")
    check(all_text(events) == "partial answer", "partial text survived the cap")
    check(llm.calls == 1, f"no second LLM call after the cap tripped (got {llm.calls})")


async def test_wall_clock_cap_prevents_second_model_call() -> None:
    llm = ScriptedLLM(
        [tool_call_response({"q": "x"}), text_response("done")], delay=0.05
    )
    mcp = ScriptedMCP([("tool ok", False)], schema=_QTOOL_SCHEMA)
    events = await collect(
        "wall-clock cap", llm, mcp, max_run_seconds=0.03, max_iterations=10
    )
    check(
        done_reason(events) == "deadline_exceeded", "done_reason == deadline_exceeded"
    )
    check(llm.calls == 1, f"only the first (slow) LLM call ran (got {llm.calls})")


async def test_final_answer_crossing_token_cap_reports_budget_exceeded() -> None:
    llm = ScriptedLLM(
        [
            AssistantMessage(
                content=[TextBlock(text="final partial")],
                stop_reason="end_turn",
                usage=CompletionUsage(total_tokens=40),
            ),
        ]
    )
    mcp = ScriptedMCP([], schema=_QTOOL_SCHEMA)
    events = await collect(
        "final-answer token cap", llm, mcp, max_run_tokens=30, max_iterations=10
    )
    check(
        done_reason(events) == "budget_exceeded",
        "final answer crossing token cap reports budget_exceeded",
    )
    check(all_text(events) == "final partial", "final partial text survived token cap")


async def test_final_answer_crossing_deadline_is_not_streamed() -> None:
    llm = ScriptedLLM([text_response("late final")], delay=0.05)
    mcp = ScriptedMCP([], schema=_QTOOL_SCHEMA)
    events = await collect(
        "final-answer deadline cap", llm, mcp, max_run_seconds=0.03, max_iterations=10
    )
    check(
        done_reason(events) == "deadline_exceeded",
        "final answer crossing wall clock reports deadline_exceeded",
    )
    check(all_text(events) == "", "timed-out LLM answer was not streamed")


async def test_slow_tool_exhausting_deadline_ends_run() -> None:
    llm = ScriptedLLM([tool_call_response({"q": "x"}), text_response("done")])
    mcp = ScriptedMCP([("tool ok", False)], schema=_QTOOL_SCHEMA)
    original_call_tool = mcp.call_tool

    async def slow_call_tool(name: str, arguments: dict[str, Any]) -> ToolCallResult:
        await asyncio.sleep(0.05)
        return await original_call_tool(name, arguments)

    mcp.call_tool = slow_call_tool  # type: ignore[method-assign]
    events = await collect(
        "slow-tool deadline cap",
        llm,
        mcp,
        max_run_seconds=0.03,
        tool_timeout_seconds=1.0,
        max_iterations=10,
    )
    check(
        done_reason(events) == "deadline_exceeded",
        "slow tool crossing wall clock reports deadline_exceeded",
    )
    check(
        llm.calls == 1, f"no second LLM call after slow-tool deadline (got {llm.calls})"
    )
    check(
        len(tool_results(events)) == 1 and tool_results(events)[0].is_error,
        "slow tool got a synthetic error result",
    )


async def test_zero_usage_trips_token_cap_via_estimator() -> None:
    llm = ScriptedLLM(
        [text_response("an estimated answer long enough to count as tokens")]
    )
    mcp = ScriptedMCP([], schema=_QTOOL_SCHEMA)
    events = await collect(
        "zero usage trips token cap via estimator",
        llm,
        mcp,
        max_run_tokens=5,
        max_iterations=10,
    )
    check(
        done_reason(events) == "budget_exceeded",
        "all-zero usage trips budget_exceeded via the estimator",
    )


async def test_zero_usage_without_cap_ends_normally() -> None:
    llm = ScriptedLLM([text_response("hi")])
    mcp = ScriptedMCP([], schema=_QTOOL_SCHEMA)
    events = await collect("zero usage without a cap", llm, mcp, max_iterations=10)
    check(
        done_reason(events) == "end_turn", "no cap set -> zero usage run ends normally"
    )


async def test_invalid_tool_arguments_are_not_dispatched() -> None:
    llm = ScriptedLLM(
        [
            tool_call_response({"q": 123}, call_id="call_1"),  # wrong type for "q"
            tool_call_response({"q": "ok"}, call_id="call_2"),  # valid
            text_response("done"),
        ]
    )
    mcp = ScriptedMCP([("tool ok", False)], schema=_QTOOL_SCHEMA)
    events = await collect("tool-argument validation", llm, mcp, max_iterations=10)
    trs = tool_results(events)
    check(len(trs) == 2, f"two tool results emitted (got {len(trs)})")
    check(trs[0].is_error, "invalid call came back as an error")
    check(
        "'q'" in trs[0].content or '"q"' in trs[0].content,
        "error names the offending field",
    )
    check(
        mcp.call_count == 1,
        f"mcp.call_tool ran only for the valid call (got {mcp.call_count})",
    )
    check(not trs[1].is_error, "the valid call executed normally")
    check(
        done_reason(events) == "end_turn", "run completed normally after the correction"
    )


async def test_consecutive_tool_failures_abort_no_progress() -> None:
    llm = ScriptedLLM(
        [tool_call_response({"q": str(i)}, call_id=f"call_{i}") for i in range(1, 6)]
    )
    mcp = ScriptedMCP([("boom", True)] * 5, schema=_QTOOL_SCHEMA)
    events = await collect(
        "no-progress abort",
        llm,
        mcp,
        abort_after_consecutive_tool_failures=5,
        max_iterations=20,
    )
    check(done_reason(events) == "no_progress", "done_reason == no_progress")
    check(
        llm.calls == 5, f"no 6th LLM call after the abort threshold (got {llm.calls})"
    )
    check(mcp.call_count == 5, f"exactly 5 tool calls ran (got {mcp.call_count})")


async def test_success_resets_tool_failure_counter() -> None:
    llm = ScriptedLLM(
        [
            tool_call_response({"q": "1"}, call_id="call_1"),
            tool_call_response({"q": "2"}, call_id="call_2"),
            tool_call_response(
                {"q": "3"}, call_id="call_3"
            ),  # succeeds, resets the streak
            tool_call_response({"q": "4"}, call_id="call_4"),
            tool_call_response({"q": "5"}, call_id="call_5"),
            text_response("done"),
        ]
    )
    mcp = ScriptedMCP(
        [("boom", True), ("boom", True), ("ok", False), ("boom", True), ("boom", True)],
        schema=_QTOOL_SCHEMA,
    )
    events = await collect(
        "no-progress counter reset by a success",
        llm,
        mcp,
        abort_after_consecutive_tool_failures=3,
        max_iterations=20,
    )
    check(
        done_reason(events) == "end_turn",
        "run completed; reset streak never hit the threshold",
    )
    check(mcp.call_count == 5, f"all 5 tool calls ran (got {mcp.call_count})")


async def test_mid_batch_abort_preserves_tool_call_result_pairing() -> None:
    llm = ScriptedLLM([multi_tool_response([{"q": "1"}, {"q": "2"}, {"q": "3"}])])
    mcp = ScriptedMCP([("boom", True)], schema=_QTOOL_SCHEMA)
    events, session = await collect_with_session(
        "mid-batch no-progress shape",
        llm,
        mcp,
        abort_after_consecutive_tool_failures=1,
        max_iterations=10,
    )
    saved_tool_results = [
        b for b in session.messages[-1].content if isinstance(b, ToolResultBlock)
    ]
    check(done_reason(events) == "no_progress", "mid-batch abort reports no_progress")
    check(
        len(tool_calls(events)) == 3,
        f"all 3 tool calls surfaced (got {len(tool_calls(events))})",
    )
    check(
        len(tool_results(events)) == 3,
        f"all 3 tool results emitted (got {len(tool_results(events))})",
    )
    check(
        len(saved_tool_results) == 3,
        f"all 3 tool results persisted (got {len(saved_tool_results)})",
    )
    check(
        mcp.call_count == 1,
        f"only the first failing tool reached MCP (got {mcp.call_count})",
    )


async def test_initial_routing_usage_blocks_downstream_call_at_run_token_limit() -> None:
    llm = ScriptedLLM([text_response("must not run")])
    mcp = ScriptedMCP(schema=_QTOOL_SCHEMA)
    session = Session()
    session.append_user("go")
    events = [
        event
        async for event in run_agent(
            session=session,
            llm=llm,
            mcp=mcp,
            limits=RunLimits(max_run_tokens=5),
            initial_usage=CompletionUsage(
                input_tokens=3, output_tokens=2, total_tokens=5
            ),
        )
    ]

    assert llm.calls == 0
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].reason == "budget_exceeded"
    assert events[-1].total_tokens == 5


@pytest.mark.parametrize(
    "reason", ["budget_exceeded", "deadline_exceeded", "no_progress"]
)
def test_bounded_done_reasons_map_to_openai_finish_reason(reason: str) -> None:
    from api.openai_compatible import _finish_reason

    mapped = _finish_reason(reason)
    assert isinstance(mapped, str) and mapped
