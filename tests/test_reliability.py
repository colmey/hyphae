"""Hermetic pytest coverage for agent-loop reliability hardening.

This hits no network: it drives run_agent with a
scripted fake LLMClient (a queue of responses/exceptions + a controllable
is_transient_error) and a scriptable fake MCP manager. That lets us exercise
the failure/edge paths deterministically:

  1. Retry-then-succeed      - transient error, then success -> one retry.
  2. Non-transient fast-fail - non-transient error -> no retry, llm_error.
  3. Empty-response retry    - stop_reason="empty" once, then a real answer.
  4. Truncation reason       - stop_reason="max_tokens" -> done "truncated".
  5. Tool timeout            - a slow tool -> is_error result, loop continues.
  6. Tool-result clip        - oversized tool output -> clipped + marker.

"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from agent import (
    DoneEvent,
    ErrorEvent,
    TextEvent,
    ToolResultEvent,
)
from llm.schemas import AssistantMessage, TextBlock, ToolUseBlock, Usage

logging.basicConfig(level=logging.WARNING, format="%(levelname)-5s %(name)s: %(message)s")
pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Scripted fakes
# ---------------------------------------------------------------------------

class TransientError(Exception):
    """An error the fake client classifies as transient (retryable)."""


# ----- AssistantMessage builders -----

def text_response(text: str, stop_reason: str = "end_turn") -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason=stop_reason,
                            usage=Usage())


def empty_response() -> AssistantMessage:
    return AssistantMessage(content=[], stop_reason="empty", usage=Usage())


def truncated_response(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="max_tokens",
                            usage=Usage())


def tool_call_response(name: str = "srv__tool") -> AssistantMessage:
    return AssistantMessage(
        content=[ToolUseBlock(id="call_1", name=name, input={})],
        stop_reason="tool_use", usage=Usage(),
    )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def check(cond: bool, msg: str) -> None:
    assert cond, msg


def done_reason(events: list[Any]) -> str | None:
    for e in events:
        if isinstance(e, DoneEvent):
            return e.reason
    return None


def all_text(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextEvent))


def tool_results(events: list[Any]) -> list[ToolResultEvent]:
    return [e for e in events if isinstance(e, ToolResultEvent)]


async def test_retry_then_succeed(
    scripted_llm_factory, scripted_mcp_factory, agent_event_collector
) -> None:
    # 1. Retry-then-succeed
    llm = scripted_llm_factory(
        [TransientError("429"), text_response("recovered")],
        transient_types=(TransientError,),
    )
    events = await agent_event_collector(
        llm=llm, mcp=scripted_mcp_factory(), max_retries=2, retry_base_delay=0.0
    )
    check(llm.calls == 2, f"complete() called twice (got {llm.calls})")
    check(done_reason(events) == "end_turn", "done_reason == end_turn")
    check(all_text(events) == "recovered", "final answer survived the retry")
    check(not any(isinstance(e, ErrorEvent) for e in events), "no ErrorEvent emitted")

async def test_non_transient_error_fails_fast(
    scripted_llm_factory, scripted_mcp_factory, agent_event_collector
) -> None:
    llm = scripted_llm_factory([ValueError("boom")], transient_types=(TransientError,))
    events = await agent_event_collector(
        llm=llm, mcp=scripted_mcp_factory(), max_retries=2, retry_base_delay=0.0
    )
    check(llm.calls == 1, f"complete() called once, no retry (got {llm.calls})")
    check(done_reason(events) == "llm_error", "done_reason == llm_error")
    check(any(isinstance(e, ErrorEvent) for e in events), "ErrorEvent emitted")

async def test_empty_response_retry(
    scripted_llm_factory, scripted_mcp_factory, agent_event_collector
) -> None:
    llm = scripted_llm_factory(
        [empty_response(), text_response("second try")], transient_types=(TransientError,)
    )
    events = await agent_event_collector(
        llm=llm, mcp=scripted_mcp_factory(), max_retries=2, retry_base_delay=0.0
    )
    check(llm.calls == 2, f"empty response retried (got {llm.calls} calls)")
    check(done_reason(events) == "end_turn", "done_reason == end_turn")
    check(all_text(events) == "second try", "answer from the retry attempt")

async def test_truncation_reason(
    scripted_llm_factory, scripted_mcp_factory, agent_event_collector
) -> None:
    llm = scripted_llm_factory([truncated_response("half an answe")])
    events = await agent_event_collector(llm=llm, mcp=scripted_mcp_factory())
    check(done_reason(events) == "truncated", "done_reason == truncated")
    check(all_text(events) == "half an answe", "partial text still streamed")

async def test_tool_timeout(
    scripted_llm_factory, scripted_mcp_factory, agent_event_collector
) -> None:
    llm = scripted_llm_factory([tool_call_response(), text_response("handled the timeout")])
    mcp = scripted_mcp_factory(delay=5.0)
    events = await agent_event_collector(llm=llm, mcp=mcp, tool_timeout_seconds=0.1)
    trs = tool_results(events)
    check(len(trs) == 1 and trs[0].is_error, "tool result is an error")
    check("timed out" in trs[0].content, "error mentions the timeout")
    check(done_reason(events) == "end_turn", "loop continued to a final answer")
    check(llm.calls == 2, f"model ran again after the timeout (got {llm.calls})")

async def test_tool_result_clip(
    scripted_llm_factory, scripted_mcp_factory, agent_event_collector
) -> None:
    big = "x" * 100_000
    llm = scripted_llm_factory([tool_call_response(), text_response("summarized")])
    mcp = scripted_mcp_factory(content=big)
    events = await agent_event_collector(
        llm=llm, mcp=mcp, tool_result_max_chars=1000
    )
    trs = tool_results(events)
    check(len(trs) == 1, "one tool result")
    check(len(trs[0].content) < 2000, f"content clipped (len={len(trs[0].content)})")
    check("[truncated," in trs[0].content, "truncation marker present")
