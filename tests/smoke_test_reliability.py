"""
Smoke test for the agent-loop reliability hardening.

Unlike smoke_test_agent.py, this hits no network: it drives run_agent with a
scripted fake LLMClient (a queue of responses/exceptions + a controllable
is_transient_error) and a scriptable fake MCP manager. That lets us exercise
the failure/edge paths deterministically:

  1. Retry-then-succeed      - transient error, then success -> one retry.
  2. Non-transient fast-fail - non-transient error -> no retry, llm_error.
  3. Empty-response retry    - stop_reason="empty" once, then a real answer.
  4. Truncation reason       - stop_reason="max_tokens" -> done "truncated".
  5. Tool timeout            - a slow tool -> is_error result, loop continues.
  6. Tool-result clip        - oversized tool output -> clipped + marker.

Run from the project root:
    ./runscript.sh tests/smoke_test_reliability.py
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent import (
    DoneEvent,
    ErrorEvent,
    InMemorySessionStore,
    TextEvent,
    ToolResultEvent,
    run_agent,
)
from llm.client import LLMClient
from llm.schemas import AssistantMessage, TextBlock, ToolUseBlock, Usage
from mcp_layer.client import ToolCallResult

logging.basicConfig(level=logging.WARNING, format="%(levelname)-5s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Scripted fakes
# ---------------------------------------------------------------------------

class TransientError(Exception):
    """An error the fake client classifies as transient (retryable)."""


class ScriptedLLM(LLMClient):
    """An LLMClient that replays a fixed script of responses/exceptions.

    Each script item is either an AssistantMessage (returned) or a BaseException
    (raised). `transient_types` controls is_transient_error so we can drive the
    loop's retry policy without provider specifics.
    """

    def __init__(self, script: list[Any], transient_types: tuple[type, ...] = ()) -> None:
        self._script = list(script)
        self._transient_types = transient_types
        self.calls = 0

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        self.calls += 1
        if not self._script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def is_transient_error(self, exc: BaseException) -> bool:
        return isinstance(exc, self._transient_types)


class FakeMCP:
    """Minimal stand-in for MCPManager: just what run_agent touches."""

    def __init__(self, *, content: str = "tool ok", is_error: bool = False,
                 delay: float = 0.0) -> None:
        self._content = content
        self._is_error = is_error
        self._delay = delay
        self.call_count = 0

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [{"name": "srv__tool", "description": "test tool", "input_schema": {}}]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return ToolCallResult(content=self._content, is_error=self._is_error)


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

_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"    [{status}] {msg}")
    if not cond:
        _failures.append(msg)


async def collect(label: str, llm: LLMClient, mcp: Any, **kwargs: Any) -> list[Any]:
    print(f"--- {label} ---")
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    events: list[Any] = []
    async for event in run_agent(session=session, llm=llm, mcp=mcp, store=store, **kwargs):
        events.append(event)
    return events


def done_reason(events: list[Any]) -> str | None:
    for e in events:
        if isinstance(e, DoneEvent):
            return e.reason
    return None


def all_text(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextEvent))


def tool_results(events: list[Any]) -> list[ToolResultEvent]:
    return [e for e in events if isinstance(e, ToolResultEvent)]


async def main() -> None:
    # 1. Retry-then-succeed
    llm = ScriptedLLM([TransientError("429"), text_response("recovered")],
                      transient_types=(TransientError,))
    events = await collect("retry-then-succeed", llm, FakeMCP(),
                           max_retries=2, retry_base_delay=0.0)
    check(llm.calls == 2, f"complete() called twice (got {llm.calls})")
    check(done_reason(events) == "end_turn", "done_reason == end_turn")
    check(all_text(events) == "recovered", "final answer survived the retry")
    check(not any(isinstance(e, ErrorEvent) for e in events), "no ErrorEvent emitted")

    # 2. Non-transient error fails fast
    llm = ScriptedLLM([ValueError("boom")], transient_types=(TransientError,))
    events = await collect("non-transient fails fast", llm, FakeMCP(),
                           max_retries=2, retry_base_delay=0.0)
    check(llm.calls == 1, f"complete() called once, no retry (got {llm.calls})")
    check(done_reason(events) == "llm_error", "done_reason == llm_error")
    check(any(isinstance(e, ErrorEvent) for e in events), "ErrorEvent emitted")

    # 3. Empty-response retry
    llm = ScriptedLLM([empty_response(), text_response("second try")],
                      transient_types=(TransientError,))
    events = await collect("empty-response retry", llm, FakeMCP(),
                           max_retries=2, retry_base_delay=0.0)
    check(llm.calls == 2, f"empty response retried (got {llm.calls} calls)")
    check(done_reason(events) == "end_turn", "done_reason == end_turn")
    check(all_text(events) == "second try", "answer from the retry attempt")

    # 4. Truncation reason
    llm = ScriptedLLM([truncated_response("half an answe")])
    events = await collect("truncation reason", llm, FakeMCP())
    check(done_reason(events) == "truncated", "done_reason == truncated")
    check(all_text(events) == "half an answe", "partial text still streamed")

    # 5. Tool timeout
    llm = ScriptedLLM([tool_call_response(), text_response("handled the timeout")])
    mcp = FakeMCP(delay=5.0)
    events = await collect("tool timeout", llm, mcp,
                           tool_timeout_seconds=0.1)
    trs = tool_results(events)
    check(len(trs) == 1 and trs[0].is_error, "tool result is an error")
    check("timed out" in trs[0].content, "error mentions the timeout")
    check(done_reason(events) == "end_turn", "loop continued to a final answer")
    check(llm.calls == 2, f"model ran again after the timeout (got {llm.calls})")

    # 6. Tool-result clip
    big = "x" * 100_000
    llm = ScriptedLLM([tool_call_response(), text_response("summarized")])
    mcp = FakeMCP(content=big)
    events = await collect("tool-result clip", llm, mcp,
                           tool_result_max_chars=1000)
    trs = tool_results(events)
    check(len(trs) == 1, "one tool result")
    check(len(trs[0].content) < 2000, f"content clipped (len={len(trs[0].content)})")
    check("[truncated," in trs[0].content, "truncation marker present")

    print()
    if _failures:
        print(f"RELIABILITY SMOKE TEST FAILED: {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("reliability smoke test complete: all checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
