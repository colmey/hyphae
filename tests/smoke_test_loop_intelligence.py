"""
Smoke test for the agent-loop intelligence scaffolding.

Like smoke_test_reliability.py, this hits no network: it drives run_agent with a
scripted fake LLMClient (a queue of responses, recording the system/tools each
call received) and a scriptable fake MCP manager. That lets us exercise the
steering behaviors deterministically:

  1. Final-iteration wrap-up - on the last allowed iteration the loop
     withholds tools and appends a wrap-up note to the system prompt, forcing a
     best-effort answer; the run still reports done "max_iterations".
  2. Stall detection         - an identical repeat tool call is
     short-circuited to an is_error result without re-executing.
  3. Distinct args not blocked    - two calls with different args both run
     (guards against over-eager dedup).
  4. Consecutive-failure nudge    - after 3 failures in a row, a steering note
     is appended (once) to the crossing result.

Run from the project root:
    ./runscript.sh tests/smoke_test_loop_intelligence.py
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent import (
    DoneEvent,
    InMemorySessionStore,
    TextEvent,
    ToolResultEvent,
    run_agent,
)
from agent.loop import _FAILURE_NUDGE, _FINAL_ITERATION_WRAPUP, _STALL_MESSAGE
from llm.client import LLMClient
from llm.schemas import AssistantMessage, TextBlock, ToolUseBlock, Usage
from mcp_layer.client import ToolCallResult

logging.basicConfig(level=logging.WARNING, format="%(levelname)-5s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Scripted fakes
# ---------------------------------------------------------------------------

class ScriptedLLM(LLMClient):
    """Replays a fixed script of responses, recording each call's system/tools.

    The recorded `systems`/`tools_seen` lists let the final-iteration wrap-up
    test assert that the final iteration withheld tools and injected the note.
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls = 0
        self.systems: list[str | None] = []
        self.tools_seen: list[Any] = []
        self.thinking_seen: list[str | None] = []

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        self.calls += 1
        self.systems.append(system)
        self.tools_seen.append(tools)
        self.thinking_seen.append(thinking_level)
        if not self._script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._script.pop(0)


class FakeMCP:
    """Minimal stand-in for MCPManager. Counts real (non-stall) executions."""

    def __init__(self, *, content: str = "tool ok", is_error: bool = False) -> None:
        self._content = content
        self._is_error = is_error
        self.call_count = 0

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [{"name": "srv__tool", "description": "test tool", "input_schema": {}}]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.call_count += 1
        return ToolCallResult(content=self._content, is_error=self._is_error)


# ----- AssistantMessage builders -----

def text_response(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="end_turn",
                            usage=Usage())


def tool_call_response(args: dict[str, Any] | None = None, *, name: str = "srv__tool",
                       call_id: str = "call_1") -> AssistantMessage:
    return AssistantMessage(
        content=[ToolUseBlock(id=call_id, name=name, input=args or {})],
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
    # 1. Final-iteration wrap-up
    llm = ScriptedLLM([tool_call_response(), text_response("best-effort answer")])
    mcp = FakeMCP()
    events = await collect("final-iteration wrap-up", llm, mcp,
                           system="You are helpful.", max_iterations=2)
    check(llm.calls == 2, f"two iterations ran (got {llm.calls})")
    check(llm.tools_seen[0] is not None, "iteration 1 saw tools")
    check(llm.tools_seen[1] is None, "final iteration withheld tools")
    check(_FINAL_ITERATION_WRAPUP in (llm.systems[1] or ""),
          "final iteration system prompt carries the wrap-up note")
    check(_FINAL_ITERATION_WRAPUP not in (llm.systems[0] or ""),
          "earlier iteration system prompt is untouched")
    check(done_reason(events) == "max_iterations", "done_reason == max_iterations")
    check(all_text(events) == "best-effort answer", "forced final answer streamed")
    check(mcp.call_count == 1, f"the one tool ran on iteration 1 (got {mcp.call_count})")

    # 2. Stall detection: identical repeat short-circuited
    llm = ScriptedLLM([
        tool_call_response({"q": "x"}),
        tool_call_response({"q": "x"}),
        text_response("answered"),
    ])
    mcp = FakeMCP()
    events = await collect("stall detection", llm, mcp, max_iterations=10)
    trs = tool_results(events)
    check(mcp.call_count == 1, f"identical repeat NOT re-executed (got {mcp.call_count})")
    check(len(trs) == 2, f"two tool results emitted (got {len(trs)})")
    check(trs[1].is_error and _STALL_MESSAGE in trs[1].content,
          "repeat returned the synthetic stall error")
    check(not trs[0].is_error, "the first (real) call was a normal result")
    check(done_reason(events) == "end_turn", "loop continued to a final answer")

    # 3. Distinct args are NOT blocked (guards against over-eager dedup)
    llm = ScriptedLLM([
        tool_call_response({"q": "x"}),
        tool_call_response({"q": "y"}),
        text_response("done"),
    ])
    mcp = FakeMCP()
    events = await collect("distinct args not blocked", llm, mcp, max_iterations=10)
    trs = tool_results(events)
    check(mcp.call_count == 2, f"both distinct calls executed (got {mcp.call_count})")
    check(not any(_STALL_MESSAGE in t.content for t in trs), "no stall message emitted")

    # 4. Consecutive-failure nudge: 3 errors in a row -> nudge on the 3rd
    llm = ScriptedLLM([
        tool_call_response({"i": 1}),
        tool_call_response({"i": 2}),
        tool_call_response({"i": 3}),
        text_response("giving up"),
    ])
    mcp = FakeMCP(content="tool failed", is_error=True)
    events = await collect("consecutive-failure nudge", llm, mcp, max_iterations=10)
    trs = tool_results(events)
    check(len(trs) == 3 and all(t.is_error for t in trs), "three error results")
    check(_FAILURE_NUDGE not in trs[0].content, "no nudge on the 1st failure")
    check(_FAILURE_NUDGE not in trs[1].content, "no nudge on the 2nd failure")
    check(_FAILURE_NUDGE in trs[2].content, "nudge appended on the 3rd consecutive failure")

    print()
    if _failures:
        print(f"LOOP-INTELLIGENCE SMOKE TEST FAILED: {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("loop-intelligence smoke test complete: all checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
