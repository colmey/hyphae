"""Hermetic pytest coverage for the tool-dispatch policy seam.

Hermetic: drives run_agent with a scripted fake LLM and a fake MCP manager that
counts executions, so we can assert a denied tool never reaches MCP while an
allowed one does. No network, no backend.

Scenarios:
  1. allow-all default   - allowed tool executes (call_count == 1), ends normally.
  2. allow_list deny      - unlisted tool blocked: call_count == 0, a
                            ToolResultEvent(is_error=True) is emitted AND a
                            ToolResultBlock(is_error=True) is persisted; run
                            continues to a normal end.
  3. allow_list allow     - a listed tool still executes.
  4. repeated denies      - a model hammering blocked tools ends no_progress.

"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from agent import (
    DoneEvent,
    InMemorySessionStore,
    RunLimits,
    TextEvent,
    ToolPolicy,
    ToolResultEvent,
    run_agent,
)
from llm.client import LLMClient
from llm.schemas import AssistantMessage, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from mcp_layer.client import ToolCallResult

logging.basicConfig(level=logging.WARNING, format="%(levelname)-5s %(name)s: %(message)s")
pytestmark = pytest.mark.anyio


# --- scripted fakes --------------------------------------------------------

class ScriptedLLM(LLMClient):
    """Replays a fixed script of AssistantMessages."""

    def __init__(self, script: list[AssistantMessage]) -> None:
        self._script = list(script)
        self.calls = 0

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        self.calls += 1
        if not self._script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._script.pop(0)

    def is_transient_error(self, exc: BaseException) -> bool:
        return False


class CountingMCP:
    """Fake MCP that records how many tool calls actually reached it."""

    def __init__(self) -> None:
        self.call_count = 0

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [
            {"name": "srv__allowed", "description": "allowed", "input_schema": {}},
            {"name": "srv__denied", "description": "denied", "input_schema": {}},
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.call_count += 1
        return ToolCallResult(content="tool ran", is_error=False)


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AssistantMessage:
    return AssistantMessage(
        content=[ToolUseBlock(id=call_id, name=name, input=args)],
        stop_reason="tool_use", usage=Usage(),
    )


def text(text_: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text_)], stop_reason="end_turn", usage=Usage())


# --- harness ---------------------------------------------------------------

def check(cond: bool, msg: str) -> None:
    assert cond, msg


async def collect(label: str, llm: LLMClient, mcp: Any, **kwargs: Any):
    print(f"--- {label} ---")
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    events: list[Any] = []
    policy = kwargs.pop("policy", None)
    async for event in run_agent(
        session=session,
        llm=llm,
        mcp=mcp,
        store=store,
        policy=policy,
        limits=RunLimits(**kwargs),
    ):
        events.append(event)
    return events, session


def done_reason(events: list[Any]) -> str | None:
    return next((e.reason for e in events if isinstance(e, DoneEvent)), None)


def tool_results(events: list[Any]) -> list[ToolResultEvent]:
    return [e for e in events if isinstance(e, ToolResultEvent)]


async def test_allow_all_default_executes_tool() -> None:
    # 1. allow-all default: allowed tool executes.
    llm = ScriptedLLM([tool_call("srv__allowed", {}), text("done")])
    mcp = CountingMCP()
    events, _ = await collect("allow-all default", llm, mcp)  # no policy -> allow-all
    check(mcp.call_count == 1, f"allowed tool executed (call_count={mcp.call_count})")
    check(done_reason(events) == "end_turn", "run ended normally")
    check(not tool_results(events)[0].is_error, "tool result is not an error")

async def test_allow_list_denies_unlisted_tool_and_persists_error() -> None:
    policy = ToolPolicy(mode="allow_list", allow=["srv__allowed"])
    llm = ScriptedLLM([tool_call("srv__denied", {}), text("adapted")])
    mcp = CountingMCP()
    events, session = await collect("allow_list deny", llm, mcp, policy=policy)
    check(mcp.call_count == 0, f"denied tool never reached MCP (call_count={mcp.call_count})")
    results = tool_results(events)
    check(len(results) == 1 and results[0].is_error, "denied call emitted an is_error ToolResultEvent")
    check("policy" in results[0].content.lower(), "teaching message mentions policy")
    persisted = [b for m in session.messages for b in getattr(m, "content", [])
                 if isinstance(b, ToolResultBlock)]
    check(len(persisted) == 1 and persisted[0].is_error,
          "denied call persisted a ToolResultBlock(is_error=True)")
    check(done_reason(events) == "end_turn", "run continued and ended normally")

async def test_allow_list_executes_listed_tool() -> None:
    policy = ToolPolicy(mode="allow_list", allow=["srv__allowed"])
    llm = ScriptedLLM([tool_call("srv__allowed", {}), text("done")])
    mcp = CountingMCP()
    events, _ = await collect("allow_list allow", llm, mcp, policy=policy)
    check(mcp.call_count == 1, f"allowed tool executed under allow_list (call_count={mcp.call_count})")
    check(done_reason(events) == "end_turn", "run ended normally")

async def test_repeated_denials_terminate_no_progress() -> None:
    # Distinct args keep stall detection from short-circuiting the policy path.
    policy = ToolPolicy(mode="allow_list", allow=["srv__allowed"])
    llm = ScriptedLLM([tool_call("srv__denied", {"n": i}, call_id=f"c{i}") for i in range(5)])
    mcp = CountingMCP()
    events, _ = await collect("repeated denies -> no_progress", llm, mcp,
                              policy=policy, abort_after_consecutive_tool_failures=3)
    check(mcp.call_count == 0, f"no denied call ever reached MCP (call_count={mcp.call_count})")
    check(done_reason(events) == "no_progress", "repeated denies ended no_progress")
