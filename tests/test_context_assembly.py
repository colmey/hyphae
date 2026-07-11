"""
Smoke test for Phase 3 context assembly: the token estimator, budget math,
the naive and compaction strategies, view-only compaction invariants, and
the estimated-usage fallback for the Phase-1 token cap.

Like the other loop smoke tests, this hits no network: a scripted fake
LLMClient (which also plays the compaction summarizer) plus a fake MCP
manager drive assemble_context() and run_agent() directly.

  1. Estimator          - non-zero sane counts for text / tool-use /
                          tool-result messages; zero for empty text; tools
                          schemas counted.
  2. Budget math        - input_budget = window - max_output - margin,
                          clamped at zero; over/under threshold.
  3. Usage estimation   - provider usage passes through untouched; a
                          zero total is filled from input+output; all-zero
                          usage is estimated non-zero from messages+response.
  4. naive              - under budget: pass-through (same list object);
                          over budget: pass-through + degraded_reason, no
                          mutation, no exception.
  5. compaction         - under budget: pass-through. Over budget: first
                          user message pinned verbatim, recent units kept
                          verbatim, summary message inserted, result under
                          budget, no orphaned tool-use/tool-result pairs,
                          original history structurally identical after.
  6. Summarizer failure - degrades to pass-through; a full run_agent run
                          with a failing summarizer still completes.
  7. Loop integration   - run_agent under compaction sends the compacted
                          view to the LLM while session.messages only grows
                          by the run's normal appends.
  8. Token-cap fallback - a zero-usage response trips budget_exceeded via
                          the estimator when max_run_tokens is low; non-zero
                          provider usage is used as-is (never double-counted).

Run from the project root with ``./runscript.sh -m pytest tests/test_context_assembly.py``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest

from agent import DoneEvent, InMemorySessionStore, TextEvent, run_agent
from agent.context import (
    _SUMMARY_HEADER,
    _SUMMARY_SYSTEM,
    ContextBudget,
    assemble_context,
    estimate_message_tokens,
    estimate_text_tokens,
    estimate_tools_tokens,
    estimate_usage_tokens,
)
from llm.client import LLMClient
from llm.schemas import (
    AssistantMessage,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from mcp_layer.client import ToolCallResult

logging.basicConfig(level=logging.ERROR, format="%(levelname)-5s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Scripted fakes
# ---------------------------------------------------------------------------

class RecordingLLM(LLMClient):
    """Replays a script for loop calls and answers summarizer calls directly.

    Summarizer calls are recognized by the compaction system prompt, so one
    client can play both roles (exactly as in production, where the selected
    model summarizes its own history).
    """

    def __init__(self, script: list[Any], *, summary_text: str = "- decided: use the API",
                 fail_summary: bool = False) -> None:
        self._script = list(script)
        self._summary_text = summary_text
        self._fail_summary = fail_summary
        self.loop_messages: list[list[Message]] = []
        self.summary_messages: list[list[Message]] = []
        self.summary_calls = 0
        self.summary_had_tools: list[bool] = []

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        if system == _SUMMARY_SYSTEM:
            self.summary_calls += 1
            self.summary_messages.append(list(messages))
            self.summary_had_tools.append(tools is not None)
            if self._fail_summary:
                raise RuntimeError("summarizer backend down")
            return AssistantMessage(
                content=[TextBlock(text=self._summary_text)],
                stop_reason="end_turn", usage=Usage(),
            )
        self.loop_messages.append(list(messages))
        if not self._script:
            raise AssertionError("RecordingLLM ran out of scripted responses")
        return self._script.pop(0)

    def is_transient_error(self, exc: BaseException) -> bool:
        return False


class ScriptedMCP:
    """Minimal MCPManager stand-in: replays (content, is_error) per call_tool()."""

    def __init__(self, script: list[tuple[str, bool]] | None = None) -> None:
        self._script = list(script or [])
        self.call_count = 0

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [{"name": "srv__tool", "description": "test tool", "input_schema": {}}]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.call_count += 1
        content, is_error = self._script.pop(0)
        return ToolCallResult(content=content, is_error=is_error)


# ----- message/history builders -----

def text_response(text: str, *, usage: Usage | None = None) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="end_turn",
                            usage=usage or Usage())


def tool_pair(call_id: str, result: str) -> list[Message]:
    """One protocol-safe unit: assistant tool_use + its tool result."""
    return [
        Message.assistant([ToolUseBlock(id=call_id, name="srv__tool", input={"q": call_id})]),
        Message.tool_results([ToolResultBlock(tool_use_id=call_id, name="srv__tool",
                                              content=result)]),
    ]


def over_budget_history() -> list[Message]:
    """Task header + a fat middle of tool exchanges + a small recent tail."""
    history: list[Message] = [Message.user("TASK: find the launch date of the probe")]
    history.append(Message.assistant([TextBlock(text="I will look this up.")]))
    for i in range(1, 4):
        history.extend(tool_pair(f"call_mid_{i}", "x" * 2000))  # ~500 tokens each
    history.append(Message.assistant([TextBlock(text="Interim: narrowed to 2031.")]))
    history.append(Message.user("Also check the landing site."))
    # recent tail (small, so it survives verbatim)
    history.extend(tool_pair("call_recent", "site=Utopia Planitia"))
    history.append(Message.user("So what is the final answer?"))
    return history


def seed_session(session, history: list[Message]) -> None:
    """Load a scripted history through the session's own append helpers."""
    for msg in history:
        if msg.role == Role.USER:
            session.append_user(next(b.text for b in msg.content if isinstance(b, TextBlock)))
        elif msg.role == Role.ASSISTANT:
            session.append_assistant(
                AssistantMessage(content=list(msg.content), stop_reason="end_turn"))
        else:
            session.append_tool_results(
                [b for b in msg.content if isinstance(b, ToolResultBlock)])


def protocol_ok(messages: list[Message]) -> bool:
    """No orphan tool results and no unanswered assistant tool calls."""
    for i, msg in enumerate(messages):
        if msg.role == Role.TOOL:
            prev = messages[i - 1] if i else None
            if prev is None or prev.role != Role.ASSISTANT:
                return False
            use_ids = {b.id for b in prev.content if isinstance(b, ToolUseBlock)}
            result_ids = {b.tool_use_id for b in msg.content if isinstance(b, ToolResultBlock)}
            if not result_ids <= use_ids:
                return False
        if msg.role == Role.ASSISTANT and any(isinstance(b, ToolUseBlock) for b in msg.content):
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if nxt is None or nxt.role != Role.TOOL:
                return False
    return True


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.anyio


def check(cond: bool, msg: str) -> None:
    assert cond, msg


def section(label: str) -> None:
    print(f"--- {label} ---")


def test_token_estimators() -> None:
    # 1. Estimator
    section("estimator")
    check(estimate_text_tokens("") == 0, "empty text estimates 0")
    check(estimate_text_tokens("ok") >= 1, "short non-empty text estimates >= 1")
    check(estimate_text_tokens("x" * 400) == 100, "chars/4 heuristic (400 chars -> 100)")
    text_msg = Message.user("What is the launch date of the probe?")
    tool_msg = Message.assistant([ToolUseBlock(id="c1", name="srv__tool",
                                               input={"query": "launch date"})])
    result_msg = Message.tool_results([ToolResultBlock(tool_use_id="c1", name="srv__tool",
                                                       content="date=2031-06-01")])
    for label, msg in [("text", text_msg), ("tool-use", tool_msg), ("tool-result", result_msg)]:
        check(estimate_message_tokens([msg]) > 0, f"{label} message estimates non-zero")
    check(
        estimate_message_tokens([text_msg, tool_msg, result_msg])
        == sum(estimate_message_tokens([m]) for m in (text_msg, tool_msg, result_msg)),
        "message list estimate is the sum of its messages",
    )
    check(estimate_tools_tokens(None) == 0, "no tools estimates 0")
    check(estimate_tools_tokens([{"name": "srv__tool", "input_schema": {"type": "object"}}]) > 0,
          "tool schemas estimate non-zero")

def test_context_budget_math() -> None:
    section("budget math")
    budget = ContextBudget(context_window=1000, max_output_tokens=200, safety_margin=100)
    check(budget.input_budget == 700, "input_budget = window - max_output - margin")
    check(ContextBudget(context_window=100, max_output_tokens=200).input_budget == 0,
          "input_budget clamps at zero")
    history = over_budget_history()
    est = estimate_message_tokens(history)
    check(est > 700, f"scripted history is over the test budget (est={est})")

def test_usage_estimation() -> None:
    section("usage estimation")
    history = over_budget_history()
    text_msg = Message.user("What is the launch date of the probe?")
    provider = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
    check(estimate_usage_tokens(provider, messages=history) is provider,
          "non-zero provider usage passes through untouched")
    partial = Usage(input_tokens=10, output_tokens=5, total_tokens=0)
    filled = estimate_usage_tokens(partial, messages=history)
    check(filled.total_tokens == 15 and filled.input_tokens == 10,
          "zero total filled from provider input+output (not estimated)")
    estimated = estimate_usage_tokens(
        Usage(), messages=[text_msg], system="be brief",
        response=text_response("a fairly long answer with some words in it"),
    )
    check(estimated.total_tokens > 0 and estimated.input_tokens > 0
          and estimated.output_tokens > 0,
          "all-zero usage estimated non-zero from messages+system+response")
    check(estimate_usage_tokens(None, messages=[text_msg]).total_tokens > 0,
          "missing usage estimated non-zero")

async def test_naive_context_strategy() -> None:
    section("naive strategy")
    budget = ContextBudget(context_window=1000, max_output_tokens=200, safety_margin=100)
    history = over_budget_history()
    small = [Message.user("hi")]
    assembled = await assemble_context(small, budget=budget, strategy="naive")
    check(assembled.messages is small, "under budget: pass-through returns the same list")
    check(not assembled.compacted and assembled.degraded_reason is None,
          "under budget: not compacted, not degraded")

    snapshot = copy.deepcopy(history)
    assembled = await assemble_context(history, budget=budget, strategy="naive")
    check(assembled.messages is history, "over budget: naive still passes through")
    check(assembled.degraded_reason == "over_budget", "over budget: degraded_reason set")
    check(history == snapshot, "naive never mutates the history")

    unknown = await assemble_context(small, budget=budget, strategy="wat")
    check(unknown.strategy == "naive", "unknown strategy degrades to naive")

async def test_compaction_strategy() -> None:
    section("compaction strategy")
    budget = ContextBudget(context_window=1000, max_output_tokens=200, safety_margin=100)
    history = over_budget_history()
    est = estimate_message_tokens(history)
    small = [Message.user("hi")]
    llm = RecordingLLM([], summary_text="- launch narrowed to 2031\n- landing site pending")
    under = await assemble_context(small, budget=budget, strategy="compaction", llm=llm)
    check(under.messages is small and not under.compacted,
          "under budget: compaction passes through, no summarizer call")
    check(llm.summary_calls == 0, "no summarizer call when under budget")

    snapshot = copy.deepcopy(history)
    assembled = await assemble_context(
        history, budget=budget, strategy="compaction", llm=llm,
        recent_messages=2, summary_max_tokens=64,
    )
    check(assembled.compacted, "over budget: compacted")
    check(llm.summary_calls == 1, "exactly one summarizer call")
    check(llm.summary_had_tools == [False], "summarizer call carries no tools")
    summary_prompt = llm.summary_messages[0][0]
    summary_transcript = next(
        b.text for b in summary_prompt.content if isinstance(b, TextBlock)
    )
    check("transcript truncated for summarization" in summary_transcript,
          "summarizer transcript is clipped with the summary marker")
    check(assembled.messages[0] == history[0], "first user message (task header) verbatim")
    summary_texts = [
        b.text for m in assembled.messages for b in m.content
        if isinstance(b, TextBlock) and b.text.startswith(_SUMMARY_HEADER)
    ]
    check(len(summary_texts) == 1 and "2031" in summary_texts[0],
          "summary message inserted with the summarizer's text")
    # recent_messages=2 -> last two protocol units verbatim: the recent tool
    # pair and the final user question.
    check(assembled.messages[-3:] == history[-3:], "recent tail verbatim")
    check(protocol_ok(assembled.messages), "no orphan tool results / unanswered tool calls")
    check(assembled.estimated_input_tokens <= budget.input_budget,
          f"compacted view under budget (est={assembled.estimated_input_tokens})")
    check(assembled.estimated_input_tokens < est, "compacted view materially smaller")
    check(history == snapshot, "compaction never mutates the original history")

async def test_too_short_history_degrades_gracefully() -> None:
    llm = RecordingLLM([])
    short = [Message.user("x" * 4000)]
    degraded = await assemble_context(
        short, budget=ContextBudget(context_window=100), strategy="compaction", llm=llm)
    check(degraded.messages is short and degraded.degraded_reason == "too_short_to_compact",
          "too-short over-budget history passes through with a reason")

async def test_summarizer_failure_degrades_to_pass_through() -> None:
    section("summarizer failure")
    budget = ContextBudget(context_window=1000, max_output_tokens=200, safety_margin=100)
    history = over_budget_history()
    failing = RecordingLLM([], fail_summary=True)
    snapshot = copy.deepcopy(history)
    assembled = await assemble_context(
        history, budget=budget, strategy="compaction", llm=failing, recent_messages=2)
    check(not assembled.compacted and assembled.degraded_reason == "summarizer_failed",
          "summarizer failure -> pass-through with degraded_reason")
    check(assembled.messages is history and history == snapshot,
          "summarizer failure leaves history untouched")

    # ...and a full run still completes
    failing = RecordingLLM([text_response("final answer after degrade")], fail_summary=True)
    store = InMemorySessionStore()
    session = await store.create()
    seed_session(session, over_budget_history())
    events = []
    async for event in run_agent(
        session=session, llm=failing, mcp=ScriptedMCP(), store=store,
        context_strategy="compaction", context_window=1000,
        context_safety_margin_tokens=100, max_tokens=200,
        context_recent_messages=2, max_iterations=5,
    ):
        events.append(event)
    done = next(e for e in events if isinstance(e, DoneEvent))
    check(done.reason == "end_turn", "run completes end_turn despite summarizer failure")
    check(failing.summary_calls == 1, "summarizer was attempted once")
    check(len(failing.loop_messages[0]) == len(session.messages) - 1,
          "degraded call sent the full history")

async def test_loop_uses_compacted_view_without_mutating_session_history() -> None:
    section("loop integration (compaction)")
    llm = RecordingLLM([text_response("the probe launches in 2031")])
    store = InMemorySessionStore()
    session = await store.create()
    seed_session(session, over_budget_history())
    before = copy.deepcopy(session.messages)
    events = []
    async for event in run_agent(
        session=session, llm=llm, mcp=ScriptedMCP(), store=store,
        context_strategy="compaction", context_window=1000,
        context_safety_margin_tokens=100, max_tokens=200,
        context_recent_messages=2, context_summary_max_tokens=64,
        max_iterations=5,
    ):
        events.append(event)
    done = next(e for e in events if isinstance(e, DoneEvent))
    sent = llm.loop_messages[0]
    check(done.reason == "end_turn", "compacted run completes end_turn")
    check(llm.summary_calls == 1, "one summarizer call for the over-budget view")
    check(len(sent) < len(before), "LLM saw the compacted view, not full history")
    check(sent[0] == before[0], "compacted view pins the task header")
    check(any(isinstance(b, TextBlock) and b.text.startswith(_SUMMARY_HEADER)
              for m in sent for b in m.content), "compacted view contains the summary")
    check(protocol_ok(sent), "compacted view is protocol-safe")
    check(session.messages[:len(before)] == before,
          "session history untouched by compaction (view-only)")
    check(len(session.messages) == len(before) + 1
          and session.messages[-1].role == Role.ASSISTANT,
          "session grew only by the run's assistant turn")
    answer = "".join(e.text for e in events if isinstance(e, TextEvent))
    check("2031" in answer, "answer text streamed normally")

async def test_agent_token_cap_falls_back_to_estimator() -> None:
    section("token cap via estimator")
    llm = RecordingLLM([text_response("a zero-usage answer that is long enough to count")])
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    events = []
    async for event in run_agent(
        session=session, llm=llm, mcp=ScriptedMCP(), store=store,
        max_run_tokens=5, max_iterations=10,
    ):
        events.append(event)
    done = next(e for e in events if isinstance(e, DoneEvent))
    check(done.reason == "budget_exceeded",
          "zero-usage response trips budget_exceeded via the estimator")
    check(done.total_tokens > 0, "DoneEvent carries the estimated (non-zero) tokens")

async def test_nonzero_provider_usage_is_authoritative() -> None:
    llm = RecordingLLM([text_response("answer", usage=Usage(input_tokens=30, output_tokens=10,
                                                            total_tokens=40))])
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    events = []
    async for event in run_agent(session=session, llm=llm, mcp=ScriptedMCP(), store=store,
                                 max_iterations=5):
        events.append(event)
    done = next(e for e in events if isinstance(e, DoneEvent))
    check(done.total_tokens == 40, "provider usage used as-is (no estimate added)")
