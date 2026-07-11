"""
Smoke test for Phase 7 token streaming.

Hermetic by design: scripted LLM clients drive run_agent directly, with no
network or provider backend. Run from the project root:
    ./runscript.sh tests/smoke_test_streaming.py
"""

from __future__ import annotations

from bootstrap import load_secrets
load_secrets()

import asyncio
from typing import Any, AsyncIterator

from agent import DoneEvent, ErrorEvent, InMemorySessionStore, TextEvent, ToolCallEvent, run_agent
from llm.client import LLMClient
from llm.providers.openai import _ReasoningStreamStripper
from llm.schemas import (
    AssistantMessage,
    Message,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    Usage,
)
from mcp_layer.client import ToolCallResult


class FakeMCP:
    def __init__(self) -> None:
        self.call_count = 0

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [{
            "name": "srv__echo",
            "description": "echo",
            "input_schema": {"type": "object", "properties": {"value": {"type": "string"}}},
        }]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.call_count += 1
        return ToolCallResult(content=f"echo: {arguments.get('value')}", is_error=False)


class NativeStreamingLLM(LLMClient):
    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        raise AssertionError("complete() should not be used in native stream test")

    async def stream(self, messages: list[Message], tools=None, system=None,
                     max_tokens=None, thinking_level=None) -> AsyncIterator[StreamChunk]:
        yield TextDelta("Hel")
        yield TextDelta("lo")
        yield StreamEnd(AssistantMessage(
            content=[TextBlock("Hello")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        ))


class CompleteOnlyLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        self.calls += 1
        return AssistantMessage(
            content=[TextBlock("fallback text")],
            stop_reason="end_turn",
            usage=Usage(total_tokens=3),
        )


class ToolStreamingLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        raise AssertionError("complete() should not be used in stream mode")

    async def stream(self, messages: list[Message], tools=None, system=None,
                     max_tokens=None, thinking_level=None) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        if self.calls == 1:
            yield StreamEnd(AssistantMessage(
                content=[ToolUseBlock(id="call_1", name="srv__echo", input={"value": "hi"})],
                stop_reason="tool_use",
                usage=Usage(total_tokens=2),
            ))
            return
        yield TextDelta("done")
        yield StreamEnd(AssistantMessage(
            content=[TextBlock("done")],
            stop_reason="end_turn",
            usage=Usage(total_tokens=2),
        ))


class ErrorAfterDeltaLLM(LLMClient):
    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        raise AssertionError("complete() should not be used in stream mode")

    async def stream(self, messages: list[Message], tools=None, system=None,
                     max_tokens=None, thinking_level=None) -> AsyncIterator[StreamChunk]:
        yield TextDelta("partial")
        raise RuntimeError("stream broke")


async def collect(llm: LLMClient, mcp: FakeMCP | None = None) -> list[Any]:
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    events: list[Any] = []
    async for event in run_agent(
        session=session,
        llm=llm,
        mcp=mcp or FakeMCP(),
        store=store,
        stream=True,
        max_retries=1,
        retry_base_delay=0.0,
    ):
        events.append(event)
    return events


def text_events(events: list[Any]) -> list[str]:
    return [event.text for event in events if isinstance(event, TextEvent)]


def done_reason(events: list[Any]) -> str | None:
    done = [event for event in events if isinstance(event, DoneEvent)]
    return done[-1].reason if done else None


def check(cond: bool, msg: str, failures: list[str]) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"    [{status}] {msg}")
    if not cond:
        failures.append(msg)


def run_stripper(pieces: list[str]) -> tuple[str, str | None]:
    stripper = _ReasoningStreamStripper()
    visible = "".join(stripper.feed(piece) for piece in pieces)
    visible += stripper.finish()
    return visible, stripper.reasoning


async def main() -> None:
    failures: list[str] = []

    print("--- native streaming emits incremental text ---")
    events = await collect(NativeStreamingLLM())
    check(text_events(events) == ["Hel", "lo"], "multiple TextEvents emitted without final duplicate", failures)
    check("".join(text_events(events)) == "Hello", "reassembled text matches final answer", failures)
    check(done_reason(events) == "end_turn", "native stream completes normally", failures)

    print("--- ABC fallback streams coarse text ---")
    fallback = CompleteOnlyLLM()
    events = await collect(fallback)
    check(fallback.calls == 1, "fallback called complete() once", failures)
    check(text_events(events) == ["fallback text"], "fallback emitted one coarse TextEvent", failures)
    check(done_reason(events) == "end_turn", "fallback completes normally", failures)

    print("--- streamed tool call reaches existing dispatch ---")
    mcp = FakeMCP()
    events = await collect(ToolStreamingLLM(), mcp)
    check(mcp.call_count == 1, "tool dispatched once", failures)
    check(any(isinstance(event, ToolCallEvent) for event in events), "ToolCallEvent emitted", failures)
    check(text_events(events) == ["done"], "follow-up streamed final text emitted once", failures)
    check(done_reason(events) == "end_turn", "tool stream completes normally", failures)

    print("--- error after first delta surfaces partials ---")
    events = await collect(ErrorAfterDeltaLLM())
    check(text_events(events) == ["partial"], "partial text survived stream error", failures)
    check(any(isinstance(event, ErrorEvent) for event in events), "ErrorEvent emitted", failures)
    check(done_reason(events) == "llm_error", "stream error terminates as llm_error", failures)

    print("--- reasoning stripper split-boundary cases ---")
    cases = [
        (["<think>secret</think>Hello"], "Hello", "secret", True),
        (["<thi", "nk>secret</think>Hello"], "Hello", "secret", True),
        (["<think>sec", "ret</thi", "nk>Hello"], "Hello", "secret", True),
        (["Hello <think>not leading</think>"], "Hello <think>not leading</think>", None, False),
        (["   <think>secret</think>  Hello"], "Hello", "secret", True),
        (["plain"], "plain", None, True),
        (["<think>unfinished"], "<think>unfinished", None, False),
    ]
    for pieces, expected_visible, expected_reasoning, assert_no_tags in cases:
        visible, reasoning = run_stripper(pieces)
        check(visible == expected_visible, f"visible output for {pieces!r}", failures)
        check(reasoning == expected_reasoning, f"reasoning output for {pieces!r}", failures)
        if assert_no_tags:
            check("<think>" not in visible and "</think>" not in visible, f"no leading tag leak for {pieces!r}", failures)

    if failures:
        print(f"\nSTREAMING SMOKE TEST FAILED: {len(failures)} check(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("\nstreaming smoke test complete: all checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
