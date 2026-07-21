"""Run-event JSONL tracing tests."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import agent.tracing as tracing_module
from agent import JSONLTracer, RunContext, RunLimits, Session, run_agent
from agent.events import DoneEvent, ToolCallEvent, ToolResultEvent, UsageEvent
from llm.client import GenerationRequest, LLMClient
from llm.schemas import AssistantMessage, TextBlock, ToolUseBlock, Usage
from mcp_layer import ToolCallResult


pytestmark = pytest.mark.anyio


def test_done_event_trace_shape_is_byte_for_byte_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing_module, "_now_iso", lambda: "fixed-timestamp")

    record = tracing_module.event_record(
        DoneEvent(
            reason="end_turn",
            iterations=2,
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
            thinking_tokens=5,
        ),
        run_id="run-fixed",
        step=9,
    )

    assert record == {
        "run_id": "run-fixed",
        "step": 9,
        "ts": "fixed-timestamp",
        "type": "done",
        "reason": "end_turn",
        "iterations": 2,
        "total_tokens": 18,
        "input_tokens": 11,
        "output_tokens": 7,
        "thinking_tokens": 5,
    }


class ScriptedLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        if self.calls == 1:
            return AssistantMessage(
                content=[
                    TextBlock(text="Let me look that up."),
                    ToolUseBlock(
                        id="call_1",
                        name="demo__lookup",
                        input={"query": "weather", "blob": b"\x00\xff\xfe"},
                        provider_metadata={"thought_signature": b"\x01\x02\x03"},
                    ),
                ],
                stop_reason="tool_use",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        return AssistantMessage(
            content=[TextBlock(text="It is sunny.")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=20, output_tokens=4, total_tokens=24),
        )


class FakeMCP:
    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [{"name": "demo__lookup", "description": "look up", "input_schema": {}}]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        return ToolCallResult(content="sunny, 24C", is_error=False)


async def _drive(tracer) -> list:
    session = Session()
    session.append_user("what's the weather?")
    limits = RunLimits()
    context = RunContext.start(
        max_run_seconds=limits.max_run_seconds,
        base_logger=logging.getLogger(__name__),
        tracer=tracer,
        run_id="run_test_123",
    )
    return [
        event
        async for event in run_agent(
            session=session,
            llm=ScriptedLLM(),
            mcp=FakeMCP(),
            limits=limits,
            context=context,
        )
    ]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


async def test_jsonl_trace_is_exact_serialized_event_log(tmp_path: Path) -> None:
    trace_path = tmp_path / "nested" / "trace.jsonl"
    tracer = JSONLTracer(trace_path)
    events = await _drive(tracer)
    tracer.close()
    records = _read_jsonl(trace_path)

    assert len(records) == len(events)
    assert [record["type"] for record in records] == [event.type for event in events]
    assert all(record["run_id"] == "run_test_123" for record in records)
    assert [record["step"] for record in records] == list(range(1, len(records) + 1))
    for record in records:
        datetime.fromisoformat(record["ts"])

    usage_records = [record for record in records if record["type"] == "usage"]
    tool_records = [record for record in records if record["type"] == "tool_result"]
    assert usage_records and all(
        isinstance(r["latency_ms"], (int, float)) for r in usage_records
    )
    assert tool_records and all(
        isinstance(r["latency_ms"], (int, float)) for r in tool_records
    )
    args = next(record["args"] for record in records if record["type"] == "tool_call")
    assert args["query"] == "weather"
    assert isinstance(args["blob"], str)
    assert any(isinstance(event, UsageEvent) for event in events)
    assert any(isinstance(event, ToolCallEvent) for event in events)
    assert any(isinstance(event, ToolResultEvent) for event in events)
    assert isinstance(events[-1], DoneEvent) and events[-1].reason == "end_turn"


async def test_none_tracer_does_not_change_event_stream_or_write(
    tmp_path: Path,
) -> None:
    traced_path = tmp_path / "trace.jsonl"
    tracer = JSONLTracer(traced_path)
    traced_events = await _drive(tracer)
    tracer.close()

    noop_path = tmp_path / "should_not_exist.jsonl"
    noop_events = await _drive(None)

    assert not noop_path.exists()
    assert [event.type for event in noop_events] == [
        event.type for event in traced_events
    ]
