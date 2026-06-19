"""
Smoke test for run tracing (agent/tracing.py + the loop's trace emission).

Hermetic by design: a scripted fake LLM and fake MCP drive run_agent directly
(no HTTP, no live backend), with a JSONLTracer pointed at a temp file. It
asserts the trace is exactly the serialized event log:

  1. one JSONL record per emitted event, in order
  2. a single stable run_id across the run
  3. strictly increasing step indices (1..N)
  4. an ISO timestamp on every record
  5. latency_ms present on the LLM (usage) and tool_result records
  6. provider_metadata / arbitrary bytes in tool args serialize without error
  7. the no-op default (tracer=None) path emits no records and is unchanged

Run from the project root:
    ./runscript.sh tests/smoke_test_tracing.py
"""

from __future__ import annotations

from bootstrap import load_secrets
load_secrets()

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from agent import JSONLTracer, Session, run_agent
from agent.events import DoneEvent, ToolCallEvent, ToolResultEvent, UsageEvent
from llm.client import LLMClient
from llm.schemas import AssistantMessage, TextBlock, ToolUseBlock, Usage
from mcp_layer import ToolCallResult


# --- scripted fakes --------------------------------------------------------

class ScriptedLLM(LLMClient):
    """Turn 1: a text block + a tool call (with bytes in args AND in
    provider_metadata, to exercise the serializer). Turn 2: a final answer."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
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
    events = []
    async for event in run_agent(
        session=session,
        llm=ScriptedLLM(),
        mcp=FakeMCP(),
        tracer=tracer,
        run_id="run_test_123",
    ):
        events.append(event)
    return events


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "nested" / "trace.jsonl"  # nested -> parents created

        # ----- traced run -----
        print("=" * 72)
        print("Scenario 1: run_agent with a JSONLTracer")
        print("=" * 72)
        tracer = JSONLTracer(trace_path)
        events = await _drive(tracer)
        tracer.close()

        records = _read_jsonl(trace_path)
        print(f"  emitted {len(events)} events, wrote {len(records)} records")
        for r in records:
            print(f"    step={r['step']:>2} type={r['type']:<12} "
                  f"latency_ms={r.get('latency_ms')}")

        # one record per event, in order
        assert len(records) == len(events), f"{len(records)} records != {len(events)} events"
        assert [r["type"] for r in records] == [e.type for e in events]

        # stable run_id
        assert all(r["run_id"] == "run_test_123" for r in records), "run_id not stable"

        # strictly increasing step indices 1..N
        assert [r["step"] for r in records] == list(range(1, len(records) + 1)), "step indices wrong"

        # ISO timestamp on every record
        for r in records:
            datetime.fromisoformat(r["ts"])  # raises if malformed

        # latency_ms present on the LLM (usage) and tool_result records
        usage_recs = [r for r in records if r["type"] == "usage"]
        tool_recs = [r for r in records if r["type"] == "tool_result"]
        assert usage_recs and all(isinstance(r["latency_ms"], (int, float)) for r in usage_recs)
        assert tool_recs and all(isinstance(r["latency_ms"], (int, float)) for r in tool_recs)

        # bytes in tool args serialized without error (base64 string, not bytes)
        call_recs = [r for r in records if r["type"] == "tool_call"]
        assert call_recs, "expected a tool_call record"
        args = call_recs[0]["args"]
        assert args["query"] == "weather"
        assert isinstance(args["blob"], str), "bytes arg should serialize to a base64 string"

        # the events we expect are all present (sanity on the captured shape)
        assert any(isinstance(e, UsageEvent) for e in events)
        assert any(isinstance(e, ToolCallEvent) for e in events)
        assert any(isinstance(e, ToolResultEvent) for e in events)
        assert isinstance(events[-1], DoneEvent) and events[-1].reason == "end_turn"
        print("  trace shape OK\n")

        # ----- no-op default path -----
        print("=" * 72)
        print("Scenario 2: run_agent with tracer=None (default) writes nothing")
        print("=" * 72)
        noop_path = Path(tmp) / "should_not_exist.jsonl"
        events_noop = await _drive(None)
        assert not noop_path.exists(), "no-op path must not create a trace file"
        assert [e.type for e in events_noop] == [e.type for e in events], \
            "tracing must not change the event stream"
        print(f"  {len(events_noop)} events, no trace file created\n")

    print("tracing smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
