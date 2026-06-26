"""
Smoke test for the native event-stream renderer (POST /chat/stream).

Hermetic by design: it verifies that the loop's typed events are forwarded live
over SSE, not a live model. The app is driven in-process via httpx + ASGITransport
with app.state populated by hand (scripted fake LLM + a one-tool fake MCP,
orchestration off), so it is deterministic and needs no backend.

This is the renderer that makes "what is the agent doing" visible: tool calls and
results show up as their own events the instant the loop reaches them, then the
answer text follows. (Text is not token-streamed — that machinery was removed to
keep the core simple; the loop emits one text event per turn.)

Scenarios:
  1. Tool-using turn  -> text? + tool_call + tool_result + done events, in order.
  2. Plain turn       -> text + done, no tool events.
  3. Empty body       -> 400.

Run from the project root:
    ./runscript.sh tests/smoke_test_event_stream.py
"""

from __future__ import annotations

from bootstrap import load_secrets
load_secrets()

import asyncio
import json
from typing import Any

import httpx
from httpx import ASGITransport

from agent import InMemorySessionStore, SessionGuard
from harness_config import get_settings
from llm.client import LLMClient
from llm.schemas import AssistantMessage, TextBlock, ToolUseBlock, Usage
from main import app


# --- scripted fakes --------------------------------------------------------

class FakeToolLLM(LLMClient):
    """Turn 1 returns a tool call; turn 2 returns the final text answer."""

    def __init__(self) -> None:
        self._n = 0

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        self._n += 1
        if self._n == 1:
            return AssistantMessage(
                content=[ToolUseBlock(id="c1", name="echo", input={"value": "hi"})],
                stop_reason="end_turn", model="fake", usage=Usage(total_tokens=5),
            )
        return AssistantMessage(
            content=[TextBlock(text="The tool said hi.")],
            stop_reason="end_turn", model="fake", usage=Usage(total_tokens=7),
        )


class FakePlainLLM(LLMClient):
    """One text block, no tools -> a clean single-iteration end_turn."""

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        return AssistantMessage(
            content=[TextBlock(text="Just an answer.")],
            stop_reason="end_turn", model="fake", usage=Usage(total_tokens=4),
        )


class _ToolResult:
    def __init__(self, content: str, is_error: bool = False) -> None:
        self.content = content
        self.is_error = is_error


class FakeMCP:
    """One 'echo' tool that returns its `value` argument."""
    connected_servers: list[str] = []

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [{
            "name": "echo",
            "description": "echo back the value",
            "input_schema": {"type": "object", "properties": {"value": {"type": "string"}}},
        }]

    def list_tools(self) -> list[Any]:
        return ["echo"]

    async def call_tool(self, name: str, args: dict) -> _ToolResult:
        return _ToolResult(content=f"echo: {args.get('value')}", is_error=False)


def _wire_state(llm: LLMClient) -> None:
    settings = get_settings()
    app.state.settings = settings
    app.state.llm = llm
    app.state.mcp = FakeMCP()
    app.state.store = InMemorySessionStore(
        ttl_seconds=settings.session_ttl_seconds,
        max_count=settings.session_max_count,
    )
    app.state.guard = SessionGuard()
    app.state.registry = None
    app.state.orchestrator = None
    app.state.tracer = None


def _parse_events(body: str) -> list[dict]:
    """Return the JSON payload of every SSE `data:` line."""
    out = []
    for line in body.splitlines():
        if line.startswith("data:"):
            out.append(json.loads(line[len("data:"):].strip()))
    return out


async def _stream_events(client: httpx.AsyncClient, prompt: str) -> tuple[list[dict], httpx.Response]:
    async with client.stream("POST", "/chat/stream", content=prompt) as resp:
        assert resp.status_code == 200, resp.status_code
        raw = "".join([chunk async for chunk in resp.aiter_text()])
        headers = resp.headers
    events = _parse_events(raw)
    return events, headers


async def main() -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://harness.local", timeout=30.0) as client:

        # ----- Scenario 1: tool-using turn -----
        print("=" * 72)
        print("Scenario 1: POST /chat/stream — tool-using turn emits live events")
        print("=" * 72)
        _wire_state(FakeToolLLM())
        events, headers = await _stream_events(client, "echo hi please")
        for e in events:
            print(f"  {e.get('type'):12} {({k: v for k, v in e.items() if k not in ('run_id','step','ts','type')})}")
        types = [e["type"] for e in events]
        assert "tool_call" in types, f"no tool_call event: {types}"
        assert "tool_result" in types, f"no tool_result event: {types}"
        assert types.index("tool_call") < types.index("tool_result"), "tool_call must precede tool_result"
        tool_call = next(e for e in events if e["type"] == "tool_call")
        assert tool_call["name"] == "echo", tool_call
        tool_result = next(e for e in events if e["type"] == "tool_result")
        assert tool_result["content"] == "echo: hi", tool_result
        text = "".join(e["text"] for e in events if e["type"] == "text")
        assert text == "The tool said hi.", f"unexpected text: {text!r}"
        done = [e for e in events if e["type"] == "done"]
        assert len(done) == 1 and done[0]["reason"] == "end_turn", done
        assert types[-1] == "done", f"stream must end on done: {types}"
        assert headers.get("x-session-id"), "missing X-Session-Id header"
        print("  -> tool_call + tool_result + text + done, in order. OK\n")

        # ----- Scenario 2: plain turn -----
        print("=" * 72)
        print("Scenario 2: POST /chat/stream — plain turn (no tools)")
        print("=" * 72)
        _wire_state(FakePlainLLM())
        events, _ = await _stream_events(client, "say something")
        types = [e["type"] for e in events]
        print(f"  event types: {types}")
        assert "tool_call" not in types and "tool_result" not in types, types
        text = "".join(e["text"] for e in events if e["type"] == "text")
        assert text == "Just an answer.", f"unexpected text: {text!r}"
        assert types[-1] == "done", types
        print("  -> text + done, no tool events. OK\n")

        # ----- Scenario 3: empty body -----
        print("=" * 72)
        print("Scenario 3: POST /chat/stream — empty body -> 400")
        print("=" * 72)
        r = await client.post("/chat/stream", content="   ")
        print(f"  status: {r.status_code}")
        assert r.status_code == 400, r.status_code
        print("  -> 400 on empty prompt. OK\n")

        print("event-stream smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
