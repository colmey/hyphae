"""Native ``/chat/stream`` event-renderer tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm.client import LLMClient
from llm.schemas import AssistantMessage, TextBlock, ToolUseBlock, Usage
from tests._app_support import wired_app


pytestmark = pytest.mark.anyio


class FakeToolLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages,
        tools=None,
        system=None,
        max_tokens=None,
        response_schema=None,
        thinking_level=None,
    ) -> AssistantMessage:
        self.calls += 1
        if self.calls == 1:
            return AssistantMessage(
                content=[ToolUseBlock(id="c1", name="echo", input={"value": "hi"})],
                stop_reason="end_turn",
                model="fake",
                usage=Usage(total_tokens=5),
            )
        return AssistantMessage(
            content=[TextBlock(text="The tool said hi.")],
            stop_reason="end_turn",
            model="fake",
            usage=Usage(total_tokens=7),
        )


class FakePlainLLM(LLMClient):
    async def complete(
        self,
        messages,
        tools=None,
        system=None,
        max_tokens=None,
        response_schema=None,
        thinking_level=None,
    ) -> AssistantMessage:
        return AssistantMessage(
            content=[TextBlock(text="Just an answer.")],
            stop_reason="end_turn",
            model="fake",
            usage=Usage(total_tokens=4),
        )


class ToolResult:
    def __init__(self, content: str, is_error: bool = False) -> None:
        self.content = content
        self.is_error = is_error


class FakeMCP:
    connected_servers: list[str] = []

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "echo",
                "description": "echo back the value",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            }
        ]

    def list_tools(self) -> list[Any]:
        return ["echo"]

    async def call_tool(self, name: str, args: dict) -> ToolResult:
        return ToolResult(content=f"echo: {args.get('value')}")


def _parse_events(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data:").strip())
        for line in body.splitlines()
        if line.startswith("data:")
    ]


async def _stream_events(client, prompt: str):
    async with client.stream("POST", "/chat/stream", content=prompt) as response:
        assert response.status_code == 200
        raw = "".join([chunk async for chunk in response.aiter_text()])
        headers = response.headers
    return _parse_events(raw), headers


async def test_tool_turn_emits_call_result_text_and_done_in_order(asgi_client) -> None:
    with wired_app(FakeToolLLM(), mcp=FakeMCP()) as (app, _settings):
        events, headers = await _stream_events(asgi_client(app), "echo hi please")

    types = [event["type"] for event in events]
    assert types.index("tool_call") < types.index("tool_result")
    assert next(e for e in events if e["type"] == "tool_call")["name"] == "echo"
    assert (
        next(e for e in events if e["type"] == "tool_result")["content"] == "echo: hi"
    )
    assert (
        "".join(e["text"] for e in events if e["type"] == "text") == "The tool said hi."
    )
    assert [e["reason"] for e in events if e["type"] == "done"] == ["end_turn"]
    assert types[-1] == "done"
    assert headers.get("x-session-id")


async def test_plain_turn_emits_only_text_usage_and_done(asgi_client) -> None:
    with wired_app(FakePlainLLM(), mcp=FakeMCP()) as (app, _settings):
        events, _headers = await _stream_events(asgi_client(app), "say something")

    types = [event["type"] for event in events]
    assert "tool_call" not in types and "tool_result" not in types
    assert (
        "".join(e["text"] for e in events if e["type"] == "text") == "Just an answer."
    )
    assert types[-1] == "done"


async def test_empty_prompt_is_rejected(asgi_client) -> None:
    with wired_app(FakePlainLLM(), mcp=FakeMCP()) as (app, _settings):
        response = await asgi_client(app).post("/chat/stream", content="   ")
    assert response.status_code == 400
