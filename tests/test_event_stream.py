"""Native ``/chat/stream`` event-renderer tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm.client import GenerationRequest, LLMClient
from llm.schemas import AssistantMessage, TextBlock, ToolUseBlock, CompletionUsage
from api.request_body import MAX_REQUEST_BODY_BYTES
from tests._app_support import wired_app


pytestmark = pytest.mark.anyio


class FakeToolLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        if self.calls == 1:
            return AssistantMessage(
                content=[ToolUseBlock(id="c1", name="echo", input={"value": "hi"})],
                stop_reason="end_turn",
                model="fake",
                usage=CompletionUsage(total_tokens=5),
            )
        return AssistantMessage(
            content=[TextBlock(text="The tool said hi.")],
            stop_reason="end_turn",
            model="fake",
            usage=CompletionUsage(total_tokens=7),
        )


class FakePlainLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        return AssistantMessage(
            content=[TextBlock(text="Just an answer.")],
            stop_reason="end_turn",
            model="fake",
            usage=CompletionUsage(total_tokens=4),
        )


class FakeOutcomeLLM(LLMClient):
    def __init__(self, stop_reason: str) -> None:
        self.stop_reason = stop_reason

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        return AssistantMessage(
            content=[
                TextBlock(
                    text="provider-visible",
                    provider_metadata={"thought_signature": b"opaque-signature"},
                )
            ],
            stop_reason=self.stop_reason,
            raw_stop_reason="raw-provider-reason",
            reasoning="private-chain-of-thought",
            model="fake",
            usage=CompletionUsage(total_tokens=4),
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


@pytest.mark.parametrize(
    "stop_reason",
    ["content_filter", "refusal", "provider_error", "incomplete_stream"],
)
async def test_native_routes_expose_abnormal_reason_without_reasoning(
    asgi_client, stop_reason: str
) -> None:
    with wired_app(FakeOutcomeLLM(stop_reason), mcp=FakeMCP()) as (app, _settings):
        client = asgi_client(app)
        buffered = await client.post("/chat", content="say something")
        events, _headers = await _stream_events(client, "say something else")

    assert buffered.status_code == 200
    assert buffered.headers["x-done-reason"] == stop_reason
    assert [event["reason"] for event in events if event["type"] == "done"] == [
        stop_reason
    ]
    assert not any(event["type"] == "reasoning" for event in events)
    rendered = json.dumps(events)
    assert "private-chain-of-thought" not in rendered
    assert "opaque-signature" not in rendered


async def test_empty_prompt_is_rejected(asgi_client) -> None:
    with wired_app(FakePlainLLM(), mcp=FakeMCP()) as (app, _settings):
        response = await asgi_client(app).post("/chat/stream", content="   ")
    assert response.status_code == 400


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
@pytest.mark.parametrize("headers", [{}, {"Content-Type": "TEXT/PLAIN; charset=utf-8"}])
async def test_native_routes_accept_missing_or_parameterized_plain_text(
    asgi_client, path: str, headers: dict[str, str]
) -> None:
    llm = FakePlainLLM()
    with wired_app(llm, mcp=FakeMCP()) as (app, _settings):
        response = await asgi_client(app).post(path, content="hello", headers=headers)

    assert response.status_code == 200
    assert llm.calls == 1


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
@pytest.mark.parametrize(
    ("content", "headers", "status"),
    [
        (b"hello", {"Content-Type": "application/json"}, 415),
        (b"\xff", {}, 400),
        (b" \n\t ", {}, 400),
    ],
)
async def test_native_body_rejections_happen_before_execution(
    asgi_client, path: str, content: bytes, headers: dict[str, str], status: int
) -> None:
    llm = FakePlainLLM()
    with wired_app(llm, mcp=FakeMCP()) as (app, _settings):
        response = await asgi_client(app).post(path, content=content, headers=headers)

    assert response.status_code == status
    assert llm.calls == 0


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
@pytest.mark.parametrize("size, status", [(MAX_REQUEST_BODY_BYTES, 200), (MAX_REQUEST_BODY_BYTES + 1, 413)])
async def test_native_routes_enforce_the_body_limit_before_execution(
    asgi_client, path: str, size: int, status: int
) -> None:
    llm = FakePlainLLM()
    content = b" " * (size - 2) + b"ok"
    with wired_app(llm, mcp=FakeMCP()) as (app, _settings):
        response = await asgi_client(app).post(path, content=content)

    assert response.status_code == status
    assert llm.calls == (1 if status == 200 else 0)
