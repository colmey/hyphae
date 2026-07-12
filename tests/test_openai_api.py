"""OpenAI-compatible HTTP and SSE wire-format tests."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from llm.client import LLMClient
from llm.schemas import AssistantMessage, StreamChunk, StreamEnd, TextBlock, TextDelta, Usage
from config import load_models_config
from orchestrator import LLMRegistry
from tests._app_support import wired_app


pytestmark = pytest.mark.anyio


class FakeLLM(LLMClient):
    def __init__(self) -> None:
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        self.complete_calls += 1
        return AssistantMessage(
            content=[TextBlock(text="Hello"), TextBlock(text=", world")],
            stop_reason="end_turn",
            model="fake",
            usage=Usage(input_tokens=11, output_tokens=3, total_tokens=14),
        )

    async def stream(self, messages, tools=None, system=None, max_tokens=None,
                     thinking_level=None) -> AsyncIterator[StreamChunk]:
        self.stream_calls += 1
        yield TextDelta(text="Hello")
        yield TextDelta(text=", ")
        yield TextDelta(text="world")
        yield StreamEnd(AssistantMessage(
            content=[TextBlock(text="Hello, world")],
            stop_reason="end_turn",
            model="fake",
            usage=Usage(input_tokens=11, output_tokens=3, total_tokens=14),
        ))


def _registry(settings) -> LLMRegistry:
    return LLMRegistry(load_models_config(settings.models_config_path), settings)


async def test_models_returns_registry_in_openai_list_shape(asgi_client) -> None:
    llm = FakeLLM()
    with wired_app(llm) as (app, settings):
        registry = _registry(settings)
        app.state.registry = registry
        response = await asgi_client(app).get("/v1/models")

    response.raise_for_status()
    body = response.json()
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"]] == registry.model_ids
    assert all(model["object"] == "model" for model in body["data"])


async def test_non_stream_completion_shape_and_usage(asgi_client) -> None:
    llm = FakeLLM()
    with wired_app(llm) as (app, _settings):
        response = await asgi_client(app).post(
            "/v1/chat/completions",
            json={
                "model": "whatever",
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Hi there"},
                ],
            },
        )

    response.raise_for_status()
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "whatever"
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hello, world"}
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
    assert llm.complete_calls == 1


async def test_stream_completion_emits_deltas_finish_and_done(asgi_client, parse_sse) -> None:
    llm = FakeLLM()
    with wired_app(llm) as (app, _settings):
        async with asgi_client(app).stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "temperature": 0.7,
            },
        ) as response:
            assert response.status_code == 200
            payloads = parse_sse("".join([chunk async for chunk in response.aiter_text()]))

    assert payloads[-1] == "[DONE]"
    frames = [payload for payload in payloads if payload != "[DONE]"]
    assert all(frame["object"] == "chat.completion.chunk" for frame in frames)
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
    deltas = [frame["choices"][0]["delta"].get("content", "") for frame in frames]
    assert deltas.count("Hello") == 1
    assert "".join(deltas) == "Hello, world"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert llm.stream_calls == 1


async def test_empty_messages_returns_openai_error(asgi_client) -> None:
    with wired_app(FakeLLM()) as (app, _settings):
        response = await asgi_client(app).post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
