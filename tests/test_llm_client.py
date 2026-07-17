"""Provider-neutral generation request interface contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import AsyncIterator

import pytest

from llm import GenerationRequest as ExportedGenerationRequest
from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    Message,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
)

pytestmark = pytest.mark.anyio


class _CompleteOnlyClient(LLMClient):
    def __init__(self, response: AssistantMessage) -> None:
        self.response = response
        self.requests: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.requests.append(request)
        return self.response


class _NativeStreamingClient(LLMClient):
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("native stream must not call complete")

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        yield StreamEnd(AssistantMessage(content=[], stop_reason="empty"))


def test_generation_request_is_exported_frozen_slotted_and_shallow() -> None:
    messages = [Message.user("hello")]
    tools = [{"name": "srv__tool", "input_schema": {}}]
    request = GenerationRequest(
        messages=messages,
        tools=tools,
        system="system",
        max_tokens=77,
        response_schema=dict,
        thinking_level="high",
    )

    assert ExportedGenerationRequest is GenerationRequest
    assert request.messages is messages
    assert request.tools is tools
    assert not hasattr(request, "__dict__")
    with pytest.raises(FrozenInstanceError):
        request.system = "changed"  # type: ignore[misc]


async def test_default_stream_forwards_same_request_and_message() -> None:
    message = AssistantMessage(
        content=[TextBlock("first"), TextBlock("second")],
        stop_reason="end_turn",
    )
    client = _CompleteOnlyClient(message)
    request = GenerationRequest(messages=[Message.user("hello")])

    chunks = [chunk async for chunk in client.stream(request)]

    assert client.requests == [request]
    assert client.requests[0] is request
    assert chunks[:2] == [TextDelta("first"), TextDelta("second")]
    assert isinstance(chunks[-1], StreamEnd)
    assert chunks[-1].message is message


async def test_default_stream_rejects_schema_before_complete() -> None:
    client = _CompleteOnlyClient(
        AssistantMessage(content=[], stop_reason="empty")
    )
    request = GenerationRequest(messages=[], response_schema=dict)

    with pytest.raises(ValueError, match="response_schema"):
        _ = [chunk async for chunk in client.stream(request)]

    assert client.requests == []


async def test_native_stream_receives_same_request_value() -> None:
    client = _NativeStreamingClient()
    request = GenerationRequest(
        messages=[Message.user("hello")],
        tools=[],
        system="system",
        max_tokens=12,
        thinking_level="low",
    )

    _ = [chunk async for chunk in client.stream(request)]

    assert client.requests[0] is request
