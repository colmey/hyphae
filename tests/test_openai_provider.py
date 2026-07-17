"""Hermetic OpenAI request and response translation contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import openai
import pytest

from llm.providers.openai import OpenAILLMClient, _merge_extra_body
from llm.schemas import Message, ModelProfile, StreamEnd, TextBlock, ToolUseBlock, Usage


def _client(
    *, compatible: bool = False, profile: ModelProfile | None = None
) -> OpenAILLMClient:
    client = object.__new__(OpenAILLMClient)
    client._model = "test-model"
    client._default_max_tokens = 123
    client._profile = profile or ModelProfile.default()
    client._compatible_endpoint = compatible
    client._warned_inert_thinking = False
    return client


def _response(
    finish_reason: Any = "stop",
    *,
    content: str | None = "answer",
    refusal: str | None = None,
    tool_calls: list[Any] | None = None,
    function_call: Any = None,
    choices: bool = True,
    usage: Any = None,
) -> Any:
    choice_items = []
    if choices:
        choice_items.append(
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    refusal=refusal,
                    tool_calls=tool_calls,
                    function_call=function_call,
                ),
            )
        )
    return SimpleNamespace(choices=choice_items, model="provider-model", usage=usage)


def test_constructor_records_real_and_compatible_endpoint_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, Any]] = []

    def fake_async_openai(**kwargs: Any) -> Any:
        constructed.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(openai, "AsyncOpenAI", fake_async_openai)

    real = OpenAILLMClient("key", "model", 10)
    empty_base_url = OpenAILLMClient("key", "model", 10, base_url="")
    compatible = OpenAILLMClient("key", "model", 10, base_url="http://local/v1")

    assert real._compatible_endpoint is False
    assert empty_base_url._compatible_endpoint is False
    assert compatible._compatible_endpoint is True
    assert [call["base_url"] for call in constructed] == [
        None,
        None,
        "http://local/v1",
    ]


@pytest.mark.parametrize(
    ("finish_reason", "content", "expected"),
    [
        ("stop", "answer", "end_turn"),
        ("stop", None, "empty"),
        ("length", "partial", "max_tokens"),
        ("content_filter", None, "content_filter"),
        ("tool_calls", "orphan", "provider_error"),
        ("function_call", "orphan", "provider_error"),
        (None, "answer", "provider_error"),
        ("future_reason", "answer", "provider_error"),
    ],
)
def test_buffered_finish_reason_translation(
    finish_reason: str | None, content: str | None, expected: str
) -> None:
    message = _client()._from_openai_response(_response(finish_reason, content=content))

    assert message.stop_reason == expected
    assert message.raw_stop_reason == finish_reason


def test_empty_choices_are_retryable_empty() -> None:
    message = _client()._from_openai_response(_response(choices=False))

    assert message.stop_reason == "empty"
    assert message.raw_stop_reason is None


def test_refusal_is_visible_and_canonical_even_with_stop() -> None:
    message = _client()._from_openai_response(
        _response("stop", content=None, refusal="I cannot help with that.")
    )

    assert message.stop_reason == "refusal"
    assert message.raw_stop_reason == "stop"
    assert message.text_blocks() == [TextBlock("I cannot help with that.")]


def test_unterminated_leading_reasoning_is_never_visible() -> None:
    message = _client()._from_openai_response(
        _response("stop", content="<think>private unfinished reasoning")
    )

    assert message.stop_reason == "empty"
    assert message.reasoning == "private unfinished reasoning"
    assert message.text_blocks() == []


@pytest.mark.parametrize("finish_reason", [None, "stop", "content_filter", "unknown"])
def test_actual_tool_content_overrides_finish_reason(finish_reason: str | None) -> None:
    tool_call = SimpleNamespace(
        id="call_provider",
        function=SimpleNamespace(name="srv__tool", arguments='{"q":"x"}'),
    )
    message = _client()._from_openai_response(
        _response(finish_reason, content=None, tool_calls=[tool_call])
    )

    assert message.stop_reason == "tool_use"
    assert message.raw_stop_reason == finish_reason
    assert message.tool_uses() == [
        ToolUseBlock(id="call_provider", name="srv__tool", input={"q": "x"})
    ]


def test_legacy_function_call_is_normalized_with_synthetic_id() -> None:
    legacy = SimpleNamespace(name="srv__legacy", arguments='{"value":1}')
    message = _client()._from_openai_response(
        _response("function_call", content=None, function_call=legacy)
    )

    assert message.stop_reason == "tool_use"
    assert message.raw_stop_reason == "function_call"
    tool_use = message.tool_uses()[0]
    assert tool_use.id.startswith("call_")
    assert tool_use.name == "srv__legacy"
    assert tool_use.input == {"value": 1}


def test_modern_tool_calls_take_precedence_over_duplicated_legacy_shape() -> None:
    modern = SimpleNamespace(
        id="call_modern",
        function=SimpleNamespace(name="srv__modern", arguments="{}"),
    )
    legacy = SimpleNamespace(name="srv__legacy", arguments="{}")

    message = _client()._from_openai_response(
        _response(
            "tool_calls",
            content=None,
            tool_calls=[modern],
            function_call=legacy,
        )
    )

    assert message.tool_uses() == [
        ToolUseBlock(id="call_modern", name="srv__modern", input={})
    ]


def test_buffered_missing_tool_id_is_nonempty_and_replayed_unchanged() -> None:
    tool_call = SimpleNamespace(
        id="", function=SimpleNamespace(name="srv__tool", arguments="{}")
    )
    client = _client()
    message = client._from_openai_response(
        _response("tool_calls", content=None, tool_calls=[tool_call])
    )
    tool_use = message.tool_uses()[0]

    assert tool_use.id.startswith("call_")
    replay = client._to_openai_messages([message.to_message()], None)
    assert replay[0]["tool_calls"][0]["id"] == tool_use.id


def test_usage_coercion_degrades_malformed_values_to_zero() -> None:
    usage = _client()._usage_from_raw(
        SimpleNamespace(
            prompt_tokens="11",
            completion_tokens="not-a-number",
            total_tokens=float("inf"),
        )
    )

    assert usage == Usage(input_tokens=11, output_tokens=0, total_tokens=0)


def test_real_openai_request_uses_supported_shape() -> None:
    profile = ModelProfile(temperature=0.2, top_p=0.8, top_k=40)
    request = _client(profile=profile)._build_request(
        messages=[Message.user("hello")],
        tools=None,
        system=None,
        max_tokens=77,
        response_schema=None,
        thinking_level=None,
    )

    assert request["max_completion_tokens"] == 77
    assert "max_tokens" not in request
    assert "top_k" not in request
    assert "extra_body" not in request
    assert request["temperature"] == 0.2
    assert request["top_p"] == 0.8


def test_compatible_request_uses_extension_shape() -> None:
    profile = ModelProfile(temperature=0.2, top_p=0.8, top_k=40)
    request = _client(compatible=True, profile=profile)._build_request(
        messages=[Message.user("hello")],
        tools=None,
        system=None,
        max_tokens=None,
        response_schema=None,
        thinking_level=None,
    )

    assert request["max_tokens"] == 123
    assert "max_completion_tokens" not in request
    assert "top_k" not in request
    assert request["extra_body"] == {"top_k": 40}


def test_extra_body_merge_preserves_existing_extensions() -> None:
    request = {"extra_body": {"vendor_flag": True, "top_k": 99}}

    _merge_extra_body(request, {"top_k": 40, "new_flag": "kept"})

    assert request["extra_body"] == {
        "vendor_flag": True,
        "top_k": 99,
        "new_flag": "kept",
    }


class _StreamingCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    async def create(self, **_request: Any):
        async def stream():
            for chunk in self._chunks:
                yield chunk

        return stream()


def _stream_client(chunks: list[Any]) -> OpenAILLMClient:
    client = _client(compatible=True)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_StreamingCompletions(chunks))
    )
    return client


@pytest.mark.anyio
async def test_streamed_refusal_is_visible_and_canonical() -> None:
    delta = SimpleNamespace(
        content=None,
        refusal="I cannot help with that.",
        function_call=None,
        tool_calls=None,
    )
    chunks = [
        SimpleNamespace(
            model="provider-model",
            usage=None,
            choices=[SimpleNamespace(finish_reason="stop", delta=delta)],
        )
    ]

    emitted = [chunk async for chunk in _stream_client(chunks).stream([])]
    end = next(chunk for chunk in emitted if isinstance(chunk, StreamEnd))

    assert end.message.stop_reason == "refusal"
    assert end.message.raw_stop_reason == "stop"
    assert end.message.text_blocks() == [TextBlock("I cannot help with that.")]


@pytest.mark.anyio
async def test_stream_with_no_choices_is_retryable_empty() -> None:
    chunks = [
        SimpleNamespace(
            model="provider-model",
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=0,
                total_tokens=1,
            ),
            choices=[],
        )
    ]

    emitted = [chunk async for chunk in _stream_client(chunks).stream([])]
    end = next(chunk for chunk in emitted if isinstance(chunk, StreamEnd))

    assert end.message.stop_reason == "empty"
    assert end.message.raw_stop_reason is None
    assert end.message.content == []


@pytest.mark.anyio
async def test_streamed_legacy_function_call_is_normalized() -> None:
    delta = SimpleNamespace(
        content=None,
        refusal=None,
        function_call=SimpleNamespace(name="srv__legacy", arguments='{"q":"x"}'),
        tool_calls=None,
    )
    chunks = [
        SimpleNamespace(
            model="provider-model",
            usage=None,
            choices=[SimpleNamespace(finish_reason="function_call", delta=delta)],
        )
    ]

    emitted = [chunk async for chunk in _stream_client(chunks).stream([])]
    end = next(chunk for chunk in emitted if isinstance(chunk, StreamEnd))
    tool_use = end.message.tool_uses()[0]

    assert end.message.stop_reason == "tool_use"
    assert end.message.raw_stop_reason == "function_call"
    assert tool_use.id.startswith("call_")
    assert tool_use.name == "srv__legacy"
    assert tool_use.input == {"q": "x"}


@pytest.mark.anyio
async def test_streamed_missing_tool_id_is_minted_once_and_stable() -> None:
    first_delta = SimpleNamespace(
        content=None,
        refusal=None,
        function_call=None,
        tool_calls=[
            SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name="srv__tool", arguments='{"q":'),
            )
        ],
    )
    second_delta = SimpleNamespace(
        content=None,
        refusal=None,
        function_call=None,
        tool_calls=[
            SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name=None, arguments='"x"}'),
            )
        ],
    )
    chunks = [
        SimpleNamespace(
            model="provider-model",
            usage=None,
            choices=[SimpleNamespace(finish_reason=None, delta=first_delta)],
        ),
        SimpleNamespace(
            model="provider-model",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            choices=[SimpleNamespace(finish_reason="tool_calls", delta=second_delta)],
        ),
    ]
    client = _stream_client(chunks)

    emitted = [chunk async for chunk in client.stream([Message.user("go")])]
    end = next(chunk for chunk in emitted if isinstance(chunk, StreamEnd))
    tool_use = end.message.tool_uses()[0]

    assert tool_use.id.startswith("call_")
    assert tool_use.input == {"q": "x"}
    assert end.message.stop_reason == "tool_use"
    assert end.message.raw_stop_reason == "tool_calls"
    replay = client._to_openai_messages([end.message.to_message()], None)
    assert replay[0]["tool_calls"][0]["id"] == tool_use.id
