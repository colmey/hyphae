"""Hermetic OpenAI adapter contracts by package owner."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
import inspect
import logging
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from llm.client import GenerationRequest, build_llm_client, supported_providers
from llm.providers.openai_compatible import OpenAICompatibleLLMClient
from llm.providers.openai_compatible.client import OpenAICompatibleClientConfig
from llm.providers.openai_compatible.codec import (
    build_request,
    merge_extra_body,
    messages_to_openai,
    build_response_format,
    response_to_message,
    usage_from_raw,
)
from llm.providers.openai_compatible.stream import (
    _ReasoningStreamStripper,
    decode_stream,
)
from llm.schemas import (
    Message,
    ModelProfile,
    ReasoningDelta,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    CompletionUsage,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(
    *,
    compatible: bool = False,
    profile: ModelProfile | None = None,
) -> OpenAICompatibleClientConfig:
    return OpenAICompatibleClientConfig(
        model="test-model",
        default_max_tokens=123,
        profile=profile or ModelProfile.default(),
        compatible_endpoint=compatible,
    )


def _client(
    *,
    compatible: bool = False,
    profile: ModelProfile | None = None,
) -> OpenAICompatibleLLMClient:
    client = object.__new__(OpenAICompatibleLLMClient)
    client._config = _config(compatible=compatible, profile=profile)
    client._warned_inert_thinking = False
    client._closed = False
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
    model: str | None = "provider-model",
    reasoning_content: str | None = None,
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
                    reasoning_content=reasoning_content,
                ),
            )
        )
    return SimpleNamespace(choices=choice_items, model=model, usage=usage)


def _message(response: Any) -> Any:
    return response_to_message(response, default_model="test-model")


def test_public_imports_remain_sdk_lazy() -> None:
    commands = [
        (
            "import sys; import llm.client; "
            "assert 'openai' not in sys.modules; "
            "assert 'llm.providers.openai_compatible' not in sys.modules"
        ),
        (
            "import sys; from llm.providers.openai_compatible import "
            "OpenAICompatibleLLMClient; "
            "assert OpenAICompatibleLLMClient.__name__ == "
            "'OpenAICompatibleLLMClient'; "
            "assert 'openai' not in sys.modules"
        ),
    ]
    for command in commands:
        subprocess.run(
            [sys.executable, "-c", command],
            check=True,
            cwd=ROOT,
        )


@pytest.mark.parametrize("provider", ["openai_compatible", "openai"])
def test_registry_accepts_canonical_provider_and_compatibility_alias(
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    monkeypatch.setattr(
        openai,
        "AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(construction=kwargs),
    )
    requested_keys: list[str] = []

    class FactorySettings:
        llm = SimpleNamespace(
            provider=provider,
            model="test-model",
            max_tokens=123,
        )
        openai_compat_base_url = "http://local.example/v1"

        def api_key_for_provider(self, requested_provider: str) -> str:
            requested_keys.append(requested_provider)
            return "test-key"

    client = build_llm_client(FactorySettings())

    assert {"openai_compatible", "openai"} <= supported_providers()
    assert isinstance(client, OpenAICompatibleLLMClient)
    assert requested_keys == ["openai_compatible"]
    assert client._client.construction == {
        "api_key": "test-key",
        "base_url": "http://local.example/v1",
    }


def test_constructor_normalizes_real_and_compatible_endpoint_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    constructed: list[dict[str, Any]] = []

    def fake_async_openai(**kwargs: Any) -> Any:
        constructed.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(openai, "AsyncOpenAI", fake_async_openai)

    signature = inspect.signature(OpenAICompatibleLLMClient)
    assert list(signature.parameters) == [
        "api_key",
        "model",
        "default_max_tokens",
        "base_url",
        "profile",
    ]
    assert signature.parameters["base_url"].default is None
    assert signature.parameters["profile"].default is None

    real = OpenAICompatibleLLMClient("key", "model", 10)
    empty_base_url = OpenAICompatibleLLMClient(
        "key", "model", 10, base_url=""
    )
    profile = ModelProfile(temperature=0.2)
    compatible = OpenAICompatibleLLMClient(
        "key",
        "model",
        10,
        base_url="http://local/v1",
        profile=profile,
    )

    assert real._config == OpenAICompatibleClientConfig(
        model="model",
        default_max_tokens=10,
        profile=ModelProfile.default(),
        compatible_endpoint=False,
    )
    assert empty_base_url._config.compatible_endpoint is False
    assert compatible._config.profile is profile
    assert compatible._config.compatible_endpoint is True
    with pytest.raises(FrozenInstanceError):
        setattr(compatible._config, "model", "changed")
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
    finish_reason: str | None,
    content: str | None,
    expected: str,
) -> None:
    message = _message(_response(finish_reason, content=content))

    assert message.stop_reason == expected
    assert message.raw_stop_reason == finish_reason


def test_empty_choices_are_retryable_empty_with_usage_and_default_model() -> None:
    message = _message(
        _response(
            choices=False,
            model="ignored-without-choices",
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
            ),
        )
    )

    assert message.stop_reason == "empty"
    assert message.raw_stop_reason is None
    assert message.model == "test-model"
    assert message.usage == CompletionUsage(input_tokens=1, output_tokens=2, total_tokens=3)


def test_refusal_is_visible_and_canonical_even_with_stop() -> None:
    message = _message(
        _response("stop", content=None, refusal="I cannot help with that.")
    )

    assert message.stop_reason == "refusal"
    assert message.raw_stop_reason == "stop"
    assert message.text_blocks() == [TextBlock("I cannot help with that.")]


def test_refusal_does_not_duplicate_identical_visible_content() -> None:
    message = _message(
        _response("stop", content="not allowed", refusal="not allowed")
    )

    assert message.stop_reason == "refusal"
    assert message.text_blocks() == [TextBlock("not allowed")]


def test_unterminated_leading_reasoning_is_never_visible() -> None:
    message = _message(
        _response("stop", content="<think>private unfinished reasoning")
    )

    assert message.stop_reason == "empty"
    assert message.reasoning == "private unfinished reasoning"
    assert message.text_blocks() == []


def test_buffered_structured_reasoning_is_normalized_without_wire_metadata() -> None:
    message = _message(
        _response(
            content="visible answer",
            reasoning_content="structured thought",
        )
    )

    assert message.reasoning == "structured thought"
    assert message.text_blocks() == [TextBlock("visible answer")]


@pytest.mark.parametrize("finish_reason", [None, "stop", "content_filter", "unknown"])
def test_actual_tool_content_overrides_finish_reason(
    finish_reason: str | None,
) -> None:
    tool_call = SimpleNamespace(
        id="call_provider",
        function=SimpleNamespace(name="srv__tool", arguments='{"q":"x"}'),
    )
    message = _message(
        _response(finish_reason, content=None, tool_calls=[tool_call])
    )

    assert message.stop_reason == "tool_use"
    assert message.raw_stop_reason == finish_reason
    assert message.tool_uses() == [
        ToolUseBlock(id="call_provider", name="srv__tool", input={"q": "x"})
    ]


def test_legacy_function_call_is_normalized_with_synthetic_id() -> None:
    legacy = SimpleNamespace(name="srv__legacy", arguments='{"value":1}')
    message = _message(
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

    message = _message(
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


def test_missing_tool_id_and_malformed_arguments_remain_replayable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("llm.providers.openai_compatible")
    caplog.set_level(logging.WARNING, logger=logger.name)
    tool_call = SimpleNamespace(
        id="",
        function=SimpleNamespace(name="srv__tool", arguments='{"broken":'),
    )
    message = response_to_message(
        _response("tool_calls", content=None, tool_calls=[tool_call]),
        default_model="test-model",
        logger=logger,
    )
    tool_use = message.tool_uses()[0]

    assert tool_use.id.startswith("call_")
    assert tool_use.input == {}
    assert tool_use.parse_error is not None
    replay = messages_to_openai([message.to_message()], None)
    assert replay[0]["tool_calls"][0]["id"] == tool_use.id
    assert "could not parse tool arguments" in caplog.text


def test_message_and_tool_result_replay() -> None:
    messages = [
        Message.user("question"),
        Message.assistant(
            [
                TextBlock("working"),
                ToolUseBlock(id="call_1", name="srv__tool", input={"q": "x"}),
            ]
        ),
        Message.tool_results(
            [
                ToolResultBlock(
                    tool_use_id="call_1",
                    name="srv__tool",
                    content="result",
                    is_error=False,
                )
            ]
        ),
    ]

    replay = messages_to_openai(messages, "system")

    assert replay == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "working",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "srv__tool",
                        "arguments": '{"q": "x"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "result",
        },
    ]


def test_usage_coercion_degrades_malformed_values_to_zero() -> None:
    usage = usage_from_raw(
        SimpleNamespace(
            prompt_tokens="11",
            completion_tokens="not-a-number",
            total_tokens=float("inf"),
        )
    )

    assert usage == CompletionUsage(input_tokens=11, output_tokens=0, total_tokens=0)


def test_real_openai_request_uses_supported_shape() -> None:
    profile = ModelProfile(temperature=0.2, top_p=0.8, top_k=40)
    built = build_request(
        GenerationRequest(
            messages=[Message.user("hello")],
            max_tokens=77,
        ),
        _config(profile=profile),
    )
    request = built.sdk_kwargs

    assert request["max_completion_tokens"] == 77
    assert "max_tokens" not in request
    assert "top_k" not in request
    assert "extra_body" not in request
    assert request["temperature"] == 0.2
    assert request["top_p"] == 0.8


def test_compatible_request_uses_extensions_tools_and_thinking_hint() -> None:
    profile = ModelProfile(
        thinking="hint-param",
        temperature=0.2,
        top_p=0.8,
        top_k=40,
    )
    built = build_request(
        GenerationRequest(
            messages=[Message.user("hello")],
            tools=[
                {
                    "name": "srv__tool",
                    "description": "tool",
                    "input_schema": {"type": "object"},
                }
            ],
            thinking_level="high",
        ),
        _config(compatible=True, profile=profile),
    )
    request = built.sdk_kwargs

    assert request["max_tokens"] == 123
    assert "max_completion_tokens" not in request
    assert "top_k" not in request
    assert request["extra_body"] == {"top_k": 40}
    assert request["tool_choice"] == "auto"
    assert request["tools"][0]["function"]["name"] == "srv__tool"
    assert request["reasoning_effort"] == "high"


def test_structured_output_removes_tools_and_reports_client_signal() -> None:
    class Structured:
        @classmethod
        def model_json_schema(cls) -> dict[str, Any]:
            return {"type": "object", "properties": {"answer": {"type": "string"}}}

    built = build_request(
        GenerationRequest(
            messages=[],
            tools=[{"name": "tool", "description": "", "input_schema": {}}],
            response_schema=Structured,
        ),
        _config(),
    )

    assert "tools" not in built.sdk_kwargs
    assert "tool_choice" not in built.sdk_kwargs
    assert built.ignored_tools_for_structured_output is True
    assert built.sdk_kwargs["response_format"] == build_response_format(Structured)


def test_extra_body_merge_preserves_existing_extensions() -> None:
    request = {"extra_body": {"vendor_flag": True, "top_k": 99}}

    merge_extra_body(request, {"top_k": 40, "new_flag": "kept"})

    assert request["extra_body"] == {
        "vendor_flag": True,
        "top_k": 99,
        "new_flag": "kept",
    }


class _CompletionCapture:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        self.requests.append(request)
        return self._response


class _OwnedSDKStream:
    def __init__(
        self,
        chunks: list[Any] | None = None,
        *,
        failure: Exception | None = None,
        blocked: bool = False,
        close_failure: Exception | None = None,
    ) -> None:
        self._chunks = list(chunks or [])
        self._failure = failure
        self._blocked = blocked
        self._blocker = asyncio.Event()
        self.entered = asyncio.Event()
        self._index = 0
        self.close_calls = 0
        self.close_failure = close_failure

    def __aiter__(self) -> _OwnedSDKStream:
        return self

    async def __anext__(self) -> Any:
        self.entered.set()
        if self._blocked:
            await self._blocker.wait()
        if self._index < len(self._chunks):
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk
        if self._failure is not None:
            raise self._failure
        raise StopAsyncIteration

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure


class _OwnedStreamingCompletions:
    def __init__(self, stream: _OwnedSDKStream) -> None:
        self.stream = stream
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> _OwnedSDKStream:
        self.requests.append(request)
        return self.stream


def _owned_stream_client(
    stream: _OwnedSDKStream,
) -> tuple[OpenAICompatibleLLMClient, _OwnedStreamingCompletions]:
    completions = _OwnedStreamingCompletions(stream)
    client = _client(compatible=True)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client, completions


def _delta(
    *,
    content: str | None = None,
    refusal: str | None = None,
    tool_calls: list[Any] | None = None,
    function_call: Any = None,
    reasoning_content: str | None = None,
) -> Any:
    return SimpleNamespace(
        content=content,
        refusal=refusal,
        function_call=function_call,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )


def _sdk_chunk(
    delta: Any | None = None,
    *,
    finish_reason: str | None = None,
    model: str | None = "provider-model",
    usage: Any = None,
    choices: bool = True,
) -> Any:
    choice_items = (
        [SimpleNamespace(finish_reason=finish_reason, delta=delta)]
        if choices
        else []
    )
    return SimpleNamespace(model=model, usage=usage, choices=choice_items)


def _text_chunk(
    text: str = "piece",
    *,
    finish_reason: str | None = "stop",
) -> Any:
    return _sdk_chunk(
        _delta(content=text),
        finish_reason=finish_reason,
    )


def _tool_chunks(argument_parts: list[str]) -> list[Any]:
    chunks: list[Any] = []
    for index, arguments in enumerate(argument_parts):
        chunks.append(
            _sdk_chunk(
                _delta(
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id=None,
                            function=SimpleNamespace(
                                name="srv__lookup" if index == 0 else None,
                                arguments=arguments,
                            ),
                        )
                    ]
                ),
                finish_reason=(
                    "tool_calls" if index == len(argument_parts) - 1 else None
                ),
            )
        )
    return chunks


async def _decode(stream: _OwnedSDKStream) -> list[Any]:
    return [
        chunk
        async for chunk in decode_stream(stream, default_model="test-model")
    ]


def _run_stripper(pieces: list[str]) -> tuple[str, str | None]:
    stripper = _ReasoningStreamStripper()
    visible = "".join(stripper.feed(piece) for piece in pieces)
    visible += stripper.finish()
    return visible, stripper.reasoning


def test_reasoning_stripper_handles_every_tag_split_point() -> None:
    source = " \t<think>private chain</think>  visible answer"
    for split_at in range(len(source) + 1):
        visible, reasoning = _run_stripper(
            [source[:split_at], source[split_at:]]
        )
        assert visible == "visible answer", split_at
        assert reasoning == "private chain", split_at

    visible, reasoning = _run_stripper(list(source))
    assert visible == "visible answer"
    assert reasoning == "private chain"


@pytest.mark.parametrize(
    ("pieces", "expected_visible", "expected_reasoning"),
    [
        (["Hello <think>not leading</think>"], "Hello <think>not leading</think>", None),
        (["plain"], "plain", None),
        (["<think>unfinished"], "", "unfinished"),
        (["   "], "   ", None),
    ],
)
def test_reasoning_stripper_nonleading_and_incomplete_cases(
    pieces: list[str],
    expected_visible: str,
    expected_reasoning: str | None,
) -> None:
    assert _run_stripper(pieces) == (expected_visible, expected_reasoning)


@pytest.mark.anyio
async def test_streamed_tool_json_handles_every_split_point() -> None:
    arguments = '{"query":"fragmented"}'
    fragmentations = [
        [arguments[:split_at], arguments[split_at:]]
        for split_at in range(len(arguments) + 1)
    ]
    fragmentations.append(list(arguments))

    for parts in fragmentations:
        chunks = _tool_chunks(parts)
        emitted = await _decode(_OwnedSDKStream(chunks))
        end = next(chunk for chunk in emitted if isinstance(chunk, StreamEnd))
        tool_use = end.message.tool_uses()[0]

        assert tool_use.id.startswith("call_"), parts
        assert tool_use.input == {"query": "fragmented"}, parts
        assert end.message.stop_reason == "tool_use"


@pytest.mark.anyio
async def test_stream_orders_indexed_tools_and_prefers_modern_over_legacy() -> None:
    modern = [
        SimpleNamespace(
            index=1,
            id="call_1",
            function=SimpleNamespace(name="second", arguments="{}"),
        ),
        SimpleNamespace(
            index=0,
            id="call_0",
            function=SimpleNamespace(name="first", arguments="{}"),
        ),
    ]
    legacy = SimpleNamespace(name="legacy", arguments="{}")
    emitted = await _decode(
        _OwnedSDKStream(
            [
                _sdk_chunk(
                    _delta(tool_calls=modern, function_call=legacy),
                    finish_reason="tool_calls",
                )
            ]
        )
    )
    end = next(chunk for chunk in emitted if isinstance(chunk, StreamEnd))

    assert [tool.name for tool in end.message.tool_uses()] == ["first", "second"]


@pytest.mark.anyio
async def test_streamed_reasoning_refusal_usage_and_model_assembly() -> None:
    usage = SimpleNamespace(
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
    )
    emitted = await _decode(
        _OwnedSDKStream(
            [
                _sdk_chunk(_delta(content="<thi"), model=None),
                _sdk_chunk(_delta(content="nk>secret</think>answer"), model=None),
                _sdk_chunk(
                    _delta(refusal=" refused"),
                    finish_reason="stop",
                    model=None,
                    usage=usage,
                ),
            ]
        )
    )
    end = next(chunk for chunk in emitted if isinstance(chunk, StreamEnd))

    assert [chunk.text for chunk in emitted if isinstance(chunk, TextDelta)] == [
        "answer",
        " refused",
    ]
    assert [
        chunk.text for chunk in emitted if isinstance(chunk, ReasoningDelta)
    ] == ["secret"]
    assert end.message.text_blocks() == [TextBlock("answer refused")]
    assert end.message.reasoning == "secret"
    assert end.message.stop_reason == "refusal"
    assert end.message.raw_stop_reason == "stop"
    assert end.message.model == "test-model"
    assert end.message.usage == CompletionUsage(input_tokens=1, output_tokens=2, total_tokens=3)


@pytest.mark.anyio
async def test_streamed_structured_reasoning_precedes_visible_content() -> None:
    emitted = await _decode(
        _OwnedSDKStream(
            [
                _sdk_chunk(
                    _delta(
                        reasoning_content="first thought",
                        content="visible answer",
                    ),
                    finish_reason="stop",
                )
            ]
        )
    )

    assert isinstance(emitted[0], ReasoningDelta)
    assert emitted[0].text == "first thought"
    assert isinstance(emitted[1], TextDelta)
    assert emitted[1].text == "visible answer"
    end = next(chunk for chunk in emitted if isinstance(chunk, StreamEnd))
    assert end.message.reasoning == "first thought"


@pytest.mark.anyio
async def test_unclosed_think_tag_is_sanitized_into_reasoning_delta() -> None:
    emitted = await _decode(
        _OwnedSDKStream(
            [
                _sdk_chunk(
                    _delta(content="<think>unfinished thought"),
                    finish_reason="length",
                )
            ]
        )
    )

    assert "".join(
        chunk.text for chunk in emitted if isinstance(chunk, ReasoningDelta)
    ) == "unfinished thought"
    assert not any(isinstance(chunk, TextDelta) for chunk in emitted)
    assert "<think>" not in repr(emitted)
    end = next(chunk for chunk in emitted if isinstance(chunk, StreamEnd))
    assert end.message.reasoning == "unfinished thought"


@pytest.mark.anyio
async def test_stream_with_no_choices_is_retryable_empty() -> None:
    emitted = await _decode(
        _OwnedSDKStream(
            [
                _sdk_chunk(
                    choices=False,
                    usage=SimpleNamespace(
                        prompt_tokens=1,
                        completion_tokens=0,
                        total_tokens=1,
                    ),
                )
            ]
        )
    )
    end = next(chunk for chunk in emitted if isinstance(chunk, StreamEnd))

    assert end.message.stop_reason == "empty"
    assert end.message.raw_stop_reason is None
    assert end.message.content == []
    assert end.message.usage.total_tokens == 1


@pytest.mark.anyio
async def test_streamed_legacy_function_call_is_normalized() -> None:
    emitted = await _decode(
        _OwnedSDKStream(
            [
                _sdk_chunk(
                    _delta(
                        function_call=SimpleNamespace(
                            name="srv__legacy",
                            arguments='{"q":"x"}',
                        )
                    ),
                    finish_reason="function_call",
                )
            ]
        )
    )
    end = next(chunk for chunk in emitted if isinstance(chunk, StreamEnd))
    tool_use = end.message.tool_uses()[0]

    assert end.message.stop_reason == "tool_use"
    assert end.message.raw_stop_reason == "function_call"
    assert tool_use.id.startswith("call_")
    assert tool_use.name == "srv__legacy"
    assert tool_use.input == {"q": "x"}


@pytest.mark.anyio
async def test_decoder_closes_sdk_stream_after_normal_completion() -> None:
    sdk_stream = _OwnedSDKStream([_text_chunk()])

    emitted = await _decode(sdk_stream)

    assert any(isinstance(chunk, StreamEnd) for chunk in emitted)
    assert sdk_stream.close_calls == 1


@pytest.mark.anyio
async def test_decoder_closes_sdk_stream_when_consumer_closes_early() -> None:
    sdk_stream = _OwnedSDKStream(
        [_text_chunk("first", finish_reason=None), _text_chunk("second")]
    )
    decoder = decode_stream(sdk_stream, default_model="test-model")

    await anext(decoder)
    await decoder.aclose()

    assert sdk_stream.close_calls == 1


@pytest.mark.anyio
async def test_decoder_closes_sdk_stream_on_timeout() -> None:
    sdk_stream = _OwnedSDKStream(blocked=True)
    decoder = decode_stream(sdk_stream, default_model="test-model")

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await anext(decoder)

    assert sdk_stream.close_calls == 1


@pytest.mark.anyio
async def test_decoder_close_failure_preserves_provider_error_without_logging_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider_error = RuntimeError("provider stream failed")
    secret = "Authorization: Bearer stream-close-secret"
    sdk_stream = _OwnedSDKStream(
        failure=provider_error,
        close_failure=RuntimeError(secret),
    )

    with pytest.raises(RuntimeError, match="provider stream failed") as raised:
        _ = [
            chunk
            async for chunk in decode_stream(
                sdk_stream,
                default_model="test-model",
                logger=logging.getLogger("openai-cleanup-test"),
            )
        ]

    assert raised.value is provider_error
    assert sdk_stream.close_calls == 1
    assert secret not in caplog.text
    assert "_OwnedSDKStream" in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.anyio
async def test_decoder_close_failure_preserves_task_cancellation() -> None:
    sdk_stream = _OwnedSDKStream(
        blocked=True,
        close_failure=RuntimeError("stream close failed"),
    )
    decoder = decode_stream(sdk_stream, default_model="test-model")
    task = asyncio.create_task(anext(decoder))
    await sdk_stream.entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert sdk_stream.close_calls == 1


@pytest.mark.anyio
async def test_client_stream_invokes_sdk_and_delegates_single_close() -> None:
    sdk_stream = _OwnedSDKStream([_text_chunk()])
    client, completions = _owned_stream_client(sdk_stream)

    emitted = [
        chunk
        async for chunk in client.stream(
            GenerationRequest(messages=[Message.user("hello")])
        )
    ]

    assert any(isinstance(chunk, StreamEnd) for chunk in emitted)
    assert completions.requests == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 123,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ]
    assert sdk_stream.close_calls == 1


@pytest.mark.anyio
async def test_client_stream_early_close_does_not_double_close_sdk_stream() -> None:
    sdk_stream = _OwnedSDKStream(
        [_text_chunk("first", finish_reason=None), _text_chunk("second")]
    )
    client, _ = _owned_stream_client(sdk_stream)
    provider_stream = client.stream(GenerationRequest(messages=[]))

    await anext(provider_stream)
    await provider_stream.aclose()

    assert sdk_stream.close_calls == 1


@pytest.mark.anyio
async def test_stream_rejects_response_schema_before_sdk_call() -> None:
    sdk_stream = _OwnedSDKStream()
    client, completions = _owned_stream_client(sdk_stream)

    with pytest.raises(ValueError, match="response_schema"):
        _ = [
            chunk
            async for chunk in client.stream(
                GenerationRequest(messages=[], response_schema=dict)
            )
        ]

    assert completions.requests == []
    assert sdk_stream.close_calls == 0


@pytest.mark.anyio
async def test_client_logs_translation_notices_with_once_only_thinking(
    caplog: pytest.LogCaptureFixture,
) -> None:
    completions = _CompletionCapture(_response(content="ok"))
    client = _client(profile=ModelProfile(thinking="none"))
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    request = GenerationRequest(
        messages=[Message.user("hello")],
        tools=[{"name": "tool", "description": "", "input_schema": {}}],
        response_schema=dict,
        thinking_level="high",
    )
    caplog.set_level(
        logging.INFO, logger="llm.providers.openai_compatible"
    )

    await client.complete(request)
    await client.complete(request)

    assert caplog.text.count("tools will be ignored") == 2
    assert caplog.text.count("the knob is inert") == 1
    assert all("tools" not in item for item in completions.requests)
    assert all("reasoning_effort" not in item for item in completions.requests)


def test_transient_error_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    monkeypatch.setattr(openai, "APIConnectionError", APIConnectionError)
    monkeypatch.setattr(openai, "APIStatusError", APIStatusError)
    monkeypatch.setattr(openai, "APITimeoutError", APITimeoutError)
    monkeypatch.setattr(openai, "RateLimitError", RateLimitError)
    client = _client()

    assert client.is_transient_error(TimeoutError())
    assert client.is_transient_error(ConnectionError())
    assert client.is_transient_error(APITimeoutError())
    assert client.is_transient_error(APIConnectionError())
    assert client.is_transient_error(RateLimitError())
    assert client.is_transient_error(APIStatusError(503))
    assert not client.is_transient_error(APIStatusError(400))
    assert not client.is_transient_error(RuntimeError())
