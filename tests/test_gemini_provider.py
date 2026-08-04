"""Hermetic Gemini adapter contracts by package owner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import logging
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
import pytest

from llm.client import GenerationRequest, build_llm_client
from llm.providers.gemini import GeminiLLMClient
from llm.providers.gemini.client import GeminiClientConfig
from llm.providers.gemini.codec import (
    build_generation_config,
    canonical_stop_reason,
    messages_to_contents,
    raw_stop_reason,
    response_to_message,
    tools_to_gemini,
    usage_from_response,
)
from llm.schemas import (
    CompletionUsage,
    Message,
    ModelProfile,
    ReasoningDelta,
    Role,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(
    *,
    profile: ModelProfile | None = None,
) -> GeminiClientConfig:
    return GeminiClientConfig(
        model="gemini-test",
        default_max_tokens=123,
        profile=profile or ModelProfile.default(),
    )


def _response(
    finish_reason: Any = genai_types.FinishReason.STOP,
    *,
    parts: list[Any] | None = None,
    candidates: bool = True,
    usage: Any = None,
) -> Any:
    candidate_items = []
    if candidates:
        candidate_items.append(
            SimpleNamespace(
                finish_reason=finish_reason,
                content=SimpleNamespace(parts=parts or []),
            )
        )
    return SimpleNamespace(candidates=candidate_items, usage_metadata=usage)


def _text_part(
    text: str = "answer",
    signature: bytes | None = None,
    *,
    thought: bool = False,
) -> Any:
    return SimpleNamespace(
        text=text,
        thought=thought,
        thought_signature=signature,
        function_call=None,
    )


class _CaptureGenerate:
    def __init__(self, response: Any | None = None) -> None:
        self.response = (
            response if response is not None else _response(parts=[_text_part()])
        )
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def _constructed_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: ModelProfile | None = None,
    capture: _CaptureGenerate | None = None,
) -> tuple[GeminiLLMClient, _CaptureGenerate]:
    generate = capture if capture is not None else _CaptureGenerate()

    async def close() -> None:
        pass

    sdk_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=generate),
            aclose=close,
        )
    )
    monkeypatch.setattr(genai, "Client", lambda **kwargs: sdk_client)
    return (
        GeminiLLMClient(
            api_key="test-key",
            model="gemini-test",
            default_max_tokens=123,
            profile=profile,
        ),
        generate,
    )


def test_public_import_and_registry_remain_lazy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import llm.client; "
                "assert 'google.genai' not in sys.modules; "
                "assert 'llm.providers.gemini' not in sys.modules"
            ),
        ],
        check=True,
        cwd=ROOT,
    )
    assert GeminiLLMClient.__name__ == "GeminiLLMClient"


def test_registry_constructs_gemini_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, Any]] = []

    def build_sdk_client(**kwargs: Any) -> Any:
        constructed.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(genai, "Client", build_sdk_client)
    requested_keys: list[str] = []

    class FactorySettings:
        llm = SimpleNamespace(
            provider="gemini",
            model="gemini-test",
            max_tokens=123,
        )
        openai_compat_base_url = ""

        def api_key_for_provider(self, provider: str) -> str:
            requested_keys.append(provider)
            return "test-key"

    client = build_llm_client(FactorySettings())

    assert isinstance(client, GeminiLLMClient)
    assert requested_keys == ["gemini"]
    assert constructed == [{"api_key": "test-key"}]


def test_constructor_preserves_signature_and_normalizes_immutable_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, Any]] = []

    def build_sdk_client(**kwargs: Any) -> Any:
        constructed.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(genai, "Client", build_sdk_client)
    profile = ModelProfile(temperature=0.2)

    client = GeminiLLMClient("key", "model", 77, profile)

    assert list(inspect.signature(GeminiLLMClient).parameters) == [
        "api_key",
        "model",
        "default_max_tokens",
        "profile",
    ]
    assert client._config == GeminiClientConfig(
        model="model",
        default_max_tokens=77,
        profile=profile,
    )
    assert constructed == [{"api_key": "key"}]
    with pytest.raises(FrozenInstanceError):
        client._config.model = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        (None, None),
        (genai_types.FinishReason.STOP, "STOP"),
        (SimpleNamespace(name="FUTURE_REASON"), "FUTURE_REASON"),
        ("provider-value", "provider-value"),
    ],
)
def test_raw_stop_reason(finish_reason: Any, expected: str | None) -> None:
    assert raw_stop_reason(finish_reason) == expected


@pytest.mark.parametrize(
    ("finish_reason", "has_tools", "has_text", "expected"),
    [
        (None, True, False, "tool_use"),
        (genai_types.FinishReason.STOP, False, True, "end_turn"),
        (genai_types.FinishReason.STOP, False, False, "empty"),
        (genai_types.FinishReason.MAX_TOKENS, False, True, "max_tokens"),
        (genai_types.FinishReason.SAFETY, False, False, "content_filter"),
        (genai_types.FinishReason.RECITATION, False, False, "content_filter"),
        (genai_types.FinishReason.BLOCKLIST, False, False, "content_filter"),
        (
            genai_types.FinishReason.PROHIBITED_CONTENT,
            False,
            False,
            "content_filter",
        ),
        (genai_types.FinishReason.SPII, False, False, "content_filter"),
        (genai_types.FinishReason.IMAGE_SAFETY, False, False, "content_filter"),
        (
            genai_types.FinishReason.IMAGE_PROHIBITED_CONTENT,
            False,
            False,
            "content_filter",
        ),
        (
            genai_types.FinishReason.IMAGE_RECITATION,
            False,
            False,
            "content_filter",
        ),
        (
            genai_types.FinishReason.FINISH_REASON_UNSPECIFIED,
            False,
            False,
            "provider_error",
        ),
        (genai_types.FinishReason.LANGUAGE, False, False, "provider_error"),
        (genai_types.FinishReason.OTHER, False, False, "provider_error"),
        (
            genai_types.FinishReason.MALFORMED_FUNCTION_CALL,
            False,
            False,
            "provider_error",
        ),
        (
            genai_types.FinishReason.UNEXPECTED_TOOL_CALL,
            False,
            False,
            "provider_error",
        ),
        (genai_types.FinishReason.NO_IMAGE, False, False, "provider_error"),
        (genai_types.FinishReason.IMAGE_OTHER, False, False, "provider_error"),
        (SimpleNamespace(name="FUTURE_REASON"), False, True, "provider_error"),
    ],
)
def test_canonical_stop_reason(
    finish_reason: Any,
    has_tools: bool,
    has_text: bool,
    expected: str,
) -> None:
    assert (
        canonical_stop_reason(
            finish_reason,
            has_tools=has_tools,
            has_visible_content=has_text,
        )
        == expected
    )


def test_missing_candidates_preserve_usage_in_empty_response() -> None:
    response = _response(
        candidates=False,
        usage=SimpleNamespace(
            prompt_token_count=2,
            candidates_token_count=0,
            total_token_count=2,
        ),
    )

    message = response_to_message(response, default_model="gemini-test")

    assert message.content == []
    assert message.stop_reason == "empty"
    assert message.raw_stop_reason is None
    assert message.model == "gemini-test"
    assert message.usage == CompletionUsage(input_tokens=2, total_tokens=2)


@pytest.mark.parametrize(
    "finish_reason",
    [None, genai_types.FinishReason.STOP, genai_types.FinishReason.SAFETY],
)
def test_function_call_is_authoritative_and_generated_id_has_exact_shape(
    finish_reason: Any,
) -> None:
    signature = b"opaque-signature"
    part = SimpleNamespace(
        text=None,
        thought_signature=signature,
        function_call=SimpleNamespace(name="srv__tool", args={"q": "x"}),
    )

    message = response_to_message(
        _response(finish_reason, parts=[part]),
        default_model="gemini-test",
    )

    assert message.stop_reason == "tool_use"
    tool_use = message.tool_uses()[0]
    assert re.fullmatch(r"call_[0-9a-f]{12}", tool_use.id)
    assert tool_use.name == "srv__tool"
    assert tool_use.input == {"q": "x"}
    assert tool_use.provider_metadata == {"thought_signature": signature}


def test_response_text_drops_noncontinuation_signature_and_empty_parts() -> None:
    signature = b"text-signature"
    ignored = SimpleNamespace(
        text="",
        thought=False,
        thought_signature=None,
        function_call=None,
    )

    message = response_to_message(
        _response(parts=[_text_part("answer", signature), ignored]),
        default_model="gemini-test",
    )

    assert message.content == [TextBlock(text="answer")]
    assert message.stop_reason == "end_turn"
    assert message.reasoning is None


def test_thought_only_response_is_reasoning_without_visible_or_replayed_text() -> None:
    signature = b"thought-signature"

    message = response_to_message(
        _response(
            parts=[
                genai_types.Part(
                    text="  private reasoning  ",
                    thought=True,
                    thought_signature=signature,
                )
            ]
        ),
        default_model="gemini-test",
    )

    assert message.reasoning == "private reasoning"
    assert message.content == []
    assert message.stop_reason == "empty"
    assert message.to_message() == Message.assistant([])
    assert messages_to_contents([message.to_message()]) == []


def test_thought_and_answer_are_separated_and_only_answer_is_replayed() -> None:
    message = response_to_message(
        _response(
            parts=[
                _text_part(" first ", thought=True),
                _text_part("second", thought=True),
                _text_part("answer", b"answer-signature"),
            ]
        ),
        default_model="gemini-test",
    )

    assert message.reasoning == "first second"
    assert message.content == [TextBlock("answer")]
    assert message.stop_reason == "end_turn"
    replay = messages_to_contents([message.to_message()])
    assert [part.text for part in replay[0].parts] == ["answer"]
    assert replay[0].parts[0].thought_signature is None


def test_thought_and_tool_preserve_only_function_call_continuation_metadata() -> None:
    thought_signature = b"thought-signature"
    call_signature = b"call-signature"
    function_call = SimpleNamespace(name="srv__tool", args={"q": "x"})

    message = response_to_message(
        _response(
            parts=[
                _text_part(
                    "private reasoning",
                    thought_signature,
                    thought=True,
                ),
                SimpleNamespace(
                    text=None,
                    thought=False,
                    thought_signature=call_signature,
                    function_call=function_call,
                ),
            ]
        ),
        default_model="gemini-test",
    )

    assert message.reasoning == "private reasoning"
    assert message.stop_reason == "tool_use"
    assert message.content == [
        ToolUseBlock(
            id=message.tool_uses()[0].id,
            name="srv__tool",
            input={"q": "x"},
            provider_metadata={"thought_signature": call_signature},
        )
    ]
    replay = messages_to_contents([message.to_message()])
    assert len(replay[0].parts) == 1
    assert replay[0].parts[0].function_call.name == function_call.name
    assert replay[0].parts[0].function_call.args == function_call.args
    assert replay[0].parts[0].thought_signature == call_signature


def test_usage_coercion_preserves_valid_counts_and_zeros_malformed() -> None:
    response = _response(
        usage=SimpleNamespace(
            prompt_token_count="10",
            candidates_token_count="bad",
            total_token_count=12,
            thoughts_token_count=float("nan"),
            cached_content_token_count=None,
        )
    )

    usage = usage_from_response(response)

    assert usage == CompletionUsage(input_tokens=10, total_tokens=12)
    assert usage_from_response(SimpleNamespace()) == CompletionUsage()


def test_messages_preserve_roles_blocks_errors_and_tool_thought_signatures() -> None:
    signature = b"opaque-signature"
    messages = [
        Message.user("hello"),
        Message(
            role=Role.ASSISTANT,
            content=[
                TextBlock(
                    "preface",
                    provider_metadata={"thought_signature": signature},
                ),
                ToolUseBlock(
                    id="call_internal",
                    name="srv__tool",
                    input={"q": "x"},
                    provider_metadata={"thought_signature": signature},
                ),
            ],
        ),
        Message.tool_results(
            [
                ToolResultBlock(
                    tool_use_id="call_internal",
                    name="srv__tool",
                    content="result",
                ),
                ToolResultBlock(
                    tool_use_id="call_error",
                    name="srv__tool",
                    content="failed",
                    is_error=True,
                ),
            ]
        ),
    ]

    contents = messages_to_contents(messages)

    assert [content.role for content in contents] == ["user", "model", "user"]
    assert contents[0].parts[0].text == "hello"
    assert contents[1].parts[0].thought_signature is None
    assert contents[1].parts[1].thought_signature == signature
    assert contents[1].parts[1].function_call.name == "srv__tool"
    assert contents[1].parts[1].function_call.args == {"q": "x"}
    assert contents[2].parts[0].function_response.response == {"content": "result"}
    assert contents[2].parts[1].function_response.response == {
        "content": "failed",
        "error": True,
    }


def test_empty_content_is_omitted_and_system_history_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="llm.providers.gemini")
    messages = [
        Message(role=Role.SYSTEM, content=[TextBlock("old system")]),
        Message(role=Role.USER, content=[]),
        Message(role=Role.USER, content=[TextBlock("")]),
    ]

    contents = messages_to_contents(messages)

    assert contents == []
    assert "system message in history was ignored" in caplog.text


def test_tool_declarations_preserve_schema_and_supply_default() -> None:
    tools = tools_to_gemini(
        [
            {
                "name": "srv__lookup",
                "description": "lookup",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
            {"name": "srv__empty", "input_schema": {}},
        ]
    )

    declarations = tools[0].function_declarations
    assert declarations[0].name == "srv__lookup"
    assert declarations[0].description == "lookup"
    assert declarations[0].parameters_json_schema == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }
    assert declarations[1].parameters_json_schema == {
        "type": "object",
        "properties": {},
    }


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, 123), (0, 123), (77, 77)],
)
def test_generation_config_preserves_token_fallback_and_override(
    requested: int | None,
    expected: int,
) -> None:
    config = build_generation_config(
        GenerationRequest(messages=[], max_tokens=requested),
        _config(),
    )

    assert config.max_output_tokens == expected
    assert config.automatic_function_calling.disable is True


def test_generation_config_preserves_system_sampling_and_tools() -> None:
    profile = ModelProfile(temperature=0.2, top_p=0.8, top_k=40)
    config = build_generation_config(
        GenerationRequest(
            messages=[],
            tools=[{"name": "srv__lookup", "input_schema": {}}],
            system="system",
        ),
        _config(profile=profile),
    )

    assert config.system_instruction == "system"
    assert config.temperature == 0.2
    assert config.top_p == 0.8
    assert config.top_k == 40
    assert config.tools[0].function_declarations[0].name == "srv__lookup"


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("low", genai_types.ThinkingLevel.LOW),
        ("MeDiUm", genai_types.ThinkingLevel.MEDIUM),
        ("HIGH", genai_types.ThinkingLevel.HIGH),
    ],
)
def test_valid_thinking_levels_are_case_normalized(
    requested: str,
    expected: genai_types.ThinkingLevel,
) -> None:
    config = build_generation_config(
        GenerationRequest(messages=[], thinking_level=requested),
        _config(),
    )

    assert config.thinking_config.thinking_level == expected


def test_invalid_thinking_level_warns_and_is_omitted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="llm.providers.gemini")

    config = build_generation_config(
        GenerationRequest(messages=[], thinking_level="maximum"),
        _config(),
    )

    assert config.thinking_config is None
    assert "ignoring unrecognized thinking_level 'maximum'" in caplog.text


def test_structured_generation_keeps_schema_and_suppresses_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="llm.providers.gemini")

    config = build_generation_config(
        GenerationRequest(
            messages=[],
            tools=[{"name": "srv__lookup", "input_schema": {}}],
            response_schema=dict,
        ),
        _config(),
    )

    assert config.tools is None
    assert config.response_mime_type == "application/json"
    assert config.response_schema is dict
    assert "tools will be ignored in structured-output mode" in caplog.text


@pytest.mark.anyio
async def test_client_invokes_sdk_and_logs_call_metadata(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    profile = ModelProfile(temperature=0.2, top_p=0.8, top_k=40)
    client, capture = _constructed_client(monkeypatch, profile=profile)
    caplog.set_level(logging.DEBUG, logger="llm.providers.gemini")
    tools = [
        {
            "name": "srv__lookup",
            "description": "lookup",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]

    message = await client.complete(
        GenerationRequest(
            messages=[Message.user("hello")],
            tools=tools,
            system="system",
            max_tokens=77,
            thinking_level="high",
        )
    )

    call = capture.calls[0]
    assert call["model"] == "gemini-test"
    assert [content.role for content in call["contents"]] == ["user"]
    assert call["config"].max_output_tokens == 77
    assert call["config"].system_instruction == "system"
    assert message.model == "gemini-test"
    assert message.text_blocks() == [TextBlock("answer")]
    assert "gemini complete: model=gemini-test messages=1 tools=1" in caplog.text


@pytest.mark.anyio
async def test_client_inherits_coarse_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _CaptureGenerate(
        _response(
            parts=[
                _text_part("private reasoning", thought=True),
                _text_part("answer"),
            ]
        )
    )
    client, _ = _constructed_client(monkeypatch, capture=capture)

    chunks = [
        chunk
        async for chunk in client.stream(
            GenerationRequest(messages=[Message.user("hello")])
        )
    ]

    assert chunks[0] == ReasoningDelta("private reasoning")
    assert chunks[1] == TextDelta("answer")
    assert isinstance(chunks[2], StreamEnd)
    assert chunks[2].message.text_blocks() == [TextBlock("answer")]
    assert chunks[2].message.to_message() == Message.assistant([TextBlock("answer")])


def test_transient_error_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class APIError(Exception):
        def __init__(self, code: int) -> None:
            self.code = code

    class ServerError(APIError):
        pass

    monkeypatch.setattr(genai_errors, "APIError", APIError)
    monkeypatch.setattr(genai_errors, "ServerError", ServerError)
    client = object.__new__(GeminiLLMClient)

    assert client.is_transient_error(TimeoutError())
    assert client.is_transient_error(ConnectionError())
    assert client.is_transient_error(ServerError(400))
    for status in (408, 429, 500, 502, 503, 504):
        assert client.is_transient_error(APIError(status))
    assert not client.is_transient_error(APIError(400))
    assert not client.is_transient_error(RuntimeError())
