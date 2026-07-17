"""Hermetic Gemini request and response translation contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import types as genai_types

from llm.providers.gemini import GeminiLLMClient
from llm.schemas import (
    Message,
    ModelProfile,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


def _client() -> GeminiLLMClient:
    client = object.__new__(GeminiLLMClient)
    client._model = "gemini-test"
    client._default_max_tokens = 123
    client._profile = ModelProfile.default()
    return client


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


def _text_part(text: str = "answer", signature: bytes | None = None) -> Any:
    return SimpleNamespace(
        text=text,
        thought_signature=signature,
        function_call=None,
    )


@pytest.mark.parametrize(
    ("finish_reason", "parts", "expected"),
    [
        (genai_types.FinishReason.STOP, [_text_part()], "end_turn"),
        (genai_types.FinishReason.STOP, [], "empty"),
        (genai_types.FinishReason.MAX_TOKENS, [_text_part()], "max_tokens"),
        (genai_types.FinishReason.SAFETY, [], "content_filter"),
        (genai_types.FinishReason.RECITATION, [], "content_filter"),
        (genai_types.FinishReason.BLOCKLIST, [], "content_filter"),
        (genai_types.FinishReason.PROHIBITED_CONTENT, [], "content_filter"),
        (genai_types.FinishReason.SPII, [], "content_filter"),
        (genai_types.FinishReason.IMAGE_SAFETY, [], "content_filter"),
        (genai_types.FinishReason.IMAGE_PROHIBITED_CONTENT, [], "content_filter"),
        (genai_types.FinishReason.IMAGE_RECITATION, [], "content_filter"),
        (genai_types.FinishReason.FINISH_REASON_UNSPECIFIED, [], "provider_error"),
        (genai_types.FinishReason.LANGUAGE, [], "provider_error"),
        (genai_types.FinishReason.OTHER, [], "provider_error"),
        (genai_types.FinishReason.MALFORMED_FUNCTION_CALL, [], "provider_error"),
        (genai_types.FinishReason.UNEXPECTED_TOOL_CALL, [], "provider_error"),
        (genai_types.FinishReason.NO_IMAGE, [], "provider_error"),
        (genai_types.FinishReason.IMAGE_OTHER, [], "provider_error"),
        (None, [_text_part()], "provider_error"),
        (SimpleNamespace(name="FUTURE_REASON"), [_text_part()], "provider_error"),
    ],
)
def test_finish_reason_translation(
    finish_reason: Any, parts: list[Any], expected: str
) -> None:
    message = _client()._from_genai_response(_response(finish_reason, parts=parts))

    assert message.stop_reason == expected
    expected_raw = (
        getattr(finish_reason, "name", None) or str(finish_reason)
        if finish_reason is not None
        else None
    )
    assert message.raw_stop_reason == expected_raw


def test_empty_candidates_are_retryable_empty() -> None:
    message = _client()._from_genai_response(_response(candidates=False))

    assert message.stop_reason == "empty"
    assert message.raw_stop_reason is None


@pytest.mark.parametrize(
    "finish_reason",
    [None, genai_types.FinishReason.STOP, genai_types.FinishReason.SAFETY],
)
def test_function_call_is_authoritative(finish_reason: Any) -> None:
    signature = b"opaque-signature"
    part = SimpleNamespace(
        text=None,
        thought_signature=signature,
        function_call=SimpleNamespace(name="srv__tool", args={"q": "x"}),
    )
    message = _client()._from_genai_response(_response(finish_reason, parts=[part]))

    assert message.stop_reason == "tool_use"
    tool_use = message.tool_uses()[0]
    assert tool_use.id.startswith("call_")
    assert tool_use.name == "srv__tool"
    assert tool_use.input == {"q": "x"}
    assert tool_use.provider_metadata == {"thought_signature": signature}


def test_usage_coercion_preserves_valid_counts_and_zeros_malformed() -> None:
    usage_metadata = SimpleNamespace(
        prompt_token_count="10",
        candidates_token_count="bad",
        total_token_count=12,
        thoughts_token_count=float("nan"),
        cached_content_token_count=None,
    )

    usage = _client()._usage_from_response(
        _response(parts=[_text_part()], usage=usage_metadata)
    )

    assert usage == Usage(input_tokens=10, output_tokens=0, total_tokens=12)


def test_thought_signatures_and_tool_results_round_trip_to_gemini_only() -> None:
    signature = b"opaque-signature"
    messages = [
        Message(
            role=Role.ASSISTANT,
            content=[
                TextBlock(
                    "preface", provider_metadata={"thought_signature": signature}
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
                    is_error=True,
                )
            ]
        ),
    ]

    contents = _client()._to_genai_contents(messages)

    assert [content.role for content in contents] == ["model", "user"]
    assert contents[0].parts[0].thought_signature == signature
    assert contents[0].parts[1].thought_signature == signature
    assert contents[0].parts[1].function_call.name == "srv__tool"
    function_response = contents[1].parts[0].function_response
    assert function_response.name == "srv__tool"
    assert function_response.response == {"content": "result", "error": True}
