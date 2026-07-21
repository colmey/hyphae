"""Tool-call presentation tests for the OpenAI-compatible adapter."""

from __future__ import annotations

import pytest

from agent import ToolResultEvent
from api.openai_compatible import (
    _ChatMessage,
    _InvalidChatRequest,
    _prepare,
    _strip_tool_blocks,
    _tool_details,
)


def _result(
    content: str,
    *,
    name: str = "web__search",
    is_error: bool = False,
    latency_ms: float | None = 12.0,
) -> ToolResultEvent:
    return ToolResultEvent(
        id="call_1",
        name=name,
        content=content,
        is_error=is_error,
        latency_ms=latency_ms,
    )


def test_render_then_strip_roundtrips() -> None:
    block = _tool_details(_result("the answer is 42"), {"q": "meaning"}, 2000)
    assert "<details>" in block and "</details>" in block
    assert "🔧 web__search ✅" in block
    assert '"q": "meaning"' in block
    assert "the answer is 42" in block
    assert _strip_tool_blocks(f"Here is what I found.{block}So, 42.") == (
        "Here is what I found.So, 42."
    )


def test_error_result_uses_error_icon() -> None:
    assert "🔧 web__search ❌" in _tool_details(
        _result("boom", is_error=True), {"q": "x"}, 2000
    )


def test_strip_preserves_model_authored_details() -> None:
    authored = "intro\n<details>\n<summary>Notes</summary>\nkeep me\n</details>\nend"
    assert _strip_tool_blocks(authored) == authored


def test_tool_result_cannot_close_details_early() -> None:
    block = _tool_details(
        _result("snippet with </details> inside it"), {"q": "html"}, 2000
    )
    assert block.count("</details>") == 1
    assert "<\u200b/details>" in block
    assert _strip_tool_blocks(f"before{block}after") == "beforeafter"


def test_prepare_strips_assistant_history_but_not_prompt() -> None:
    block = _tool_details(_result("tool output"), {"q": "y"}, 2000)
    messages = [
        _ChatMessage(role="system", content="be terse"),
        _ChatMessage(role="user", content="first question"),
        _ChatMessage(role="assistant", content=f"I checked.{block}Done."),
        _ChatMessage(role="user", content="follow-up question"),
    ]

    prepared = _prepare(messages)

    assert prepared.system_override == "be terse"
    assert prepared.prompt == "follow-up question"
    assert [text for role, text in prepared.history if role == "assistant"] == [
        "I checked.Done."
    ]
    assert all("<details>" not in text for _role, text in prepared.history)


def test_prepare_preserves_order_and_combines_system_messages() -> None:
    prepared = _prepare(
        [
            _ChatMessage(role="system", content="first system"),
            _ChatMessage(role="user", content="first user"),
            _ChatMessage(role="assistant", content="first answer"),
            _ChatMessage(role="system", content="second system"),
            _ChatMessage(role="user", content="active prompt"),
        ]
    )

    assert prepared.system_override == "first system\n\nsecond system"
    assert prepared.history == (
        ("user", "first user"),
        ("assistant", "first answer"),
    )
    assert prepared.prompt == "active prompt"


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        ([], "no user message found in 'messages'"),
        (
            [_ChatMessage(role="assistant", content="prefill")],
            "no user message found in 'messages'",
        ),
        (
            [
                _ChatMessage(role="user", content="question"),
                _ChatMessage(role="assistant", content="prefill"),
            ],
            "the final conversational message must have role 'user'",
        ),
        (
            [_ChatMessage(role="user", content="")],
            "no user message found in 'messages'",
        ),
    ],
)
def test_prepare_preserves_validation_errors(messages, error: str) -> None:
    with pytest.raises(_InvalidChatRequest) as exc_info:
        _prepare(messages)
    assert str(exc_info.value) == error
