"""Tool-activity presentation tests for the OpenAI-compatible adapter."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from typing import Literal

import pytest

from agent import (
    DoneEvent,
    ReasoningEvent,
    Session,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from api.openai_compatible import (
    _ChatMessage,
    _InvalidChatRequest,
    _format_tool_call_activity,
    _format_tool_result_activity,
    _prepare_chat_request,
    _stream_chat_completion,
    _strip_legacy_tool_blocks,
)
from api.turn import (
    PersistencePolicy,
    TurnExecution,
    TurnMetadata,
    TurnRequest,
)


def _call(
    args: dict | None = None,
    *,
    call_id: str = "call_1",
    name: str = "web__search",
) -> ToolCallEvent:
    return ToolCallEvent(
        id=call_id,
        name=name,
        input=args or {},
    )


def _result(
    result: str,
    *,
    call_id: str = "call_1",
    name: str = "web__search",
    is_error: bool = False,
    latency_ms: float | None = 12.0,
) -> ToolResultEvent:
    return ToolResultEvent(
        id=call_id,
        name=name,
        content=result,
        is_error=is_error,
        latency_ms=latency_ms,
    )


def _legacy_block(content: str = "tool output") -> str:
    return (
        "\n\n<details>\n"
        "<summary>🔧 web__search ✅ · 12 ms</summary>\n"
        f"{content}\n"
        "</details>\n\n"
    )


def test_call_activity_is_deterministic_unicode_markdown() -> None:
    activity = _format_tool_call_activity(
        _call({"z": "café\nsecond", "a": 1}),
        2000,
        include_details=True,
    )

    assert activity == (
        "> **Tool** `web.search` — running\n\n"
        "**Arguments**\n\n"
        "```json\n"
        '{"a": 1,"z": "café\\nsecond"}\n'
        "```\n\n"
    )
    assert "<details>" not in activity
    assert "<think>" not in activity


def test_result_activity_formats_success_unicode_and_multiline_content() -> None:
    activity = _format_tool_result_activity(
        _result("first line\nnaïve second", latency_ms=12.6),
        2000,
        include_details=True,
    )

    assert activity == (
        "> **Tool** `web.search` — completed in 13 ms\n\n"
        "**Result**\n\n"
        "```text\n"
        "first line\n"
        "naïve second\n"
        "```\n\n"
    )
    assert "<details>" not in activity
    assert "<think>" not in activity


def test_error_result_uses_failure_status_without_missing_latency() -> None:
    assert _format_tool_result_activity(
        _result("boom", is_error=True, latency_ms=None),
        2000,
        include_details=True,
    ) == ("> **Tool** `web.search` — failed\n\n**Result**\n\n```text\nboom\n```\n\n")


def test_summary_activity_is_plain_and_omits_bodies() -> None:
    call = _format_tool_call_activity(
        _call({"secret": "not displayed"}),
        2000,
        include_details=False,
    )
    result = _format_tool_result_activity(
        _result("not displayed", latency_ms=12.6),
        2000,
        include_details=False,
    )

    assert call == "> **Tool** `web.search` — running\n\n"
    assert result == "> **Tool** `web.search` — completed in 13 ms\n\n"
    assert "secret" not in call
    assert "not displayed" not in call
    assert "not displayed" not in result
    assert all(marker not in call + result for marker in ("🔧", "✅", "❌"))


def test_call_and_result_bodies_are_independently_truncated() -> None:
    call = _format_tool_call_activity(
        _call({"query": "abcdefghij"}),
        10,
        include_details=True,
    )
    result = _format_tool_result_activity(
        _result("0123456789abcdef"),
        10,
        include_details=True,
    )

    assert '{"query": \n…[truncated]' in call
    assert "abcdefghij" not in call
    assert "0123456789\n…[truncated]" in result
    assert "abcdef" not in result


def test_activity_fences_prevent_url_linkification_and_embedded_fence_escape() -> None:
    call = _format_tool_call_activity(
        _call({"url": "https://example.test/article?q=one&sort=two"}),
        2000,
        include_details=True,
    )
    result = _format_tool_result_activity(
        _result("literal ``` fence\nhttps://example.test/result"),
        2000,
        include_details=True,
    )

    assert "```json\n" in call
    assert '"https://example.test/article?q=one&sort=two"' in call
    assert "    https://" not in call
    assert "````text\nliteral ``` fence\nhttps://example.test/result\n````" in result


def test_strip_legacy_tool_blocks_preserves_model_authored_markup() -> None:
    authored = (
        "intro\n"
        "<details>\n<summary>Notes</summary>\nkeep me\n</details>\n"
        "<think>model-authored</think>\n"
        "end"
    )
    assert _strip_legacy_tool_blocks(authored) == authored


def test_prepare_strips_legacy_assistant_history_but_not_user_content() -> None:
    block = _legacy_block()
    user_text = f"user-authored block stays{block}after"
    messages = [
        _ChatMessage(role="system", content="be terse"),
        _ChatMessage(role="user", content=user_text),
        _ChatMessage(role="assistant", content=f"I checked.{block}Done."),
        _ChatMessage(role="user", content="follow-up question"),
    ]

    prepared = _prepare_chat_request(messages)

    assert prepared.system_override == "be terse"
    assert prepared.prompt == "follow-up question"
    assert prepared.history == (
        ("user", user_text),
        ("assistant", "I checked.Done."),
    )


def test_prepare_preserves_order_and_combines_system_messages() -> None:
    prepared = _prepare_chat_request(
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
        _prepare_chat_request(messages)
    assert str(exc_info.value) == error


class _EventsRunner:
    def __init__(self, events) -> None:
        self._events = events

    @asynccontextmanager
    async def open(self, _turn):
        async def events():
            for event in self._events:
                yield event

        yield TurnExecution(
            metadata=TurnMetadata(run_id="test-run", model_id="test-model"),
            events=events(),
        )


def _collect_stream(
    events,
    *,
    activity: Literal["reasoning", "reasoning_full", "hidden"] = "reasoning",
) -> list[dict | str]:
    async def collect() -> list[dict | str]:
        turn = TurnRequest(
            prompt="hello",
            session=Session(),
            persistence=PersistencePolicy.EPHEMERAL,
            stream=True,
        )
        items = [
            item
            async for item in _stream_chat_completion(
                _EventsRunner(events),
                turn,
                activity,
                2000,
            )
        ]
        return [
            json.loads(item["data"]) if item["data"] != "[DONE]" else "[DONE]"
            for item in items
        ]

    return asyncio.run(collect())


def _delta(frame: dict) -> dict:
    return frame["choices"][0]["delta"]


def test_reasoning_mode_streams_ordered_activity_and_clean_answer() -> None:
    frames = _collect_stream(
        [
            _call({"q": "first"}, call_id="same", name="first_tool"),
            _result("one", call_id="same", name="first_tool"),
            ReasoningEvent("model narration"),
            _call({"q": "second"}, call_id="same", name="second_tool"),
            _result(
                "two",
                call_id="same",
                name="second_tool",
                is_error=True,
                latency_ms=None,
            ),
            TextEvent("clean "),
            TextEvent("answer"),
            DoneEvent("end_turn", 2),
        ]
    )

    assert frames[-1] == "[DONE]"
    chunks = [frame for frame in frames[:-1] if isinstance(frame, dict)]
    deltas = [_delta(frame) for frame in chunks]
    assert deltas[0] == {"role": "assistant"}
    assert [next(iter(delta)) for delta in deltas[1:-1]] == [
        "reasoning_content",
        "reasoning_content",
        "reasoning_content",
        "reasoning_content",
        "reasoning_content",
        "content",
        "content",
    ]
    activity = [
        delta["reasoning_content"] for delta in deltas if "reasoning_content" in delta
    ]
    assert ["first_tool" in text for text in activity] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert ["second_tool" in text for text in activity] == [
        False,
        False,
        False,
        True,
        True,
    ]
    assert "running" in activity[0]
    assert "completed" in activity[1]
    assert activity[2] == "model narration"
    assert "running" in activity[3]
    assert "failed" in activity[4]
    assert "".join(delta.get("content", "") for delta in deltas) == "clean answer"
    assert all("tool_calls" not in delta for delta in deltas)
    assert "<think>" not in json.dumps(frames)
    assert "</think>" not in json.dumps(frames)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_reasoning_fragments_are_streamed_without_whitespace_changes() -> None:
    fragments = [
        "The",
        " user",
        " simply",
        ' said "',
        "hello",
        '".',
        "\nSecond",
        " line.",
    ]
    frames = _collect_stream(
        [
            *(ReasoningEvent(fragment) for fragment in fragments),
            TextEvent("Hello!"),
            DoneEvent("end_turn", 1),
        ]
    )

    chunks = [frame for frame in frames[:-1] if isinstance(frame, dict)]
    deltas = [_delta(frame) for frame in chunks]
    streamed_reasoning = [
        delta["reasoning_content"] for delta in deltas if "reasoning_content" in delta
    ]

    assert streamed_reasoning == fragments
    assert "".join(streamed_reasoning) == (
        'The user simply said "hello".\nSecond line.'
    )
    assert "".join(delta.get("content", "") for delta in deltas) == "Hello!"


def test_tool_activity_adds_boundaries_only_between_semantic_phases() -> None:
    frames = _collect_stream(
        [
            ReasoningEvent("Need"),
            ReasoningEvent(" a search."),
            _call({"query": "weather"}),
            _result("sunny"),
            ReasoningEvent("I"),
            ReasoningEvent(" found it."),
            TextEvent("It is sunny."),
            DoneEvent("end_turn", 1),
        ]
    )

    chunks = [frame for frame in frames[:-1] if isinstance(frame, dict)]
    deltas = [_delta(frame) for frame in chunks]
    reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)

    assert reasoning == (
        "Need a search.\n\n"
        "> **Tool** `web.search` — running\n\n"
        "> **Tool** `web.search` — completed in 12 ms\n\n"
        "I found it."
    )
    assert "".join(delta.get("content", "") for delta in deltas) == "It is sunny."


def test_reasoning_full_mode_includes_bounded_tool_details() -> None:
    frames = _collect_stream(
        [
            _call({"query": "café"}),
            _result("first\nsecond"),
            TextEvent("answer"),
            DoneEvent("end_turn", 1),
        ],
        activity="reasoning_full",
    )

    chunks = [frame for frame in frames[:-1] if isinstance(frame, dict)]
    deltas = [_delta(frame) for frame in chunks]
    activity = [
        delta["reasoning_content"] for delta in deltas if "reasoning_content" in delta
    ]

    assert '"query": "café"' in activity[0]
    assert "**Arguments**" in activity[0]
    assert "first\nsecond" in activity[1]
    assert "**Result**" in activity[1]
    assert "".join(delta.get("content", "") for delta in deltas) == "answer"


def test_hidden_mode_omits_tool_activity_without_changing_answer() -> None:
    frames = _collect_stream(
        [
            ReasoningEvent("hidden reasoning"),
            _call({"q": "hidden"}),
            _result("hidden result"),
            TextEvent("visible"),
            DoneEvent("end_turn", 2),
        ],
        activity="hidden",
    )

    chunks = [frame for frame in frames[:-1] if isinstance(frame, dict)]
    deltas = [_delta(frame) for frame in chunks]
    assert deltas == [{"role": "assistant"}, {"content": "visible"}, {}]
    assert all("reasoning_content" not in delta for delta in deltas)
    assert "hidden reasoning" not in json.dumps(frames)
    assert all("tool_calls" not in delta for delta in deltas)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert frames[-1] == "[DONE]"
