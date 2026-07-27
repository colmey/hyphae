"""Focused failure-state tests for the OpenAI-compatible API renderer."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from agent import DoneEvent, ErrorEvent, Session, TextEvent
from api.openai_compatible import (
    _FINISH_REASONS,
    _finish_reason,
    _stream_chat_completion,
    chat_completions,
)
from api.schemas import TokenUsage
from api.turn import (
    PersistencePolicy,
    TurnExecution,
    TurnMetadata,
    TurnRequest,
    TurnResult,
)


class _EventsRunner:
    def __init__(self, events, *, failure: Exception | None = None) -> None:
        self._events = events
        self._failure = failure

    @asynccontextmanager
    async def open(self, _turn):
        async def events():
            for event in self._events:
                yield event
            if self._failure is not None:
                raise self._failure

        yield TurnExecution(
            metadata=TurnMetadata(run_id="test-run", model_id="test-model"),
            events=events(),
        )


class _RunRunner:
    def __init__(self, done_reason: str) -> None:
        self.done_reason = done_reason

    def available_model_ids(self) -> list[str]:
        return ["test-model"]

    def validate_model_id(self, model_id: str | None) -> None:
        if model_id is not None and model_id not in self.available_model_ids():
            raise HTTPException(status_code=400, detail="invalid model")

    async def run(self, _turn):
        return TurnResult(
            answer="partial answer",
            done_reason=self.done_reason,
            usage=TokenUsage(total_tokens=3),
            metadata=TurnMetadata(run_id="test-run", model_id="test-model"),
        )


class _HTTPErrorRunner(_RunRunner):
    async def run(self, _turn):
        raise HTTPException(status_code=409, detail="turn is busy")


class _Request:
    async def json(self):
        return {"messages": [{"role": "user", "content": "hello"}]}


def _collect_stream(runner: _EventsRunner) -> list[dict]:
    async def collect() -> list[dict]:
        turn = TurnRequest(
            prompt="hello",
            session=Session(),
            persistence=PersistencePolicy.EPHEMERAL,
            stream=True,
        )
        return [item async for item in _stream_chat_completion(runner, turn, "reasoning", 2000)]

    return asyncio.run(collect())


def _decoded(items: list[dict]) -> list[dict | str]:
    return [
        json.loads(item["data"]) if item["data"] != "[DONE]" else "[DONE]"
        for item in items
    ]


@pytest.mark.parametrize("done_reason, expected", sorted(_FINISH_REASONS.items()))
def test_finish_reason_maps_only_supported_success_reasons(
    done_reason: str, expected: str
) -> None:
    assert _finish_reason(done_reason) == expected


@pytest.mark.parametrize(
    "done_reason",
    [
        "llm_error",
        "provider_error",
        "incomplete_stream",
        "unknown",
        "provider_surprise",
    ],
)
def test_finish_reason_fails_closed(done_reason: str) -> None:
    with pytest.raises(ValueError, match="unsupported completion reason"):
        _finish_reason(done_reason)


@pytest.mark.parametrize(
    "events",
    [
        [
            ErrorEvent("backend unavailable"),
            TextEvent("must be ignored"),
            DoneEvent("end_turn", 1),
        ],
        [
            TextEvent("partial"),
            ErrorEvent("backend unavailable"),
            DoneEvent("llm_error", 1),
        ],
    ],
)
def test_sse_error_event_is_terminal_and_emitted_once(events) -> None:
    frames = _decoded(_collect_stream(_EventsRunner(events)))

    errors = [frame for frame in frames if isinstance(frame, dict) and "error" in frame]
    finish_frames = [
        frame
        for frame in frames
        if isinstance(frame, dict)
        and frame.get("choices")
        and frame["choices"][0]["finish_reason"] is not None
    ]
    content = [
        frame["choices"][0]["delta"].get("content", "")
        for frame in frames
        if isinstance(frame, dict) and frame.get("choices")
    ]

    assert errors == [
        {
            "error": {
                "message": "backend unavailable",
                "type": "server_error",
                "param": None,
                "code": None,
            }
        }
    ]
    assert "must be ignored" not in content
    assert finish_frames == []
    assert frames[-1] == "[DONE]"


@pytest.mark.parametrize(
    "reason", ["llm_error", "provider_error", "incomplete_stream", "unexpected_reason"]
)
def test_sse_invalid_done_reason_becomes_error_without_success_finish(
    reason: str,
) -> None:
    frames = _decoded(
        _collect_stream(_EventsRunner([TextEvent("partial"), DoneEvent(reason, 1)]))
    )

    assert sum(isinstance(frame, dict) and "error" in frame for frame in frames) == 1
    assert not any(
        isinstance(frame, dict)
        and frame.get("choices")
        and frame["choices"][0]["finish_reason"] is not None
        for frame in frames
    )
    assert frames[-1] == "[DONE]"


def test_sse_renderer_exception_uses_error_envelope_and_done_sentinel() -> None:
    frames = _decoded(
        _collect_stream(
            _EventsRunner([TextEvent("partial")], failure=RuntimeError("boom"))
        )
    )

    assert frames[-2] == {
        "error": {
            "message": "boom",
            "type": "server_error",
            "param": None,
            "code": None,
        }
    }
    assert frames[-1] == "[DONE]"


def test_sse_missing_terminal_event_fails_instead_of_defaulting_to_stop() -> None:
    frames = _decoded(_collect_stream(_EventsRunner([TextEvent("partial")])))

    assert frames[-2] == {
        "error": {
            "message": "completion stream ended without a terminal event",
            "type": "server_error",
            "param": None,
            "code": None,
        }
    }
    assert frames[-1] == "[DONE]"


@pytest.mark.parametrize(
    "reason", ["llm_error", "provider_error", "incomplete_stream", "unexpected_reason"]
)
def test_nonstream_invalid_done_reason_returns_openai_500(reason: str) -> None:
    response = asyncio.run(
        chat_completions(
            _Request(),
            settings=SimpleNamespace(
                llm=SimpleNamespace(model_name="test-model")
            ),
            runner=_RunRunner(reason),
        )
    )

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"]["type"] == "server_error"
    assert body["error"]["param"] is None
    assert body["error"]["code"] is None


def test_nonstream_http_exception_preserves_status_and_openai_envelope() -> None:
    response = asyncio.run(
        chat_completions(
            _Request(),
            settings=SimpleNamespace(
                llm=SimpleNamespace(model_name="test-model")
            ),
            runner=_HTTPErrorRunner("end_turn"),
        )
    )

    assert response.status_code == 409
    assert json.loads(response.body) == {
        "error": {
            "message": "turn is busy",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }
