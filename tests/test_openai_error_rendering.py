"""Focused failure-state tests for the OpenAI-compatible API renderer."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from starlette.requests import ClientDisconnect

from agent import (
    DoneEvent,
    ErrorEvent,
    Session,
    SessionBusyError,
    SessionCapacityError,
    SessionHistoryLimitExceeded,
    SessionNotFoundError,
    TextEvent,
)
from api.openai_compatible import (
    _FINISH_REASONS,
    _finish_reason,
    _stream_chat_completion,
    chat_completions,
)
from application import (
    ExecutionProtocolError,
    InvalidModelError,
    ModelInventoryError,
    PersistencePolicy,
    RuntimeConfigurationError,
    TurnExecution,
    TurnMetadata,
    TurnRequest,
    TurnResult,
)
from api.public_errors import native_error_body, openai_error_body, public_error_from_exception
from llm.schemas import CompletionUsage
from orchestrator import ModelUnavailableError


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
            raise InvalidModelError(model_id, self.available_model_ids())

    async def run(self, _turn):
        return TurnResult(
            answer="partial answer",
            done_reason=self.done_reason,
            usage=CompletionUsage(total_tokens=3),
            metadata=TurnMetadata(run_id="test-run", model_id="test-model"),
        )


def _runtime(runner):
    return SimpleNamespace(
        settings=SimpleNamespace(
            llm=SimpleNamespace(model="test-model"),
            openai_compat_tool_activity_mode="hidden",
            openai_compat_tool_activity_max_chars=2000,
        ),
        turn_runner=lambda: runner,
    )


class _HTTPErrorRunner(_RunRunner):
    async def run(self, _turn):
        raise SessionBusyError("test-session")


class _Request:
    headers: dict[str, str] = {}

    async def stream(self):
        yield b'{"messages":[{"role":"user","content":"hello"}]}'


class _DisconnectingRequest:
    headers: dict[str, str] = {}

    async def stream(self):
        raise ClientDisconnect()
        yield b""


def _collect_stream(runner: _EventsRunner) -> list[dict]:
    async def collect() -> list[dict]:
        turn = TurnRequest(
            prompt="hello",
            session=Session(),
            persistence=PersistencePolicy.EPHEMERAL,
            stream=True,
        )
        return [
            item
            async for item in _stream_chat_completion(
                runner,
                turn,
                "reasoning",
                2000,
            )
        ]

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
                "message": "The model provider failed to complete the request.",
                "type": "server_error",
                "param": None,
                "code": "provider_failure",
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
            "message": "An internal server error occurred.",
            "type": "server_error",
            "param": None,
            "code": "internal_error",
        }
    }
    assert frames[-1] == "[DONE]"


def test_sse_missing_terminal_event_fails_instead_of_defaulting_to_stop() -> None:
    frames = _decoded(_collect_stream(_EventsRunner([TextEvent("partial")])))

    assert frames[-2] == {
        "error": {
            "message": "The service could not complete the request.",
            "type": "server_error",
            "param": None,
            "code": "execution_protocol_error",
        }
    }
    assert frames[-1] == "[DONE]"


def test_sse_defers_an_error_event_to_the_history_limit_terminal_reason() -> None:
    frames = _decoded(
        _collect_stream(
            _EventsRunner(
                [
                    ErrorEvent("raw-history-limit-sentinel"),
                    DoneEvent("session_history_limit", 1),
                ]
            )
        )
    )

    assert frames[-2] == {
        "error": {
            "message": "Session history limit reached; start a new session.",
            "type": "invalid_request_error",
            "param": None,
            "code": "session_history_limit",
        }
    }
    assert "raw-history-limit-sentinel" not in repr(frames)
    assert frames[-1] == "[DONE]"


def test_sse_error_event_with_unknown_terminal_reason_is_a_protocol_error() -> None:
    frames = _decoded(
        _collect_stream(
            _EventsRunner(
                [ErrorEvent("raw-error-sentinel"), DoneEvent("unknown-reason", 1)]
            )
        )
    )

    assert frames[-2]["error"] == {
        "message": "The service could not complete the request.",
        "type": "server_error",
        "param": None,
        "code": "execution_protocol_error",
    }
    assert "raw-error-sentinel" not in repr(frames)
    assert frames[-1] == "[DONE]"


@pytest.mark.parametrize(
    "reason", ["llm_error", "provider_error", "incomplete_stream", "unexpected_reason"]
)
def test_nonstream_invalid_done_reason_returns_openai_500(reason: str) -> None:
    response = asyncio.run(
        chat_completions(
            _Request(),
            runtime=_runtime(_RunRunner(reason)),
        )
    )

    expected_status = 502 if reason in {"llm_error", "provider_error", "incomplete_stream"} else 500
    assert response.status_code == expected_status
    body = json.loads(response.body)
    assert body["error"]["type"] == "server_error"
    assert body["error"]["param"] is None
    assert body["error"]["code"] == (
        "provider_failure"
        if reason in {"llm_error", "provider_error", "incomplete_stream"}
        else "execution_protocol_error"
    )


def test_nonstream_http_exception_preserves_status_and_openai_envelope() -> None:
    response = asyncio.run(
        chat_completions(
            _Request(),
            runtime=_runtime(_HTTPErrorRunner("end_turn")),
        )
    )

    assert response.status_code == 409
    assert json.loads(response.body) == {
        "error": {
            "message": "Session is processing another request.",
            "type": "invalid_request_error",
            "param": None,
            "code": "session_busy",
        }
    }


def test_openai_endpoint_propagates_client_disconnect() -> None:
    with pytest.raises(ClientDisconnect):
        asyncio.run(
            chat_completions(
                _DisconnectingRequest(),
                runtime=_runtime(_RunRunner("end_turn")),
            )
        )


@pytest.mark.parametrize(
    ("exc", "status", "code", "message"),
    [
        (InvalidModelError("raw-invalid-model", []), 400, "invalid_model", "The requested model is not available."),
        (ModelUnavailableError("raw-unavailable-model"), 503, "model_unavailable", "The requested model is temporarily unavailable."),
        (ModelInventoryError(), 500, "internal_error", "An internal server error occurred."),
        (RuntimeConfigurationError("raw-runtime-config"), 500, "internal_error", "An internal server error occurred."),
        (ExecutionProtocolError("raw-protocol"), 500, "execution_protocol_error", "The service could not complete the request."),
        (SessionNotFoundError("raw-session"), 404, "session_not_found", "Session not found."),
        (SessionBusyError("raw-session"), 409, "session_busy", "Session is processing another request."),
        (SessionCapacityError(), 503, "session_capacity_unavailable", "Session capacity is temporarily unavailable."),
        (SessionHistoryLimitExceeded(), 409, "session_history_limit", "Session history limit reached; start a new session."),
        (RuntimeError("raw-unexpected-sentinel"), 500, "internal_error", "An internal server error occurred."),
    ],
)
def test_public_error_matrix_hides_private_exception_detail(
    exc: Exception, status: int, code: str, message: str
) -> None:
    error = public_error_from_exception(exc)

    assert (error.status, error.code, error.message) == (status, code, message)
    assert native_error_body(error) == {"code": code, "message": message}
    assert openai_error_body(error)["error"]["code"] == code
    assert "raw-" not in repr(native_error_body(error))
    assert "raw-" not in repr(openai_error_body(error))
