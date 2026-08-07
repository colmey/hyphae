"""Stable public errors shared by the native and OpenAI HTTP adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyphae.agent import (
    SessionBusyError,
    SessionCapacityError,
    SessionHistoryLimitExceeded,
    SessionNotFoundError,
)
from hyphae.application import (
    ExecutionProtocolError,
    InvalidModelError,
    ModelInventoryError,
    RuntimeConfigurationError,
)
from hyphae.orchestrator import ModelUnavailableError


@dataclass(frozen=True, slots=True)
class PublicError:
    """One matrix-defined error safe to serialize to an HTTP client."""

    status: int
    code: str
    message: str
    openai_type: str


def _error(status: int, code: str, message: str, openai_type: str) -> PublicError:
    return PublicError(status, code, message, openai_type)


def invalid_request_error() -> PublicError:
    return _error(400, "invalid_request", "The request is invalid.", "invalid_request_error")


def authentication_failed_error() -> PublicError:
    return _error(401, "authentication_failed", "Invalid or missing API key.", "invalid_request_error")


def request_too_large_error() -> PublicError:
    return _error(413, "request_too_large", "Request body too large.", "invalid_request_error")


def unsupported_media_type_error() -> PublicError:
    return _error(415, "unsupported_media_type", "Content-Type must be text/plain.", "invalid_request_error")


def provider_failure_error() -> PublicError:
    return _error(502, "provider_failure", "The model provider failed to complete the request.", "server_error")


def execution_protocol_error() -> PublicError:
    return _error(500, "execution_protocol_error", "The service could not complete the request.", "server_error")


def internal_error() -> PublicError:
    return _error(500, "internal_error", "An internal server error occurred.", "server_error")


def public_error_from_exception(exc: Exception) -> PublicError:
    """Map a known internal outcome without exposing its private detail."""
    if isinstance(exc, InvalidModelError):
        return _error(400, "invalid_model", "The requested model is not available.", "invalid_request_error")
    if isinstance(exc, ModelUnavailableError):
        return _error(503, "model_unavailable", "The requested model is temporarily unavailable.", "server_error")
    if isinstance(exc, SessionNotFoundError):
        return _error(404, "session_not_found", "Session not found.", "invalid_request_error")
    if isinstance(exc, SessionBusyError):
        return _error(409, "session_busy", "Session is processing another request.", "invalid_request_error")
    if isinstance(exc, SessionCapacityError):
        return _error(503, "session_capacity_unavailable", "Session capacity is temporarily unavailable.", "server_error")
    if isinstance(exc, SessionHistoryLimitExceeded):
        return _error(409, "session_history_limit", "Session history limit reached; start a new session.", "invalid_request_error")
    if isinstance(exc, ExecutionProtocolError):
        return execution_protocol_error()
    if isinstance(exc, (ModelInventoryError, RuntimeConfigurationError)):
        return internal_error()
    return internal_error()


def public_error_from_done_reason(done_reason: str) -> PublicError | None:
    """Return an error for terminal reasons that cannot represent success."""
    if done_reason in {"llm_error", "provider_error", "incomplete_stream"}:
        return provider_failure_error()
    if done_reason == "session_history_limit":
        return _error(
            409,
            "session_history_limit",
            "Session history limit reached; start a new session.",
            "invalid_request_error",
        )
    if done_reason in {
        "end_turn",
        "empty",
        "truncated",
        "max_tokens",
        "max_iterations",
        "budget_exceeded",
        "deadline_exceeded",
        "no_progress",
        "content_filter",
        "refusal",
    }:
        return None
    return execution_protocol_error()


def native_error_body(error: PublicError) -> dict[str, str]:
    return {"code": error.code, "message": error.message}


def native_sse_error_body(error: PublicError) -> dict[str, str]:
    return {"type": "error", **native_error_body(error)}


def openai_error_body(error: PublicError) -> dict[str, Any]:
    return {
        "error": {
            "message": error.message,
            "type": error.openai_type,
            "param": None,
            "code": error.code,
        }
    }
