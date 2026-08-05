"""Focused bounded-request-reader tests using direct ASGI receive frames."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import ClientDisconnect, Request

from api.request_body import MAX_REQUEST_BODY_BYTES, read_request_body
from api.public_errors import PublicError


def _request(
    frames: list[dict[str, object]], *, content_length: str | None = None
) -> tuple[Request, list[int]]:
    receives: list[int] = []

    async def receive() -> dict[str, object]:
        receives.append(1)
        return frames.pop(0)

    headers = []
    if content_length is not None:
        headers.append((b"content-length", content_length.encode()))
    return (
        Request({"type": "http", "method": "POST", "headers": headers}, receive),
        receives,
    )


def _body_frames(chunks: list[bytes]) -> list[dict[str, object]]:
    return [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ]


@pytest.mark.anyio
async def test_reader_allows_the_exact_limit_across_multiple_receive_frames() -> None:
    request, receives = _request(
        _body_frames([b"a" * (MAX_REQUEST_BODY_BYTES - 1), b"b"])
    )

    assert await read_request_body(request) == b"a" * (MAX_REQUEST_BODY_BYTES - 1) + b"b"
    assert len(receives) == 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    "chunks",
    [
        [b"a" * MAX_REQUEST_BODY_BYTES, b"b"],
        [b"a" * (MAX_REQUEST_BODY_BYTES - 1), b"bb"],
    ],
)
async def test_reader_rejects_an_observed_body_over_the_limit(chunks: list[bytes]) -> None:
    request, _receives = _request(_body_frames(chunks))

    with pytest.raises(HTTPException) as raised:
        await read_request_body(request)

    assert raised.value.status_code == 413
    assert raised.value.detail == PublicError(
        413,
        "request_too_large",
        "Request body too large.",
        "invalid_request_error",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("declared", [None, "not-a-number", "false", "-1", "1"])
async def test_reader_counts_streamed_bytes_despite_unusable_or_understated_length(
    declared: str | None,
) -> None:
    request, _receives = _request(
        _body_frames([b"a" * MAX_REQUEST_BODY_BYTES, b"b"]),
        content_length=declared,
    )

    with pytest.raises(HTTPException) as raised:
        await read_request_body(request)

    assert raised.value.status_code == 413
    assert isinstance(raised.value.detail, PublicError)
    assert raised.value.detail.code == "request_too_large"


@pytest.mark.anyio
async def test_reader_rejects_an_oversized_declared_length_before_receiving() -> None:
    request, receives = _request(
        _body_frames([b"never read"]), content_length=str(MAX_REQUEST_BODY_BYTES + 1)
    )

    with pytest.raises(HTTPException) as raised:
        await read_request_body(request)

    assert raised.value.status_code == 413
    assert isinstance(raised.value.detail, PublicError)
    assert raised.value.detail.code == "request_too_large"
    assert receives == []


@pytest.mark.anyio
async def test_reader_propagates_starlette_disconnect_unchanged() -> None:
    request, _receives = _request([{"type": "http.disconnect"}])

    with pytest.raises(ClientDisconnect):
        await read_request_body(request)
