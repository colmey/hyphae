"""Provider-neutral generation attempts, retries, and stream normalization."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
import logging
import random
from typing import Any

from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    CompletionUsage,
    ReasoningDelta,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
)

from .runtime import RunContext, RunDeadlineExceeded, RunLimits


logger = logging.getLogger(__name__)

_RETRY_BACKOFF_CAP_SECONDS = 30.0
_REASONING_RETRY_NOTICE = (
    "\n\n[Generation was interrupted; retrying...]\n\n"
)


@dataclass(frozen=True)
class _TimeoutBound:
    """One timeout computed against the current absolute run deadline."""

    seconds: float | None
    deadline_limited: bool


@dataclass(frozen=True)
class _AttemptController:
    """Shared attempt, deadline, and backoff policy for one LLM generation."""

    limits: RunLimits
    context: RunContext
    log: logging.Logger | logging.LoggerAdapter

    @property
    def total_attempts(self) -> int:
        return max(0, self.limits.max_retries) + 1

    def indexes(self) -> range:
        return range(self.total_attempts)

    def ensure_before_attempt(self) -> None:
        if self.context.deadline_exceeded():
            raise RunDeadlineExceeded()

    def timeout_bound(self) -> _TimeoutBound:
        """Return the smaller enabled LLM timeout and remaining run budget."""
        remaining = self.context.remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise RunDeadlineExceeded()

        per_attempt = self.limits.llm_timeout_seconds
        if per_attempt is not None and per_attempt <= 0:
            per_attempt = None
        if remaining is None:
            return _TimeoutBound(per_attempt, False)
        if per_attempt is None or remaining <= per_attempt:
            return _TimeoutBound(remaining, True)
        return _TimeoutBound(per_attempt, False)

    async def retry(
        self,
        attempt_index: int,
        *,
        eligible: bool,
        mode: str,
        reason: str,
        cause: BaseException | None = None,
    ) -> bool:
        """Sleep for the next retry, or report that this attempt is final."""
        if not eligible or attempt_index >= self.total_attempts - 1:
            return False

        delay = _backoff_delay_seconds(
            self.limits.retry_base_delay,
            attempt_index,
        )
        remaining = self.context.remaining_seconds()
        if remaining is not None:
            if remaining <= 0:
                raise RunDeadlineExceeded() from cause
            delay = min(delay, remaining)

        suffix = f": {cause}" if cause is not None else ""
        self.log.warning(
            "retrying %s LLM after %s (attempt %d/%d) in %.2fs%s",
            mode,
            reason,
            attempt_index + 1,
            self.total_attempts,
            delay,
            suffix,
        )
        await asyncio.sleep(delay)
        if self.context.deadline_exceeded():
            raise RunDeadlineExceeded() from cause
        return True


async def _complete_with_retry(
    llm: LLMClient,
    *,
    request: GenerationRequest,
    attempts: _AttemptController,
) -> AssistantMessage:
    """Call complete with bounded retries and return the final response."""
    for attempt_index in attempts.indexes():
        attempts.ensure_before_attempt()
        timeout = attempts.timeout_bound()
        try:
            if timeout.seconds is not None and timeout.seconds > 0:
                async with asyncio.timeout(timeout.seconds):
                    response = await llm.complete(request)
            else:
                response = await llm.complete(request)
        except TimeoutError as exc:
            if timeout.deadline_limited or attempts.context.deadline_exceeded():
                raise RunDeadlineExceeded() from exc
            if await attempts.retry(
                attempt_index,
                eligible=True,
                mode="buffered",
                reason="timeout",
                cause=exc,
            ):
                continue
            raise
        except Exception as exc:
            if attempts.context.deadline_exceeded():
                raise RunDeadlineExceeded() from exc
            if await attempts.retry(
                attempt_index,
                eligible=llm.is_transient_error(exc),
                mode="buffered",
                reason="transient provider error",
                cause=exc,
            ):
                continue
            raise

        if await attempts.retry(
            attempt_index,
            eligible=response.stop_reason == "empty",
            mode="buffered",
            reason="empty response",
        ):
            continue
        return response

    raise AssertionError("attempt controller produced no buffered attempts")


async def _read_stream_chunk(
    chunks: AsyncIterator[StreamChunk],
    attempts: _AttemptController,
) -> StreamChunk:
    """Read one stream chunk within the idle timeout and absolute deadline."""
    timeout = attempts.timeout_bound()
    if timeout.seconds is None or timeout.seconds <= 0:
        return await anext(chunks)
    try:
        async with asyncio.timeout(timeout.seconds):
            return await anext(chunks)
    except TimeoutError as exc:
        if timeout.deadline_limited or attempts.context.deadline_exceeded():
            raise RunDeadlineExceeded() from exc
        raise


async def _close_stream(
    stream: Any,
    *,
    log: logging.Logger | logging.LoggerAdapter = logger,
) -> None:
    """Best-effort close one provider iterator without hiding its outcome."""
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except asyncio.CancelledError as exc:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise
        log.warning(
            "LLM stream cleanup was cancelled for %s (%s)",
            type(stream).__name__,
            type(exc).__name__,
        )
    except Exception as exc:  # noqa: BLE001 -- preserve the primary outcome.
        log.warning(
            "failed to close LLM stream %s (%s)",
            type(stream).__name__,
            type(exc).__name__,
        )


def _backoff_delay_seconds(base_delay: float, attempt: int) -> float:
    """Return capped exponential backoff plus the existing jitter range."""
    return min(_RETRY_BACKOFF_CAP_SECONDS, base_delay * (2**attempt)) + random.uniform(
        0, base_delay
    )


async def generate_with_retry(
    llm: LLMClient,
    request: GenerationRequest,
    *,
    limits: RunLimits,
    context: RunContext,
    stream: bool,
    log: logging.Logger | logging.LoggerAdapter,
) -> AsyncGenerator[StreamChunk, None]:
    """Normalize buffered or streaming attempts into canonical stream chunks."""
    attempts = _AttemptController(limits=limits, context=context, log=log)

    if not stream:
        response = await _complete_with_retry(
            llm,
            request=request,
            attempts=attempts,
        )
        yield StreamEnd(response)
        return

    for attempt_index in attempts.indexes():
        attempts.ensure_before_attempt()
        visible_text: list[str] = []
        emitted_reasoning = False
        chunks: AsyncIterator[StreamChunk] | None = None
        attempt_response: AssistantMessage | None = None
        attempt_error: Exception | None = None
        retryable_error = False
        try:
            chunks = llm.stream(request)
            while True:
                try:
                    chunk = await _read_stream_chunk(chunks, attempts)
                except StopAsyncIteration:
                    break

                if isinstance(chunk, TextDelta):
                    if chunk.text:
                        visible_text.append(chunk.text)
                        yield chunk
                elif isinstance(chunk, ReasoningDelta):
                    if chunk.text:
                        emitted_reasoning = True
                        yield chunk
                elif isinstance(chunk, StreamEnd):
                    attempt_response = chunk.message
                    break

            if attempt_response is None:
                if visible_text:
                    attempt_response = AssistantMessage(
                        content=[TextBlock(text="".join(visible_text))],
                        stop_reason="incomplete_stream",
                        model=None,
                        usage=CompletionUsage(),
                    )
                else:
                    attempt_response = AssistantMessage(
                        content=[],
                        stop_reason="empty",
                        model=None,
                        usage=CompletionUsage(),
                    )
        except RunDeadlineExceeded:
            raise
        except Exception as exc:
            if context.deadline_exceeded():
                raise RunDeadlineExceeded() from exc
            attempt_error = exc
            retryable_error = not visible_text and (
                isinstance(exc, TimeoutError) or llm.is_transient_error(exc)
            )
        finally:
            if chunks is not None:
                await _close_stream(chunks, log=log)

        if attempt_error is not None:
            retry_reason = (
                "timeout"
                if isinstance(attempt_error, TimeoutError)
                else "transient provider error"
            )
            if await attempts.retry(
                attempt_index,
                eligible=retryable_error,
                mode="streaming",
                reason=retry_reason,
                cause=attempt_error,
            ):
                if emitted_reasoning:
                    yield ReasoningDelta(_REASONING_RETRY_NOTICE)
                continue
            raise attempt_error

        assert attempt_response is not None
        if await attempts.retry(
            attempt_index,
            eligible=(
                attempt_response.stop_reason == "empty" and not visible_text
            ),
            mode="streaming",
            reason="empty response",
        ):
            if emitted_reasoning:
                yield ReasoningDelta(_REASONING_RETRY_NOTICE)
            continue
        yield StreamEnd(attempt_response)
        return

    raise AssertionError("attempt controller produced no streaming attempts")
