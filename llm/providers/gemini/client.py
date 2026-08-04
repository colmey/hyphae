"""Gemini SDK client ownership and invocation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging

from llm.client import GenerationRequest, LLMClient
from llm.schemas import AssistantMessage, ModelProfile

from .codec import (
    build_generation_config,
    messages_to_contents,
    response_to_message,
)


logger = logging.getLogger("llm.providers.gemini")


@dataclass(frozen=True, slots=True)
class GeminiClientConfig:
    """Immutable provider-owned translation configuration."""

    model: str
    default_max_tokens: int
    profile: ModelProfile


class GeminiLLMClient(LLMClient):
    """LLMClient implementation backed by the google-genai SDK."""

    def __init__(
        self,
        api_key: str,
        model: str,
        default_max_tokens: int,
        profile: ModelProfile | None = None,
    ) -> None:
        from google import genai

        # Pass explicitly so env/config failures surface early.
        self._client = genai.Client(api_key=api_key)
        self._config = GeminiClientConfig(
            model=model,
            default_max_tokens=default_max_tokens,
            profile=profile or ModelProfile.default(),
        )
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def aclose(self) -> None:
        """Join or retry closing the owned async google-genai client."""
        if self._closed:
            return

        task = self._close_task
        if task is not None and task.done():
            if task.cancelled() or task.exception() is not None:
                self._close_task = None
                task = None
        if task is None:
            task = asyncio.create_task(self._finish_close(), name="gemini-client-close")
            self._close_task = task
            task.add_done_callback(self._close_finished)
        await asyncio.shield(task)

    async def _finish_close(self) -> None:
        await self._client.aio.aclose()
        self._closed = True

    def _close_finished(self, task: asyncio.Task[None]) -> None:
        """Observe failed background cleanup and leave it retryable."""
        failed = task.cancelled() or task.exception() is not None
        if failed and self._close_task is task:
            self._close_task = None

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        contents = messages_to_contents(request.messages, logger=logger)
        config = build_generation_config(request, self._config, logger=logger)

        logger.debug(
            "gemini complete: model=%s messages=%d tools=%s schema=%s thinking=%s",
            self._config.model,
            len(contents),
            len(request.tools or []),
            request.response_schema.__name__ if request.response_schema else None,
            request.thinking_level,
        )

        response = await self._client.aio.models.generate_content(
            model=self._config.model,
            contents=contents,
            config=config,
        )

        return response_to_message(
            response,
            default_model=self._config.model,
            logger=logger,
        )

    _RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

    def is_transient_error(self, exc: BaseException) -> bool:
        """Retry rate limits, transient 5xx, and timeouts/connection resets."""
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True

        from google.genai import errors as genai_errors

        if isinstance(exc, genai_errors.ServerError):
            return True
        if isinstance(exc, genai_errors.APIError):
            return exc.code in self._RETRYABLE_STATUS
        return False
