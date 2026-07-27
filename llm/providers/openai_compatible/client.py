"""OpenAI-compatible SDK client ownership and invocation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
import logging

from llm.client import GenerationRequest, LLMClient
from llm.schemas import AssistantMessage, ModelProfile, StreamChunk

from .codec import PreparedOpenAIRequest, build_request, response_to_message
from .stream import decode_stream


logger = logging.getLogger("llm.providers.openai_compatible")


@dataclass(frozen=True, slots=True)
class OpenAICompatibleClientConfig:
    """Immutable provider-owned translation configuration."""

    model: str
    default_max_tokens: int
    profile: ModelProfile
    compatible_endpoint: bool


class OpenAICompatibleLLMClient(LLMClient):
    """LLMClient implementation for OpenAI-compatible chat-completion APIs."""

    def __init__(
        self,
        api_key: str,
        model: str,
        default_max_tokens: int,
        base_url: str | None = None,
        profile: ModelProfile | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        resolved_base_url = base_url or None
        self._client = AsyncOpenAI(api_key=api_key, base_url=resolved_base_url)
        self._config = OpenAICompatibleClientConfig(
            model=model,
            default_max_tokens=default_max_tokens,
            profile=profile or ModelProfile.default(),
            compatible_endpoint=resolved_base_url is not None,
        )
        self._warned_inert_thinking = False
        self._closed = False

    async def aclose(self) -> None:
        """Close the AsyncOpenAI client once."""
        if self._closed:
            return
        self._closed = True
        await self._client.close()

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        built = self._prepare_request(request)
        sdk_request = built.sdk_kwargs

        logger.debug(
            "openai complete: model=%s messages=%d tools=%d schema=%s",
            self._config.model,
            len(sdk_request["messages"]),
            len(sdk_request.get("tools") or []),
            request.response_schema.__name__ if request.response_schema else None,
        )

        response = await self._client.chat.completions.create(**sdk_request)
        return response_to_message(
            response,
            default_model=self._config.model,
            logger=logger,
        )

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        if request.response_schema is not None:
            raise ValueError("response_schema is not supported for streaming")

        built = self._prepare_request(request)
        sdk_request = dict(built.sdk_kwargs)
        sdk_request["stream"] = True
        sdk_request["stream_options"] = {"include_usage": True}

        logger.debug(
            "openai stream: model=%s messages=%d tools=%d",
            self._config.model,
            len(sdk_request["messages"]),
            len(sdk_request.get("tools") or []),
        )

        sdk_stream = await self._client.chat.completions.create(**sdk_request)
        decoder = decode_stream(
            sdk_stream,
            default_model=self._config.model,
            logger=logger,
        )
        async with aclosing(decoder):
            async for chunk in decoder:
                yield chunk

    def _prepare_request(
        self, generation: GenerationRequest
    ) -> PreparedOpenAIRequest:
        """Translate one request and emit client-owned adapter notices."""
        built = build_request(generation, self._config, logger=logger)
        if built.ignored_tools_for_structured_output:
            logger.warning(
                "OpenAI call received both tools and response_schema; "
                "tools will be ignored in structured-output mode."
            )
        if built.inert_thinking_requested and not self._warned_inert_thinking:
            logger.info(
                "thinking_level=%r requested but model %s declares thinking:none; "
                "the knob is inert for this model",
                generation.thinking_level,
                self._config.model,
            )
            self._warned_inert_thinking = True
        return built

    _RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

    def is_transient_error(self, exc: BaseException) -> bool:
        """Retry rate limits, transient 5xx, and timeouts/connection resets."""
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        try:
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                RateLimitError,
            )
        except ImportError:  # pragma: no cover - SDK is a hard dep when used
            return False
        if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            return getattr(exc, "status_code", None) in self._RETRYABLE_STATUS
        return False
