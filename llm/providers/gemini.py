# llm/providers/gemini.py
"""Gemini LLMClient backed by the google-genai SDK."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    CanonicalStopReason,
    Message,
    ModelProfile,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    coerce_usage_count,
)

logger = logging.getLogger(__name__)


_CONTENT_FILTER_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    }
)


def _raw_stop_reason(finish_reason: Any) -> str | None:
    if finish_reason is None:
        return None
    return getattr(finish_reason, "name", None) or str(finish_reason)


def _canonical_stop_reason(
    finish_reason: Any, *, has_tools: bool, has_visible_content: bool
) -> CanonicalStopReason:
    """Translate Gemini terminal state, with function-call parts authoritative."""
    if has_tools:
        return "tool_use"
    raw_reason = _raw_stop_reason(finish_reason)
    if raw_reason == "STOP":
        return "end_turn" if has_visible_content else "empty"
    if raw_reason == "MAX_TOKENS":
        return "max_tokens"
    if raw_reason in _CONTENT_FILTER_REASONS:
        return "content_filter"
    return "provider_error"


class GeminiLLMClient(LLMClient):
    """LLMClient implementation backed by the google-genai SDK."""

    def __init__(
        self,
        api_key: str,
        model: str,
        default_max_tokens: int,
        profile: ModelProfile | None = None,
    ) -> None:
        # Pass explicitly so env/config failures surface early.
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._default_max_tokens = default_max_tokens
        self._profile = profile or ModelProfile.default()

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        contents = self._to_genai_contents(request.messages)
        genai_tools = self._to_genai_tools(request.tools) if request.tools else None

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": request.max_tokens or self._default_max_tokens,
            "tools": genai_tools,
            "automatic_function_calling": genai_types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
            "system_instruction": request.system,
        }
        p = self._profile
        if p.temperature is not None:
            config_kwargs["temperature"] = p.temperature
        if p.top_p is not None:
            config_kwargs["top_p"] = p.top_p
        if p.top_k is not None:
            config_kwargs["top_k"] = p.top_k

        # Invalid thinking_level is skipped rather than failing the request.
        if request.thinking_level is not None:
            try:
                config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_level=genai_types.ThinkingLevel(
                        request.thinking_level.upper()
                    ),
                )
            except ValueError:
                logger.warning(
                    "ignoring unrecognized thinking_level %r",
                    request.thinking_level,
                )

        # Structured output and tools are typically mutually exclusive.
        if request.response_schema is not None:
            if genai_tools:
                logger.warning(
                    "Gemini call received both tools and response_schema; "
                    "tools will be ignored in structured-output mode."
                )
                config_kwargs["tools"] = None
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = request.response_schema

        config = genai_types.GenerateContentConfig(**config_kwargs)

        logger.debug(
            "gemini complete: model=%s messages=%d tools=%s schema=%s thinking=%s",
            self._model,
            len(contents),
            len(genai_tools[0].function_declarations) if genai_tools else 0,
            request.response_schema.__name__ if request.response_schema else None,
            request.thinking_level,
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        return self._from_genai_response(response)

    _RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

    def is_transient_error(self, exc: BaseException) -> bool:
        """Retry rate limits, transient 5xx, and timeouts/connection resets."""
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        if isinstance(exc, genai_errors.ServerError):
            return True
        if isinstance(exc, genai_errors.APIError):
            return exc.code in self._RETRYABLE_STATUS
        return False

    def _to_genai_contents(
        self, messages: Sequence[Message]
    ) -> list[genai_types.Content]:
        """Translate internal Message list to Gemini's Content list.

        Tool results are user-role function_response parts. System text is
        passed through `system_instruction`, not message history.
        """
        out: list[genai_types.Content] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                logger.warning(
                    "system message in history was ignored; use the system= param"
                )
                continue

            parts: list[genai_types.Part] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    if block.text:
                        part_kwargs: dict[str, Any] = {"text": block.text}
                        # Gemini 3+ uses thought signatures for turn continuity.
                        sig = block.provider_metadata.get("thought_signature")
                        if sig is not None:
                            part_kwargs["thought_signature"] = sig
                        parts.append(genai_types.Part(**part_kwargs))
                elif isinstance(block, ToolUseBlock):
                    # Echo the original thought_signature or Gemini may return 400.
                    part_kwargs: dict[str, Any] = {
                        "function_call": genai_types.FunctionCall(
                            name=block.name,
                            args=block.input,
                        )
                    }
                    sig = block.provider_metadata.get("thought_signature")
                    if sig is not None:
                        part_kwargs["thought_signature"] = sig
                    parts.append(genai_types.Part(**part_kwargs))
                elif isinstance(block, ToolResultBlock):
                    # Keep both tool content and error state visible to Gemini.
                    response_payload: dict[str, Any] = {"content": block.content}
                    if block.is_error:
                        response_payload["error"] = True
                    parts.append(
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=block.name,
                                response=response_payload,
                            )
                        )
                    )

            if not parts:
                continue

            gemini_role = "model" if msg.role == Role.ASSISTANT else "user"
            out.append(genai_types.Content(role=gemini_role, parts=parts))

        return out

    def _to_genai_tools(
        self, tools: Sequence[Mapping[str, Any]]
    ) -> list[genai_types.Tool]:
        """Translate generic tools into one Gemini Tool object."""
        declarations = [
            genai_types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters_json_schema=t.get("input_schema")
                or {
                    "type": "object",
                    "properties": {},
                },
            )
            for t in tools
        ]
        return [genai_types.Tool(function_declarations=declarations)]

    def _from_genai_response(self, response: Any) -> AssistantMessage:
        """Convert a google-genai GenerateContentResponse into AssistantMessage."""
        blocks: list[Any] = []

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return AssistantMessage(
                content=[],
                stop_reason="empty",
                raw_stop_reason=None,
                model=self._model,
                usage=self._usage_from_response(response),
                reasoning=None,
            )

        candidate = candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None

        for part in parts or []:
            # Preserve thought_signature for the next Gemini turn.
            signature = getattr(part, "thought_signature", None)
            metadata: dict[str, Any] = {}
            if signature is not None:
                metadata["thought_signature"] = signature

            text = getattr(part, "text", None)
            if text:
                blocks.append(TextBlock(text=text, provider_metadata=metadata))
                continue

            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                # Gemini omits call IDs; mint one for the internal protocol.
                call_id = f"call_{uuid.uuid4().hex[:12]}"
                blocks.append(
                    ToolUseBlock(
                        id=call_id,
                        name=function_call.name,
                        input=dict(function_call.args or {}),
                        provider_metadata=metadata,
                    )
                )
                continue

            logger.debug("unhandled gemini part type: %r", part)

        has_tools = any(isinstance(block, ToolUseBlock) for block in blocks)
        has_visible_content = any(
            isinstance(block, TextBlock) and bool(block.text) for block in blocks
        )
        raw_stop_reason = _raw_stop_reason(finish_reason)

        return AssistantMessage(
            content=blocks,
            stop_reason=_canonical_stop_reason(
                finish_reason,
                has_tools=has_tools,
                has_visible_content=has_visible_content,
            ),
            raw_stop_reason=raw_stop_reason,
            model=self._model,
            usage=self._usage_from_response(response),
        )

    def _usage_from_response(self, response: Any) -> Usage:
        """Map google-genai usage_metadata onto our provider-agnostic Usage."""
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return Usage()

        return Usage(
            input_tokens=coerce_usage_count(getattr(um, "prompt_token_count", 0)),
            output_tokens=coerce_usage_count(getattr(um, "candidates_token_count", 0)),
            total_tokens=coerce_usage_count(getattr(um, "total_token_count", 0)),
            thinking_tokens=coerce_usage_count(getattr(um, "thoughts_token_count", 0)),
            cached_tokens=coerce_usage_count(
                getattr(um, "cached_content_token_count", 0)
            ),
        )
