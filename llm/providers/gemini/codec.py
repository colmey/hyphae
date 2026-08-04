"""SDK-call-free Gemini request and response translation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from typing import TYPE_CHECKING, Any
import uuid

from google.genai import types as genai_types

from llm.client import GenerationRequest
from llm.schemas import (
    AssistantMessage,
    CanonicalStopReason,
    CompletionUsage,
    ContentBlock,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    coerce_usage_count,
)

if TYPE_CHECKING:
    from .client import GeminiClientConfig


_LOGGER = logging.getLogger("llm.providers.gemini")

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


def raw_stop_reason(finish_reason: Any) -> str | None:
    """Return the provider-native terminal reason name when present."""
    if finish_reason is None:
        return None
    return getattr(finish_reason, "name", None) or str(finish_reason)


def canonical_stop_reason(
    finish_reason: Any, *, has_tools: bool, has_visible_content: bool
) -> CanonicalStopReason:
    """Translate Gemini terminal state, with function-call parts authoritative."""
    if has_tools:
        return "tool_use"
    raw_reason = raw_stop_reason(finish_reason)
    if raw_reason == "STOP":
        return "end_turn" if has_visible_content else "empty"
    if raw_reason == "MAX_TOKENS":
        return "max_tokens"
    if raw_reason in _CONTENT_FILTER_REASONS:
        return "content_filter"
    return "provider_error"


def messages_to_contents(
    messages: Sequence[Message],
    *,
    logger: logging.Logger | None = None,
) -> list[genai_types.Content]:
    """Translate canonical message history into Gemini content."""
    active_logger = logger or _LOGGER
    out: list[genai_types.Content] = []
    for message in messages:
        if message.role == Role.SYSTEM:
            active_logger.warning(
                "system message in history was ignored; use the system= param"
            )
            continue

        parts: list[genai_types.Part] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                if block.text:
                    parts.append(genai_types.Part(text=block.text))
            elif isinstance(block, ToolUseBlock):
                tool_part_kwargs: dict[str, Any] = {
                    "function_call": genai_types.FunctionCall(
                        name=block.name,
                        args=block.input,
                    )
                }
                signature = block.provider_metadata.get("thought_signature")
                if signature is not None:
                    tool_part_kwargs["thought_signature"] = signature
                parts.append(genai_types.Part(**tool_part_kwargs))
            elif isinstance(block, ToolResultBlock):
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

        gemini_role = "model" if message.role == Role.ASSISTANT else "user"
        out.append(genai_types.Content(role=gemini_role, parts=parts))

    return out


def tools_to_gemini(
    tools: Sequence[Mapping[str, Any]],
) -> list[genai_types.Tool]:
    """Translate generic tools into one Gemini tool declaration."""
    declarations = [
        genai_types.FunctionDeclaration(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters_json_schema=tool.get("input_schema")
            or {
                "type": "object",
                "properties": {},
            },
        )
        for tool in tools
    ]
    return [genai_types.Tool(function_declarations=declarations)]


def build_generation_config(
    generation: GenerationRequest,
    config: GeminiClientConfig,
    *,
    logger: logging.Logger | None = None,
) -> genai_types.GenerateContentConfig:
    """Translate provider-neutral generation controls into Gemini config."""
    active_logger = logger or _LOGGER
    genai_tools = tools_to_gemini(generation.tools) if generation.tools else None
    config_kwargs: dict[str, Any] = {
        "max_output_tokens": (generation.max_tokens or config.default_max_tokens),
        "tools": genai_tools,
        "automatic_function_calling": genai_types.AutomaticFunctionCallingConfig(
            disable=True,
        ),
        "system_instruction": generation.system,
    }

    profile = config.profile
    if profile.temperature is not None:
        config_kwargs["temperature"] = profile.temperature
    if profile.top_p is not None:
        config_kwargs["top_p"] = profile.top_p
    if profile.top_k is not None:
        config_kwargs["top_k"] = profile.top_k

    if generation.thinking_level is not None:
        normalized_level = generation.thinking_level.upper()
        try:
            thinking_level = genai_types.ThinkingLevel.__members__[normalized_level]
        except KeyError:
            active_logger.warning(
                "ignoring unrecognized thinking_level %r",
                generation.thinking_level,
            )
        else:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_level=thinking_level,
            )

    if generation.response_schema is not None:
        if genai_tools:
            active_logger.warning(
                "Gemini call received both tools and response_schema; "
                "tools will be ignored in structured-output mode."
            )
            config_kwargs["tools"] = None
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = generation.response_schema

    return genai_types.GenerateContentConfig(**config_kwargs)


def response_to_message(
    response: Any,
    *,
    default_model: str,
    logger: logging.Logger | None = None,
) -> AssistantMessage:
    """Convert a Gemini generation response into a canonical assistant message."""
    active_logger = logger or _LOGGER
    blocks: list[ContentBlock] = []
    reasoning_parts: list[str] = []

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return AssistantMessage(
            content=[],
            stop_reason="empty",
            raw_stop_reason=None,
            model=default_model,
            usage=usage_from_response(response),
            reasoning=None,
        )

    candidate = candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content else None

    for part in parts or []:
        text = getattr(part, "text", None)
        if getattr(part, "thought", False):
            if text:
                reasoning_parts.append(text)
            continue

        signature = getattr(part, "thought_signature", None)

        if text:
            blocks.append(TextBlock(text=text))
            continue

        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            metadata = {"thought_signature": signature} if signature is not None else {}
            blocks.append(
                ToolUseBlock(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=function_call.name,
                    input=dict(function_call.args or {}),
                    provider_metadata=metadata,
                )
            )
            continue

        active_logger.debug("unhandled gemini part type: %r", part)

    has_tools = any(isinstance(block, ToolUseBlock) for block in blocks)
    has_visible_content = any(
        isinstance(block, TextBlock) and bool(block.text) for block in blocks
    )

    return AssistantMessage(
        content=blocks,
        stop_reason=canonical_stop_reason(
            finish_reason,
            has_tools=has_tools,
            has_visible_content=has_visible_content,
        ),
        raw_stop_reason=raw_stop_reason(finish_reason),
        model=default_model,
        usage=usage_from_response(response),
        reasoning="".join(reasoning_parts).strip() or None,
    )


def usage_from_response(response: Any) -> CompletionUsage:
    """Map Gemini usage metadata onto canonical completion usage."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return CompletionUsage()

    return CompletionUsage(
        input_tokens=coerce_usage_count(getattr(usage, "prompt_token_count", 0)),
        output_tokens=coerce_usage_count(getattr(usage, "candidates_token_count", 0)),
        total_tokens=coerce_usage_count(getattr(usage, "total_token_count", 0)),
        thinking_tokens=coerce_usage_count(getattr(usage, "thoughts_token_count", 0)),
        cached_tokens=coerce_usage_count(
            getattr(usage, "cached_content_token_count", 0)
        ),
    )
