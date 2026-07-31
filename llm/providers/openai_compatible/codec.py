"""SDK-call-free OpenAI request and response translation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import re
from typing import TYPE_CHECKING, Any
import uuid

from llm.client import GenerationRequest
from llm.schemas import (
    AssistantMessage,
    CanonicalStopReason,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    CompletionUsage,
    coerce_usage_count,
)

if TYPE_CHECKING:
    from .client import OpenAICompatibleClientConfig


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK = re.compile(
    rf"^\s*{re.escape(_THINK_OPEN)}.*?{re.escape(_THINK_CLOSE)}\s*",
    re.DOTALL,
)

_STOP_REASON_MAP: dict[str, CanonicalStopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
}


@dataclass(frozen=True, slots=True)
class PreparedOpenAIRequest:
    """One translated SDK request plus client-owned logging signals."""

    sdk_kwargs: dict[str, Any]
    ignored_tools_for_structured_output: bool = False
    inert_thinking_requested: bool = False


def raw_stop_reason(finish_reason: Any) -> str | None:
    return None if finish_reason is None else str(finish_reason)


def canonical_stop_reason(
    finish_reason: Any,
    *,
    has_tools: bool,
    has_visible_content: bool,
    has_refusal: bool = False,
) -> CanonicalStopReason:
    """Translate OpenAI terminal state, with structured content authoritative."""
    if has_tools:
        return "tool_use"
    if has_refusal:
        return "refusal"

    raw_reason = raw_stop_reason(finish_reason)
    if raw_reason == "stop" and not has_visible_content:
        return "empty"
    canonical = _STOP_REASON_MAP.get(raw_reason or "")
    if canonical == "tool_use" or canonical is None:
        return "provider_error"
    return canonical


def _tool_call_id(raw_id: Any) -> str:
    value = str(raw_id).strip() if raw_id is not None else ""
    return value or f"call_{uuid.uuid4().hex[:12]}"


def build_tool_use_block(
    *,
    call_id: Any,
    name: Any,
    raw_arguments: Any,
    logger: logging.Logger | None = None,
) -> ToolUseBlock:
    """Finalize one provider call into the canonical, replayable tool shape."""
    try:
        arguments = json.loads(raw_arguments) if raw_arguments else {}
        parse_error = None
    except (json.JSONDecodeError, TypeError):
        if logger is not None:
            logger.warning("could not parse tool arguments: %r", raw_arguments)
        arguments = {}
        parse_error = f"arguments were not valid JSON: {raw_arguments!r}"
    return ToolUseBlock(
        id=_tool_call_id(call_id),
        name=str(name) if name is not None else "",
        input=arguments,
        parse_error=parse_error,
    )


def merge_extra_body(
    request: dict[str, Any], extensions: Mapping[str, Any]
) -> None:
    """Merge compatible-endpoint extensions without replacing existing values."""
    merged = dict(request.get("extra_body") or {})
    for key, value in extensions.items():
        merged.setdefault(key, value)
    request["extra_body"] = merged


def _split_reasoning(text: str | None) -> tuple[str | None, str]:
    """Return provider-emitted reasoning and visible content separately."""
    if not text:
        return None, ""
    reasoning: str | None
    match = _THINK_BLOCK.match(text)
    if not match:
        candidate = text.lstrip()
        if candidate.startswith(_THINK_OPEN):
            reasoning = candidate[len(_THINK_OPEN) :].strip()
            return reasoning or None, ""
        return None, text
    reasoning = (
        match.group(0)
        .replace(_THINK_OPEN, "")
        .replace(_THINK_CLOSE, "")
        .strip()
        or None
    )
    return reasoning, text[match.end() :]


def structured_reasoning_text(value: Any) -> str | None:
    """Read supported OpenAI-compatible reasoning text fields."""
    for field_name in ("reasoning_content", "reasoning", "thinking"):
        text = getattr(value, field_name, None)
        if isinstance(text, str) and text:
            return text
    return None


def messages_to_openai(
    messages: Sequence[Message],
    system: str | None,
    *,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Translate canonical messages into OpenAI chat messages."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for message in messages:
        if message.role == Role.SYSTEM:
            if logger is not None:
                logger.warning(
                    "system message in history was ignored; use the system= param"
                )
            continue

        if message.role == Role.TOOL:
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": block.content,
                        }
                    )
            continue

        if message.role == Role.ASSISTANT:
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    if block.text:
                        text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input or {}),
                            },
                        }
                    )
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts) or None,
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            out.append(assistant_message)
            continue

        text_parts = [
            block.text
            for block in message.content
            if isinstance(block, TextBlock) and block.text
        ]
        out.append({"role": "user", "content": "".join(text_parts)})

    return out


def tools_to_openai(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Translate generic tools into OpenAI function-tool dictionaries."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema")
                or {
                    "type": "object",
                    "properties": {},
                },
            },
        }
        for tool in tools
    ]


def build_response_format(response_schema: type) -> dict[str, Any]:
    """Build an OpenAI response format for structured output."""
    try:
        schema = response_schema.model_json_schema()  # type: ignore[attr-defined]
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.__name__,
                "schema": schema,
            },
        }
    except Exception:  # pragma: no cover - non-Pydantic or unsupported
        return {"type": "json_object"}


def build_request(
    generation: GenerationRequest,
    config: OpenAICompatibleClientConfig,
    *,
    logger: logging.Logger | None = None,
) -> PreparedOpenAIRequest:
    """Translate a provider-neutral generation request into SDK kwargs."""
    openai_messages = messages_to_openai(
        generation.messages,
        generation.system,
        logger=logger,
    )
    openai_tools = tools_to_openai(generation.tools) if generation.tools else None

    request: dict[str, Any] = {
        "model": config.model,
        "messages": openai_messages,
    }
    token_limit = generation.max_tokens or config.default_max_tokens
    if config.compatible_endpoint:
        request["max_tokens"] = token_limit
    else:
        request["max_completion_tokens"] = token_limit

    profile = config.profile
    if profile.temperature is not None:
        request["temperature"] = profile.temperature
    if profile.top_p is not None:
        request["top_p"] = profile.top_p
    if config.compatible_endpoint and profile.top_k is not None:
        merge_extra_body(request, {"top_k": profile.top_k})
    if openai_tools:
        request["tools"] = openai_tools
        request["tool_choice"] = "auto"

    ignored_tools = False
    if generation.response_schema is not None:
        if openai_tools:
            ignored_tools = True
            request.pop("tools", None)
            request.pop("tool_choice", None)
        request["response_format"] = build_response_format(generation.response_schema)

    inert_thinking = False
    if generation.thinking_level is not None:
        if profile.thinking == "hint-param":
            request["reasoning_effort"] = generation.thinking_level
        elif profile.thinking == "none":
            inert_thinking = True

    return PreparedOpenAIRequest(
        sdk_kwargs=request,
        ignored_tools_for_structured_output=ignored_tools,
        inert_thinking_requested=inert_thinking,
    )


def response_to_message(
    response: Any,
    *,
    default_model: str,
    logger: logging.Logger | None = None,
) -> AssistantMessage:
    """Convert an OpenAI ChatCompletion into a canonical assistant message."""
    blocks: list[Any] = []
    choices = getattr(response, "choices", None) or []
    if not choices:
        return AssistantMessage(
            content=[],
            stop_reason="empty",
            raw_stop_reason=None,
            model=default_model,
            usage=usage_from_raw(getattr(response, "usage", None)),
            reasoning=None,
        )

    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    message = getattr(choice, "message", None)

    tagged_reasoning, content = _split_reasoning(getattr(message, "content", None))
    reasoning = structured_reasoning_text(message) or tagged_reasoning
    if content:
        blocks.append(TextBlock(text=content))

    refusal = getattr(message, "refusal", None)
    if refusal and refusal != content:
        blocks.append(TextBlock(text=refusal))

    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        blocks.append(
            build_tool_use_block(
                call_id=getattr(tool_call, "id", None),
                name=getattr(function, "name", None) if function else None,
                raw_arguments=(
                    getattr(function, "arguments", None) if function else None
                ),
                logger=logger,
            )
        )

    legacy_function = getattr(message, "function_call", None)
    if not tool_calls and legacy_function is not None:
        blocks.append(
            build_tool_use_block(
                call_id=None,
                name=getattr(legacy_function, "name", None),
                raw_arguments=getattr(legacy_function, "arguments", None),
                logger=logger,
            )
        )

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
            has_refusal=bool(refusal),
        ),
        raw_stop_reason=raw_stop_reason(finish_reason),
        model=getattr(response, "model", None) or default_model,
        usage=usage_from_raw(getattr(response, "usage", None)),
        reasoning=reasoning,
    )


def usage_from_raw(raw_usage: Any) -> CompletionUsage:
    """Map OpenAI usage onto the provider-neutral usage schema."""
    if raw_usage is None:
        return CompletionUsage()
    return CompletionUsage(
        input_tokens=coerce_usage_count(getattr(raw_usage, "prompt_tokens", 0)),
        output_tokens=coerce_usage_count(
            getattr(raw_usage, "completion_tokens", 0)
        ),
        total_tokens=coerce_usage_count(getattr(raw_usage, "total_tokens", 0)),
    )
