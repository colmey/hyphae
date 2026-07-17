# llm/providers/openai.py
"""OpenAI-compatible LLMClient backed by the official `openai` SDK.

The same provider serves real OpenAI and local OpenAI-compatible servers via
`base_url`. Tool calls are returned to the agent loop; the SDK never executes
them. Leading `<think>...</think>` blocks are stripped from reasoning models so
chain-of-thought is not replayed as assistant content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, AsyncIterator

from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    CanonicalStopReason,
    Message,
    ModelProfile,
    Role,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    coerce_usage_count,
)

logger = logging.getLogger(__name__)

# Some OpenAI-compatible reasoning models inline a leading thought block.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


_STOP_REASON_MAP: dict[str, CanonicalStopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
}


def _raw_stop_reason(finish_reason: Any) -> str | None:
    return None if finish_reason is None else str(finish_reason)


def _canonical_stop_reason(
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

    raw_reason = _raw_stop_reason(finish_reason)
    if raw_reason == "stop" and not has_visible_content:
        return "empty"
    canonical = _STOP_REASON_MAP.get(raw_reason or "")
    if canonical == "tool_use" or canonical is None:
        return "provider_error"
    return canonical


def _tool_call_id(raw_id: Any) -> str:
    value = str(raw_id).strip() if raw_id is not None else ""
    return value or f"call_{uuid.uuid4().hex[:12]}"


def _tool_use_block(*, call_id: Any, name: Any, raw_arguments: Any) -> ToolUseBlock:
    """Finalize one provider call into the canonical, replayable tool shape."""
    try:
        arguments = json.loads(raw_arguments) if raw_arguments else {}
        parse_error = None
    except (json.JSONDecodeError, TypeError):
        logger.warning("could not parse tool arguments: %r", raw_arguments)
        arguments = {}
        parse_error = f"arguments were not valid JSON: {raw_arguments!r}"
    return ToolUseBlock(
        id=_tool_call_id(call_id),
        name=str(name) if name is not None else "",
        input=arguments,
        parse_error=parse_error,
    )


@dataclass
class _ToolCallAccumulator:
    """Collect streamed call fragments and finalize exactly once."""

    call_id: str = ""
    name: str = ""
    argument_parts: list[str] = field(default_factory=list)

    def update(self, *, call_id: Any, name: Any, arguments: Any) -> None:
        if call_id:
            self.call_id = str(call_id)
        if name:
            self.name = str(name)
        if arguments:
            self.argument_parts.append(str(arguments))

    def finalize(self) -> ToolUseBlock:
        return _tool_use_block(
            call_id=self.call_id,
            name=self.name,
            raw_arguments="".join(self.argument_parts),
        )


def _merge_extra_body(request: dict[str, Any], extensions: dict[str, Any]) -> None:
    """Merge compatible-endpoint extensions without replacing existing values."""
    merged = dict(request.get("extra_body") or {})
    for key, value in extensions.items():
        merged.setdefault(key, value)
    request["extra_body"] = merged


def _split_reasoning(text: str | None) -> tuple[str | None, str]:
    """Return provider-emitted reasoning and visible content separately."""
    if not text:
        return None, ""
    m = _THINK_BLOCK.match(text)
    if not m:
        candidate = text.lstrip()
        if candidate.startswith(_THINK_OPEN):
            reasoning = candidate[len(_THINK_OPEN) :].strip()
            return reasoning or None, ""
        return None, text
    reasoning = re.sub(r"</?think>", "", m.group(0)).strip() or None
    return reasoning, text[m.end() :]


class _ReasoningStreamStripper:
    """Incrementally strip one leading <think>...</think> block."""

    _OPEN = _THINK_OPEN
    _CLOSE = _THINK_CLOSE

    def __init__(self) -> None:
        self._state = "pending"
        self._pending = ""
        self._reasoning_parts: list[str] = []
        self._raw_prefix = ""

    @property
    def reasoning(self) -> str | None:
        text = "".join(self._reasoning_parts).strip()
        return text or None

    def feed(self, piece: str) -> str:
        if not piece:
            return ""
        if self._state == "pass":
            return piece
        self._raw_prefix += piece
        if self._state == "pending":
            return self._feed_pending(piece)
        if self._state == "reasoning":
            return self._feed_reasoning(piece)
        if self._state == "after_reasoning":
            return self._feed_after_reasoning(piece)
        return piece

    def finish(self) -> str:
        """Flush buffered visible text if the stream ended before a think decision."""
        if self._state == "pending":
            out = self._pending
            self._pending = ""
            self._state = "pass"
            return out
        if self._state == "reasoning":
            self._reasoning_parts.append(self._pending)
            self._pending = ""
            self._state = "pass"
            return ""
        return ""

    def _feed_pending(self, piece: str) -> str:
        self._pending += piece
        candidate = self._pending.lstrip()

        if not candidate:
            return ""
        if candidate.startswith(self._OPEN):
            rest = candidate[len(self._OPEN) :]
            self._pending = ""
            self._state = "reasoning"
            return self._feed_reasoning(rest)
        if self._OPEN.startswith(candidate):
            return ""

        out = self._pending
        self._pending = ""
        self._state = "pass"
        return out

    def _feed_reasoning(self, piece: str) -> str:
        self._pending += piece
        close_at = self._pending.find(self._CLOSE)
        if close_at == -1:
            keep = max(0, len(self._pending) - (len(self._CLOSE) - 1))
            if keep:
                self._reasoning_parts.append(self._pending[:keep])
                self._pending = self._pending[keep:]
            return ""

        self._reasoning_parts.append(self._pending[:close_at])
        rest = self._pending[close_at + len(self._CLOSE) :]
        self._pending = ""
        self._state = "after_reasoning"
        return self._feed_after_reasoning(rest)

    def _feed_after_reasoning(self, piece: str) -> str:
        visible = piece.lstrip()
        if not visible:
            return ""
        self._state = "pass"
        return visible


class OpenAILLMClient(LLMClient):
    """LLMClient implementation backed by the openai AsyncOpenAI SDK."""

    def __init__(
        self,
        api_key: str,
        model: str,
        default_max_tokens: int,
        base_url: str | None = None,
        profile: ModelProfile | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        # The SDK requires a non-empty api_key even for local servers.
        resolved_base_url = base_url or None
        self._client = AsyncOpenAI(api_key=api_key, base_url=resolved_base_url)
        self._compatible_endpoint = resolved_base_url is not None
        self._model = model
        self._default_max_tokens = default_max_tokens
        self._profile = profile or ModelProfile.default()
        self._warned_inert_thinking = False

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        sdk_request = self._build_request(request)

        logger.debug(
            "openai complete: model=%s messages=%d tools=%d schema=%s",
            self._model,
            len(sdk_request["messages"]),
            len(sdk_request.get("tools") or []),
            request.response_schema.__name__ if request.response_schema else None,
        )

        response = await self._client.chat.completions.create(**sdk_request)
        return self._from_openai_response(response)

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        if request.response_schema is not None:
            raise ValueError("response_schema is not supported for streaming")
        sdk_request = self._build_request(request)
        sdk_request["stream"] = True
        sdk_request["stream_options"] = {"include_usage": True}

        logger.debug(
            "openai stream: model=%s messages=%d tools=%d",
            self._model,
            len(sdk_request["messages"]),
            len(sdk_request.get("tools") or []),
        )

        stripper = _ReasoningStreamStripper()
        text_parts: list[str] = []
        tool_accs: dict[int, _ToolCallAccumulator] = {}
        legacy_tool_acc: _ToolCallAccumulator | None = None
        has_refusal = False
        saw_choice = False
        finish_reason: Any = None
        raw_usage: Any = None
        response_model: str | None = None

        sdk_stream = await self._client.chat.completions.create(**sdk_request)
        async for chunk in sdk_stream:
            response_model = getattr(chunk, "model", None) or response_model
            if getattr(chunk, "usage", None) is not None:
                raw_usage = chunk.usage

            for choice in getattr(chunk, "choices", None) or []:
                saw_choice = True
                if getattr(choice, "finish_reason", None) is not None:
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                piece = getattr(delta, "content", None)
                if piece:
                    visible = stripper.feed(piece)
                    if visible:
                        text_parts.append(visible)
                        yield TextDelta(text=visible)

                refusal_piece = getattr(delta, "refusal", None)
                if refusal_piece:
                    has_refusal = True
                    text_parts.append(refusal_piece)
                    yield TextDelta(text=refusal_piece)

                for tc in getattr(delta, "tool_calls", None) or []:
                    index = int(getattr(tc, "index", 0) or 0)
                    acc = tool_accs.setdefault(index, _ToolCallAccumulator())
                    fn = getattr(tc, "function", None)
                    acc.update(
                        call_id=getattr(tc, "id", None),
                        name=getattr(fn, "name", None) if fn else None,
                        arguments=getattr(fn, "arguments", None) if fn else None,
                    )

                legacy_fn = getattr(delta, "function_call", None)
                if legacy_fn is not None:
                    if legacy_tool_acc is None:
                        legacy_tool_acc = _ToolCallAccumulator()
                    legacy_tool_acc.update(
                        call_id=None,
                        name=getattr(legacy_fn, "name", None),
                        arguments=getattr(legacy_fn, "arguments", None),
                    )

        tail = stripper.finish()
        if tail:
            text_parts.append(tail)
            yield TextDelta(text=tail)

        blocks: list[Any] = []
        full_text = "".join(text_parts)
        if full_text:
            blocks.append(TextBlock(text=full_text))

        for index in sorted(tool_accs):
            blocks.append(tool_accs[index].finalize())

        # Prefer the modern representation if a compatible server emits both.
        if not tool_accs and legacy_tool_acc is not None:
            blocks.append(legacy_tool_acc.finalize())

        has_tools = any(isinstance(block, ToolUseBlock) for block in blocks)
        has_visible_content = any(
            isinstance(block, TextBlock) and bool(block.text) for block in blocks
        )
        raw_stop_reason = _raw_stop_reason(finish_reason)
        stop_reason = (
            _canonical_stop_reason(
                finish_reason,
                has_tools=has_tools,
                has_visible_content=has_visible_content,
                has_refusal=has_refusal,
            )
            if saw_choice
            else "empty"
        )

        yield StreamEnd(
            message=AssistantMessage(
                content=blocks,
                stop_reason=stop_reason,
                raw_stop_reason=raw_stop_reason,
                model=response_model or self._model,
                usage=self._usage_from_raw(raw_usage),
                reasoning=stripper.reasoning,
            )
        )

    def _build_request(self, generation: GenerationRequest) -> dict[str, Any]:
        oai_messages = self._to_openai_messages(
            generation.messages, generation.system
        )
        oai_tools = self._to_openai_tools(generation.tools) if generation.tools else None

        request: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
        }
        token_limit = generation.max_tokens or self._default_max_tokens
        if self._compatible_endpoint:
            request["max_tokens"] = token_limit
        else:
            request["max_completion_tokens"] = token_limit
        p = self._profile
        if p.temperature is not None:
            request["temperature"] = p.temperature
        if p.top_p is not None:
            request["top_p"] = p.top_p
        if self._compatible_endpoint and p.top_k is not None:
            _merge_extra_body(request, {"top_k": p.top_k})
        if oai_tools:
            request["tools"] = oai_tools
            request["tool_choice"] = "auto"

        # Structured output and tools are typically mutually exclusive.
        if generation.response_schema is not None:
            if oai_tools:
                logger.warning(
                    "OpenAI call received both tools and response_schema; "
                    "tools will be ignored in structured-output mode."
                )
                request.pop("tools", None)
                request.pop("tool_choice", None)
            request["response_format"] = self._response_format(
                generation.response_schema
            )

        if generation.thinking_level is not None:
            if p.thinking == "hint-param":
                request["reasoning_effort"] = generation.thinking_level
            elif p.thinking == "none" and not self._warned_inert_thinking:
                logger.info(
                    "thinking_level=%r requested but model %s declares thinking:none; "
                    "the knob is inert for this model",
                    generation.thinking_level,
                    self._model,
                )
                self._warned_inert_thinking = True

        return request

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

    def _to_openai_messages(
        self, messages: Sequence[Message], system: str | None
    ) -> list[dict[str, Any]]:
        """Translate internal Message list to OpenAI's chat message list.

        System text is passed through the `system` arg; Role.SYSTEM history is
        ignored so providers share one convention.
        """
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == Role.SYSTEM:
                logger.warning(
                    "system message in history was ignored; use the system= param"
                )
                continue

            if msg.role == Role.TOOL:
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.tool_use_id,
                                "content": block.content,
                            }
                        )
                continue

            if msg.role == Role.ASSISTANT:
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for block in msg.content:
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
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                out.append(assistant_msg)
                continue

            text_parts = [
                block.text
                for block in msg.content
                if isinstance(block, TextBlock) and block.text
            ]
            out.append({"role": "user", "content": "".join(text_parts)})

        return out

    def _to_openai_tools(
        self, tools: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Translate generic tools into OpenAI function-tool dicts."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema")
                    or {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            for t in tools
        ]

    def _response_format(self, response_schema: type) -> dict[str, Any]:
        """Build a response_format for structured output."""
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

    def _from_openai_response(self, response: Any) -> AssistantMessage:
        """Convert an openai ChatCompletion into AssistantMessage."""
        blocks: list[Any] = []

        choices = getattr(response, "choices", None) or []
        if not choices:
            return AssistantMessage(
                content=[],
                stop_reason="empty",
                raw_stop_reason=None,
                model=self._model,
                usage=self._usage_from_response(response),
                reasoning=None,
            )

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)

        reasoning, content = _split_reasoning(getattr(message, "content", None))
        if content:
            blocks.append(TextBlock(text=content))

        refusal = getattr(message, "refusal", None)
        if refusal and refusal != content:
            blocks.append(TextBlock(text=refusal))

        tool_calls = getattr(message, "tool_calls", None) or []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            blocks.append(
                _tool_use_block(
                    call_id=getattr(tc, "id", None),
                    name=getattr(fn, "name", None) if fn else None,
                    raw_arguments=getattr(fn, "arguments", None) if fn else None,
                )
            )

        legacy_fn = getattr(message, "function_call", None)
        if not tool_calls and legacy_fn is not None:
            blocks.append(
                _tool_use_block(
                    call_id=None,
                    name=getattr(legacy_fn, "name", None),
                    raw_arguments=getattr(legacy_fn, "arguments", None),
                )
            )

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
                has_refusal=bool(refusal),
            ),
            raw_stop_reason=raw_stop_reason,
            model=getattr(response, "model", None) or self._model,
            usage=self._usage_from_response(response),
            reasoning=reasoning,
        )

    def _usage_from_response(self, response: Any) -> Usage:
        """Map OpenAI usage onto our provider-agnostic Usage."""
        return self._usage_from_raw(getattr(response, "usage", None))

    def _usage_from_raw(self, u: Any) -> Usage:
        """Map an OpenAI usage object onto our provider-agnostic Usage."""
        if u is None:
            return Usage()

        return Usage(
            input_tokens=coerce_usage_count(getattr(u, "prompt_tokens", 0)),
            output_tokens=coerce_usage_count(getattr(u, "completion_tokens", 0)),
            total_tokens=coerce_usage_count(getattr(u, "total_tokens", 0)),
        )
