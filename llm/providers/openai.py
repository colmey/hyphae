# llm/providers/openai.py
"""OpenAI-compatible LLMClient backed by the official `openai` SDK.

The same provider serves real OpenAI and local OpenAI-compatible servers via
`base_url`. Tool calls are returned to the agent loop; the SDK never executes
them. Leading `<think>...</think>` blocks are stripped from reasoning models so
chain-of-thought is not replayed as assistant content.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

from llm.client import LLMClient
from llm.schemas import (
    AssistantMessage,
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
)

logger = logging.getLogger(__name__)

# Some OpenAI-compatible reasoning models inline a leading thought block.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _canonical_stop_reason(finish_reason: Any) -> str | None:
    """Normalize an OpenAI finish_reason into the harness vocabulary."""
    if finish_reason is None:
        return None
    name = str(finish_reason)
    if name in ("stop", "tool_calls"):
        return "end_turn"
    if name == "length":
        return "max_tokens"
    return name.lower()


def _split_reasoning(text: str | None) -> tuple[str | None, str]:
    """Return provider-emitted reasoning and visible content separately."""
    if not text:
        return None, ""
    m = _THINK_BLOCK.match(text)
    if not m:
        return None, text
    reasoning = re.sub(r"</?think>", "", m.group(0)).strip() or None
    return reasoning, text[m.end():]


class _ReasoningStreamStripper:
    """Incrementally strip one leading <think>...</think> block."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

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
            out = self._raw_prefix
            self._pending = ""
            self._reasoning_parts.clear()
            self._state = "pass"
            return out
        return ""

    def _feed_pending(self, piece: str) -> str:
        self._pending += piece
        candidate = self._pending.lstrip()

        if not candidate:
            return ""
        if candidate.startswith(self._OPEN):
            rest = candidate[len(self._OPEN):]
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
        rest = self._pending[close_at + len(self._CLOSE):]
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
        default_max_tokens: int = 4096,
        base_url: str | None = None,
        profile: ModelProfile | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        # The SDK requires a non-empty api_key even for local servers.
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        self._default_max_tokens = default_max_tokens
        self._profile = profile or ModelProfile.default()
        self._warned_inert_thinking = False

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        response_schema: type | None = None,
        thinking_level: str | None = None,
    ) -> AssistantMessage:
        request = self._build_request(
            messages=messages,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
            response_schema=response_schema,
            thinking_level=thinking_level,
        )

        logger.debug(
            "openai complete: model=%s messages=%d tools=%d schema=%s",
            self._model, len(request["messages"]), len(request.get("tools") or []),
            response_schema.__name__ if response_schema else None,
        )

        response = await self._client.chat.completions.create(**request)
        return self._from_openai_response(response)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        thinking_level: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        request = self._build_request(
            messages=messages,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
            response_schema=None,
            thinking_level=thinking_level,
        )
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}

        logger.debug(
            "openai stream: model=%s messages=%d tools=%d",
            self._model, len(request["messages"]), len(request.get("tools") or []),
        )

        stripper = _ReasoningStreamStripper()
        text_parts: list[str] = []
        tool_accs: dict[int, dict[str, str]] = {}
        finish_reason: Any = None
        raw_usage: Any = None
        response_model: str | None = None

        sdk_stream = await self._client.chat.completions.create(**request)
        async for chunk in sdk_stream:
            response_model = getattr(chunk, "model", None) or response_model
            if getattr(chunk, "usage", None) is not None:
                raw_usage = chunk.usage

            for choice in getattr(chunk, "choices", None) or []:
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

                for tc in getattr(delta, "tool_calls", None) or []:
                    index = int(getattr(tc, "index", 0) or 0)
                    acc = tool_accs.setdefault(index, {"id": "", "name": "", "args": ""})
                    if getattr(tc, "id", None):
                        acc["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn and getattr(fn, "name", None):
                        acc["name"] = fn.name
                    if fn and getattr(fn, "arguments", None):
                        acc["args"] += fn.arguments

        tail = stripper.finish()
        if tail:
            text_parts.append(tail)
            yield TextDelta(text=tail)

        blocks: list[Any] = []
        full_text = "".join(text_parts)
        if full_text:
            blocks.append(TextBlock(text=full_text))

        for index in sorted(tool_accs):
            acc = tool_accs[index]
            raw_args = acc["args"]
            try:
                args = json.loads(raw_args) if raw_args else {}
                parse_error = None
            except (json.JSONDecodeError, TypeError):
                logger.warning("could not parse streamed tool arguments: %r", raw_args)
                args = {}
                parse_error = f"arguments were not valid JSON: {raw_args!r}"
            blocks.append(
                ToolUseBlock(
                    id=acc["id"],
                    name=acc["name"],
                    input=args,
                    parse_error=parse_error,
                )
            )

        yield StreamEnd(
            message=AssistantMessage(
                content=blocks,
                stop_reason=_canonical_stop_reason(finish_reason),
                model=response_model or self._model,
                usage=self._usage_from_raw(raw_usage),
                reasoning=stripper.reasoning,
            )
        )

    def _build_request(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        system: str | None,
        max_tokens: int | None,
        response_schema: type | None,
        thinking_level: str | None,
    ) -> dict[str, Any]:
        oai_messages = self._to_openai_messages(messages, system)
        oai_tools = self._to_openai_tools(tools) if tools else None

        request: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "max_tokens": max_tokens or self._default_max_tokens,
        }
        p = self._profile
        if p.temperature is not None:
            request["temperature"] = p.temperature
        if p.top_p is not None:
            request["top_p"] = p.top_p
        if p.top_k is not None:
            request["top_k"] = p.top_k
        if oai_tools:
            request["tools"] = oai_tools
            request["tool_choice"] = "auto"

        # Structured output and tools are typically mutually exclusive.
        if response_schema is not None:
            if oai_tools:
                logger.warning(
                    "OpenAI call received both tools and response_schema; "
                    "tools will be ignored in structured-output mode."
                )
                request.pop("tools", None)
                request.pop("tool_choice", None)
            request["response_format"] = self._response_format(response_schema)

        if thinking_level is not None:
            if p.thinking == "hint-param":
                request["reasoning_effort"] = thinking_level
            elif p.thinking == "none" and not self._warned_inert_thinking:
                logger.info(
                    "thinking_level=%r requested but model %s declares thinking:none; "
                    "the knob is inert for this model",
                    thinking_level,
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
        self, messages: list[Message], system: str | None
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
                        out.append({
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": block.content,
                        })
                continue

            if msg.role == Role.ASSISTANT:
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        if block.text:
                            text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input or {}),
                            },
                        })
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

    def _to_openai_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate generic tools into OpenAI function-tool dicts."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {
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
                content=[], stop_reason="empty", model=self._model,
                usage=self._usage_from_response(response),
                reasoning=None,
            )

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)

        reasoning, content = _split_reasoning(getattr(message, "content", None))
        if content:
            blocks.append(TextBlock(text=content))

        for tc in getattr(message, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            raw_args = getattr(fn, "arguments", None) if fn else None
            try:
                args = json.loads(raw_args) if raw_args else {}
                parse_error = None
            except (json.JSONDecodeError, TypeError):
                logger.warning("could not parse tool arguments: %r", raw_args)
                args = {}
                parse_error = f"arguments were not valid JSON: {raw_args!r}"
            blocks.append(
                ToolUseBlock(
                    id=getattr(tc, "id", "") or "",
                    name=getattr(fn, "name", "") if fn else "",
                    input=args,
                    parse_error=parse_error,
                )
            )

        return AssistantMessage(
            content=blocks,
            stop_reason=_canonical_stop_reason(finish_reason),
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

        def _n(v: Any) -> int:
            return int(v) if v else 0

        return Usage(
            input_tokens=_n(getattr(u, "prompt_tokens", 0)),
            output_tokens=_n(getattr(u, "completion_tokens", 0)),
            total_tokens=_n(getattr(u, "total_tokens", 0)),
        )
