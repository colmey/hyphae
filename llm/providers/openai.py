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
from typing import Any

from llm.client import LLMClient
from llm.schemas import (
    AssistantMessage,
    Message,
    ModelProfile,
    Role,
    TextBlock,
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

        logger.debug(
            "openai complete: model=%s messages=%d tools=%d schema=%s",
            self._model, len(oai_messages), len(oai_tools or []),
            response_schema.__name__ if response_schema else None,
        )

        response = await self._client.chat.completions.create(**request)
        return self._from_openai_response(response)

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
        u = getattr(response, "usage", None)
        if u is None:
            return Usage()

        def _n(v: Any) -> int:
            return int(v) if v else 0

        return Usage(
            input_tokens=_n(getattr(u, "prompt_tokens", 0)),
            output_tokens=_n(getattr(u, "completion_tokens", 0)),
            total_tokens=_n(getattr(u, "total_tokens", 0)),
        )
