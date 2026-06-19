# llm/providers/openai.py
"""
OpenAI-compatible provider: an LLMClient backed by the official `openai` SDK
(AsyncOpenAI).

Because the SDK targets the OpenAI wire protocol, this one client serves real
OpenAI *and* any OpenAI-compatible server (notably a local Ollama instance) by
pointing `base_url` at the alternate endpoint -- e.g.
`http://localhost:11434/v1`. The provider name in the registry is `openai`; the
endpoint is selected via Settings.openai_base_url (empty = real OpenAI).

This module owns every OpenAI-specific concern -- request shaping, response
parsing, transient-error classification, and the `openai` import itself. It is
imported lazily by the provider registry in `llm/client.py` so the LLM layer's
abstraction never pulls in the SDK.

Manual function calling: tool calls are returned to the agent loop as
ToolUseBlocks; we never let the SDK execute tools (the loop is the orchestrator,
per CLAUDE.md).

Ollama / reasoning-model notes:
  - Tool calling only works with tools-capable models (llama3.1, qwen2.5/3, ...).
    A model without tool support simply never emits tool_calls.
  - response_schema is wired (response_format) but only best-effort against
    Ollama; the agent loop never sends it.
  - thinking_level has no OpenAI-compatible-for-Ollama equivalent, so it is
    ignored (the LLMClient contract permits this).
  - Reasoning models (e.g. qwen3) surface chain-of-thought either inline in
    message.content as a leading <think>...</think> block or in a separate
    reasoning field. We strip a leading <think> block from content and ignore
    any separate reasoning field, so reasoning never pollutes the TextBlock or
    the replayed history.
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
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

logger = logging.getLogger(__name__)

# Matches a single leading <think>...</think> block (and trailing whitespace),
# the way Qwen3 and similar reasoning models inline chain-of-thought when an
# OpenAI-compatible server doesn't split it into a separate field.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _canonical_stop_reason(finish_reason: Any) -> str | None:
    """Normalize an OpenAI finish_reason into the harness's canonical vocabulary.

    OpenAI emits "stop", "tool_calls", "length", "content_filter", etc. The
    agent loop reads stop_reason to detect truncation, so we map to the tokens
    documented on AssistantMessage: "end_turn" (natural stop or a tool call) and
    "max_tokens" (truncation). Anything else passes through lowercased. None
    passes through. The no-choices case is handled separately as "empty".
    """
    if finish_reason is None:
        return None
    name = str(finish_reason)
    if name in ("stop", "tool_calls"):
        return "end_turn"
    if name == "length":
        return "max_tokens"
    return name.lower()


def _strip_reasoning(text: str | None) -> str:
    """Remove a leading <think>...</think> block from content, if present."""
    if not text:
        return ""
    return _THINK_BLOCK.sub("", text, count=1)


class OpenAILLMClient(LLMClient):
    """LLMClient implementation backed by the openai AsyncOpenAI SDK."""

    def __init__(
        self,
        api_key: str,
        model: str,
        default_max_tokens: int = 4096,
        base_url: str | None = None,
    ) -> None:
        # Lazy import so importing the LLM layer never drags in the SDK.
        from openai import AsyncOpenAI

        # base_url=None lets the SDK target real OpenAI; a value (e.g. Ollama's
        # /v1) points it at an OpenAI-compatible server. The SDK requires a
        # non-empty api_key string even when the server ignores it.
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        self._default_max_tokens = default_max_tokens

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
        if oai_tools:
            request["tools"] = oai_tools
            # We orchestrate tool calls ourselves in the agent loop; "auto"
            # lets the model decide whether to call, never forces it.
            request["tool_choice"] = "auto"

        # Structured output. Tools and structured output are typically mutually
        # exclusive; the orchestrator never passes both, but be defensive.
        if response_schema is not None:
            if oai_tools:
                logger.warning(
                    "OpenAI call received both tools and response_schema; "
                    "tools will be ignored in structured-output mode."
                )
                request.pop("tools", None)
                request.pop("tool_choice", None)
            request["response_format"] = self._response_format(response_schema)

        # thinking_level has no OpenAI-compatible-for-Ollama mapping; ignore it.
        if thinking_level is not None:
            logger.debug("ignoring thinking_level=%r (unsupported)", thinking_level)

        logger.debug(
            "openai complete: model=%s messages=%d tools=%d schema=%s",
            self._model, len(oai_messages), len(oai_tools or []),
            response_schema.__name__ if response_schema else None,
        )

        response = await self._client.chat.completions.create(**request)
        return self._from_openai_response(response)

    # HTTP statuses worth retrying: request timeout, rate limit, and the
    # transient 5xx family.
    _RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

    def is_transient_error(self, exc: BaseException) -> bool:
        """Retry rate limits, transient 5xx, and timeouts/connection resets."""
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        # Lazy import so error classification doesn't force the SDK at module load.
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

    # ----- request translation -----

    def _to_openai_messages(
        self, messages: list[Message], system: str | None
    ) -> list[dict[str, Any]]:
        """Translate internal Message list to OpenAI's chat message list.

        The `system` argument becomes a leading system-role message. Internal
        Role.SYSTEM messages in history are logged and skipped (system text is
        passed via the `system` param, mirroring the Gemini client).
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
                # Each tool result becomes its own tool-role message keyed by
                # the originating tool_call id.
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
                    # content must be present; null is valid when only tools.
                    "content": "".join(text_parts) or None,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                out.append(assistant_msg)
                continue

            # USER (and any other) role: flatten text blocks.
            text_parts = [
                block.text
                for block in msg.content
                if isinstance(block, TextBlock) and block.text
            ]
            out.append({"role": "user", "content": "".join(text_parts)})

        return out

    def _to_openai_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate the generic tool list into OpenAI function-tool dicts.

        MCP gives a JSON Schema dict in input_schema; OpenAI's function
        `parameters` accepts that directly, so no reshape is needed.
        """
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
        """Build a response_format for structured output.

        Prefer a json_schema (strict OpenAI mode) derived from the Pydantic
        model; fall back to plain json_object if the schema can't be produced
        (e.g. against a server that only supports json_object).
        """
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

    # ----- response translation -----

    def _from_openai_response(self, response: Any) -> AssistantMessage:
        """Convert an openai ChatCompletion into AssistantMessage."""
        blocks: list[Any] = []

        choices = getattr(response, "choices", None) or []
        if not choices:
            return AssistantMessage(
                content=[], stop_reason="empty", model=self._model,
                usage=self._usage_from_response(response),
            )

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)

        # Text content (strip a leading <think>...</think> reasoning block).
        content = _strip_reasoning(getattr(message, "content", None))
        if content:
            blocks.append(TextBlock(text=content))

        # Tool calls. OpenAI supplies a real id, so no synthetic minting.
        for tc in getattr(message, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            raw_args = getattr(fn, "arguments", None) if fn else None
            try:
                args = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError):
                logger.warning("could not parse tool arguments: %r", raw_args)
                args = {}
            blocks.append(
                ToolUseBlock(
                    id=getattr(tc, "id", "") or "",
                    name=getattr(fn, "name", "") if fn else "",
                    input=args,
                )
            )

        return AssistantMessage(
            content=blocks,
            stop_reason=_canonical_stop_reason(finish_reason),
            model=getattr(response, "model", None) or self._model,
            usage=self._usage_from_response(response),
        )

    def _usage_from_response(self, response: Any) -> Usage:
        """Map openai usage onto our provider-agnostic Usage.

        OpenAI-compatible servers report prompt/completion/total tokens; there
        are no separate thinking/cached fields in the basic shape, so those
        stay 0.
        """
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
