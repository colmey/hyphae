# llm/providers/gemini.py
"""
Gemini provider: an LLMClient backed by the google-genai SDK.

This module owns every Gemini-specific concern -- request shaping, response
parsing, transient-error classification, and the google-genai imports
themselves. It is imported lazily by the provider registry in `llm/client.py`
so the LLM layer's abstraction never pulls in the google SDK.

Manual function calling: automatic function calling is disabled so the agent
loop stays the orchestrator (see CLAUDE.md).

Structured output (response_schema): wired to GenerateContentConfig.
response_schema + response_mime_type="application/json".
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

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


def _canonical_stop_reason(finish_reason: Any) -> str | None:
    """Normalize a Gemini FinishReason into the harness's canonical vocabulary.

    Gemini's FinishReason is a str-enum whose str() is verbose
    ("FinishReason.MAX_TOKENS"). The agent loop reads stop_reason to detect
    truncation, so we map to the canonical tokens documented on
    AssistantMessage: "end_turn" (natural stop), "max_tokens" (truncation),
    and the provider's lowercased name for everything else (safety, recitation,
    ...). None passes through. The no-candidates case is handled separately as
    "empty".
    """
    if finish_reason is None:
        return None
    name = getattr(finish_reason, "name", None) or str(finish_reason)
    if name == "STOP":
        return "end_turn"
    if name == "MAX_TOKENS":
        return "max_tokens"
    return name.lower()


class GeminiLLMClient(LLMClient):
    """LLMClient implementation backed by the google-genai SDK."""

    def __init__(self, api_key: str, model: str, default_max_tokens: int = 4096) -> None:
        # google-genai picks up GEMINI_API_KEY automatically, but we pass
        # explicitly so we fail fast if the bootstrap script didn't run.
        self._client = genai.Client(api_key=api_key)
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
        contents = self._to_genai_contents(messages)
        genai_tools = self._to_genai_tools(tools) if tools else None

        # Build config kwargs incrementally so structured-output mode is
        # easy to opt into without disturbing the normal path.
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens or self._default_max_tokens,
            "tools": genai_tools,
            # We orchestrate tool calls ourselves in the agent loop.
            "automatic_function_calling": genai_types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
            "system_instruction": system,
        }

        # Thinking budget. Gemini 3 exposes a native thinking_level
        # ("LOW"/"MEDIUM"/"HIGH"); we map our lowercase tier onto it. None
        # leaves the model default untouched (legacy path, orchestrator's own
        # call). An unrecognized value is logged and skipped rather than
        # failing the request.
        if thinking_level is not None:
            try:
                config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_level=genai_types.ThinkingLevel(thinking_level.upper()),
                )
            except ValueError:
                logger.warning(
                    "ignoring unrecognized thinking_level %r", thinking_level
                )

        # Structured output. Tools and structured output are typically
        # mutually exclusive in provider SDKs -- the orchestrator never
        # passes both, but be defensive.
        if response_schema is not None:
            if genai_tools:
                logger.warning(
                    "Gemini call received both tools and response_schema; "
                    "tools will be ignored in structured-output mode."
                )
                config_kwargs["tools"] = None
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        config = genai_types.GenerateContentConfig(**config_kwargs)

        logger.debug(
            "gemini complete: model=%s messages=%d tools=%s schema=%s thinking=%s",
            self._model, len(contents),
            len(genai_tools[0].function_declarations) if genai_tools else 0,
            response_schema.__name__ if response_schema else None,
            thinking_level,
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        return self._from_genai_response(response)

    # HTTP statuses worth retrying: request timeout, rate limit, and the
    # transient 5xx family.
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

    # ----- request translation -----

    def _to_genai_contents(self, messages: list[Message]) -> list[genai_types.Content]:
        """Translate internal Message list to Gemini's Content list.

        Gemini roles are "user" and "model". Tool results are sent as "user"
        Content with function_response parts. Internal Role.SYSTEM messages
        are NOT included here -- the agent passes system text via the
        `system` argument, which we map to system_instruction.
        """
        out: list[genai_types.Content] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                # System messages should be passed via the `system` param.
                # If one shows up here, log and skip rather than crashing.
                logger.warning("system message in history was ignored; use the system= param")
                continue

            parts: list[genai_types.Part] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    if block.text:
                        part_kwargs: dict[str, Any] = {"text": block.text}
                        # Gemini 3+ recommends round-tripping thought
                        # signatures on every Part type; required for
                        # function_call parts (see ToolUseBlock branch).
                        sig = block.provider_metadata.get("thought_signature")
                        if sig is not None:
                            part_kwargs["thought_signature"] = sig
                        parts.append(genai_types.Part(**part_kwargs))
                elif isinstance(block, ToolUseBlock):
                    # Gemini 3+ requires the original thought_signature to be
                    # echoed back here, or the model returns 400. We stashed
                    # it in provider_metadata on parse.
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
                    # Gemini's function_response carries a name (matching the
                    # original function_call) and a response dict. We surface
                    # tool content under "content" and error state under
                    # "error" so the model can see both.
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

            # Map roles: USER -> "user", ASSISTANT -> "model", TOOL -> "user"
            # (Gemini sends function_responses as user-role content).
            gemini_role = "model" if msg.role == Role.ASSISTANT else "user"
            out.append(genai_types.Content(role=gemini_role, parts=parts))

        return out

    def _to_genai_tools(self, tools: list[dict[str, Any]]) -> list[genai_types.Tool]:
        """Translate the generic tool list into a single Gemini Tool object.

        Gemini accepts a list of Tool objects, each containing a list of
        function declarations. We put all our function declarations into one
        Tool for simplicity.
        """
        declarations = [
            genai_types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                # MCP gives us a JSON Schema dict in input_schema; Gemini's
                # parameters_json_schema accepts that directly. No reshape.
                parameters_json_schema=t.get("input_schema") or {
                    "type": "object",
                    "properties": {},
                },
            )
            for t in tools
        ]
        return [genai_types.Tool(function_declarations=declarations)]

    # ----- response translation -----

    def _from_genai_response(self, response: Any) -> AssistantMessage:
        """Convert a google-genai GenerateContentResponse into AssistantMessage."""
        blocks: list[Any] = []

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return AssistantMessage(
                content=[], stop_reason="empty", model=self._model,
                usage=self._usage_from_response(response),
            )

        candidate = candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None

        for part in parts or []:
            # Gemini 3+ may attach a thought_signature to any Part. Capture
            # it so we can round-trip it back on the next turn (required for
            # function_call parts; recommended for text parts).
            signature = getattr(part, "thought_signature", None)
            metadata: dict[str, Any] = {}
            if signature is not None:
                metadata["thought_signature"] = signature

            # Text part
            text = getattr(part, "text", None)
            if text:
                blocks.append(TextBlock(text=text, provider_metadata=metadata))
                continue

            # Function call part
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                # Gemini doesn't provide a call ID, so we mint one. The name
                # is preserved separately on the ToolUseBlock, and the agent
                # loop will echo it back on the matching ToolResultBlock so
                # the function_response round-trip works.
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

            # Other part types (thought signatures, executable code, etc.)
            # are not handled in v1. Log and skip.
            logger.debug("unhandled gemini part type: %r", part)

        return AssistantMessage(
            content=blocks,
            stop_reason=_canonical_stop_reason(finish_reason),
            model=self._model,
            usage=self._usage_from_response(response),
        )

    def _usage_from_response(self, response: Any) -> Usage:
        """Map google-genai usage_metadata onto our provider-agnostic Usage."""
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return Usage()

        def _n(v: Any) -> int:
            return int(v) if v else 0

        return Usage(
            input_tokens=_n(getattr(um, "prompt_token_count", 0)),
            output_tokens=_n(getattr(um, "candidates_token_count", 0)),
            total_tokens=_n(getattr(um, "total_token_count", 0)),
            thinking_tokens=_n(getattr(um, "thoughts_token_count", 0)),
            cached_tokens=_n(getattr(um, "cached_content_token_count", 0)),
        )
