# hyphae/llm/tool_prompt_protocol.py

"""Prompt protocol adapter for models without native tool calling."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from hyphae.llm.client import GenerationRequest, LLMClient
from hyphae.llm.schemas import (
    AssistantMessage,
    Message,
    TextBlock,
    ToolUseBlock,
    CompletionUsage,
)


_ACTION_INSTRUCTIONS = """\
You can call one tool by returning exactly one JSON object in a ```json fence:
{"tool":"server__tool_name","arguments":{"arg":"value"}}

Rules:
- Call at most one tool.
- Use only a listed tool name.
- "arguments" must be a JSON object matching that tool's schema.
- If no tool is needed, answer normally and do not include action JSON.
"""

_REPAIR_INSTRUCTIONS = """\
Your previous tool action could not be parsed:
{error}

Return exactly one corrected JSON action object in a ```json fence, or answer normally
if no tool is needed.
"""

_FENCED_JSON = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class PromptedAction:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class PromptParseResult:
    action: PromptedAction | None = None
    text: str = ""
    error: str | None = None
    error_tool_name: str | None = None


def render_prompted_tools(tools: Sequence[Mapping[str, Any]]) -> str:
    """Return compact tool instructions for the model-visible system prompt."""
    if not tools:
        return ""

    lines = [_ACTION_INSTRUCTIONS.rstrip(), "", "Available tools:"]
    for tool in tools:
        name = str(tool.get("name", "")).strip()
        if not name:
            continue
        description = " ".join(str(tool.get("description") or "").split())
        schema = tool.get("input_schema") or {}
        schema_text = json.dumps(
            schema,
            separators=(",", ":"),
            ensure_ascii=False,
            sort_keys=True,
        )
        lines.append(f"- {name}: {description}")
        lines.append(f"  schema: {schema_text}")
    return "\n".join(lines).rstrip()


def parse_prompted_action(text: str, allowed_tools: set[str]) -> PromptParseResult:
    """Parse a single prompted action from model text, or return final text."""
    candidate = _extract_action_candidate(text)
    if candidate is None:
        return PromptParseResult(text=text)

    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return PromptParseResult(
            text=text,
            error=f"invalid JSON tool action: {exc.msg}",
        )

    if not isinstance(raw, dict):
        return PromptParseResult(
            text=text,
            error="tool action must be a JSON object",
        )

    name = raw.get("tool", raw.get("name"))
    if not isinstance(name, str) or not name.strip():
        return PromptParseResult(text=text, error="tool action missing tool name")
    name = name.strip()
    if name not in allowed_tools:
        return PromptParseResult(
            text=text,
            error=f"unknown prompted tool {name!r}",
            error_tool_name=name,
        )

    sentinel = object()
    arguments = raw.get("arguments", sentinel)
    if arguments is sentinel:
        arguments = raw.get("args", sentinel)
    if arguments is sentinel:
        arguments = raw.get("input", {})
    if not isinstance(arguments, dict):
        return PromptParseResult(
            text=text,
            error=f"arguments must be a JSON object for prompted tool {name!r}",
            error_tool_name=name,
        )

    return PromptParseResult(action=PromptedAction(name=name, arguments=arguments))


class PromptedToolLLMClient(LLMClient):
    """LLMClient decorator that encodes tool calling into ordinary prompt text."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        model: str | None = None,
        max_repairs: int = 1,
    ) -> None:
        self._inner = inner
        self._model = model
        self._max_repairs = max_repairs
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def inner(self) -> LLMClient:
        """Wrapped client, exposed for smoke tests and diagnostics."""
        return self._inner

    def is_transient_error(self, exc: BaseException) -> bool:
        return self._inner.is_transient_error(exc)

    async def aclose(self) -> None:
        """Join or retry closing the lifecycle-owned wrapped client."""
        if self._closed:
            return

        task = self._close_task
        if task is not None and task.done():
            if task.cancelled() or task.exception() is not None:
                self._close_task = None
                task = None
        if task is None:
            task = asyncio.create_task(
                self._finish_close(), name="prompted-tool-client-close"
            )
            self._close_task = task
            task.add_done_callback(self._close_finished)
        await asyncio.shield(task)

    async def _finish_close(self) -> None:
        await self._inner.aclose()
        self._closed = True

    def _close_finished(self, task: asyncio.Task[None]) -> None:
        """Observe failed background cleanup and leave it retryable."""
        failed = task.cancelled() or task.exception() is not None
        if failed and self._close_task is task:
            self._close_task = None

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        if not request.tools or request.response_schema is not None:
            return await self._inner.complete(request)

        tool_prompt = render_prompted_tools(request.tools)
        effective_system = _join_system(request.system, tool_prompt)
        allowed_tools = {
            str(tool.get("name", "")).strip()
            for tool in request.tools
            if str(tool.get("name", "")).strip()
        }

        prompted_request = replace(
            request,
            tools=None,
            system=effective_system,
            response_schema=None,
        )
        first = await self._inner.complete(prompted_request)
        if first.stop_reason != "end_turn":
            return first
        parsed = parse_prompted_action(_visible_text(first), allowed_tools)
        if parsed.action is not None:
            return _action_message(first, parsed.action, model=self._model)
        if parsed.error is not None and parsed.error_tool_name is not None:
            return _parse_error_tool_message(first, parsed, model=self._model)
        if parsed.error is None:
            return first
        if self._max_repairs <= 0:
            return _parse_failure_message(first, parsed.error, self._model)

        last_error = parsed.error
        usage = first.usage
        repair_messages: tuple[Message, ...] = (
            *request.messages,
            Message.assistant(first.content),
            Message.user(_REPAIR_INSTRUCTIONS.format(error=last_error)),
        )

        repair: AssistantMessage | None = None
        for _ in range(self._max_repairs):
            repair = await self._inner.complete(
                replace(prompted_request, messages=repair_messages)
            )
            usage = _combine_usage(usage, repair.usage)
            if repair.stop_reason != "end_turn":
                repair.usage = usage
                return repair
            parsed = parse_prompted_action(_visible_text(repair), allowed_tools)
            if parsed.action is not None:
                return _action_message(
                    repair, parsed.action, model=self._model, usage=usage
                )
            if parsed.error is not None and parsed.error_tool_name is not None:
                return _parse_error_tool_message(
                    repair, parsed, model=self._model, usage=usage
                )
            if parsed.error is None:
                repair.usage = usage
                return repair
            last_error = parsed.error
            repair_messages = (
                *repair_messages,
                Message.assistant(repair.content),
                Message.user(_REPAIR_INSTRUCTIONS.format(error=last_error)),
            )

        source = repair or first
        return _parse_failure_message(source, last_error, self._model, usage=usage)


def _extract_action_candidate(text: str) -> str | None:
    match = _FENCED_JSON.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    return None


def _visible_text(message: AssistantMessage) -> str:
    return "".join(
        block.text for block in message.content if isinstance(block, TextBlock)
    )


def _join_system(system: str | None, tool_prompt: str) -> str:
    if system and tool_prompt:
        return f"{system.rstrip()}\n\n{tool_prompt}"
    return system or tool_prompt


def _new_tool_id() -> str:
    return f"prompted_{uuid.uuid4().hex[:12]}"


def _action_message(
    source: AssistantMessage,
    action: PromptedAction,
    *,
    model: str | None,
    usage: CompletionUsage | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolUseBlock(
                id=_new_tool_id(),
                name=action.name,
                input=action.arguments,
            )
        ],
        stop_reason="tool_use",
        model=source.model or model,
        usage=usage if usage is not None else source.usage,
        reasoning=source.reasoning,
        raw_stop_reason=source.raw_stop_reason,
    )


def _parse_failure_message(
    source: AssistantMessage,
    error: str,
    model: str | None,
    *,
    usage: CompletionUsage | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=f"I could not parse a valid tool action: {error}")],
        stop_reason="end_turn",
        model=source.model or model,
        usage=usage if usage is not None else source.usage,
        reasoning=source.reasoning,
        raw_stop_reason=source.raw_stop_reason,
    )


def _parse_error_tool_message(
    source: AssistantMessage,
    parsed: PromptParseResult,
    *,
    model: str | None,
    usage: CompletionUsage | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolUseBlock(
                id=_new_tool_id(),
                name=parsed.error_tool_name or "prompted_tool_parse_error",
                input={},
                parse_error=parsed.error or "invalid prompted tool action",
            )
        ],
        stop_reason="tool_use",
        model=source.model or model,
        usage=usage if usage is not None else source.usage,
        reasoning=source.reasoning,
        raw_stop_reason=source.raw_stop_reason,
    )


def _combine_usage(
    left: CompletionUsage | None, right: CompletionUsage | None
) -> CompletionUsage | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right
