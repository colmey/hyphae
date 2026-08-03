"""OpenAI-compatible stream decoding and SDK stream ownership."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
import inspect
import logging
from typing import Any

from llm.schemas import (
    AssistantMessage,
    ReasoningDelta,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)

from .codec import (
    _THINK_CLOSE,
    _THINK_OPEN,
    canonical_stop_reason,
    raw_stop_reason,
    build_tool_use_block,
    structured_reasoning_text,
    usage_from_raw,
)

async def _close_sdk_stream(
    stream: Any,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Close one SDK stream without replacing its primary outcome."""
    close = getattr(stream, "aclose", None)
    if close is None:
        close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except asyncio.CancelledError as exc:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise
        if logger is not None:
            logger.warning(
                "OpenAI SDK stream cleanup was cancelled for %s (%s)",
                type(stream).__name__,
                type(exc).__name__,
            )
    except Exception as exc:  # noqa: BLE001 -- preserve the active stream outcome.
        if logger is not None:
            logger.warning(
                "failed to close OpenAI SDK stream %s (%s)",
                type(stream).__name__,
                type(exc).__name__,
            )


@dataclass
class _ToolCallAccumulator:
    """Collect streamed call fragments and finalize exactly once."""

    call_id: Any = ""
    name: Any = ""
    argument_parts: list[str] = field(default_factory=list)
    malformed_arguments: bool = False

    def update(self, *, call_id: Any, name: Any, arguments: Any) -> None:
        if call_id:
            self.call_id = call_id
        if name is not None:
            self.name = name
        if arguments is not None:
            if isinstance(arguments, str):
                self.argument_parts.append(arguments)
            else:
                self.malformed_arguments = True

    def finalize(
        self, *, logger: logging.Logger | None = None
    ) -> ToolUseBlock:
        return build_tool_use_block(
            call_id=self.call_id,
            name=self.name,
            raw_arguments=(
                None if self.malformed_arguments else "".join(self.argument_parts)
            ),
            logger=logger,
        )


class _ReasoningStreamStripper:
    """Incrementally strip one leading <think>...</think> block."""

    _OPEN = _THINK_OPEN
    _CLOSE = _THINK_CLOSE

    def __init__(self) -> None:
        self._state = "pending"
        self._pending = ""
        self._reasoning_parts: list[str] = []
        self._reasoning_deltas: list[str] = []

    @property
    def reasoning(self) -> str | None:
        text = "".join(self._reasoning_parts).strip()
        return text or None

    def drain_reasoning(self) -> str:
        """Return newly extracted reasoning without its wrapper tags."""
        text = "".join(self._reasoning_deltas)
        self._reasoning_deltas.clear()
        return text

    def _append_reasoning(self, text: str) -> None:
        if text:
            self._reasoning_parts.append(text)
            self._reasoning_deltas.append(text)

    def feed(self, piece: str) -> str:
        if not piece:
            return ""
        if self._state == "pass":
            return piece
        if self._state == "pending":
            return self._feed_pending(piece)
        if self._state == "reasoning":
            return self._feed_reasoning(piece)
        if self._state == "after_reasoning":
            return self._feed_after_reasoning(piece)
        return piece

    def finish(self) -> str:
        """Flush buffered visible text if the stream ended before a decision."""
        if self._state == "pending":
            out = self._pending
            self._pending = ""
            self._state = "pass"
            return out
        if self._state == "reasoning":
            self._append_reasoning(self._pending)
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
                self._append_reasoning(self._pending[:keep])
                self._pending = self._pending[keep:]
            return ""

        self._append_reasoning(self._pending[:close_at])
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


async def decode_stream(
    sdk_stream: Any,
    *,
    default_model: str,
    logger: logging.Logger | None = None,
) -> AsyncGenerator[StreamChunk, None]:
    """Decode one SDK stream and own its cleanup for the full iteration."""
    stripper = _ReasoningStreamStripper()
    text_parts: list[str] = []
    tool_accumulators: dict[int, _ToolCallAccumulator] = {}
    legacy_tool_accumulator: _ToolCallAccumulator | None = None
    has_refusal = False
    saw_choice = False
    finish_reason: Any = None
    raw_usage: Any = None
    response_model: str | None = None
    structured_reasoning_parts: list[str] = []

    try:
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

                structured_reasoning = structured_reasoning_text(delta)
                if structured_reasoning:
                    structured_reasoning_parts.append(structured_reasoning)
                    yield ReasoningDelta(text=structured_reasoning)

                piece = getattr(delta, "content", None)
                if piece:
                    visible = stripper.feed(piece)
                    tagged_reasoning = stripper.drain_reasoning()
                    if tagged_reasoning:
                        yield ReasoningDelta(text=tagged_reasoning)
                    if visible:
                        text_parts.append(visible)
                        yield TextDelta(text=visible)

                refusal_piece = getattr(delta, "refusal", None)
                if refusal_piece:
                    has_refusal = True
                    text_parts.append(refusal_piece)
                    yield TextDelta(text=refusal_piece)

                for tool_call in getattr(delta, "tool_calls", None) or []:
                    index = int(getattr(tool_call, "index", 0) or 0)
                    accumulator = tool_accumulators.setdefault(
                        index, _ToolCallAccumulator()
                    )
                    function = getattr(tool_call, "function", None)
                    accumulator.update(
                        call_id=getattr(tool_call, "id", None),
                        name=getattr(function, "name", None) if function else None,
                        arguments=(
                            getattr(function, "arguments", None)
                            if function
                            else None
                        ),
                    )

                legacy_function = getattr(delta, "function_call", None)
                if legacy_function is not None:
                    if legacy_tool_accumulator is None:
                        legacy_tool_accumulator = _ToolCallAccumulator()
                    legacy_tool_accumulator.update(
                        call_id=None,
                        name=getattr(legacy_function, "name", None),
                        arguments=getattr(legacy_function, "arguments", None),
                    )

        tail = stripper.finish()
        tagged_reasoning = stripper.drain_reasoning()
        if tagged_reasoning:
            yield ReasoningDelta(text=tagged_reasoning)
        if tail:
            text_parts.append(tail)
            yield TextDelta(text=tail)

        blocks: list[Any] = []
        full_text = "".join(text_parts)
        if full_text:
            blocks.append(TextBlock(text=full_text))

        for index in sorted(tool_accumulators):
            blocks.append(tool_accumulators[index].finalize(logger=logger))

        if not tool_accumulators and legacy_tool_accumulator is not None:
            blocks.append(legacy_tool_accumulator.finalize(logger=logger))

        has_tools = any(isinstance(block, ToolUseBlock) for block in blocks)
        has_visible_content = any(
            isinstance(block, TextBlock) and bool(block.text) for block in blocks
        )
        stop_reason = (
            canonical_stop_reason(
                finish_reason,
                has_tools=has_tools,
                has_visible_content=has_visible_content,
                has_refusal=has_refusal,
            )
            if saw_choice
            else "empty"
        )

        reasoning = "\n".join(
            part
            for part in (
                "".join(structured_reasoning_parts).strip(),
                stripper.reasoning,
            )
            if part
        ) or None

        yield StreamEnd(
            message=AssistantMessage(
                content=blocks,
                stop_reason=stop_reason,
                raw_stop_reason=raw_stop_reason(finish_reason),
                model=response_model or default_model,
                usage=usage_from_raw(raw_usage),
                reasoning=reasoning,
            )
        )
    finally:
        await _close_sdk_stream(sdk_stream, logger=logger)
