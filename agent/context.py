# agent/context.py

"""
Context assembly: the seam between session history and the outgoing LLM call.

The loop calls assemble_context() instead of handing session.messages to the
LLM raw. The seam estimates the request's input tokens against an explicit
budget (context_window − max_output_tokens − safety_margin) and, per the
configured strategy, decides what the model sees:

  - "naive" (default): pass-through. Over budget only logs a warning — the
    exact behavior the harness had before this module existed.
  - "compaction": when over budget, keep the first user message (the task
    header) and the last N protocol-safe units verbatim, and replace the
    middle with a one-call LLM summary that preserves decisions, constraints,
    facts learned, failed approaches, and open questions.

Compaction shapes the OUTGOING VIEW only. session.messages remains the
append-only source of truth; nothing here mutates it (callers must not feed
the assembled list back into the session).

Protocol safety: providers reject tool results whose triggering assistant
tool_use is missing (and vice versa), so history is grouped into units that
are kept or summarized atomically — an assistant message carrying tool_use
blocks travels with the tool message(s) that answer it. A retained view can
therefore never start with an orphan tool result.

Degrade, never break: a failed summarization, malformed history, or a
history too short to compact all fall back to pass-through with a warning
(`degraded_reason` says why). The context layer must never kill a request.

The module also owns the cheap local token estimator (chars/4 heuristic plus
per-message/block overhead). estimate_usage_tokens() backs the loop's
Phase-1 token cap when a provider reports absent/all-zero usage — common on
local OpenAI-compatible servers.

Import rule: this seam may import LLM schema types and LLMClient (for the
summarizer call) but never MCP — the loop stays the only LLM⇄MCP bridge.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    ContentBlock,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

logger = logging.getLogger(__name__)

# chars/4 is the standard cheap heuristic; good enough for budget guarding.
# The per-message/block overheads stand in for role tags, ids, and framing
# tokens so the estimate errs slightly high (a budget guard should).
_CHARS_PER_TOKEN = 4
_MESSAGE_OVERHEAD_TOKENS = 4
_BLOCK_OVERHEAD_TOKENS = 4

_SUMMARY_SYSTEM = (
    "You compact an agent conversation into a terse factual summary. Preserve: "
    "decisions made, constraints, facts learned, tool results that still matter, "
    "approaches that failed, and open questions. Discard redundant tool output "
    "and superseded reasoning. Output compact bullet points, not prose."
)
_SUMMARY_HEADER = "Conversation summary so far:"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _render_json(obj: object) -> str:
    """Render JSON-ish values for estimates/transcripts; repr on odd values."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(obj)


def clip_content(
    content: str, max_chars: int | None, *, marker: str = "truncated"
) -> str:
    """Bound content while recording how many characters were omitted."""
    if not max_chars or max_chars <= 0 or len(content) <= max_chars:
        return content
    omitted = len(content) - max_chars
    return f"{content[:max_chars]}\n…[{marker}, {omitted} chars omitted]"


def estimate_text_tokens(text: str) -> int:
    """Cheap token estimate for a string: chars/4, at least 1 if non-empty."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _estimate_block_tokens(block: ContentBlock) -> int:
    if isinstance(block, TextBlock):
        return estimate_text_tokens(block.text)
    if isinstance(block, ToolUseBlock):
        args = _render_json(block.input)
        return (
            _BLOCK_OVERHEAD_TOKENS
            + estimate_text_tokens(block.name)
            + estimate_text_tokens(args)
        )
    if isinstance(block, ToolResultBlock):
        return (
            _BLOCK_OVERHEAD_TOKENS
            + estimate_text_tokens(block.name)
            + estimate_text_tokens(block.content)
        )
    return _BLOCK_OVERHEAD_TOKENS


def estimate_message_tokens(messages: list[Message]) -> int:
    """Estimate the token cost of a message list (content + framing overhead)."""
    total = 0
    for msg in messages:
        total += _MESSAGE_OVERHEAD_TOKENS
        for block in msg.content:
            total += _estimate_block_tokens(block)
    return total


def estimate_tools_tokens(tools: list[dict] | None) -> int:
    """Estimate the token cost of the tool schemas sent with a request."""
    if not tools:
        return 0
    return estimate_text_tokens(_render_json(tools))


def estimate_usage_tokens(
    usage: Usage | None,
    messages: list[Message] | None = None,
    system: str | None = None,
    response: AssistantMessage | None = None,
) -> Usage:
    """Return provider usage when it reported anything, else a local estimate.

    Local/OpenAI-compatible servers often report all-zero usage, which left
    the Phase-1 token cap inert. This fills the gap: absent/all-zero usage is
    estimated from the outgoing messages (+ system prompt) and the assistant
    response. Non-zero provider usage passes through untouched — never
    double-counted — except that a missing total is filled from input+output
    (arithmetic on provider figures, not an estimate).
    """
    if usage is not None and (
        usage.input_tokens or usage.output_tokens or usage.total_tokens
    ):
        if usage.total_tokens:
            return usage
        return Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            thinking_tokens=usage.thinking_tokens,
            cached_tokens=usage.cached_tokens,
        )
    input_est = estimate_message_tokens(messages or []) + estimate_text_tokens(
        system or ""
    )
    output_est = (
        estimate_message_tokens([response.to_message()])
        if response is not None and response.content
        else 0
    )
    return Usage(
        input_tokens=input_est,
        output_tokens=output_est,
        total_tokens=input_est + output_est,
    )


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextBudget:
    """Explicit token budget for one request's input side."""

    context_window: int
    max_output_tokens: int = 0
    safety_margin: int = 1024

    @property
    def input_budget(self) -> int:
        return max(0, self.context_window - self.max_output_tokens - self.safety_margin)


@dataclass
class AssembledContext:
    """What assemble_context() decided to send, and why."""

    messages: list[Message]
    estimated_input_tokens: int
    budget: ContextBudget
    strategy: str
    compacted: bool = False
    # Set when the strategy could not do its job (over budget under naive,
    # summarizer failure, malformed/too-short history) and the full history
    # was passed through instead.
    degraded_reason: str | None = None


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


async def assemble_context(
    messages: list[Message],
    *,
    budget: ContextBudget,
    strategy: str = "naive",
    system: str | None = None,
    tools: list[dict] | None = None,
    llm: LLMClient | None = None,
    recent_messages: int = 6,
    summary_max_tokens: int = 512,
) -> AssembledContext:
    """Build the outgoing message view for one LLM call.

    `system` and `tools` must be the *effective* values the call will use
    (e.g. the final-iteration wrap-up prompt with tools withheld) so their
    token cost is counted against the same budget the request will face.
    `llm` is the summarizer client for "compaction"; it is called once, with
    no tools and no recursive context assembly.

    Never raises for strategy-level failures — degrades to pass-through with
    `degraded_reason` set and a warning logged.
    """
    if strategy not in ("naive", "compaction"):
        logger.warning("unknown context_strategy %r; using 'naive'", strategy)
        strategy = "naive"

    overhead = estimate_text_tokens(system or "") + estimate_tools_tokens(tools)
    estimated = overhead + estimate_message_tokens(messages)

    if estimated <= budget.input_budget:
        return AssembledContext(messages, estimated, budget, strategy)

    if strategy == "naive":
        logger.warning(
            "context over budget under 'naive' (est=%d > budget=%d); passing through",
            estimated,
            budget.input_budget,
        )
        return AssembledContext(
            messages, estimated, budget, strategy, degraded_reason="over_budget"
        )

    if llm is None:
        logger.warning("compaction requested but no summarizer client; passing through")
        return AssembledContext(
            messages,
            estimated,
            budget,
            strategy,
            degraded_reason="no_summarizer_client",
        )

    try:
        return await _compact(
            messages,
            budget=budget,
            overhead=overhead,
            estimated=estimated,
            llm=llm,
            recent_messages=recent_messages,
            summary_max_tokens=summary_max_tokens,
        )
    except Exception:  # noqa: BLE001 -- the context layer must never kill a request
        logger.warning("compaction failed; passing full history through", exc_info=True)
        return AssembledContext(
            messages, estimated, budget, strategy, degraded_reason="compaction_error"
        )


def _protocol_units(messages: list[Message]) -> list[list[Message]]:
    """Group history into units safe to retain or summarize atomically.

    A tool message attaches to the preceding unit when that unit contains the
    assistant tool_use it answers; anything else starts its own unit. Keeping
    units whole is what guarantees compaction never splits a tool-use/
    tool-result pair across the retained/summarized boundary.
    """
    units: list[list[Message]] = []
    for msg in messages:
        if (
            msg.role == Role.TOOL
            and units
            and any(isinstance(b, ToolUseBlock) for m in units[-1] for b in m.content)
        ):
            units[-1].append(msg)
        else:
            units.append([msg])
    return units


async def _compact(
    messages: list[Message],
    *,
    budget: ContextBudget,
    overhead: int,
    estimated: int,
    llm: LLMClient,
    recent_messages: int,
    summary_max_tokens: int,
) -> AssembledContext:
    units = _protocol_units(messages)

    # Pin the task header: the first user message stays verbatim so the goal
    # can't rot away. (If history doesn't start with one, nothing is pinned.)
    pinned: list[Message] = []
    start = 0
    if units and units[0][0].role == Role.USER:
        pinned = units[0]
        start = 1

    available = len(units) - start
    if available < 2:
        logger.warning(
            "history too short to compact (%d units); passing through", len(units)
        )
        return AssembledContext(
            messages,
            estimated,
            budget,
            "compaction",
            degraded_reason="too_short_to_compact",
        )

    # Choose how many recent units survive verbatim: start from the configured
    # count (leaving at least one unit to summarize) and shrink while the
    # fixed parts (overhead + pinned + a summary at its output cap) plus the
    # recent tail still can't fit. keep=1 is the smallest protocol-safe view;
    # if even that overflows we proceed anyway and log below.
    keep = max(1, min(recent_messages, available - 1))
    fixed = (
        overhead
        + estimate_message_tokens(pinned)
        + _MESSAGE_OVERHEAD_TOKENS
        + summary_max_tokens
    )
    while (
        keep > 1
        and fixed
        + estimate_message_tokens(
            [m for unit in units[len(units) - keep :] for m in unit]
        )
        > budget.input_budget
    ):
        keep -= 1

    recent = [m for unit in units[len(units) - keep :] for m in unit]
    middle = [m for unit in units[start : len(units) - keep] for m in unit]

    if recent and recent[0].role == Role.TOOL:
        # Only possible on malformed history (a tool result with no prior
        # assistant tool_use formed its own unit). Refuse rather than send an
        # orphan the provider will reject.
        logger.warning(
            "compaction boundary would orphan a tool result; passing through"
        )
        return AssembledContext(
            messages,
            estimated,
            budget,
            "compaction",
            degraded_reason="malformed_history",
        )

    summary_text = await _summarize(
        middle,
        llm=llm,
        max_tokens=summary_max_tokens,
        transcript_char_cap=budget.input_budget * _CHARS_PER_TOKEN,
    )
    if summary_text is None:
        return AssembledContext(
            messages,
            estimated,
            budget,
            "compaction",
            degraded_reason="summarizer_failed",
        )

    # A plain user message, clearly marked — no new role or block type, and
    # nothing that looks like a tool call/result to any provider.
    summary_msg = Message.user(f"{_SUMMARY_HEADER}\n{summary_text}")
    assembled = [*pinned, summary_msg, *recent]
    assembled_estimate = overhead + estimate_message_tokens(assembled)

    if assembled_estimate > budget.input_budget:
        logger.warning(
            "compacted context still over budget (est=%d > budget=%d); "
            "sending the smallest protocol-safe view",
            assembled_estimate,
            budget.input_budget,
        )
    logger.info(
        "context compacted: %d -> %d messages (est %d -> %d tokens, budget %d)",
        len(messages),
        len(assembled),
        estimated,
        assembled_estimate,
        budget.input_budget,
    )
    return AssembledContext(
        assembled, assembled_estimate, budget, "compaction", compacted=True
    )


def _flatten_for_summary(messages: list[Message]) -> str:
    """Render the middle of the history as a plain transcript for the summarizer."""
    lines: list[str] = []
    for msg in messages:
        for block in msg.content:
            if isinstance(block, TextBlock):
                if block.text:
                    lines.append(f"{msg.role.value}: {block.text}")
            elif isinstance(block, ToolUseBlock):
                args = _render_json(block.input)
                lines.append(f"assistant called tool {block.name} with {args}")
            elif isinstance(block, ToolResultBlock):
                marker = " (error)" if block.is_error else ""
                lines.append(f"tool result{marker} from {block.name}: {block.content}")
    return "\n".join(lines)


async def _summarize(
    messages: list[Message],
    *,
    llm: LLMClient,
    max_tokens: int,
    transcript_char_cap: int,
) -> str | None:
    """One tool-less LLM call summarizing `messages`. None on any failure.

    The transcript is clipped to roughly the input budget so the summarizer
    call itself can't overflow the same window — no recursive assembly.
    """
    transcript = _flatten_for_summary(messages)
    transcript = clip_content(
        transcript,
        transcript_char_cap,
        marker="transcript truncated for summarization",
    )
    try:
        request = GenerationRequest(
            messages=[Message.user(transcript)],
            system=_SUMMARY_SYSTEM,
            max_tokens=max_tokens,
        )
        response = await llm.complete(request)
        text = "\n".join(
            b.text for b in response.content if isinstance(b, TextBlock) and b.text
        ).strip()
    except Exception:  # noqa: BLE001
        logger.warning(
            "history summarization failed; degrading to pass-through", exc_info=True
        )
        return None
    if not text:
        logger.warning(
            "history summarization returned no text; degrading to pass-through"
        )
        return None
    return text
