# agent/loop.py

"""Async reasoning loop bridging LLM clients and MCP tools."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    Message,
    ReasoningDelta,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    CompletionUsage,
)
from .contracts import ToolRuntime
from .context import (
    ContextBudget,
    assemble_context,
    clip_content,
    estimate_usage_tokens,
)
from .events import (
    DoneEvent,
    ErrorEvent,
    Event,
    ReasoningEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from .generation import generate_with_retry
from .session import Session, SessionStore
from .runtime import RunContext, RunDeadlineExceeded, RunLimits
from .tool_execution import (
    ActiveToolBatch,
    ToolDispatcher,
    make_skipped_tool_result,
)
from .tool_policy import ToolPolicy

logger = logging.getLogger(__name__)

# Always-on steering for common failure modes.
_FINAL_ITERATION_WRAPUP = (
    "This is your final step; you cannot call any more tools after this. "
    "Provide your best final answer using the information you already have."
)

# Appended once when consecutive tool failures cross the threshold.
_FAILURE_NUDGE = (
    "Multiple tool calls in a row have failed. Re-read the errors above and "
    "reconsider your tool choice and arguments before trying again."
)
_CONSECUTIVE_ERROR_NUDGE_THRESHOLD = 3
_SKIPPED_TOOL_ABORT_MESSAGE = (
    "tool call skipped because the run aborted after consecutive tool failures"
)
_SKIPPED_TOOL_DEADLINE_MESSAGE = (
    "tool call skipped because the run deadline was exceeded"
)
_SKIPPED_TOOL_BUDGET_MESSAGE = (
    "tool call skipped because the run token budget was exceeded"
)


def _with_wrapup(system: str | None) -> str:
    """Append the final-iteration wrap-up note to a per-call system prompt."""
    if system:
        return f"{system}\n\n{_FINAL_ITERATION_WRAPUP}"
    return _FINAL_ITERATION_WRAPUP


async def run_agent(
    session: Session,
    llm: LLMClient,
    mcp: ToolRuntime,
    *,
    store: SessionStore | None = None,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    thinking_level: str | None = None,
    limits: RunLimits | None = None,
    context: RunContext | None = None,
    policy: ToolPolicy | None = None,
    stream: bool = False,
) -> AsyncGenerator[Event, None]:
    """Drive a conversation to completion, yielding events along the way.

    The caller appends the user turn first. The loop handles assistant turns,
    tool round-trips, optional context shaping, dispatch policy, and tracing.

    ``limits`` is immutable policy; ``context`` carries the original turn
    deadline, run identity, and trace sequence. Direct callers may omit them
    to use defaults, while TurnRunner always supplies both.
    """
    limits = limits or RunLimits()
    context = context or RunContext.start(
        max_run_seconds=limits.max_run_seconds,
        base_logger=logger,
    )
    run_log = context.logger
    max_iterations = limits.max_iterations
    max_tokens = limits.max_tokens
    llm_timeout_seconds = limits.llm_timeout_seconds
    tool_result_max_chars = limits.tool_result_max_chars
    max_run_tokens = limits.max_run_tokens
    max_run_seconds = limits.max_run_seconds
    abort_after_consecutive_tool_failures = limits.abort_after_consecutive_tool_failures
    context_strategy = limits.context_strategy
    context_window = limits.context_window
    context_safety_margin_tokens = limits.context_safety_margin_tokens
    context_recent_messages = limits.context_recent_messages
    context_summary_max_tokens = limits.context_summary_max_tokens

    # `tools` controls what the model sees; MCP dispatch still routes by name.
    if tools is None:
        tools = mcp.get_tools_for_llm()
    # Policy controls what may run.
    policy = policy if policy is not None else ToolPolicy()
    tool_dispatcher = ToolDispatcher(
        runtime=mcp,
        policy=policy,
        tools=tools,
        limits=limits,
        context=context,
        log=run_log,
    )
    iteration = 0
    cumulative = CompletionUsage()

    # Run-scoped state for failure nudging and cancellation repair.
    consecutive_tool_errors = 0
    active_tool_batch: ActiveToolBatch | None = None

    _emit = context.emit

    def _done(reason: str) -> DoneEvent:
        return DoneEvent(
            reason=reason,
            iterations=iteration,
            total_tokens=cumulative.total_tokens,
            input_tokens=cumulative.input_tokens,
            output_tokens=cumulative.output_tokens,
            thinking_tokens=cumulative.thinking_tokens,
        )

    def _deadline_exceeded() -> bool:
        return context.deadline_exceeded()

    def _effective_timeout(per_call_timeout: float | None) -> float | None:
        return context.effective_timeout(per_call_timeout)

    def _token_budget_exceeded() -> bool:
        return (
            max_run_tokens is not None
            and max_run_tokens > 0
            and cumulative.total_tokens >= max_run_tokens
        )

    async def _publish_checkpoint(*, continue_work: bool) -> None:
        """Publish only safe state, then detach before further mutation."""
        nonlocal session
        if store is not None:
            await store.save(session)
        if continue_work:
            session = session.staged_copy()

    async def _publish_tool_batch(
        batch: ActiveToolBatch,
        *,
        continue_work: bool,
    ) -> None:
        batch.append_to(session)
        await _publish_checkpoint(continue_work=continue_work)

    async def _balance_interrupted_tool_batch(batch: ActiveToolBatch) -> None:
        """Best-effort protocol balancing without replacing the primary signal."""
        batch.balance_after_interruption()
        await _publish_tool_batch(batch, continue_work=False)

    def _skipped_tool_events(
        skipped_tools: list[ToolUseBlock],
        content: str,
        *,
        first_call_already_emitted: bool = False,
    ) -> tuple[list[Event], list[ToolResultBlock]]:
        events: list[Event] = []
        results: list[ToolResultBlock] = []
        for index, skipped in enumerate(skipped_tools):
            if not (first_call_already_emitted and index == 0):
                events.append(
                    ToolCallEvent(
                        id=skipped.id,
                        name=skipped.name,
                        input=skipped.input,
                    )
                )
            result = make_skipped_tool_result(skipped, content)
            events.append(
                ToolResultEvent(
                    id=skipped.id,
                    name=skipped.name,
                    content=result.content,
                    is_error=result.is_error,
                    latency_ms=None,
                )
            )
            results.append(result)
        return events, results

    while iteration < max_iterations:
        # Bounded-run guards before spending another LLM call.
        if _deadline_exceeded():
            elapsed = context.elapsed_seconds()
            run_log.warning(
                "run exceeded max_run_seconds=%.1f (elapsed=%.1fs)",
                max_run_seconds,
                elapsed,
            )
            yield await _emit(_done(reason="deadline_exceeded"))
            return
        if _token_budget_exceeded():
            run_log.warning(
                "run exceeded max_run_tokens=%d (used=%d)",
                max_run_tokens,
                cumulative.total_tokens,
            )
            yield await _emit(_done(reason="budget_exceeded"))
            return

        iteration += 1

        # Final iteration: withhold tools and ask for a best-effort answer.
        is_final_iteration = iteration >= max_iterations
        effective_system: str | None
        effective_tools: list[dict[str, Any]] | None
        if is_final_iteration:
            effective_system = _with_wrapup(system)
            effective_tools = None
        else:
            effective_system = system
            effective_tools = tools

        run_log.debug(
            "agent loop iteration %d (history=%d msgs, tools=%d, final=%s)",
            iteration,
            len(session.messages),
            len(effective_tools) if effective_tools else 0,
            is_final_iteration,
        )

        # Reassemble the view each call; context failures degrade to full history.
        messages_for_llm: list[Message] = session.messages
        if context_window and context_window > 0:
            try:
                assembly = assemble_context(
                    session.messages,
                    budget=ContextBudget(
                        context_window=context_window,
                        max_output_tokens=max_tokens or 0,
                        safety_margin=context_safety_margin_tokens,
                    ),
                    strategy=context_strategy,
                    system=effective_system,
                    tools=effective_tools or None,
                    llm=llm,
                    recent_messages=context_recent_messages,
                    summary_max_tokens=context_summary_max_tokens,
                )
                assembly_timeout = _effective_timeout(llm_timeout_seconds)
                if assembly_timeout and assembly_timeout > 0:
                    async with asyncio.timeout(assembly_timeout):
                        messages_for_llm = (await assembly).messages
                else:
                    messages_for_llm = (await assembly).messages
            except Exception:  # noqa: BLE001
                run_log.warning(
                    "context assembly failed; sending full history", exc_info=True
                )

        # Consume provider-neutral generation signals; retry mechanics stay below.
        generation_request = GenerationRequest(
            messages=tuple(messages_for_llm),
            tools=effective_tools or None,
            system=effective_system,
            max_tokens=max_tokens,
            thinking_level=thinking_level,
        )
        llm_started = time.perf_counter()
        streamed_reasoning = False
        response: AssistantMessage | None = None
        try:
            generation = generate_with_retry(
                llm,
                generation_request,
                limits=limits,
                context=context,
                stream=stream,
                log=run_log,
            )
            async with aclosing(generation):
                async for chunk in generation:
                    if isinstance(chunk, TextDelta):
                        yield await _emit(TextEvent(text=chunk.text))
                    elif isinstance(chunk, ReasoningDelta):
                        streamed_reasoning = True
                        yield await _emit(ReasoningEvent(text=chunk.text))
                    elif isinstance(chunk, StreamEnd):
                        response = chunk.message
            assert response is not None
        except RunDeadlineExceeded:
            elapsed = context.elapsed_seconds()
            run_log.warning(
                "run exceeded max_run_seconds=%.1f during LLM call (elapsed=%.1fs)",
                max_run_seconds,
                elapsed,
            )
            yield await _emit(_done(reason="deadline_exceeded"))
            return
        except Exception as e:
            # The model never sees unrecoverable LLM failures; the caller does.
            run_log.exception("LLM completion failed on iteration %d", iteration)
            yield await _emit(ErrorEvent(message=f"LLM call failed: {e}"))
            yield await _emit(_done(reason="llm_error"))
            return
        llm_latency_ms = round((time.perf_counter() - llm_started) * 1000, 2)

        # Establish tool-batch state before any post-response cancellation point.
        session.append_assistant(response)
        tool_uses = response.tool_uses()
        active_tool_batch = (
            ActiveToolBatch(tuple(tool_uses)) if tool_uses else None
        )

        try:
            # A complete non-tool response is terminal and safe immediately.
            if active_tool_batch is None:
                await _publish_checkpoint(continue_work=False)

            # Estimate usage when providers omit it so local servers still honor caps.
            usage = estimate_usage_tokens(
                response.usage,
                messages=messages_for_llm,
                system=effective_system,
                response=response,
            )
            cumulative = cumulative + usage
            yield await _emit(
                UsageEvent(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                    thinking_tokens=usage.thinking_tokens,
                    cached_tokens=usage.cached_tokens,
                    iteration=iteration,
                    latency_ms=llm_latency_ms,
                )
            )

            if response.reasoning and not streamed_reasoning:
                yield await _emit(ReasoningEvent(text=response.reasoning))

            for block in response.content:
                if not stream and isinstance(block, TextBlock) and block.text:
                    yield await _emit(TextEvent(text=block.text))

            if (
                active_tool_batch is None
                and response.stop_reason == "incomplete_stream"
            ):
                yield await _emit(
                    ErrorEvent(
                        message="LLM stream ended without a terminal provider message"
                    )
                )
                yield await _emit(_done(reason="incomplete_stream"))
                return

            # If a post-LLM guard trips, close requested tools synthetically.
            guard_reason: str | None = None
            skipped_message: str | None = None
            if _deadline_exceeded():
                guard_reason = "deadline_exceeded"
                skipped_message = _SKIPPED_TOOL_DEADLINE_MESSAGE
            elif _token_budget_exceeded():
                guard_reason = "budget_exceeded"
                skipped_message = _SKIPPED_TOOL_BUDGET_MESSAGE

            if guard_reason is not None:
                if active_tool_batch is not None:
                    assert skipped_message is not None
                    events, skipped_results = _skipped_tool_events(
                        tool_uses, skipped_message
                    )
                    active_tool_batch.complete_remaining(skipped_results)
                    for event in events:
                        yield await _emit(event)
                    await _publish_tool_batch(
                        active_tool_batch,
                        continue_work=False,
                    )
                    active_tool_batch = None
                if guard_reason == "deadline_exceeded":
                    elapsed = context.elapsed_seconds()
                    run_log.warning(
                        "run exceeded max_run_seconds=%.1f (elapsed=%.1fs)",
                        max_run_seconds,
                        elapsed,
                    )
                else:
                    run_log.warning(
                        "run exceeded max_run_tokens=%d (used=%d)",
                        max_run_tokens,
                        cumulative.total_tokens,
                    )
                yield await _emit(_done(reason=guard_reason))
                return

            if active_tool_batch is None:
                # Provider outcomes remain explicit at the native boundary.
                stop_reason = response.stop_reason
                if stop_reason == "max_tokens":
                    yield await _emit(_done(reason="truncated"))
                elif stop_reason == "content_filter":
                    yield await _emit(_done(reason="content_filter"))
                elif stop_reason == "refusal":
                    yield await _emit(_done(reason="refusal"))
                elif stop_reason == "empty":
                    yield await _emit(_done(reason="empty"))
                elif stop_reason == "end_turn" and is_final_iteration:
                    yield await _emit(_done(reason="max_iterations"))
                elif stop_reason == "end_turn":
                    yield await _emit(_done(reason="end_turn"))
                else:
                    run_log.warning(
                        "provider terminated abnormally: canonical=%r raw=%r",
                        stop_reason,
                        response.raw_stop_reason,
                    )
                    yield await _emit(
                        ErrorEvent(message="LLM provider terminated abnormally")
                    )
                    yield await _emit(_done(reason="provider_error"))
                return

            # Execute tools sequentially; some MCP tools may have side effects.
            for tool_index, tool_use in enumerate(tool_uses):
                yield await _emit(
                    ToolCallEvent(
                        id=tool_use.id,
                        name=tool_use.name,
                        input=tool_use.input,
                    )
                )

                if _deadline_exceeded():
                    events, skipped_results = _skipped_tool_events(
                        tool_uses[tool_index:],
                        _SKIPPED_TOOL_DEADLINE_MESSAGE,
                        first_call_already_emitted=True,
                    )
                    active_tool_batch.complete_remaining(skipped_results)
                    for event in events:
                        yield await _emit(event)
                    await _publish_tool_batch(active_tool_batch, continue_work=False)
                    active_tool_batch = None
                    elapsed = context.elapsed_seconds()
                    run_log.warning(
                        "run exceeded max_run_seconds=%.1f before tool dispatch (elapsed=%.1fs)",
                        max_run_seconds,
                        elapsed,
                    )
                    yield await _emit(_done(reason="deadline_exceeded"))
                    return

                try:
                    dispatch_result = await tool_dispatcher.dispatch(
                        active_tool_batch,
                        tool_use,
                    )
                except RunDeadlineExceeded:
                    events, skipped_results = _skipped_tool_events(
                        tool_uses[tool_index:],
                        _SKIPPED_TOOL_DEADLINE_MESSAGE,
                        first_call_already_emitted=True,
                    )
                    active_tool_batch.complete_remaining(skipped_results)
                    for event in events:
                        yield await _emit(event)
                    await _publish_tool_batch(
                        active_tool_batch,
                        continue_work=False,
                    )
                    active_tool_batch = None
                    elapsed = context.elapsed_seconds()
                    run_log.warning(
                        "run exceeded max_run_seconds=%.1f during tool "
                        "dispatch (elapsed=%.1fs)",
                        max_run_seconds,
                        elapsed,
                    )
                    yield await _emit(_done(reason="deadline_exceeded"))
                    return

                content = clip_content(
                    dispatch_result.content,
                    tool_result_max_chars,
                )
                is_error = dispatch_result.is_error
                consecutive_tool_errors = consecutive_tool_errors + 1 if is_error else 0
                if (
                    is_error
                    and consecutive_tool_errors == _CONSECUTIVE_ERROR_NUDGE_THRESHOLD
                ):
                    content = f"{content}\n\n{_FAILURE_NUDGE}"

                result = ToolResultBlock(
                    tool_use_id=tool_use.id,
                    name=tool_use.name,
                    content=content,
                    is_error=is_error,
                )
                active_tool_batch.complete(result)
                yield await _emit(
                    ToolResultEvent(
                        id=tool_use.id,
                        name=tool_use.name,
                        content=content,
                        is_error=is_error,
                        latency_ms=dispatch_result.latency_ms,
                    )
                )

                if (
                    is_error
                    and abort_after_consecutive_tool_failures
                    and abort_after_consecutive_tool_failures > 0
                    and consecutive_tool_errors >= abort_after_consecutive_tool_failures
                ):
                    events, skipped_results = _skipped_tool_events(
                        tool_uses[tool_index + 1 :],
                        _SKIPPED_TOOL_ABORT_MESSAGE,
                    )
                    active_tool_batch.complete_remaining(skipped_results)
                    for event in events:
                        yield await _emit(event)
                    run_log.warning(
                        "aborting run: %d consecutive tool failures (threshold=%d)",
                        consecutive_tool_errors,
                        abort_after_consecutive_tool_failures,
                    )
                    await _publish_tool_batch(active_tool_batch, continue_work=False)
                    active_tool_batch = None
                    yield await _emit(_done(reason="no_progress"))
                    return

            await _publish_tool_batch(active_tool_batch, continue_work=True)
            active_tool_batch = None
        except (asyncio.CancelledError, GeneratorExit):
            interrupted_batch = active_tool_batch
            if interrupted_batch is not None:
                try:
                    await _balance_interrupted_tool_batch(interrupted_batch)
                except BaseException:  # noqa: BLE001 -- preserve cancellation/close.
                    run_log.warning(
                        "failed to publish balanced tool checkpoint during interruption",
                        exc_info=True,
                    )
            raise

    # Structural fallback: the generator must always end on a DoneEvent.
    run_log.warning("agent loop hit max_iterations=%d without end_turn", max_iterations)
    yield await _emit(_done(reason="max_iterations"))
