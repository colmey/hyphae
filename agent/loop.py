# agent/loop.py

"""Async reasoning loop bridging LLM clients and MCP tools."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Literal

from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    CompletionUsage,
    ReasoningDelta,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
)
from tooling import ToolRuntime

from .context import (
    ContextAssembly,
    ContextBudget,
    ContextBudgetExceeded,
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
from .session import (
    Session,
    SessionHistoryLimitExceeded,
    SessionStore,
    TranscriptValidationError,
    validate_transcript,
)
from .runtime import RunContext, RunDeadlineExceeded, RunLimits
from .tool_execution import (
    ActiveToolBatch,
    ToolDispatchResult,
    ToolDispatcher,
    make_skipped_tool_result,
)
from .tool_policy import ToolPolicy

logger = logging.getLogger(__name__)

type _GuardReason = Literal["deadline_exceeded", "budget_exceeded"]

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
_LLM_CALL_FAILED_MESSAGE = "LLM call failed"


def _with_wrapup(system: str | None) -> str:
    """Append the final-iteration wrap-up note to a per-call system prompt."""
    if system:
        return f"{system}\n\n{_FINAL_ITERATION_WRAPUP}"
    return _FINAL_ITERATION_WRAPUP


@dataclass(frozen=True, slots=True)
class _PreparedGeneration:
    """Immutable request inputs passed between the preparation and run phases."""

    request: GenerationRequest
    context_assembly: ContextAssembly
    final_iteration: bool


@dataclass
class _AgentRun:
    """Explicit state and high-level phase composition for one agent run."""

    session: Session
    llm: LLMClient
    store: SessionStore | None
    system: str | None
    visible_tools: list[dict[str, Any]]
    thinking_level: str | None
    limits: RunLimits
    context: RunContext
    stream: bool
    tool_dispatcher: ToolDispatcher
    iteration: int = 0
    cumulative_usage: CompletionUsage = field(default_factory=CompletionUsage)
    consecutive_tool_errors: int = 0
    active_tool_batch: ActiveToolBatch | None = None
    _pending_response: AssistantMessage | None = None
    _pending_streamed_reasoning: bool = False
    _pending_latency_ms: float | None = None
    _pending_terminal_reason: str | None = None

    @classmethod
    def from_inputs(
        cls,
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
        initial_usage: CompletionUsage | None = None,
    ) -> "_AgentRun":
        """Resolve defaults and construct exactly one run-scoped dispatcher."""
        resolved_limits = limits or RunLimits()
        resolved_context = context or RunContext.start(
            max_run_seconds=resolved_limits.max_run_seconds,
            base_logger=logger,
        )
        resolved_tools = tools if tools is not None else mcp.get_tools_for_llm()
        resolved_policy = policy if policy is not None else ToolPolicy()
        dispatcher = ToolDispatcher(
            runtime=mcp,
            policy=resolved_policy,
            tools=resolved_tools,
            limits=resolved_limits,
            context=resolved_context,
            log=resolved_context.logger,
        )
        return cls(
            session=session,
            llm=llm,
            store=store,
            system=system,
            visible_tools=resolved_tools,
            thinking_level=thinking_level,
            limits=resolved_limits,
            context=resolved_context,
            stream=stream,
            tool_dispatcher=dispatcher,
            cumulative_usage=initial_usage or CompletionUsage(),
        )

    def _done(self, reason: str) -> DoneEvent:
        return DoneEvent(
            reason=reason,
            iterations=self.iteration,
            total_tokens=self.cumulative_usage.total_tokens,
            input_tokens=self.cumulative_usage.input_tokens,
            output_tokens=self.cumulative_usage.output_tokens,
            thinking_tokens=self.cumulative_usage.thinking_tokens,
        )

    def _token_budget_exceeded(self) -> bool:
        max_run_tokens = self.limits.max_run_tokens
        return (
            max_run_tokens is not None
            and max_run_tokens > 0
            and self.cumulative_usage.total_tokens >= max_run_tokens
        )

    def _bounded_run_reason(self) -> _GuardReason | None:
        if self.context.deadline_exceeded():
            return "deadline_exceeded"
        if self._token_budget_exceeded():
            return "budget_exceeded"
        return None

    def _clear_pending_generation(self) -> None:
        self._pending_response = None
        self._pending_streamed_reasoning = False
        self._pending_latency_ms = None

    @staticmethod
    def _terminal_reason(
        response: AssistantMessage,
        final_iteration: bool,
    ) -> str:
        stop_reason = response.stop_reason
        if stop_reason == "max_tokens":
            return "truncated"
        if stop_reason == "content_filter":
            return "content_filter"
        if stop_reason == "refusal":
            return "refusal"
        if stop_reason == "empty":
            return "empty"
        if stop_reason == "incomplete_stream":
            return "incomplete_stream"
        if stop_reason == "end_turn" and final_iteration:
            return "max_iterations"
        if stop_reason == "end_turn":
            return "end_turn"
        return "provider_error"

    async def _prepare_generation(self) -> _PreparedGeneration:
        """Assemble one immutable generation request or a typed budget outcome."""
        final_iteration = self.iteration >= self.limits.max_iterations
        effective_system: str | None
        effective_tools: list[dict[str, Any]] | None
        if final_iteration:
            effective_system = _with_wrapup(self.system)
            effective_tools = None
        else:
            effective_system = self.system
            effective_tools = self.visible_tools

        self.context.logger.debug(
            "agent loop iteration %d (history=%d msgs, tools=%d, final=%s)",
            self.iteration,
            len(self.session.messages),
            len(effective_tools) if effective_tools else 0,
            final_iteration,
        )

        context_assembly = await self._assemble_generation_context(
            system=effective_system,
            tools=effective_tools,
        )
        self.cumulative_usage = self.cumulative_usage + context_assembly.auxiliary_usage
        request = GenerationRequest(
            messages=context_assembly.messages,
            tools=effective_tools or None,
            system=effective_system,
            max_tokens=self.limits.max_tokens,
            thinking_level=self.thinking_level,
        )
        return _PreparedGeneration(
            request=request,
            context_assembly=context_assembly,
            final_iteration=final_iteration,
        )

    async def _assemble_generation_context(
        self,
        *,
        system: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> ContextAssembly:
        """Assemble a request view that is known to fit its configured budget."""
        context_window = self.limits.context_window
        if not context_window or context_window <= 0:
            estimated = estimate_usage_tokens(
                None,
                messages=self.session.messages,
                system=system,
                tools=tools,
            ).input_tokens
            return ContextAssembly(tuple(self.session.messages), estimated)

        assembly_timeout = self.context.effective_timeout(
            self.limits.llm_timeout_seconds
        )
        if assembly_timeout is not None and assembly_timeout <= 0:
            raise RunDeadlineExceeded
        try:
            assembly = assemble_context(
                self.session.messages,
                budget=ContextBudget(
                    context_window=context_window,
                    max_output_tokens=self.limits.max_tokens or 0,
                    safety_margin=self.limits.context_safety_margin_tokens,
                ),
                strategy=self.limits.context_strategy,
                system=system,
                tools=tools or None,
                llm=self.llm,
                recent_messages=self.limits.context_recent_messages,
                summary_max_tokens=self.limits.context_summary_max_tokens,
            )
            if assembly_timeout and assembly_timeout > 0:
                async with asyncio.timeout(assembly_timeout):
                    return await assembly
            return await assembly
        except ContextBudgetExceeded:
            raise
        except TimeoutError:
            if self.context.deadline_exceeded():
                raise RunDeadlineExceeded from None
            self.context.logger.warning("context compaction timed out")
            raise ContextBudgetExceeded from None
        except Exception:  # noqa: BLE001
            self.context.logger.warning("context assembly failed", exc_info=True)
            raise ContextBudgetExceeded from None

    async def _consume_generation(
        self,
        prepared: _PreparedGeneration,
    ) -> AsyncGenerator[Event, None]:
        """Forward live deltas and retain exactly one terminal response."""
        self._clear_pending_generation()
        llm_started = time.perf_counter()
        generation = generate_with_retry(
            self.llm,
            prepared.request,
            limits=self.limits,
            context=self.context,
            stream=self.stream,
            log=self.context.logger,
        )
        async with aclosing(generation):
            async for chunk in generation:
                if isinstance(chunk, TextDelta):
                    yield await self.context.emit(TextEvent(text=chunk.text))
                elif isinstance(chunk, ReasoningDelta):
                    self._pending_streamed_reasoning = True
                    yield await self.context.emit(ReasoningEvent(text=chunk.text))
                elif isinstance(chunk, StreamEnd):
                    if self._pending_response is not None:
                        raise RuntimeError(
                            "generation produced more than one terminal response"
                        )
                    self._pending_response = chunk.message
        assert self._pending_response is not None
        self._pending_latency_ms = round(
            (time.perf_counter() - llm_started) * 1000,
            2,
        )

    async def _record_response(
        self,
        prepared: _PreparedGeneration,
    ) -> AsyncGenerator[Event, None]:
        """Record a completed response and present its buffered public events."""
        response = self._pending_response
        assert response is not None
        streamed_reasoning = self._pending_streamed_reasoning
        latency_ms = self._pending_latency_ms
        assert latency_ms is not None

        self._clear_pending_generation()
        self._pending_terminal_reason = None

        self.session.append_assistant(response)
        tool_uses = response.tool_uses()
        transcript_error: TranscriptValidationError | None = None
        try:
            validate_transcript(self.session.messages, allow_pending=bool(tool_uses))
        except TranscriptValidationError as exc:
            transcript_error = exc
        self.active_tool_batch = (
            ActiveToolBatch(tuple(tool_uses))
            if tool_uses and transcript_error is None
            else None
        )

        checkpoint_error: SessionHistoryLimitExceeded | None = None
        if self.active_tool_batch is None and transcript_error is None:
            try:
                await self._publish_checkpoint(continue_work=False)
            except SessionHistoryLimitExceeded as exc:
                checkpoint_error = exc

        usage = estimate_usage_tokens(
            response.usage,
            messages=list(prepared.context_assembly.messages),
            system=prepared.request.system,
            response=response,
            tools=prepared.request.tools,
        )
        self.cumulative_usage = self.cumulative_usage + usage
        yield await self.context.emit(
            UsageEvent(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                thinking_tokens=usage.thinking_tokens,
                cached_tokens=usage.cached_tokens,
                iteration=self.iteration,
                latency_ms=latency_ms,
            )
        )

        if transcript_error is not None:
            raise transcript_error
        if checkpoint_error is not None:
            raise checkpoint_error

        if response.reasoning and not streamed_reasoning:
            yield await self.context.emit(ReasoningEvent(text=response.reasoning))

        for block in response.content:
            if not self.stream and isinstance(block, TextBlock) and block.text:
                yield await self.context.emit(TextEvent(text=block.text))

        if self.active_tool_batch is None:
            terminal_reason = self._terminal_reason(
                response,
                prepared.final_iteration,
            )
            if terminal_reason == "provider_error":
                self.context.logger.warning(
                    "provider terminated abnormally: canonical=%r raw=%r",
                    response.stop_reason,
                    response.raw_stop_reason,
                )
            self._pending_terminal_reason = terminal_reason

    async def _publish_checkpoint(self, *, continue_work: bool) -> None:
        """Publish only protocol-safe state and detach before continuing."""
        if self.store is not None:
            await self.store.save(self.session)
        if continue_work:
            self.session = self.session.staged_copy()

    async def _publish_tool_batch(
        self,
        batch: ActiveToolBatch,
        *,
        continue_work: bool,
    ) -> None:
        batch.append_to(self.session)
        await self._publish_checkpoint(continue_work=continue_work)

    async def _publish_active_batch(
        self,
        *,
        continue_work: bool,
        terminal_reason: str | None = None,
    ) -> None:
        """Publish the active batch, then clear it and record its outcome."""
        batch = self.active_tool_batch
        assert batch is not None
        await self._publish_tool_batch(batch, continue_work=continue_work)
        self.active_tool_batch = None
        self._pending_terminal_reason = terminal_reason

    async def _balance_interrupted_batch(self) -> None:
        """Best-effort protocol repair without replacing the primary signal."""
        batch = self.active_tool_batch
        if batch is None:
            return
        batch.balance_after_interruption()
        await self._publish_tool_batch(batch, continue_work=False)

    @staticmethod
    def _complete_skipped_tools(
        batch: ActiveToolBatch,
        skipped_tools: list[ToolUseBlock],
        content: str,
        *,
        first_call_already_emitted: bool = False,
    ) -> list[Event]:
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
        batch.complete_remaining(results)
        return events

    def _take_terminal_reason(self) -> str | None:
        reason = self._pending_terminal_reason
        self._pending_terminal_reason = None
        return reason

    def _record_tool_result(
        self,
        batch: ActiveToolBatch,
        tool_use: ToolUseBlock,
        dispatch_result: ToolDispatchResult,
    ) -> ToolResultEvent:
        """Record one dispatched result and return its public presentation."""
        content = clip_content(
            dispatch_result.content,
            self.limits.tool_result_max_chars,
        )
        is_error = dispatch_result.is_error
        self.consecutive_tool_errors = (
            self.consecutive_tool_errors + 1 if is_error else 0
        )
        if (
            is_error
            and self.consecutive_tool_errors == _CONSECUTIVE_ERROR_NUDGE_THRESHOLD
        ):
            content = f"{content}\n\n{_FAILURE_NUDGE}"

        result = ToolResultBlock(
            tool_use_id=tool_use.id,
            name=tool_use.name,
            content=content,
            is_error=is_error,
        )
        batch.complete(result)
        return ToolResultEvent(
            id=tool_use.id,
            name=tool_use.name,
            content=content,
            is_error=is_error,
            latency_ms=dispatch_result.latency_ms,
        )

    async def _tool_events(self) -> AsyncGenerator[Event, None]:
        """Present and publish the current assistant-declared tool batch."""
        batch = self.active_tool_batch
        assert batch is not None
        tool_uses = list(batch.tool_uses)

        for tool_index, tool_use in enumerate(tool_uses):
            yield await self.context.emit(
                ToolCallEvent(
                    id=tool_use.id,
                    name=tool_use.name,
                    input=tool_use.input,
                )
            )

            if self.context.deadline_exceeded():
                events = self._complete_skipped_tools(
                    batch,
                    tool_uses[tool_index:],
                    _SKIPPED_TOOL_DEADLINE_MESSAGE,
                    first_call_already_emitted=True,
                )
                for event in events:
                    yield await self.context.emit(event)
                await self._publish_active_batch(
                    continue_work=False,
                    terminal_reason="deadline_exceeded",
                )
                self.context.logger.warning(
                    "run exceeded max_run_seconds=%.1f before tool dispatch "
                    "(elapsed=%.1fs)",
                    self.limits.max_run_seconds,
                    self.context.elapsed_seconds(),
                )
                return

            try:
                dispatch_result = await self.tool_dispatcher.dispatch(batch, tool_use)
            except RunDeadlineExceeded:
                events = self._complete_skipped_tools(
                    batch,
                    tool_uses[tool_index:],
                    _SKIPPED_TOOL_DEADLINE_MESSAGE,
                    first_call_already_emitted=True,
                )
                for event in events:
                    yield await self.context.emit(event)
                await self._publish_active_batch(
                    continue_work=False,
                    terminal_reason="deadline_exceeded",
                )
                self.context.logger.warning(
                    "run exceeded max_run_seconds=%.1f during tool dispatch "
                    "(elapsed=%.1fs)",
                    self.limits.max_run_seconds,
                    self.context.elapsed_seconds(),
                )
                return

            result_event = self._record_tool_result(
                batch,
                tool_use,
                dispatch_result,
            )
            yield await self.context.emit(result_event)

            failure_limit = self.limits.abort_after_consecutive_tool_failures
            if (
                result_event.is_error
                and failure_limit
                and failure_limit > 0
                and self.consecutive_tool_errors >= failure_limit
            ):
                events = self._complete_skipped_tools(
                    batch,
                    tool_uses[tool_index + 1 :],
                    _SKIPPED_TOOL_ABORT_MESSAGE,
                )
                for event in events:
                    yield await self.context.emit(event)
                self.context.logger.warning(
                    "aborting run: %d consecutive tool failures (threshold=%d)",
                    self.consecutive_tool_errors,
                    failure_limit,
                )
                await self._publish_active_batch(
                    continue_work=False,
                    terminal_reason="no_progress",
                )
                return

        await self._publish_active_batch(continue_work=True)

    def _log_guard(self, reason: _GuardReason) -> None:
        if reason == "deadline_exceeded":
            self.context.logger.warning(
                "run exceeded max_run_seconds=%.1f (elapsed=%.1fs)",
                self.limits.max_run_seconds,
                self.context.elapsed_seconds(),
            )
        else:
            if self.limits.max_run_tokens is None:
                self.context.logger.warning(
                    "context could not fit within its configured input budget "
                    "(run usage=%d)",
                    self.cumulative_usage.total_tokens,
                )
            else:
                self.context.logger.warning(
                    "run exceeded max_run_tokens=%d (used=%d)",
                    self.limits.max_run_tokens,
                    self.cumulative_usage.total_tokens,
                )

    def _auxiliary_usage_event(
        self,
        usage: CompletionUsage,
        latency_ms: float,
    ) -> UsageEvent:
        """Present one compaction completion without folding it into generation."""
        return UsageEvent(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            thinking_tokens=usage.thinking_tokens,
            cached_tokens=usage.cached_tokens,
            iteration=self.iteration,
            latency_ms=latency_ms,
        )

    async def events(self) -> AsyncGenerator[Event, None]:
        """Drive the explicit preparation, generation, record, and tool phases."""
        while self.iteration < self.limits.max_iterations:
            preflight_reason = self._bounded_run_reason()
            if preflight_reason is not None:
                self._log_guard(preflight_reason)
                yield await self.context.emit(self._done(preflight_reason))
                return

            self.iteration += 1
            try:
                prepared = await self._prepare_generation()
            except ContextBudgetExceeded as exc:
                if exc.auxiliary_usage.total_tokens:
                    self.cumulative_usage = self.cumulative_usage + exc.auxiliary_usage
                    yield await self.context.emit(
                        self._auxiliary_usage_event(
                            exc.auxiliary_usage,
                            exc.auxiliary_latency_ms,
                        )
                    )
                context_terminal_reason = (
                    self._bounded_run_reason() or "budget_exceeded"
                )
                self._log_guard(context_terminal_reason)
                yield await self.context.emit(self._done(context_terminal_reason))
                return
            except RunDeadlineExceeded:
                yield await self.context.emit(self._done("deadline_exceeded"))
                return

            assembly = prepared.context_assembly
            if assembly.compacted:
                yield await self.context.emit(
                    self._auxiliary_usage_event(
                        assembly.auxiliary_usage,
                        assembly.auxiliary_latency_ms,
                    )
                )
                guard_reason = self._bounded_run_reason()
                if guard_reason is not None:
                    self._log_guard(guard_reason)
                    yield await self.context.emit(self._done(guard_reason))
                    return

            try:
                generation_events = self._consume_generation(prepared)
                async with aclosing(generation_events):
                    async for event in generation_events:
                        yield event
            except RunDeadlineExceeded:
                self._clear_pending_generation()
                self.context.logger.warning(
                    "run exceeded max_run_seconds=%.1f during LLM call (elapsed=%.1fs)",
                    self.limits.max_run_seconds,
                    self.context.elapsed_seconds(),
                )
                yield await self.context.emit(self._done("deadline_exceeded"))
                return
            except Exception:
                self._clear_pending_generation()
                self.context.logger.exception(
                    "LLM completion failed on iteration %d",
                    self.iteration,
                )
                yield await self.context.emit(ErrorEvent(message=_LLM_CALL_FAILED_MESSAGE))
                yield await self.context.emit(self._done("llm_error"))
                return
            except (asyncio.CancelledError, GeneratorExit):
                self._clear_pending_generation()
                raise

            try:
                response_events = self._record_response(prepared)
                async with aclosing(response_events):
                    async for event in response_events:
                        yield event

                if self._pending_terminal_reason == "incomplete_stream":
                    self._take_terminal_reason()
                    yield await self.context.emit(
                        ErrorEvent(
                            message=(
                                "LLM stream ended without a terminal provider message"
                            )
                        )
                    )
                    yield await self.context.emit(self._done("incomplete_stream"))
                    return

                guard_reason = self._bounded_run_reason()
                if guard_reason is not None:
                    batch = self.active_tool_batch
                    if batch is not None:
                        skipped_message = (
                            _SKIPPED_TOOL_DEADLINE_MESSAGE
                            if guard_reason == "deadline_exceeded"
                            else _SKIPPED_TOOL_BUDGET_MESSAGE
                        )
                        events = self._complete_skipped_tools(
                            batch,
                            list(batch.tool_uses),
                            skipped_message,
                        )
                        for event in events:
                            yield await self.context.emit(event)
                        await self._publish_active_batch(continue_work=False)
                    self._pending_terminal_reason = None
                    self._log_guard(guard_reason)
                    yield await self.context.emit(self._done(guard_reason))
                    return

                if self.active_tool_batch is None:
                    terminal_reason = self._take_terminal_reason()
                    assert terminal_reason is not None
                    if terminal_reason == "provider_error":
                        yield await self.context.emit(
                            ErrorEvent(message="LLM provider terminated abnormally")
                        )
                    yield await self.context.emit(self._done(terminal_reason))
                    return

                tool_events = self._tool_events()
                async with aclosing(tool_events):
                    async for event in tool_events:
                        yield event
                tool_terminal_reason = self._take_terminal_reason()
                if tool_terminal_reason is not None:
                    yield await self.context.emit(self._done(tool_terminal_reason))
                    return
            except TranscriptValidationError:
                self.active_tool_batch = None
                self._pending_terminal_reason = None
                self.context.logger.warning(
                    "provider returned an invalid canonical tool transcript"
                )
                yield await self.context.emit(
                    ErrorEvent(message="LLM provider returned invalid tool-call output")
                )
                yield await self.context.emit(self._done("provider_error"))
                return
            except SessionHistoryLimitExceeded as exc:
                self.active_tool_batch = None
                self._pending_terminal_reason = None
                self.context.logger.warning(
                    "session checkpoint rejected by retained-history limit"
                )
                yield await self.context.emit(ErrorEvent(message=str(exc)))
                yield await self.context.emit(self._done("session_history_limit"))
                return
            except (asyncio.CancelledError, GeneratorExit):
                if self.active_tool_batch is not None:
                    try:
                        await self._balance_interrupted_batch()
                    except BaseException:  # noqa: BLE001
                        self.context.logger.warning(
                            "failed to publish balanced tool checkpoint during "
                            "interruption",
                            exc_info=True,
                        )
                raise

        self.context.logger.warning(
            "agent loop hit max_iterations=%d without end_turn",
            self.limits.max_iterations,
        )
        yield await self.context.emit(self._done("max_iterations"))


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
    initial_usage: CompletionUsage | None = None,
) -> AsyncGenerator[Event, None]:
    """Drive a conversation to completion, yielding events along the way.

    The caller appends the user turn first. The loop handles assistant turns,
    tool round-trips, optional context shaping, dispatch policy, and tracing.

    ``limits`` is immutable policy; ``context`` carries the original turn
    deadline, run identity, and trace sequence. Direct callers may omit them
    to use defaults, while TurnRunner always supplies both. ``initial_usage``
    seeds one already-completed auxiliary operation into terminal totals and
    preflight token bounds without emitting a second usage event.
    """
    run = _AgentRun.from_inputs(
        session,
        llm,
        mcp,
        store=store,
        system=system,
        tools=tools,
        thinking_level=thinking_level,
        limits=limits,
        context=context,
        policy=policy,
        stream=stream,
        initial_usage=initial_usage,
    )
    events = run.events()
    async with aclosing(events):
        async for event in events:
            yield event
