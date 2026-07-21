# agent/loop.py

"""Async reasoning loop bridging LLM clients and MCP tools."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

import jsonschema
from jsonschema.validators import validator_for

from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    Message,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from mcp_layer import MCPManager

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
from .session import Session, SessionStore
from .tool_policy import ToolPolicy, Verdict
from .runtime import RunContext, RunLimits

logger = logging.getLogger(__name__)

# Cap exponential retry sleeps.
_RETRY_BACKOFF_CAP_SECONDS = 30.0

# Always-on steering for common failure modes.
_FINAL_ITERATION_WRAPUP = (
    "This is your final step; you cannot call any more tools after this. "
    "Provide your best final answer using the information you already have."
)

# Synthetic error for exact repeat calls in the same run.
_STALL_MESSAGE = (
    "You already called this tool with identical arguments; re-running it will "
    "not produce a different result. Try different arguments or a different "
    "approach."
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
_CANCELLED_TOOL_UNKNOWN_MESSAGE = "tool call outcome is unknown because execution was cancelled while the call was in flight"
_CANCELLED_TOOL_NOT_STARTED_MESSAGE = (
    "tool call was not executed because execution was cancelled"
)


class _RunDeadlineExceeded(TimeoutError):
    """Internal signal that the run-level wall clock expired."""


@dataclass(frozen=True)
class _TimeoutBound:
    """One timeout computed against the current absolute run deadline."""

    seconds: float | None
    deadline_limited: bool


@dataclass(frozen=True)
class _AttemptController:
    """Shared attempt, deadline, and backoff policy for one LLM generation."""

    limits: RunLimits
    context: RunContext
    log: logging.Logger | logging.LoggerAdapter

    @property
    def total_attempts(self) -> int:
        return max(0, self.limits.max_retries) + 1

    def indexes(self) -> range:
        return range(self.total_attempts)

    def ensure_before_attempt(self) -> None:
        if self.context.deadline_exceeded():
            raise _RunDeadlineExceeded()

    def timeout_bound(self) -> _TimeoutBound:
        """Return the smaller enabled LLM timeout and remaining run budget."""
        remaining = self.context.remaining()
        if remaining is not None and remaining <= 0:
            raise _RunDeadlineExceeded()

        per_attempt = self.limits.llm_timeout_seconds
        if per_attempt is not None and per_attempt <= 0:
            per_attempt = None
        if remaining is None:
            return _TimeoutBound(per_attempt, False)
        if per_attempt is None or remaining <= per_attempt:
            return _TimeoutBound(remaining, True)
        return _TimeoutBound(per_attempt, False)

    async def retry(
        self,
        attempt_index: int,
        *,
        eligible: bool,
        mode: str,
        reason: str,
        cause: BaseException | None = None,
    ) -> bool:
        """Sleep for the next retry, or report that this attempt is final."""
        if not eligible or attempt_index >= self.total_attempts - 1:
            return False

        delay = _backoff_delay(self.limits.retry_base_delay, attempt_index)
        remaining = self.context.remaining()
        if remaining is not None:
            if remaining <= 0:
                raise _RunDeadlineExceeded() from cause
            delay = min(delay, remaining)

        suffix = f": {cause}" if cause is not None else ""
        self.log.warning(
            "retrying %s LLM after %s (attempt %d/%d) in %.2fs%s",
            mode,
            reason,
            attempt_index + 1,
            self.total_attempts,
            delay,
            suffix,
        )
        await asyncio.sleep(delay)
        if self.context.deadline_exceeded():
            raise _RunDeadlineExceeded() from cause
        return True


@dataclass
class _ActiveToolBatch:
    """Cancellation state for one sequential assistant tool-use batch."""

    tool_uses: list[ToolUseBlock]
    _results: list[ToolResultBlock] = field(default_factory=list, init=False)
    _next_index: int = field(default=0, init=False)
    _in_flight: ToolUseBlock | None = field(default=None, init=False)
    _appended: bool = field(default=False, init=False)

    def start_dispatch(self, tool_use: ToolUseBlock) -> None:
        if self._in_flight is not None:
            raise RuntimeError("another tool call is already in flight")
        expected = self.tool_uses[self._next_index]
        if tool_use is not expected:
            raise ValueError("tool calls must be dispatched in batch order")
        self._in_flight = tool_use

    def complete(self, result: ToolResultBlock) -> None:
        expected = self.tool_uses[self._next_index]
        if (result.tool_use_id, result.name) != (expected.id, expected.name):
            raise ValueError("tool result does not match the next tool call")
        self._results.append(result)
        self._next_index += 1
        self._in_flight = None

    def complete_remaining(self, results: list[ToolResultBlock]) -> None:
        remaining = self.tool_uses[self._next_index :]
        expected = [(tool_use.id, tool_use.name) for tool_use in remaining]
        actual = [(result.tool_use_id, result.name) for result in results]
        if actual != expected:
            raise ValueError("synthetic results must match every remaining tool call")
        self._results.extend(results)
        self._next_index = len(self.tool_uses)
        self._in_flight = None

    def balance_after_interruption(self) -> None:
        if self._appended:
            return
        if self._in_flight is not None:
            self._results.append(
                ToolResultBlock(
                    tool_use_id=self._in_flight.id,
                    name=self._in_flight.name,
                    content=_CANCELLED_TOOL_UNKNOWN_MESSAGE,
                    is_error=True,
                )
            )
            self._next_index += 1
            self._in_flight = None
        for tool_use in self.tool_uses[self._next_index :]:
            self._results.append(
                ToolResultBlock(
                    tool_use_id=tool_use.id,
                    name=tool_use.name,
                    content=_CANCELLED_TOOL_NOT_STARTED_MESSAGE,
                    is_error=True,
                )
            )
        self._next_index = len(self.tool_uses)

    def append_to(self, session: Session) -> None:
        if self._appended:
            return
        if self._next_index != len(self.tool_uses):
            raise ValueError("cannot append an incomplete tool-result batch")
        session.append_tool_results(self._results)
        self._appended = True


def _canonical_args(args: dict[str, Any]) -> str:
    """Stable string key for a tool call's arguments (for stall detection).

    Sort keys so reordered-but-equivalent calls match; fall back to repr() so
    stall detection never breaks dispatch.
    """
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(args)


class _ArgumentValidator(Protocol):
    """Run-local argument-validation seam used by tool dispatch."""

    def validate(self, args: dict[str, Any]) -> str | None: ...


class _ConcreteSchemaValidator(Protocol):
    """Minimal jsonschema validator surface retained by the run wrapper."""

    def validate(self, instance: Any) -> None: ...


class _PermissiveArgumentValidator:
    """Sentinel used for absent or unusable schemas."""

    def validate(self, args: dict[str, Any]) -> None:
        return None


_PERMISSIVE_VALIDATOR = _PermissiveArgumentValidator()


@dataclass
class _CompiledArgumentValidator:
    """Concrete JSON Schema validator isolated to one run and tool."""

    name: str
    concrete: _ConcreteSchemaValidator
    log: logging.Logger | logging.LoggerAdapter
    _disabled: bool = field(default=False, init=False)

    def validate(self, args: dict[str, Any]) -> str | None:
        if self._disabled:
            return None
        try:
            self.concrete.validate(args)
        except jsonschema.ValidationError as exc:
            field_path = "/".join(str(part) for part in exc.path) or "(top level)"
            return (
                f"invalid arguments for field {field_path!r}: {exc.message}. "
                f"Expected shape: {json.dumps(exc.schema, ensure_ascii=False)}"
            )
        except Exception:  # noqa: BLE001 -- a broken schema stays permissive.
            self._disabled = True
            self.log.warning(
                "tool %s input validator failed; disabling arg validation for this run",
                self.name,
                exc_info=True,
            )
        return None


def _compile_tool_validators(
    tools: list[dict[str, Any]],
    *,
    log: logging.Logger | logging.LoggerAdapter = logger,
) -> dict[str, _ArgumentValidator]:
    """Compile each advertised tool schema once for this run."""
    schemas = {tool["name"]: tool.get("input_schema") or {} for tool in tools}
    validators: dict[str, _ArgumentValidator] = {}
    for name, schema in schemas.items():
        try:
            validator_cls = validator_for(schema)
            validator_cls.check_schema(schema)
            validators[name] = _CompiledArgumentValidator(
                name=name,
                concrete=validator_cls(schema),
                log=log,
            )
        except Exception:  # noqa: BLE001 -- malformed tool schemas are permissive.
            log.warning(
                "tool %s input_schema is invalid; skipping arg validation",
                name,
                exc_info=True,
            )
            validators[name] = _PERMISSIVE_VALIDATOR
    return validators


def _with_wrapup(system: str | None) -> str:
    """Append the final-iteration wrap-up note to a per-call system prompt."""
    if system:
        return f"{system}\n\n{_FINAL_ITERATION_WRAPUP}"
    return _FINAL_ITERATION_WRAPUP


async def _complete_with_retry(
    llm: LLMClient,
    *,
    request: GenerationRequest,
    attempts: _AttemptController,
) -> AssistantMessage:
    """Call llm.complete() with a per-attempt timeout and bounded retries.

    Retries transient errors, timeouts, and empty responses. Exhausted transient
    errors re-raise; exhausted empty responses return the last empty response.
    """
    for attempt_index in attempts.indexes():
        attempts.ensure_before_attempt()
        timeout = attempts.timeout_bound()
        try:
            if timeout.seconds is not None and timeout.seconds > 0:
                async with asyncio.timeout(timeout.seconds):
                    response = await llm.complete(request)
            else:
                response = await llm.complete(request)
        except TimeoutError as exc:
            if timeout.deadline_limited or attempts.context.deadline_exceeded():
                raise _RunDeadlineExceeded() from exc
            if await attempts.retry(
                attempt_index,
                eligible=True,
                mode="buffered",
                reason="timeout",
                cause=exc,
            ):
                continue
            raise
        except Exception as exc:
            if attempts.context.deadline_exceeded():
                raise _RunDeadlineExceeded() from exc
            if await attempts.retry(
                attempt_index,
                eligible=llm.is_transient_error(exc),
                mode="buffered",
                reason="transient provider error",
                cause=exc,
            ):
                continue
            raise

        # Success. An empty-candidates response is retryable up to the budget.
        if await attempts.retry(
            attempt_index,
            eligible=response.stop_reason == "empty",
            mode="buffered",
            reason="empty response",
        ):
            continue
        return response

    raise AssertionError("attempt controller produced no buffered attempts")


async def _read_stream_chunk(
    chunks: AsyncIterator[StreamChunk],
    attempts: _AttemptController,
) -> StreamChunk:
    """Read one stream chunk within the idle timeout and absolute deadline."""
    timeout = attempts.timeout_bound()
    if timeout.seconds is None or timeout.seconds <= 0:
        return await anext(chunks)
    try:
        async with asyncio.timeout(timeout.seconds):
            return await anext(chunks)
    except TimeoutError as exc:
        if timeout.deadline_limited or attempts.context.deadline_exceeded():
            raise _RunDeadlineExceeded() from exc
        raise


async def _close_stream(
    stream: Any,
    *,
    log: logging.Logger | logging.LoggerAdapter = logger,
) -> None:
    """Close a provider stream when its iterator exposes ``aclose``.

    Stream cleanup is best-effort: a provider cleanup failure must not replace
    the timeout, deadline, or provider exception that ended the attempt.
    """
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise
        log.warning("LLM stream cleanup was cancelled", exc_info=True)
    except Exception:  # noqa: BLE001 -- cleanup must preserve the primary outcome.
        log.warning("failed to close LLM stream", exc_info=True)


def _backoff_delay(base_delay: float, attempt: int) -> float:
    """Jittered exponential backoff, capped."""
    return min(_RETRY_BACKOFF_CAP_SECONDS, base_delay * (2**attempt)) + random.uniform(
        0, base_delay
    )


async def run_agent(
    session: Session,
    llm: LLMClient,
    mcp: MCPManager,
    *,
    store: SessionStore | None = None,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    thinking_level: str | None = None,
    limits: RunLimits | None = None,
    context: RunContext | None = None,
    policy: ToolPolicy | None = None,
    stream: bool = False,
) -> AsyncIterator[Event]:
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
    tool_timeout_seconds = limits.tool_timeout_seconds
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
    tool_validators = _compile_tool_validators(tools, log=run_log)
    # Policy controls what may run.
    policy = policy if policy is not None else ToolPolicy()
    iteration = 0
    cumulative = Usage()

    # Run-scoped state for repeat-call detection and failure nudging.
    seen_calls: set[tuple[str, str]] = set()
    consecutive_tool_errors = 0
    active_tool_batch: _ActiveToolBatch | None = None

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
            bool(max_run_tokens)
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
        batch: _ActiveToolBatch,
        *,
        continue_work: bool,
    ) -> None:
        batch.append_to(session)
        await _publish_checkpoint(continue_work=continue_work)

    async def _balance_interrupted_tool_batch(batch: _ActiveToolBatch) -> None:
        """Best-effort protocol balancing without replacing the primary signal."""
        batch.balance_after_interruption()
        await _publish_tool_batch(batch, continue_work=False)

    def _skipped_result(tu: ToolUseBlock, content: str) -> ToolResultBlock:
        return ToolResultBlock(
            tool_use_id=tu.id,
            name=tu.name,
            content=content,
            is_error=True,
        )

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
            result = _skipped_result(skipped, content)
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
            elapsed = context.elapsed()
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
        if is_final_iteration:
            effective_system = _with_wrapup(system)
            effective_tools: list[dict[str, Any]] | None = None
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

        # LLM call with per-attempt timeout and bounded retry.
        generation_request = GenerationRequest(
            messages=tuple(messages_for_llm),
            tools=effective_tools or None,
            system=effective_system,
            max_tokens=max_tokens,
            thinking_level=thinking_level,
        )
        llm_started = time.perf_counter()
        attempts = _AttemptController(limits=limits, context=context, log=run_log)
        try:
            if stream:
                response: AssistantMessage | None = None

                for attempt_index in attempts.indexes():
                    attempts.ensure_before_attempt()
                    visible_deltas: list[str] = []
                    chunks: AsyncIterator[StreamChunk] | None = None
                    attempt_response: AssistantMessage | None = None
                    attempt_error: Exception | None = None
                    retryable_error = False
                    try:
                        chunks = llm.stream(generation_request)
                        while True:
                            try:
                                chunk = await _read_stream_chunk(chunks, attempts)
                            except StopAsyncIteration:
                                break

                            if isinstance(chunk, TextDelta):
                                if chunk.text:
                                    visible_deltas.append(chunk.text)
                                    yield await _emit(TextEvent(text=chunk.text))
                            elif isinstance(chunk, StreamEnd):
                                attempt_response = chunk.message
                                break

                        if attempt_response is None:
                            if visible_deltas:
                                attempt_response = AssistantMessage(
                                    content=[TextBlock(text="".join(visible_deltas))],
                                    stop_reason="incomplete_stream",
                                    model=None,
                                    usage=Usage(),
                                )
                            else:
                                attempt_response = AssistantMessage(
                                    content=[],
                                    stop_reason="empty",
                                    model=None,
                                    usage=Usage(),
                                )
                    except _RunDeadlineExceeded:
                        raise
                    except Exception as exc:
                        if _deadline_exceeded():
                            raise _RunDeadlineExceeded() from exc
                        attempt_error = exc
                        retryable_error = not visible_deltas and (
                            isinstance(exc, TimeoutError)
                            or llm.is_transient_error(exc)
                        )
                    finally:
                        if chunks is not None:
                            await _close_stream(chunks, log=run_log)

                    if attempt_error is not None:
                        retry_reason = (
                            "timeout"
                            if isinstance(attempt_error, TimeoutError)
                            else "transient provider error"
                        )
                        if await attempts.retry(
                            attempt_index,
                            eligible=retryable_error,
                            mode="streaming",
                            reason=retry_reason,
                            cause=attempt_error,
                        ):
                            continue
                        raise attempt_error

                    assert attempt_response is not None
                    if await attempts.retry(
                        attempt_index,
                        eligible=(
                            attempt_response.stop_reason == "empty"
                            and not visible_deltas
                        ),
                        mode="streaming",
                        reason="empty response",
                    ):
                        continue
                    response = attempt_response
                    break

                assert response is not None
            else:
                response = await _complete_with_retry(
                    llm,
                    request=generation_request,
                    attempts=attempts,
                )
        except _RunDeadlineExceeded:
            elapsed = context.elapsed()
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
        active_tool_batch = _ActiveToolBatch(tool_uses) if tool_uses else None

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

            if response.reasoning:
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
                    elapsed = context.elapsed()
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
            for tool_index, tu in enumerate(tool_uses):
                yield await _emit(ToolCallEvent(id=tu.id, name=tu.name, input=tu.input))

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
                    elapsed = context.elapsed()
                    run_log.warning(
                        "run exceeded max_run_seconds=%.1f before tool dispatch (elapsed=%.1fs)",
                        max_run_seconds,
                        elapsed,
                    )
                    yield await _emit(_done(reason="deadline_exceeded"))
                    return

                # Exact repeat calls get a synthetic error instead of re-execution.
                call_key = (tu.name, _canonical_args(tu.input))
                tool_latency_ms: float | None = None
                if call_key in seen_calls:
                    run_log.info(
                        "stall: repeat call to %s with identical args; skipping",
                        tu.name,
                    )
                    content = _STALL_MESSAGE
                    is_error = True
                else:
                    seen_calls.add(call_key)
                    if tu.parse_error is not None:
                        validation_error = (
                            f"tool call arguments were not valid JSON ({tu.parse_error}); "
                            "return the arguments as a JSON object matching the tool schema."
                        )
                    else:
                        validator = tool_validators.get(
                            tu.name, _PERMISSIVE_VALIDATOR
                        )
                        validation_error = validator.validate(tu.input)

                    decision = (
                        policy.check(tu.name, tu.input)
                        if validation_error is None
                        else None
                    )
                    if validation_error is not None:
                        run_log.info(
                            "invalid args for %s: %s", tu.name, validation_error
                        )
                        content = validation_error
                        is_error = True
                    elif decision is not None and decision.verdict is Verdict.DENY:
                        run_log.info("policy denied %s", tu.name)
                        content = decision.reason
                        is_error = True
                    else:
                        tool_started = time.perf_counter()
                        active_tool_batch.start_dispatch(tu)
                        try:
                            effective_tool_timeout = _effective_timeout(
                                tool_timeout_seconds
                            )
                            if effective_tool_timeout and effective_tool_timeout > 0:
                                async with asyncio.timeout(effective_tool_timeout):
                                    call_result = await mcp.call_tool(tu.name, tu.input)
                            else:
                                call_result = await mcp.call_tool(tu.name, tu.input)
                            content = call_result.content
                            is_error = call_result.is_error
                        except TimeoutError:
                            if _deadline_exceeded():
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
                                elapsed = context.elapsed()
                                run_log.warning(
                                    "run exceeded max_run_seconds=%.1f during tool "
                                    "dispatch (elapsed=%.1fs)",
                                    max_run_seconds,
                                    elapsed,
                                )
                                yield await _emit(_done(reason="deadline_exceeded"))
                                return
                            run_log.warning(
                                "tool %s timed out after %ss",
                                tu.name,
                                tool_timeout_seconds,
                            )
                            content = f"tool {tu.name!r} timed out after {tool_timeout_seconds}s"
                            is_error = True
                        except Exception as exc:
                            run_log.exception("tool execution raised for %s", tu.name)
                            content = f"tool execution raised: {exc}"
                            is_error = True
                        tool_latency_ms = round(
                            (time.perf_counter() - tool_started) * 1000, 2
                        )

                content = clip_content(content, tool_result_max_chars)
                consecutive_tool_errors = consecutive_tool_errors + 1 if is_error else 0
                if (
                    is_error
                    and consecutive_tool_errors == _CONSECUTIVE_ERROR_NUDGE_THRESHOLD
                ):
                    content = f"{content}\n\n{_FAILURE_NUDGE}"

                result = ToolResultBlock(
                    tool_use_id=tu.id,
                    name=tu.name,
                    content=content,
                    is_error=is_error,
                )
                active_tool_batch.complete(result)
                yield await _emit(
                    ToolResultEvent(
                        id=tu.id,
                        name=tu.name,
                        content=content,
                        is_error=is_error,
                        latency_ms=tool_latency_ms,
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
