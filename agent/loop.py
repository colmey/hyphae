# agent/loop.py

"""
The reasoning loop.

This is the only module in the harness that imports both LLMClient and
MCPManager. Both of those subsystems are unaware of each other; the loop
is the bridge.

Algorithm (one iteration):
  1. Ask the LLM to complete given the current session history + MCP tools.
  2. Append the assistant response to the session.
  3. Stream out the model's text blocks as TextEvents.
  4. If no tool_use blocks: yield DoneEvent(end_turn) and stop.
  5. Otherwise, for each tool_use:
       a. Yield ToolCallEvent
       b. Execute via MCPManager.call_tool() (unless it is an exact repeat of
          a call already made this run -- see stall detection below)
       c. Yield ToolResultEvent
       d. Build a ToolResultBlock with the same id and name
  6. Append all results as one Role.TOOL message.
  7. Loop until DoneEvent or max_iterations.

Loop-intelligence scaffolding (always on, no config knob):
  - Final-iteration wrap-up: on the last allowed iteration the model can
    make no further tool round trip, so the loop withholds tools and appends a
    wrap-up note to the per-call system prompt, coaxing a best-effort answer
    instead of fragments. The run still reports done_reason="max_iterations".
  - Stall detection: a tool call byte-for-byte identical (name + canonical
    args) to one already run is short-circuited to a synthetic is_error result
    rather than re-executed, breaking model fixation. After several failures in
    a row a one-line nudge is appended steering the model to reconsider.

Design notes:
  - Sequential tool execution. Some MCP tools have side effects; we don't
    want surprises. Wrap a future iteration in asyncio.gather() if we want
    parallel.
  - System prompt comes in via the function arg, not the Session. The
    Session is conversation state; system prompt is per-call configuration.
  - The session is save()'d after each iteration to keep the store
    consistent if anything later in the loop crashes. In-memory backend
    treats this as a no-op; future durable backends will commit.
  - Tool execution failures become ToolResultBlock(is_error=True) — the
    model sees them and can recover or apologize. The LLM-call itself
    raising propagates: we catch once and surface as ErrorEvent + DoneEvent.
  - The `tools` parameter is optional. When None, the loop pulls the full
    inventory from MCPManager (legacy behavior). When provided (typically
    by the orchestrator-aware route), the loop uses it verbatim and does
    not call mcp.get_tools_for_llm(). This is how the orchestration layer
    plugs in: it filters the inventory ahead of time and hands the loop
    only the tools the model is meant to see.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, AsyncIterator, Callable

import jsonschema

from llm.client import LLMClient
from llm.schemas import AssistantMessage, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from mcp_layer import MCPManager

from .events import (
    DoneEvent,
    ErrorEvent,
    Event,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from .session import Session, SessionStore
from .tracing import Tracer, event_record

logger = logging.getLogger(__name__)

# Upper bound on a single backoff sleep, so an exponential schedule can't
# stretch a retry into an absurd wait.
_RETRY_BACKOFF_CAP_SECONDS = 30.0

# --- Loop-intelligence scaffolding -----------------------------------------
# These steer the model at the moments it is about to fail. They are always-on
# loop logic (no Settings knob): none has a failure mode that warrants a kill
# switch, and they are pure intelligence improvements for every caller.

# Final-iteration wrap-up: appended to the per-call system prompt on the final
# allowed iteration, where the model can no longer call tools (we withhold them
# too). A model that knows it is on its last step produces a best-effort summary
# instead of dying mid-investigation.
_FINAL_ITERATION_WRAPUP = (
    "This is your final step; you cannot call any more tools after this. "
    "Provide your best final answer using the information you already have."
)

# Stall message: returned as a synthetic is_error result when the model asks
# for a tool call byte-for-byte identical to one already executed this run,
# instead of re-running it. Breaks the most common fixation loop.
_STALL_MESSAGE = (
    "You already called this tool with identical arguments; re-running it will "
    "not produce a different result. Try different arguments or a different "
    "approach."
)

# Consecutive-failure nudge: appended once, when this many tool results have
# failed in a row, telling the model to step back and reconsider.
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


class _RunDeadlineExceeded(TimeoutError):
    """Internal signal that the run-level wall clock expired."""


def _canonical_args(args: dict[str, Any]) -> str:
    """Stable string key for a tool call's arguments (for stall detection).

    `json.dumps(sort_keys=True)` makes `{"a":1,"b":2}` and `{"b":2,"a":1}` hash
    identically, so reordered-but-equivalent calls are caught. Tool inputs are
    JSON in practice (they came from the model), but we fall back to repr() if a
    value is ever non-serializable rather than letting stall detection raise.
    """
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(args)


def _validate_tool_args(schema: dict[str, Any] | None, args: dict[str, Any]) -> str | None:
    """Validate a tool call's arguments against its declared `input_schema`.

    Returns a teaching message (bad/missing field + expected shape) on
    failure, else None. A missing schema is treated as permissive -- we
    enforce what the tool declares, not constraints we'd have to invent. A
    malformed schema (the tool's own bug) is logged and skipped rather than
    blocking an otherwise-valid call.
    """
    if not schema:
        return None
    try:
        jsonschema.validate(instance=args, schema=schema)
    except jsonschema.ValidationError as exc:
        field = "/".join(str(p) for p in exc.path) or "(top level)"
        return (
            f"invalid arguments for field {field!r}: {exc.message}. "
            f"Expected shape: {json.dumps(exc.schema, ensure_ascii=False)}"
        )
    except jsonschema.SchemaError:
        logger.warning("tool input_schema is invalid; skipping arg validation", exc_info=True)
        return None
    return None


def _with_wrapup(system: str | None) -> str:
    """Append the final-iteration wrap-up note to a per-call system prompt."""
    if system:
        return f"{system}\n\n{_FINAL_ITERATION_WRAPUP}"
    return _FINAL_ITERATION_WRAPUP


def _clip_tool_content(content: str, max_chars: int | None) -> str:
    """Bound a flattened tool result so one large output can't flood context.

    The clipped text re-enters session history and is re-sent to the model on
    every later iteration, so the cap protects both the context window and the
    model's attention. A marker records how much was dropped.
    """
    if not max_chars or max_chars <= 0 or len(content) <= max_chars:
        return content
    omitted = len(content) - max_chars
    return f"{content[:max_chars]}\n…[truncated, {omitted} chars omitted]"


async def _complete_with_retry(
    llm: LLMClient,
    *,
    messages: list[Any],
    tools: list[dict[str, Any]] | None,
    system: str | None,
    max_tokens: int | None,
    timeout: float | None,
    max_retries: int,
    base_delay: float,
    thinking_level: str | None = None,
    deadline_expired: Callable[[], bool] | None = None,
    remaining_seconds: Callable[[], float | None] | None = None,
) -> AssistantMessage:
    """Call llm.complete() with a per-attempt timeout and bounded retries.

    Retries on conditions the client deems transient (rate limits, transient
    5xx, connection resets) and on per-attempt timeouts, plus once-more on an
    empty-candidates response (`stop_reason == "empty"`) — empty shares the
    same retry budget. Backoff is jittered exponential, capped.

    After the budget is exhausted: the last transient exception is re-raised
    (the caller turns it into ErrorEvent + DoneEvent("llm_error")), or the last
    empty response is returned (preserving the loop's existing "empty" floor).
    Non-transient exceptions are re-raised immediately without retrying.
    """
    attempts = max(0, max_retries) + 1
    last_response: AssistantMessage | None = None

    for attempt in range(attempts):
        if deadline_expired is not None and deadline_expired():
            raise _RunDeadlineExceeded()
        try:
            if timeout and timeout > 0:
                async with asyncio.timeout(timeout):
                    response = await llm.complete(
                        messages=messages,
                        tools=tools,
                        system=system,
                        max_tokens=max_tokens,
                        thinking_level=thinking_level,
                    )
            else:
                response = await llm.complete(
                    messages=messages,
                    tools=tools,
                    system=system,
                    max_tokens=max_tokens,
                    thinking_level=thinking_level,
                )
        except Exception as exc:
            if deadline_expired is not None and deadline_expired():
                raise _RunDeadlineExceeded() from exc
            transient = isinstance(exc, TimeoutError) or llm.is_transient_error(exc)
            if not transient or attempt == attempts - 1:
                raise
            delay = _backoff_delay(base_delay, attempt)
            logger.warning(
                "transient LLM error (attempt %d/%d), retrying in %.2fs: %s",
                attempt + 1, attempts, delay, exc,
            )
            if remaining_seconds is not None:
                remaining = remaining_seconds()
                if remaining is not None:
                    if remaining <= 0:
                        raise _RunDeadlineExceeded() from exc
                    delay = min(delay, remaining)
            await asyncio.sleep(delay)
            continue

        # Success. An empty-candidates response is retryable up to the budget.
        if response.stop_reason == "empty" and attempt < attempts - 1:
            last_response = response
            delay = _backoff_delay(base_delay, attempt)
            logger.warning(
                "empty LLM response (attempt %d/%d), retrying in %.2fs",
                attempt + 1, attempts, delay,
            )
            if remaining_seconds is not None:
                remaining = remaining_seconds()
                if remaining is not None:
                    if remaining <= 0:
                        raise _RunDeadlineExceeded()
                    delay = min(delay, remaining)
            await asyncio.sleep(delay)
            continue
        return response

    # Only reachable if the final attempt produced an empty response.
    assert last_response is not None
    return last_response


def _backoff_delay(base_delay: float, attempt: int) -> float:
    """Jittered exponential backoff, capped."""
    return min(_RETRY_BACKOFF_CAP_SECONDS, base_delay * (2 ** attempt)) + random.uniform(
        0, base_delay
    )


async def run_agent(
    session: Session,
    llm: LLMClient,
    mcp: MCPManager,
    *,
    store: SessionStore | None = None,
    system: str | None = None,
    max_iterations: int = 25,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    llm_timeout_seconds: float | None = None,
    tool_timeout_seconds: float | None = None,
    max_retries: int = 0,
    retry_base_delay: float = 0.5,
    tool_result_max_chars: int | None = None,
    max_run_tokens: int | None = None,
    max_run_seconds: float | None = None,
    abort_after_consecutive_tool_failures: int | None = None,
    thinking_level: str | None = None,
    tracer: Tracer | None = None,
    run_id: str | None = None,
) -> AsyncIterator[Event]:
    """Drive a conversation to completion, yielding events along the way.

    The caller is responsible for appending the user's new message to the
    session *before* calling this. The loop only handles assistant turns
    and the tool round-trips that follow.

    Args:
      session: the conversation to advance. Mutated in place.
      llm: the configured LLM client.
      mcp: the configured MCP manager (must already be started).
      store: optional session store; if provided, save() is called after
             each iteration. Pass None during smoke tests / one-shot scripts.
      system: optional system instruction for this call.
      max_iterations: safety cap on how many LLM round-trips we'll do for
                      one user message. 25 is generous; tune later.
      max_tokens: optional override for the LLM's max_tokens. None uses the
                  client's default (set from Settings).
      tools: optional pre-filtered tool list (generic schema, as produced
             by MCPManager.get_tools_for_llm()). When None, the loop pulls
             the full inventory. Used by the orchestrator-aware route to
             expose only a subset of tools to the model.
      llm_timeout_seconds: per-attempt cap on each llm.complete() call. None
             (or <=0) disables the timeout.
      tool_timeout_seconds: cap on each mcp.call_tool() call; on timeout the
             tool yields an is_error result so the model can react. None
             (or <=0) disables it.
      max_retries: retries on transient LLM failures / empty responses. 0
             preserves the legacy no-retry behavior.
      retry_base_delay: base seconds for jittered exponential backoff.
      tool_result_max_chars: clip threshold for a single flattened tool
             result before it enters session history. None (or <=0) disables
             clipping.
      max_run_tokens: hard ceiling on cumulative Usage.total_tokens for this
             run. On trip, exits done_reason="budget_exceeded" with the
             partial answer. None (or <=0) disables. Inert when the provider
             reports all-zero usage -- there is no token estimator (yet).
      max_run_seconds: hard wall-clock ceiling on this run, measured from
             just before the first iteration. On trip, exits
             done_reason="deadline_exceeded" with the partial answer. None
             (or <=0) disables.
      abort_after_consecutive_tool_failures: abort the run
             done_reason="no_progress" after this many tool-call failures in
             a row (a success resets the count). Checked in addition to the
             always-on consecutive-failure nudge at 3; set higher than 3 so
             the model gets a chance to recover first. None (or <=0)
             disables.
      thinking_level: optional "low"|"medium"|"high" deliberation hint passed
             to llm.complete() on every iteration of this run. None leaves the
             model default. The orchestrator-aware route supplies this from its
             routing decision.
      tracer: optional Tracer; when set, every yielded event is also serialized
             to it as a trace record (tagged with run_id, a monotonic step
             index, an ISO timestamp). None disables tracing with zero hot-path
             cost. Emit failures are logged and swallowed — tracing never breaks
             a run.
      run_id: opaque per-request id stamped on every trace record so a run's
             events stay correlated. Minted by the route.

    Reliability params default to legacy behavior (no timeout / no retry / no
    clip) so direct callers like smoke tests are unaffected; the /chat route
    opts in via Settings.
    """
    # If the caller hasn't pre-filtered, expose everything. Either way, the
    # actual call to mcp.call_tool() below dispatches by name -- the `tools`
    # list governs what the model SEES, not what the manager can ROUTE.
    if tools is None:
        tools = mcp.get_tools_for_llm()
    tool_schemas: dict[str, dict[str, Any] | None] = {
        t["name"]: t.get("input_schema") for t in tools
    }
    iteration = 0
    cumulative = Usage()
    run_started = time.perf_counter()

    # Run-scoped loop-intelligence state. `seen_calls` keys every executed
    # tool call so an identical repeat is short-circuited; `consecutive_tool_errors`
    # counts failures in a row so we can nudge the model once it starts thrashing.
    seen_calls: set[tuple[str, str]] = set()
    consecutive_tool_errors = 0

    # Trace step counter. Incremented per emitted event so the trace reconstructs
    # the run in order. Only advances when a tracer is attached.
    step = 0

    async def _emit(event: Event) -> Event:
        """Serialize an event to the tracer (best-effort) and return it to yield.

        Centralizes tracing at the one place every event passes through so the
        trace is exactly the serialized event log. A failing/slow sink is logged
        and swallowed: tracing must never break the request.
        """
        nonlocal step
        if tracer is not None:
            step += 1
            try:
                tracer.emit(event_record(event, run_id=run_id, step=step))
            except Exception:  # noqa: BLE001
                logger.warning("trace emit failed (run_id=%s)", run_id, exc_info=True)
        return event

    def _done(reason: str) -> DoneEvent:
        return DoneEvent(
            reason=reason,
            iterations=iteration,
            total_tokens=cumulative.total_tokens,
            input_tokens=cumulative.input_tokens,
            output_tokens=cumulative.output_tokens,
            thinking_tokens=cumulative.thinking_tokens,
        )

    def _remaining_run_seconds() -> float | None:
        if not max_run_seconds or max_run_seconds <= 0:
            return None
        return max_run_seconds - (time.perf_counter() - run_started)

    def _deadline_exceeded() -> bool:
        remaining = _remaining_run_seconds()
        return remaining is not None and remaining <= 0

    def _effective_timeout(per_call_timeout: float | None) -> float | None:
        timeouts = [
            t for t in (per_call_timeout, _remaining_run_seconds())
            if t is not None and t > 0
        ]
        return min(timeouts) if timeouts else None

    def _token_budget_exceeded() -> bool:
        return (
            bool(max_run_tokens)
            and max_run_tokens > 0
            and cumulative.total_tokens >= max_run_tokens
        )

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
                events.append(ToolCallEvent(
                    id=skipped.id,
                    name=skipped.name,
                    input=skipped.input,
                ))
            result = _skipped_result(skipped, content)
            events.append(ToolResultEvent(
                id=skipped.id,
                name=skipped.name,
                content=result.content,
                is_error=result.is_error,
                latency_ms=None,
            ))
            results.append(result)
        return events, results

    while iteration < max_iterations:
        # ----- bounded-run guards before spending another LLM call -----
        if _deadline_exceeded():
            elapsed = time.perf_counter() - run_started
            logger.warning("run exceeded max_run_seconds=%.1f (elapsed=%.1fs)",
                           max_run_seconds, elapsed)
            yield await _emit(_done(reason="deadline_exceeded"))
            return
        if _token_budget_exceeded():
            logger.warning("run exceeded max_run_tokens=%d (used=%d)",
                           max_run_tokens, cumulative.total_tokens)
            yield await _emit(_done(reason="budget_exceeded"))
            return

        iteration += 1

        # ----- final-iteration wrap-up -----
        # On the last allowed iteration the model cannot make another tool round
        # trip, so we withhold tools (forcing a final answer) and tell it so via
        # the per-call system prompt. system/tools are per-call configuration,
        # not session state, so this never pollutes history.
        is_final_iteration = iteration >= max_iterations
        if is_final_iteration:
            effective_system = _with_wrapup(system)
            effective_tools: list[dict[str, Any]] | None = None
        else:
            effective_system = system
            effective_tools = tools

        logger.debug("agent loop iteration %d (history=%d msgs, tools=%d, final=%s)",
                     iteration, len(session.messages),
                     len(effective_tools) if effective_tools else 0, is_final_iteration)

        # ----- LLM call (with per-attempt timeout + bounded retry) -----
        llm_started = time.perf_counter()
        try:
            response = await _complete_with_retry(
                llm,
                messages=session.messages,
                tools=effective_tools or None,
                system=effective_system,
                max_tokens=max_tokens,
                timeout=_effective_timeout(llm_timeout_seconds),
                max_retries=max_retries,
                base_delay=retry_base_delay,
                thinking_level=thinking_level,
                deadline_expired=_deadline_exceeded,
                remaining_seconds=_remaining_run_seconds,
            )
        except _RunDeadlineExceeded:
            elapsed = time.perf_counter() - run_started
            logger.warning("run exceeded max_run_seconds=%.1f during LLM call (elapsed=%.1fs)",
                           max_run_seconds, elapsed)
            yield await _emit(_done(reason="deadline_exceeded"))
            return
        except Exception as e:
            # LLM failures we cannot recover from (after retries). The model
            # never sees this; the caller does.
            logger.exception("LLM completion failed on iteration %d", iteration)
            yield await _emit(ErrorEvent(message=f"LLM call failed: {e}"))
            yield await _emit(_done(reason="llm_error"))
            return
        llm_latency_ms = round((time.perf_counter() - llm_started) * 1000, 2)

        # ----- record the assistant turn before doing anything else -----
        # If we crash later in this iteration, the session at least reflects
        # what the model said.
        session.append_assistant(response)

        usage = response.usage or Usage()
        cumulative = cumulative + usage
        yield await _emit(UsageEvent(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            thinking_tokens=usage.thinking_tokens,
            cached_tokens=usage.cached_tokens,
            iteration=iteration,
            latency_ms=llm_latency_ms,
        ))

        if store is not None:
            await store.save(session)

        # ----- stream out the text blocks -----
        for block in response.content:
            if isinstance(block, TextBlock) and block.text:
                yield await _emit(TextEvent(text=block.text))

        # ----- did the model want to call any tools? -----
        tool_uses: list[ToolUseBlock] = [
            b for b in response.content if isinstance(b, ToolUseBlock)
        ]

        # A guard can trip immediately after this LLM call's usage/latency is
        # known. If the assistant already requested tools, close that protocol
        # turn with synthetic skipped results before terminating the run.
        guard_reason: str | None = None
        skipped_message: str | None = None
        if _deadline_exceeded():
            guard_reason = "deadline_exceeded"
            skipped_message = _SKIPPED_TOOL_DEADLINE_MESSAGE
        elif _token_budget_exceeded():
            guard_reason = "budget_exceeded"
            skipped_message = _SKIPPED_TOOL_BUDGET_MESSAGE

        if guard_reason is not None:
            if tool_uses:
                assert skipped_message is not None
                events, results = _skipped_tool_events(tool_uses, skipped_message)
                for event in events:
                    yield await _emit(event)
                session.append_tool_results(results)
                if store is not None:
                    await store.save(session)
            if guard_reason == "deadline_exceeded":
                elapsed = time.perf_counter() - run_started
                logger.warning("run exceeded max_run_seconds=%.1f (elapsed=%.1fs)",
                               max_run_seconds, elapsed)
            else:
                logger.warning("run exceeded max_run_tokens=%d (used=%d)",
                               max_run_tokens, cumulative.total_tokens)
            yield await _emit(_done(reason=guard_reason))
            return

        if not tool_uses:
            # No tools requested -> the model is done. Distinguish a truncated
            # answer (stopped on max_tokens) from a genuine end_turn so callers
            # aren't handed a clipped response that looks complete. On the forced
            # final iteration we still report "max_iterations" (the run did hit
            # the cap) so the signal isn't lost just because we coaxed an answer.
            if response.stop_reason == "max_tokens":
                yield await _emit(_done(reason="truncated"))
            elif is_final_iteration:
                yield await _emit(_done(reason="max_iterations"))
            else:
                yield await _emit(_done(reason="end_turn" if response.content else "empty"))
            return

        # ----- execute the tools sequentially -----
        results: list[ToolResultBlock] = []
        for tool_index, tu in enumerate(tool_uses):
            yield await _emit(ToolCallEvent(id=tu.id, name=tu.name, input=tu.input))

            if _deadline_exceeded():
                events, skipped_results = _skipped_tool_events(
                    tool_uses[tool_index:],
                    _SKIPPED_TOOL_DEADLINE_MESSAGE,
                    first_call_already_emitted=True,
                )
                for event in events:
                    yield await _emit(event)
                results.extend(skipped_results)
                session.append_tool_results(results)
                if store is not None:
                    await store.save(session)
                elapsed = time.perf_counter() - run_started
                logger.warning("run exceeded max_run_seconds=%.1f before tool dispatch (elapsed=%.1fs)",
                               max_run_seconds, elapsed)
                yield await _emit(_done(reason="deadline_exceeded"))
                return

            # ----- stall detection -----
            # If the model asks for a call identical to one already run this
            # run, the result won't change. Skip execution and hand back a
            # synthetic error so the model breaks out instead of fixating.
            call_key = (tu.name, _canonical_args(tu.input))
            tool_latency_ms: float | None = None
            if call_key in seen_calls:
                logger.info("stall: repeat call to %s with identical args; skipping", tu.name)
                content = _STALL_MESSAGE
                is_error = True
            else:
                seen_calls.add(call_key)

                # ----- tool-argument validation -----
                # Model-hallucinated args would otherwise reach the MCP server
                # and come back as an opaque remote error. Validate against the
                # tool's own input_schema and short-circuit with a teaching
                # message naming the bad field -- mcp.call_tool never runs.
                validation_error = _validate_tool_args(tool_schemas.get(tu.name), tu.input)
                if validation_error is not None:
                    logger.info("invalid args for %s: %s", tu.name, validation_error)
                    content = validation_error
                    is_error = True
                    tool_latency_ms = None
                else:
                    tool_started = time.perf_counter()
                    try:
                        effective_tool_timeout = _effective_timeout(tool_timeout_seconds)
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
                            for event in events:
                                yield await _emit(event)
                            results.extend(skipped_results)
                            session.append_tool_results(results)
                            if store is not None:
                                await store.save(session)
                            elapsed = time.perf_counter() - run_started
                            logger.warning(
                                "run exceeded max_run_seconds=%.1f during tool dispatch (elapsed=%.1fs)",
                                max_run_seconds,
                                elapsed,
                            )
                            yield await _emit(_done(reason="deadline_exceeded"))
                            return
                        # A hung tool would otherwise hang the request (and lock the
                        # session via SessionGuard). Surface it as a tool-result error
                        # so the model can react and the loop keeps going.
                        logger.warning("tool %s timed out after %ss", tu.name, tool_timeout_seconds)
                        content = f"tool {tu.name!r} timed out after {tool_timeout_seconds}s"
                        is_error = True
                    except Exception as e:
                        # An exception escaping mcp.call_tool() is unusual (it normally
                        # returns ToolCallResult(is_error=True) on failures). Still,
                        # we convert to a tool-result-shaped error so the model can
                        # react rather than the whole loop dying.
                        logger.exception("tool execution raised for %s", tu.name)
                        content = f"tool execution raised: {e}"
                        is_error = True
                    tool_latency_ms = round((time.perf_counter() - tool_started) * 1000, 2)

            # Bound the result so one large output can't flood context for the
            # rest of the run. Clip once, before both the event and the block,
            # so streamed and stored content stay identical.
            content = _clip_tool_content(content, tool_result_max_chars)

            # ----- consecutive-failure nudge -----
            # Track failures in a row across the whole run; when the model starts
            # thrashing, append a one-line steering note (once, at the crossing)
            # so it re-reads the errors. Clip first so the nudge survives.
            consecutive_tool_errors = consecutive_tool_errors + 1 if is_error else 0
            if is_error and consecutive_tool_errors == _CONSECUTIVE_ERROR_NUDGE_THRESHOLD:
                content = f"{content}\n\n{_FAILURE_NUDGE}"

            yield await _emit(ToolResultEvent(
                id=tu.id,
                name=tu.name,
                content=content,
                is_error=is_error,
                latency_ms=tool_latency_ms,
            ))
            results.append(ToolResultBlock(
                tool_use_id=tu.id,
                name=tu.name,
                content=content,
                is_error=is_error,
            ))

            # ----- no-progress abort -----
            # The nudge (above) informs the model; this stops a cascade it
            # doesn't recover from. Checked after the nudge so the threshold
            # ordering (abort > nudge) always gives the model its shot first.
            # Ends the run mid-batch -- whatever results were already
            # collected this iteration are still recorded before returning.
            if (
                is_error
                and abort_after_consecutive_tool_failures
                and abort_after_consecutive_tool_failures > 0
                and consecutive_tool_errors >= abort_after_consecutive_tool_failures
            ):
                events, skipped_results = _skipped_tool_events(
                    tool_uses[tool_index + 1:],
                    _SKIPPED_TOOL_ABORT_MESSAGE,
                )
                for event in events:
                    yield await _emit(event)
                results.extend(skipped_results)
                logger.warning(
                    "aborting run: %d consecutive tool failures (threshold=%d)",
                    consecutive_tool_errors, abort_after_consecutive_tool_failures,
                )
                session.append_tool_results(results)
                if store is not None:
                    await store.save(session)
                yield await _emit(_done(reason="no_progress"))
                return

        # ----- record the tool results turn and loop -----
        session.append_tool_results(results)
        if store is not None:
            await store.save(session)

    # Structural terminal: the generator must always end on a DoneEvent. In
    # practice the final-iteration wrap-up makes this unreachable — it withholds tools,
    # so the model can't return a tool call and instead lands in the
    # no-tool-calls branch above (which already yields "max_iterations"). Kept as
    # a guaranteed terminal in case max_iterations is ever 0 or the wrap-up is
    # bypassed.
    logger.warning("agent loop hit max_iterations=%d without end_turn",
                   max_iterations)
    yield await _emit(_done(reason="max_iterations"))
