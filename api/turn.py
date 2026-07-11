# api/turn.py

"""Shared orchestrate-to-loop core used by native and /v1 route renderers."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from fastapi import HTTPException

from agent import (
    DoneEvent,
    Event,
    Session,
    SessionBusyError,
    SessionGuard,
    SessionStore,
    TextEvent,
    ToolPolicy,
    Tracer,
    run_agent,
    run_logger,
)
from llm.client import LLMClient
from mcp_layer import MCPManager
from orchestrator import LLMRegistry, Orchestrator, ToolPreferences

from .schemas import OrchestrationInfo, TokenUsage

logger = logging.getLogger(__name__)


async def _resolve_routing(
    *,
    prompt: str,
    system_override: str | None,
    preferences: ToolPreferences | None,
    model_id: str | None,
    orchestrator: Optional[Orchestrator],
    registry: Optional[LLMRegistry],
    mcp: MCPManager,
    default_llm: LLMClient,
    history: list | None,
) -> tuple[LLMClient, list[dict], str | None, str | None, OrchestrationInfo | None, Any]:
    """Decide model, tools, thinking level, and system prompt for one request.

    Returns selected LLM, tools, system prompt, thinking level, orchestration
    metadata, and selected model entry (when available for context budgeting).
    """
    if orchestrator is None or registry is None:
        return default_llm, mcp.get_tools_for_llm(), system_override, None, None, None

    decision = await orchestrator.decide(prompt, preferences=preferences, history=history)
    result = decision.result
    if decision.fallback_used:
        logger.info("orchestration fallback in effect: %s", decision.fallback_reason)

    # A valid model hint wins over the orchestrator's model pick.
    chosen_id = model_id if (model_id and model_id in registry.model_ids) else result.selected_model_id
    resolved_id, llm = registry.get_or_default(chosen_id)

    # Best effort: fake registries/tests may not expose get_entry.
    try:
        model_entry = registry.get_entry(resolved_id)
    except Exception:  # noqa: BLE001
        model_entry = None

    # Filter against live inventory so disabled tools cannot slip through.
    selected = set(result.selected_tools)
    tools_for_llm = [t for t in mcp.get_tools_for_llm() if t["name"] in selected]

    system_prompt = system_override if system_override is not None else result.generated_system_prompt

    info = OrchestrationInfo(
        model_id=resolved_id,
        tools=[t["name"] for t in tools_for_llm],
        system_prompt=system_prompt,
        fallback_used=decision.fallback_used,
        thinking_level=result.thinking_level,
    )
    return llm, tools_for_llm, system_prompt, result.thinking_level, info, model_entry


async def _turn_events(
    *,
    prompt: str,
    session: Session,
    system_override: str | None,
    preferences: ToolPreferences | None,
    model_id: str | None,
    llm: LLMClient,
    mcp: MCPManager,
    store: SessionStore,
    guard: SessionGuard,
    settings,
    orchestrator: Optional[Orchestrator],
    registry: Optional[LLMRegistry],
    policy: Optional[ToolPolicy] = None,
    tracer: Optional[Tracer] = None,
    stream: bool = False,
) -> AsyncIterator[Event]:
    """Run one user turn end to end, yielding the loop's events as they happen.

    Resolves routing, appends the prompt under the same-session guard, and
    drives the agent loop with the selected model/tools/system.
    """
    run_id = uuid.uuid4().hex
    rlog = run_logger(logger, run_id)

    selected_llm, selected_tools, system_prompt, thinking_level, orch_info, model_entry = await _resolve_routing(
        prompt=prompt,
        system_override=system_override,
        preferences=preferences,
        model_id=model_id,
        orchestrator=orchestrator,
        registry=registry,
        mcp=mcp,
        default_llm=llm,
        history=session.messages,
    )

    if orch_info is not None:
        rlog.info(
            "chat: orchestrator picked model=%s tools=%d thinking=%s fallback=%s",
            orch_info.model_id, len(orch_info.tools),
            orch_info.thinking_level, orch_info.fallback_used,
        )
        rlog.debug("chat: system_prompt=%r tools=%r", orch_info.system_prompt, orch_info.tools)
    else:
        rlog.info("chat: legacy mode (orchestration disabled), tools=%d", len(selected_tools))

    # Selected model entry wins; Settings defaults fill gaps.
    context_window = (
        getattr(model_entry, "context_window", None)
        or settings.context_default_window_tokens
    )
    max_output_tokens = (
        getattr(model_entry, "max_tokens", None)
        or settings.llm_max_tokens
    )

    # Reject same-session concurrency instead of interleaving turns.
    try:
        async with guard.claim(session.session_id):
            session.append_user(prompt)
            async for event in run_agent(
                session=session,
                llm=selected_llm,
                mcp=mcp,
                store=store,
                system=system_prompt,
                max_iterations=settings.max_loop_iterations,
                max_tokens=max_output_tokens,
                tools=selected_tools,
                llm_timeout_seconds=settings.llm_timeout_seconds,
                tool_timeout_seconds=settings.tool_timeout_seconds,
                max_retries=settings.llm_max_retries,
                retry_base_delay=settings.llm_retry_base_delay,
                tool_result_max_chars=settings.tool_result_max_chars,
                max_run_tokens=settings.max_run_tokens,
                max_run_seconds=settings.max_run_seconds,
                abort_after_consecutive_tool_failures=settings.abort_after_consecutive_tool_failures,
                thinking_level=thinking_level,
                context_strategy=settings.context_strategy,
                context_window=context_window,
                context_safety_margin_tokens=settings.context_safety_margin_tokens,
                context_recent_messages=settings.context_recent_messages,
                context_summary_max_tokens=settings.context_summary_max_tokens,
                policy=policy,
                tracer=tracer,
                run_id=run_id,
                stream=stream,
            ):
                if isinstance(event, DoneEvent):
                    rlog.info(
                        "chat: done reason=%s iterations=%d tokens=%d session=%s",
                        event.reason, event.iterations, event.total_tokens, session.session_id,
                    )
                yield event
    except SessionBusyError:
        raise HTTPException(
            status_code=409,
            detail=f"session {session.session_id!r} is processing another request",
        )


async def _collect(events: AsyncIterator[Event]) -> tuple[str, str, TokenUsage]:
    """Collect a turn's events into (answer, done_reason, usage)."""
    text_parts: list[str] = []
    done_reason = "unknown"
    usage = TokenUsage()
    async for event in events:
        if isinstance(event, TextEvent):
            text_parts.append(event.text)
        elif isinstance(event, DoneEvent):
            done_reason = event.reason
            usage = TokenUsage(
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                total_tokens=event.total_tokens,
                thinking_tokens=event.thinking_tokens,
            )
    return "".join(text_parts).strip(), done_reason, usage


@dataclass(frozen=True)
class TurnRunner:
    """The seam between an API renderer and the orchestrate->loop core.

    Holds process-wide singletons; methods take only per-request primitives.
    """

    llm: LLMClient
    mcp: MCPManager
    store: SessionStore
    guard: SessionGuard
    settings: Any
    orchestrator: Optional[Orchestrator]
    registry: Optional[LLMRegistry]
    policy: Optional[ToolPolicy]
    tracer: Optional[Tracer]

    def events(
        self,
        *,
        prompt: str,
        session: Session,
        system_override: str | None = None,
        preferences: ToolPreferences | None = None,
        model_id: str | None = None,
        stream: bool = False,
    ) -> AsyncIterator[Event]:
        """Run one turn, yielding the loop's events live. See `_turn_events`."""
        return _turn_events(
            prompt=prompt,
            session=session,
            system_override=system_override,
            preferences=preferences,
            model_id=model_id,
            llm=self.llm,
            mcp=self.mcp,
            store=self.store,
            guard=self.guard,
            settings=self.settings,
            orchestrator=self.orchestrator,
            registry=self.registry,
            policy=self.policy,
            tracer=self.tracer,
            stream=stream,
        )

    async def run(
        self,
        *,
        prompt: str,
        session: Session,
        system_override: str | None = None,
        preferences: ToolPreferences | None = None,
        model_id: str | None = None,
    ) -> tuple[str, str, TokenUsage]:
        """Run one turn and collect it into (answer, done_reason, usage)."""
        return await _collect(
            self.events(
                prompt=prompt,
                session=session,
                system_override=system_override,
                preferences=preferences,
                model_id=model_id,
                stream=False,
            )
        )
