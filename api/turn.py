# api/turn.py

"""
The shared orchestrate->loop core for one user turn.

Every API surface is a thin renderer over this module: `routes.py` (`/chat`
plain text, `/chat/stream` SSE events) and `openai_compatible.py` (`/v1` JSON
and SSE) all drive the same `_turn_events` generator and choose how to present
its events. Keeping the core here -- rather than inside any one route module --
lets those surfaces be peers over a neutral seam instead of importing each other.

`_turn_events` is the single place orchestration and the agent loop are wired
together; `_resolve_routing` decides model/tools/system for a turn.

`TurnRunner` is the narrow seam every renderer depends on: it bundles the
process-wide singletons so a route wires *one* object instead of eight, and
exposes `events()` (the live event stream) and `run()` (collect to a final
answer). It is also the extraction seam -- an out-of-process adapter would swap
only this class's body for an HTTP/SSE client, leaving the renderers unchanged.
"""

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
) -> tuple[LLMClient, list[dict], str | None, str | None, OrchestrationInfo | None]:
    """Decide model, tools, thinking level, and system prompt for one request.

    Returns (llm_client, tools_for_llm, system_prompt, thinking_level, info).

    Legacy mode (no orchestrator/registry): default LLM, full tool inventory,
    `system_override`, no thinking level. Orchestrated mode: `orchestrator.decide()`
    drives the choice, routed with `history` in view; `system_override`, when set,
    wins over the generated prompt. `model_id`, when it names a registered model,
    pins that model (the OpenAI adapter's `model` hint) while the orchestrator
    still selects tools and the system prompt.
    """
    if orchestrator is None or registry is None:
        return default_llm, mcp.get_tools_for_llm(), system_override, None, None

    decision = await orchestrator.decide(prompt, preferences=preferences, history=history)
    result = decision.result
    if decision.fallback_used:
        logger.info("orchestration fallback in effect: %s", decision.fallback_reason)

    # An explicit, valid model hint wins over the orchestrator's pick;
    # get_or_default still guards against a sanitized id we lost track of.
    chosen_id = model_id if (model_id and model_id in registry.model_ids) else result.selected_model_id
    resolved_id, llm = registry.get_or_default(chosen_id)

    # Filter against the live inventory so a disabled server can't slip a tool through.
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
    return llm, tools_for_llm, system_prompt, result.thinking_level, info


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
    tracer: Optional[Tracer] = None,
) -> AsyncIterator[Event]:
    """Run one user turn end to end, yielding the loop's events as they happen.

    Resolves routing, appends `prompt` to the session under a same-session guard,
    and drives the agent loop with the orchestrator's selections. This is the one
    place orchestration and the loop are wired together; callers choose how to
    render the events (plain text, OpenAI JSON, SSE deltas).

    Mints the per-request `run_id` here (covering both /chat and the /v1 adapter)
    and threads it into the loop's tracer and into a run-scoped logger so every
    log line for this turn carries it.
    """
    run_id = uuid.uuid4().hex
    rlog = run_logger(logger, run_id)

    selected_llm, selected_tools, system_prompt, thinking_level, orch_info = await _resolve_routing(
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

    # Distinct sessions are already isolated; the guard rejects a second
    # concurrent request on the SAME session (409) instead of interleaving.
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
                tools=selected_tools,
                llm_timeout_seconds=settings.llm_timeout_seconds,
                tool_timeout_seconds=settings.tool_timeout_seconds,
                max_retries=settings.llm_max_retries,
                retry_base_delay=settings.llm_retry_base_delay,
                tool_result_max_chars=settings.tool_result_max_chars,
                thinking_level=thinking_level,
                tracer=tracer,
                run_id=run_id,
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
    """Collect a turn's events into a final answer. Returns (answer, done_reason, usage).

    The non-streaming reduction over an event stream; streaming callers iterate
    the stream directly.
    """
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

    Holds the process-wide singletons (built once in main.py's lifespan, assembled
    on demand by `get_turn_runner`) so a renderer depends on this one object rather
    than wiring eight. `events()` yields the loop's typed events for one turn;
    `run()` reduces them to a final answer. Both take only per-request primitives,
    which is what makes this the clean extraction point for a future out-of-process
    adapter: swap the body, keep the renderers.
    """

    llm: LLMClient
    mcp: MCPManager
    store: SessionStore
    guard: SessionGuard
    settings: Any
    orchestrator: Optional[Orchestrator]
    registry: Optional[LLMRegistry]
    tracer: Optional[Tracer]

    def events(
        self,
        *,
        prompt: str,
        session: Session,
        system_override: str | None = None,
        preferences: ToolPreferences | None = None,
        model_id: str | None = None,
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
            tracer=self.tracer,
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
            )
        )
