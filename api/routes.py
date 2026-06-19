# api/routes.py

"""
HTTP routes for the harness.

Two endpoints:
  POST /chat    -- run the agent loop on a plain-text prompt, return the answer
  GET  /health  -- liveness + connected MCP server summary

`/chat` is a plain-text dumb pipe: the request body IS the prompt and the
response body IS the answer. Session continuation rides on the `X-Session-Id`
header (echoed back on the response, alongside `X-Done-Reason`). The bespoke
JSON request/response schema was removed in favor of this minimal contract; an
OpenAI-compatible `/v1/chat/completions` endpoint (for OpenWebUI et al.) is a
separate adapter over the same core.

Per-request flow for /chat:
  1. Read the prompt (plain-text body); resolve/create the session.
  2. Orchestrate -> pick model + filter tools + generate system. If
     orchestration is disabled, fall back to the default LLM and all tools.
  3. Run the agent loop and stream-collect the final answer.
"""

from __future__ import annotations

import logging
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse

from agent import (
    DoneEvent,
    Event,
    Session,
    SessionBusyError,
    SessionGuard,
    SessionNotFoundError,
    SessionStore,
    TextEvent,
    Tracer,
    run_agent,
    run_logger,
)
from llm.client import LLMClient
from mcp_layer import MCPManager
from orchestrator import LLMRegistry, Orchestrator, ToolPreferences

from .dependencies import (
    get_guard,
    get_llm,
    get_mcp,
    get_orchestrator,
    get_registry,
    get_settings_obj,
    get_store,
    get_tracer,
)
from .schemas import HealthResponse, OrchestrationInfo, TokenUsage

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    mcp: MCPManager = Depends(get_mcp),
    settings = Depends(get_settings_obj),
    registry: Optional[LLMRegistry] = Depends(get_registry),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        provider=settings.llm_provider,
        model=settings.llm_model,
        connected_servers=mcp.connected_servers,
        tool_count=len(mcp.list_tools()),
        orchestration_enabled=registry is not None,
        available_model_ids=registry.model_ids if registry is not None else [],
    )


# ---------------------------------------------------------------------------
# Shared core: resolve routing, run the loop, surface its events.
# `_turn_events` is the single orchestrate->loop seam; `/chat` collects it into
# plain text and the OpenAI-compatible adapter (api/openai_compat.py) reuses it
# for both non-stream JSON and SSE.
# ---------------------------------------------------------------------------

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


async def _run_turn(**kwargs) -> tuple[str, str, TokenUsage]:
    """Collect a turn's events into a final answer. Returns (answer, done_reason, usage).

    A thin non-streaming wrapper over `_turn_events`; takes the same keyword
    arguments. Streaming callers iterate `_turn_events` directly.
    """
    text_parts: list[str] = []
    done_reason = "unknown"
    usage = TokenUsage()
    async for event in _turn_events(**kwargs):
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


@router.post("/chat", response_class=PlainTextResponse)
async def chat(
    request: Request,
    llm: LLMClient = Depends(get_llm),
    mcp: MCPManager = Depends(get_mcp),
    store: SessionStore = Depends(get_store),
    guard: SessionGuard = Depends(get_guard),
    settings = Depends(get_settings_obj),
    orchestrator: Optional[Orchestrator] = Depends(get_orchestrator),
    registry: Optional[LLMRegistry] = Depends(get_registry),
    tracer: Optional[Tracer] = Depends(get_tracer),
) -> PlainTextResponse:
    prompt = (await request.body()).decode("utf-8").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty body; send the prompt as plain text")

    # Resolve the session before routing so a follow-up turn is routed with the
    # conversation in view. The new prompt is appended later, inside _run_turn.
    session_id = request.headers.get("X-Session-Id")
    if session_id:
        try:
            session = await store.get(session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"session {session_id!r} not found")
    else:
        session = await store.create()

    answer, done_reason, _usage = await _run_turn(
        prompt=prompt,
        session=session,
        system_override=None,
        preferences=None,
        model_id=None,
        llm=llm,
        mcp=mcp,
        store=store,
        guard=guard,
        settings=settings,
        orchestrator=orchestrator,
        registry=registry,
        tracer=tracer,
    )
    return PlainTextResponse(
        answer,
        headers={"X-Session-Id": session.session_id, "X-Done-Reason": done_reason},
    )
