"""Guarded turn execution shared by native and OpenAI-compatible routes."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, AsyncIterator, Optional

from fastapi import HTTPException

from agent import (
    DoneEvent,
    Event,
    OrchestrationDecisionEvent,
    RunContext,
    RunLimits,
    Session,
    SessionBusyError,
    SessionGuard,
    SessionStore,
    TextEvent,
    ToolPolicy,
    Tracer,
    run_agent,
)
from llm.client import LLMClient
from mcp_layer import MCPManager, ToolSnapshot
from orchestrator import LLMRegistry, Orchestrator

from .schemas import TokenUsage

logger = logging.getLogger(__name__)


class PersistencePolicy(Enum):
    """Whether a turn publishes its session mutations to the native store."""

    PERSISTENT = "persistent"
    EPHEMERAL = "ephemeral"


@dataclass(frozen=True)
class TurnRequest:
    """All request-scoped input needed to execute one turn."""

    prompt: str
    session: Session
    persistence: PersistencePolicy
    system_override: str | None = None
    model_id: str | None = None
    stream: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.persistence, PersistencePolicy):
            raise TypeError("persistence must be a PersistencePolicy")


@dataclass(frozen=True)
class TurnMetadata:
    """Stable identity and sanitized routing facts for one accepted turn."""

    run_id: str
    model_id: str
    orchestration: OrchestrationDecisionEvent | None = None


@dataclass(frozen=True)
class TurnExecution:
    """Metadata plus events whose lifetime is owned by ``TurnRunner.open``."""

    metadata: TurnMetadata
    events: AsyncIterator[Event]


@dataclass(frozen=True)
class TurnResult:
    """Buffered terminal representation of a turn."""

    answer: str
    done_reason: str
    usage: TokenUsage
    metadata: TurnMetadata


@dataclass(frozen=True)
class _ResolvedRouting:
    llm: LLMClient
    tools: list[dict[str, Any]]
    system_prompt: str | None
    thinking_level: str | None
    model_id: str
    model_entry: Any = None
    orchestration: OrchestrationDecisionEvent | None = None


async def _collect(events: AsyncIterator[Event], metadata: TurnMetadata) -> TurnResult:
    text_parts: list[str] = []
    done_reason = "unknown"
    usage = TokenUsage()
    async for event in events:
        if isinstance(event, TextEvent):
            text_parts.append(event.text)
        elif isinstance(event, DoneEvent):
            done_reason = event.reason
            usage = TokenUsage.from_done_event(event)
    return TurnResult(
        answer="".join(text_parts).strip(),
        done_reason=done_reason,
        usage=usage,
        metadata=metadata,
    )


@dataclass(frozen=True)
class TurnRunner:
    """Own the complete lifecycle of an accepted turn."""

    llm: LLMClient
    mcp: MCPManager
    store: SessionStore
    guard: SessionGuard
    settings: Any
    orchestrator: Optional[Orchestrator]
    registry: Optional[LLMRegistry]
    policy: Optional[ToolPolicy]
    tracer: Optional[Tracer]
    limits: RunLimits = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "limits", RunLimits.from_settings(self.settings))

    def available_model_ids(self) -> list[str]:
        """Return the model IDs accepted by the OpenAI-compatible boundary."""
        try:
            if self.registry is not None:
                return self.registry.model_ids
            return [self.settings.llm_model]
        except Exception as exc:
            logger.exception("failed to read model inventory")
            raise HTTPException(
                status_code=500,
                detail="model inventory unavailable",
            ) from exc

    def validate_model_id(self, model_id: str | None) -> None:
        """Reject an explicit model ID that this runner cannot execute."""
        if model_id is None:
            return
        available = self.available_model_ids()
        if model_id not in available:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"invalid model {model_id!r}; available model IDs: "
                    f"{', '.join(available)}"
                ),
            )

    async def _resolve_routing(
        self,
        request: TurnRequest,
        tools: ToolSnapshot,
        context: RunContext,
    ) -> _ResolvedRouting:
        if self.orchestrator is None or self.registry is None:
            context.logger.info(
                "chat: legacy mode (orchestration disabled), tools=%d", len(tools.tools)
            )
            return _ResolvedRouting(
                llm=self.llm,
                tools=tools.as_llm_tools(),
                system_prompt=request.system_override,
                thinking_level=None,
                model_id=self.settings.llm_model,
            )

        decision = await self.orchestrator.decide(
            request.prompt,
            tools,
            history=request.session.messages,
            timeout=context.effective_timeout(self.limits.llm_timeout_seconds),
            log=context.logger,
        )
        result = decision.result
        if decision.fallback_used:
            context.logger.info(
                "orchestration fallback in effect: %s", decision.fallback_reason
            )

        if request.model_id is not None:
            resolved_id = request.model_id
            selected_llm = self.registry.get(resolved_id)
        else:
            resolved_id, selected_llm = self.registry.get_or_default(
                result.selected_model_id
            )
        try:
            model_entry = self.registry.get_entry(resolved_id)
        except (AttributeError, KeyError):
            model_entry = None

        selected_tools = tools.selected(result.selected_tools)
        system_prompt = (
            request.system_override
            if request.system_override is not None
            else result.generated_system_prompt
        )
        event = OrchestrationDecisionEvent(
            model_id=resolved_id,
            tools=[tool.name for tool in selected_tools.tools],
            system_prompt=system_prompt,
            fallback_used=decision.fallback_used,
            thinking_level=result.thinking_level,
        )
        context.logger.info(
            "chat: orchestrator picked model=%s tools=%d thinking=%s fallback=%s",
            resolved_id,
            len(selected_tools.tools),
            result.thinking_level,
            decision.fallback_used,
        )
        context.logger.debug(
            "chat: system_prompt=%r tools=%r", system_prompt, event.tools
        )
        return _ResolvedRouting(
            llm=selected_llm,
            tools=selected_tools.as_llm_tools(),
            system_prompt=system_prompt,
            thinking_level=result.thinking_level,
            model_id=resolved_id,
            model_entry=model_entry,
            orchestration=event,
        )

    async def _events(
        self,
        *,
        request: TurnRequest,
        routing: _ResolvedRouting,
        limits: RunLimits,
        context: RunContext,
    ) -> AsyncIterator[Event]:
        request.session.append_user(request.prompt)

        if routing.orchestration is not None:
            yield await context.emit(routing.orchestration)

        if context.deadline_exceeded():
            context.logger.warning(
                "turn deadline exhausted during routing (elapsed=%.3fs)",
                context.elapsed(),
            )
            yield await context.emit(
                DoneEvent(reason="deadline_exceeded", iterations=0)
            )
            return

        if request.persistence is PersistencePolicy.PERSISTENT:
            store = self.store
        elif request.persistence is PersistencePolicy.EPHEMERAL:
            store = None
        else:  # Defensive against future enum members.
            raise ValueError(f"unsupported persistence policy: {request.persistence!r}")

        agent_events = run_agent(
            session=request.session,
            llm=routing.llm,
            mcp=self.mcp,
            store=store,
            system=routing.system_prompt,
            tools=routing.tools,
            thinking_level=routing.thinking_level,
            limits=limits,
            context=context,
            policy=self.policy,
            stream=request.stream,
        )
        try:
            async for event in agent_events:
                if isinstance(event, DoneEvent):
                    context.logger.info(
                        "chat: done reason=%s iterations=%d tokens=%d session=%s",
                        event.reason,
                        event.iterations,
                        event.total_tokens,
                        request.session.session_id,
                    )
                yield event
        finally:
            await agent_events.aclose()

    @asynccontextmanager
    async def open(self, request: TurnRequest) -> AsyncIterator[TurnExecution]:
        """Claim a session and hold it through routing, iteration, and cleanup."""
        self.validate_model_id(request.model_id)
        try:
            async with self.guard.claim(request.session.session_id):
                context = RunContext.start(
                    max_run_seconds=self.limits.max_run_seconds,
                    base_logger=logger,
                    tracer=self.tracer,
                )
                if request.persistence is PersistencePolicy.PERSISTENT:
                    source_session = await self.store.get(request.session.session_id)
                elif request.persistence is PersistencePolicy.EPHEMERAL:
                    source_session = request.session
                else:  # Defensive against future enum members.
                    raise ValueError(
                        f"unsupported persistence policy: {request.persistence!r}"
                    )
                tool_snapshot = ToolSnapshot.from_llm_tools(
                    self.mcp.get_tools_for_llm()
                )
                staged_request = replace(
                    request,
                    session=source_session.staged_copy(),
                )
                routing = await self._resolve_routing(
                    staged_request, tool_snapshot, context
                )
                limits = self.limits.for_model(routing.model_entry)
                metadata = TurnMetadata(
                    run_id=context.run_id,
                    model_id=routing.model_id,
                    orchestration=routing.orchestration,
                )
                events = self._events(
                    request=staged_request,
                    routing=routing,
                    limits=limits,
                    context=context,
                )
                try:
                    yield TurnExecution(metadata=metadata, events=events)
                finally:
                    await events.aclose()
        except SessionBusyError:
            raise HTTPException(
                status_code=409,
                detail=f"session {request.session.session_id!r} is processing another request",
            )

    async def run(self, request: TurnRequest) -> TurnResult:
        """Buffer one turn while preserving the same guarded execution envelope."""
        async with self.open(request) as execution:
            return await _collect(execution.events, execution.metadata)
