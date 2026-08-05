"""Framework-neutral execution of one accepted user turn."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, AsyncContextManager, Protocol, runtime_checkable

from agent import (
    DoneEvent,
    Event,
    OrchestrationDecisionEvent,
    RunContext,
    RunLimits,
    Session,
    SessionGuard,
    SessionHistoryLimitExceeded,
    SessionStore,
    TextEvent,
    PolicyVerdict,
    ToolPolicy,
    Tracer,
    run_agent,
)
from agent.session import session_history_chars
from agent.runtime import ModelLimits
from llm.client import LLMClient
from llm.schemas import CompletionUsage
from mcp_runtime import MCPServerStatus, Tool
from orchestrator import ModelUnavailableError
from orchestrator.contracts import ModelRegistry, RoutingService
from tooling import ToolRuntime, ToolSnapshot

from config import Settings

logger = logging.getLogger(__name__)


class PersistencePolicy(Enum):
    """Whether a turn publishes its session mutations to the native store."""

    PERSISTENT = "persistent"
    EPHEMERAL = "ephemeral"


class InvalidModelError(ValueError):
    """An explicit model ID is not executable in this runtime."""

    code = "invalid_model"

    def __init__(self, model_id: str, available_model_ids: list[str]) -> None:
        self.model_id = model_id
        self.available_model_ids = tuple(available_model_ids)
        super().__init__(
            f"invalid model {model_id!r}; available model IDs: "
            f"{', '.join(available_model_ids)}"
        )


class ModelInventoryError(RuntimeError):
    """The runtime cannot safely read its executable model inventory."""

    code = "model_inventory_unavailable"

    def __init__(self) -> None:
        super().__init__("model inventory unavailable")


class ExecutionProtocolError(RuntimeError):
    """A buffered event sequence did not contain one final terminal event."""

    code = "execution_protocol_error"


class RuntimeConfigurationError(RuntimeError):
    """Application composition contains an impossible routing arrangement."""

    code = "runtime_configuration_error"


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
    usage: CompletionUsage
    metadata: TurnMetadata


@dataclass(frozen=True, slots=True)
class UnorchestratedRouting:
    """Fixed client and model used when no routing service is active."""

    llm: LLMClient
    model_id: str
    inventory: ModelRegistry | None = None
    advertised_model_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.llm is None:
            raise RuntimeConfigurationError("direct routing requires an executable client")
        if self.inventory is None:
            if self.advertised_model_ids is not None:
                raise RuntimeConfigurationError(
                    "direct routing cannot advertise a separate inventory"
                )
            return
        if self.advertised_model_ids != (self.model_id,):
            raise RuntimeConfigurationError(
                "fixed degraded routing must advertise only its executable default"
            )


@dataclass(frozen=True, slots=True)
class OrchestratedRouting:
    """Routing service and registry used for per-turn model selection."""

    orchestrator: RoutingService
    registry: ModelRegistry
    agent_system_prompt: str

    def __post_init__(self) -> None:
        if self.orchestrator is None or self.registry is None:
            raise RuntimeConfigurationError(
                "orchestrated routing requires both orchestrator and registry"
            )


type RoutingRuntime = UnorchestratedRouting | OrchestratedRouting


class TurnToolProvider(Protocol):
    """Accepted-turn factory for a neutral, turn-local tool runtime."""

    def open_turn(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncContextManager[ToolRuntime]: ...


@runtime_checkable
class ApplicationMCP(TurnToolProvider, Protocol):
    """Turn ownership plus the HTTP health/catalog views of one MCP owner."""

    @property
    def connected_servers(self) -> list[str]: ...

    def status_snapshot(self) -> tuple[MCPServerStatus, ...]: ...

    def list_tools(self) -> list[tuple[str, Tool]]: ...


@dataclass(frozen=True, slots=True)
class ApplicationRuntime:
    """Process-lifetime composition from which requests derive a TurnRunner."""

    settings: Settings
    routing: RoutingRuntime
    limits: RunLimits
    mcp: ApplicationMCP
    store: SessionStore
    guard: SessionGuard
    policy: ToolPolicy | None
    tracer: Tracer | None

    def turn_runner(self) -> "TurnRunner":
        return TurnRunner(
            routing=self.routing,
            limits=self.limits,
            mcp=self.mcp,
            store=self.store,
            guard=self.guard,
            policy=self.policy,
            tracer=self.tracer,
        )


@dataclass(frozen=True)
class _ResolvedRouting:
    llm: LLMClient
    tools: list[dict[str, Any]]
    system_prompt: str | None
    thinking_level: str | None
    model_id: str
    model_entry: ModelLimits | None = None
    orchestration: OrchestrationDecisionEvent | None = None


async def _collect(events: AsyncIterator[Event], metadata: TurnMetadata) -> TurnResult:
    text_parts: list[str] = []
    terminal: DoneEvent | None = None
    async for event in events:
        if terminal is not None:
            raise ExecutionProtocolError("event received after terminal event")
        if isinstance(event, TextEvent):
            text_parts.append(event.text)
        elif isinstance(event, DoneEvent):
            terminal = event
    if terminal is None:
        raise ExecutionProtocolError("event stream ended without a terminal event")
    return TurnResult(
        answer="".join(text_parts).strip(),
        done_reason=terminal.reason,
        usage=CompletionUsage(
            input_tokens=terminal.input_tokens,
            output_tokens=terminal.output_tokens,
            total_tokens=terminal.total_tokens,
            thinking_tokens=terminal.thinking_tokens,
        ),
        metadata=metadata,
    )


@dataclass(frozen=True)
class TurnRunner:
    """Own the complete lifecycle of an accepted turn."""

    routing: RoutingRuntime
    limits: RunLimits
    mcp: TurnToolProvider
    store: SessionStore
    guard: SessionGuard
    policy: ToolPolicy | None
    tracer: Tracer | None

    async def create_session(self) -> Session:
        """Create a persistent session while preserving active checkpoints."""
        return await self.store.create(
            protected_session_ids=self.guard.claimed_session_ids
        )

    def available_model_ids(self) -> list[str]:
        """Return the model IDs accepted by the OpenAI-compatible boundary."""
        try:
            if isinstance(self.routing, OrchestratedRouting):
                return self.routing.registry.model_ids
            if self.routing.advertised_model_ids is not None:
                return list(self.routing.advertised_model_ids)
            return [self.routing.model_id]
        except Exception as exc:
            logger.exception("failed to read model inventory")
            raise ModelInventoryError() from exc

    def validate_model_id(self, model_id: str | None) -> None:
        """Reject an explicit model ID that this runner cannot execute."""
        if model_id is None:
            return
        available = self.available_model_ids()
        if model_id not in available:
            registry = (
                self.routing.registry
                if isinstance(self.routing, OrchestratedRouting)
                else self.routing.inventory
            )
            if registry is not None and registry.is_configured(model_id):
                raise ModelUnavailableError(model_id)
            raise InvalidModelError(model_id, available)

    async def _resolve_routing(
        self,
        request: TurnRequest,
        tools: ToolSnapshot,
        context: RunContext,
    ) -> _ResolvedRouting:
        if isinstance(self.routing, UnorchestratedRouting):
            context.logger.info("chat: unorchestrated mode, tools=%d", len(tools.tools))
            return _ResolvedRouting(
                llm=self.routing.llm,
                tools=tools.as_llm_tools(),
                system_prompt=request.system_override,
                thinking_level=None,
                model_id=self.routing.model_id,
            )

        decision = await self.routing.orchestrator.decide(
            request.prompt,
            tools,
            history=request.session.messages,
            timeout=context.effective_timeout(self.limits.llm_timeout_seconds),
            log=context.logger,
        )
        proposal = decision.result
        if decision.fallback_used:
            context.logger.info(
                "orchestration fallback in effect: %s", decision.fallback_reason
            )

        if request.model_id is not None:
            resolved_id = request.model_id
            selected_llm = self.routing.registry.get(resolved_id)
        else:
            resolved_id, selected_llm = self.routing.registry.get_or_default(
                proposal.selected_model_id
            )
        model_entry = self.routing.registry.get_entry(resolved_id)

        selected_tools = tools.select(proposal.selected_tools)
        system_prompt = (
            request.system_override
            if request.system_override is not None
            else self.routing.agent_system_prompt
        )
        event = OrchestrationDecisionEvent(
            model_id=resolved_id,
            tools=[tool.name for tool in selected_tools.tools],
            fallback_used=decision.fallback_used,
            fallback_reason=decision.fallback_reason,
            corrections=decision.corrections,
            control_model_id=decision.control_model_id,
            input_tokens=decision.usage.input_tokens,
            output_tokens=decision.usage.output_tokens,
            total_tokens=decision.usage.total_tokens,
            thinking_tokens=decision.usage.thinking_tokens,
            cached_tokens=decision.usage.cached_tokens,
            latency_ms=decision.latency_ms,
            thinking_level=proposal.thinking_level,
        )
        context.logger.info(
            "chat: orchestrator picked model=%s tools=%d thinking=%s fallback=%s",
            resolved_id,
            len(selected_tools.tools),
            proposal.thinking_level,
            decision.fallback_used,
        )
        return _ResolvedRouting(
            llm=selected_llm,
            tools=selected_tools.as_llm_tools(),
            system_prompt=system_prompt,
            thinking_level=proposal.thinking_level,
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
        tool_runtime: ToolRuntime,
    ) -> AsyncGenerator[Event, None]:
        request.session.append_user(request.prompt)

        if routing.orchestration is not None:
            yield await context.emit(routing.orchestration)

        if context.deadline_exceeded():
            context.logger.warning(
                "turn deadline exhausted during routing (elapsed=%.3fs)",
                context.elapsed_seconds(),
            )
            yield await context.emit(
                DoneEvent(
                    reason="deadline_exceeded",
                    iterations=0,
                    total_tokens=routing.orchestration.total_tokens
                    if routing.orchestration is not None else 0,
                    input_tokens=routing.orchestration.input_tokens
                    if routing.orchestration is not None else 0,
                    output_tokens=routing.orchestration.output_tokens
                    if routing.orchestration is not None else 0,
                    thinking_tokens=routing.orchestration.thinking_tokens
                    if routing.orchestration is not None else 0,
                )
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
            mcp=tool_runtime,
            store=store,
            protected_session_ids=(
                self.guard.claimed_session_ids
                if request.persistence is PersistencePolicy.PERSISTENT
                else None
            ),
            system=routing.system_prompt,
            tools=routing.tools,
            thinking_level=routing.thinking_level,
            limits=limits,
            context=context,
            policy=self.policy,
            stream=request.stream,
            initial_usage=(
                CompletionUsage(
                    input_tokens=routing.orchestration.input_tokens,
                    output_tokens=routing.orchestration.output_tokens,
                    total_tokens=routing.orchestration.total_tokens,
                    thinking_tokens=routing.orchestration.thinking_tokens,
                    cached_tokens=routing.orchestration.cached_tokens,
                )
                if routing.orchestration is not None
                else None
            ),
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
        async with self.guard.claim(request.session.session_id):
            context = RunContext.start(
                max_run_seconds=self.limits.max_run_seconds,
                base_logger=logger,
                tracer=self.tracer,
            )
            if request.persistence is PersistencePolicy.PERSISTENT:
                source_session = await self.store.get(request.session.session_id)
                prospective = source_session.staged_copy()
                prospective.append_user(request.prompt)
                if (
                    session_history_chars(prospective.messages)
                    > self.store.session_history_max_chars
                ):
                    raise SessionHistoryLimitExceeded
            elif request.persistence is PersistencePolicy.EPHEMERAL:
                source_session = request.session
            else:  # Defensive against future enum members.
                raise ValueError(f"unsupported persistence policy: {request.persistence!r}")
            async with self.mcp.open_turn(
                timeout_seconds=context.remaining_seconds()
            ) as tool_runtime:
                tool_snapshot = ToolSnapshot.from_llm_tools(tool_runtime.get_tools_for_llm())
                policy = self.policy or ToolPolicy()
                visible_tools = ToolSnapshot(
                    tool
                    for tool in tool_snapshot.tools
                    if policy.check(tool.name).verdict is PolicyVerdict.ALLOW
                )
                staged_request = replace(request, session=source_session.staged_copy())
                routing = await self._resolve_routing(staged_request, visible_tools, context)
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
                    tool_runtime=tool_runtime,
                )
                try:
                    yield TurnExecution(metadata=metadata, events=events)
                finally:
                    await events.aclose()
    async def run(self, request: TurnRequest) -> TurnResult:
        """Buffer one turn while preserving the same guarded execution envelope."""
        async with self.open(request) as execution:
            return await _collect(execution.events, execution.metadata)
