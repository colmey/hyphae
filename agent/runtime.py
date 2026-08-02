"""Per-run policy and live execution state for the agent loop."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, replace
from typing import Protocol

from .events import Event
from .tracing import Tracer, event_record, run_logger


class RunDeadlineExceeded(TimeoutError):
    """Internal signal that the run-level wall clock expired."""


class LLMRunSettings(Protocol):
    """Nested LLM execution values used to construct immutable run limits."""

    max_tokens: int
    timeout_seconds: float
    max_retries: int
    retry_base_delay: float


class RunSettings(Protocol):
    """Settings values used to construct immutable run limits."""

    loop_max_iterations: int
    llm: LLMRunSettings
    tool_timeout_seconds: float
    tool_result_max_chars: int
    run_max_tokens: int
    run_max_seconds: float
    abort_after_consecutive_tool_failures: int
    context_strategy: str
    context_default_window_tokens: int
    context_safety_margin_tokens: int
    context_recent_messages: int
    context_summary_max_tokens: int


class ModelLimits(Protocol):
    """Per-model output and context limits used by the agent."""

    @property
    def max_tokens(self) -> int | None: ...

    @property
    def context_window(self) -> int | None: ...


@dataclass(frozen=True)
class RunLimits:
    """Frozen policy values used throughout one run."""

    max_iterations: int = 10
    max_tokens: int | None = None
    llm_timeout_seconds: float | None = None
    tool_timeout_seconds: float | None = None
    max_retries: int = 0
    retry_base_delay: float = 0.5
    tool_result_max_chars: int | None = None
    max_run_tokens: int | None = None
    max_run_seconds: float | None = None
    abort_after_consecutive_tool_failures: int | None = None
    context_strategy: str = "naive"
    context_window: int | None = None
    context_safety_margin_tokens: int = 1024
    context_recent_messages: int = 6
    context_summary_max_tokens: int = 512

    @classmethod
    def from_settings(cls, settings: RunSettings) -> "RunLimits":
        return cls(
            max_iterations=settings.loop_max_iterations,
            max_tokens=settings.llm.max_tokens,
            llm_timeout_seconds=settings.llm.timeout_seconds,
            tool_timeout_seconds=settings.tool_timeout_seconds,
            max_retries=settings.llm.max_retries,
            retry_base_delay=settings.llm.retry_base_delay,
            tool_result_max_chars=settings.tool_result_max_chars,
            max_run_tokens=settings.run_max_tokens,
            max_run_seconds=settings.run_max_seconds,
            abort_after_consecutive_tool_failures=settings.abort_after_consecutive_tool_failures,
            context_strategy=settings.context_strategy,
            context_window=settings.context_default_window_tokens,
            context_safety_margin_tokens=settings.context_safety_margin_tokens,
            context_recent_messages=settings.context_recent_messages,
            context_summary_max_tokens=settings.context_summary_max_tokens,
        )

    def for_model(self, model_entry: ModelLimits | None) -> "RunLimits":
        if model_entry is None:
            return self
        return replace(
            self,
            max_tokens=model_entry.max_tokens or self.max_tokens,
            context_window=model_entry.context_window or self.context_window,
        )


@dataclass
class RunContext:
    """One run ID, absolute deadline, logger, and trace sequence."""

    started_at: float
    deadline: float | None
    run_id: str
    logger: logging.LoggerAdapter
    tracer: Tracer | None = None
    trace_step: int = 0

    @classmethod
    def start(
        cls,
        *,
        max_run_seconds: float | None,
        base_logger: logging.Logger,
        tracer: Tracer | None = None,
        run_id: str | None = None,
    ) -> "RunContext":
        started_at = time.perf_counter()
        deadline = (
            started_at + max_run_seconds
            if max_run_seconds is not None and max_run_seconds > 0
            else None
        )
        resolved_run_id = run_id or uuid.uuid4().hex
        return cls(
            started_at=started_at,
            deadline=deadline,
            run_id=resolved_run_id,
            logger=run_logger(base_logger, resolved_run_id),
            tracer=tracer,
        )

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def remaining_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return self.deadline - time.perf_counter()

    def deadline_exceeded(self) -> bool:
        remaining = self.remaining_seconds()
        return remaining is not None and remaining <= 0

    def effective_timeout(self, per_call_timeout: float | None) -> float | None:
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            return 0.0
        enabled = [
            value
            for value in (per_call_timeout, remaining)
            if value is not None and value > 0
        ]
        return min(enabled) if enabled else None

    async def emit(self, event: Event) -> Event:
        if self.tracer is not None:
            self.trace_step += 1
            try:
                self.tracer.emit(
                    event_record(event, run_id=self.run_id, step=self.trace_step)
                )
            except Exception:  # noqa: BLE001 -- tracing must never break a run.
                self.logger.warning("trace emit failed")
        return event
