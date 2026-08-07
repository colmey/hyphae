"""Run-scoped validation, dispatch, and protocol state for tool calls."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Protocol

import jsonschema
from jsonschema.validators import validator_for

from hyphae.llm.schemas import Message, ToolResultBlock, ToolUseBlock
from hyphae.tooling import ToolRuntime

from .runtime import RunContext, RunDeadlineExceeded, RunLimits
from .tool_policy import PolicyVerdict, ToolPolicy


logger = logging.getLogger(__name__)

_STALL_MESSAGE = (
    "You already called this tool with identical arguments; re-running it will "
    "not produce a different result. Try different arguments or a different "
    "approach."
)
_CANCELLED_TOOL_UNKNOWN_MESSAGE = (
    "tool call outcome is unknown because execution was cancelled while the "
    "call was in flight"
)
_CANCELLED_TOOL_NOT_STARTED_MESSAGE = (
    "tool call was not executed because execution was cancelled"
)
_TOOL_EXECUTION_FAILED_MESSAGE = (
    "tool execution failed; try again later or use a different approach"
)


@dataclass(frozen=True, slots=True)
class ToolDispatchResult:
    """Provider-neutral outcome returned to the agent loop."""

    content: str
    is_error: bool
    latency_ms: float | None


class _ToolResultSink(Protocol):
    """Minimal transcript capability needed to append a complete batch."""

    def append_tool_results(self, results: list[ToolResultBlock]) -> Message: ...


@dataclass
class ActiveToolBatch:
    """Ordered results and in-flight state for one assistant tool-use batch."""

    tool_uses: tuple[ToolUseBlock, ...]
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
        if (result.tool_use_id, result.name) != (
            expected.id,
            expected.name,
        ):
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

    def append_to(self, sink: _ToolResultSink) -> None:
        if self._appended:
            return
        if self._next_index != len(self.tool_uses):
            raise ValueError("cannot append an incomplete tool-result batch")
        sink.append_tool_results(self._results)
        self._appended = True


def make_skipped_tool_result(
    tool_use: ToolUseBlock,
    content: str,
) -> ToolResultBlock:
    """Create one synthetic error result for a call that was not dispatched."""
    return ToolResultBlock(
        tool_use_id=tool_use.id,
        name=tool_use.name,
        content=content,
        is_error=True,
    )


def _canonical_args(args: dict[str, Any]) -> str:
    """Return a stable argument key without allowing serialization to fail."""
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(args)


class _ArgumentValidator(Protocol):
    """Run-local argument-validation seam used by tool dispatch."""

    def validate(self, args: dict[str, Any]) -> str | None: ...


class _ConcreteSchemaValidator(Protocol):
    """Minimal jsonschema validator surface retained by the dispatcher."""

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
    log: logging.Logger | logging.LoggerAdapter[logging.Logger]
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
    tools: Sequence[Mapping[str, Any]],
    *,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] = logger,
) -> dict[str, _ArgumentValidator]:
    """Compile each advertised tool schema once for this run."""
    schemas = {str(tool["name"]): tool.get("input_schema") or {} for tool in tools}
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


class ToolDispatcher:
    """Run-scoped validation, authorization, repeat detection, and dispatch."""

    def __init__(
        self,
        *,
        runtime: ToolRuntime,
        policy: ToolPolicy,
        tools: Sequence[Mapping[str, Any]],
        limits: RunLimits,
        context: RunContext,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    ) -> None:
        self._runtime = runtime
        self._policy = policy
        self._validators = _compile_tool_validators(tools, log=log)
        self._limits = limits
        self._context = context
        self._log = log
        self._seen_calls: set[tuple[str, str]] = set()

    async def dispatch(
        self,
        batch: ActiveToolBatch,
        tool_use: ToolUseBlock,
    ) -> ToolDispatchResult:
        """Dispatch one call after applying all run-local synthetic guards."""
        guard_result = self._guard_result(tool_use)
        if guard_result is not None:
            return guard_result
        return await self._invoke(batch, tool_use)

    def _guard_result(
        self,
        tool_use: ToolUseBlock,
    ) -> ToolDispatchResult | None:
        """Return a synthetic result or authorize one real invocation."""
        call_key = (tool_use.name, _canonical_args(tool_use.input))
        if call_key in self._seen_calls:
            self._log.info(
                "stall: repeat call to %s with identical args; skipping",
                tool_use.name,
            )
            return ToolDispatchResult(_STALL_MESSAGE, True, None)
        self._seen_calls.add(call_key)

        validation_error: str | None
        if tool_use.parse_error is not None:
            validation_error = (
                "tool call arguments were not valid JSON "
                f"({tool_use.parse_error}); "
                "return the arguments as a JSON object matching the tool schema."
            )
        else:
            validator = self._validators.get(tool_use.name, _PERMISSIVE_VALIDATOR)
            validation_error = validator.validate(tool_use.input)

        if validation_error is not None:
            self._log.info(
                "invalid args for %s: %s",
                tool_use.name,
                validation_error,
            )
            return ToolDispatchResult(validation_error, True, None)

        decision = self._policy.check(tool_use.name, tool_use.input)
        if decision.verdict is PolicyVerdict.DENY:
            self._log.info("policy denied %s", tool_use.name)
            reason = decision.reason
            assert reason is not None
            return ToolDispatchResult(reason, True, None)

        return None

    async def _invoke(
        self,
        batch: ActiveToolBatch,
        tool_use: ToolUseBlock,
    ) -> ToolDispatchResult:
        """Invoke one authorized tool with timeout and deadline classification."""
        effective_timeout = self._context.effective_timeout(
            self._limits.tool_timeout_seconds
        )
        if effective_timeout == 0.0 and self._context.deadline_exceeded():
            raise RunDeadlineExceeded()

        tool_started = time.perf_counter()
        batch.start_dispatch(tool_use)
        try:
            if effective_timeout is not None and effective_timeout > 0:
                async with asyncio.timeout(effective_timeout):
                    call_result = await self._runtime.call_tool(
                        tool_use.name,
                        tool_use.input,
                    )
            else:
                call_result = await self._runtime.call_tool(
                    tool_use.name,
                    tool_use.input,
                )
            content = call_result.content
            is_error = call_result.is_error
        except TimeoutError as exc:
            if self._context.deadline_exceeded():
                raise RunDeadlineExceeded() from exc
            self._log.warning(
                "tool %s timed out after %ss",
                tool_use.name,
                self._limits.tool_timeout_seconds,
            )
            content = (
                f"tool {tool_use.name!r} timed out after "
                f"{self._limits.tool_timeout_seconds}s"
            )
            is_error = True
        except Exception:
            self._log.exception("tool execution raised for %s", tool_use.name)
            content = _TOOL_EXECUTION_FAILED_MESSAGE
            is_error = True

        latency_ms = round((time.perf_counter() - tool_started) * 1000, 2)
        return ToolDispatchResult(content, is_error, latency_ms)
