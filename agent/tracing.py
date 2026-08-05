# agent/tracing.py

"""
Run tracing: serialize the loop's event stream as an append-only JSONL trace.

The agent loop already keeps an append-only event log as its run state (see
`agent/events.py`). A *trace* is just that log written to disk — one JSON record
per event, tagged with the run's `run_id`, a monotonic step index, an ISO
timestamp, and (for LLM/tool steps) latency. This is observability for nearly
free: we serialize the events the loop emits anyway rather than running a second
logging subsystem (research Doc 08, "the trace is the state log").

The `Tracer` ABC is the swappable seam, mirroring `SessionStore`: a no-op
default, a JSONL implementation today, and a clean path to an OpenTelemetry
exporter later (instrument once, send anywhere) without touching the loop.

A tracer is neither the LLM client nor the MCP manager, so threading one into
`run_agent` preserves the "loop is the only bridge" invariant. Tracing is also
strictly optional: a missing or failing sink degrades silently and never breaks
a request — the loop wraps every emit, and a tracer that can't open its file
falls back to no tracing at startup.

Sensitivity: the JSONL trace captures *full* message text, tool arguments, and
tool results by default. Route authentication does not protect the local trace
file, so full bodies are appropriate only for the single-operator dev harness;
treat the trace file as sensitive. A
metadata-only mode (omit bodies, keep names/usage/latency) is the natural next
toggle once multi-tenant exposure exists.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, MutableMapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Any, Protocol

from .events import (
    DoneEvent,
    ErrorEvent,
    Event,
    OrchestrationDecisionEvent,
    ReasoningEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)

logger = logging.getLogger(__name__)

TRACE_QUEUE_CAPACITY = 4096
TRACE_BATCH_SIZE = 100
TRACE_FLUSH_INTERVAL_SECONDS = 0.250
TRACE_OVERFLOW_WARNING_INTERVAL_SECONDS = 60.0
_TRACE_PATH_DISPLAY_MAX_CHARS = 512


def _safe_path_display(path: Path) -> str:
    printable = "".join(
        character if character.isprintable() else " " for character in str(path)
    )
    display = " ".join(printable.split()).strip()
    if len(display) <= _TRACE_PATH_DISPLAY_MAX_CHARS:
        return display
    return f"{display[: _TRACE_PATH_DISPLAY_MAX_CHARS - 3]}..."


class _TracerState(Enum):
    NEW = "new"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    CLOSED = "closed"


class _Stop:
    pass


_STOP = _Stop()
_QueueItem = str | _Stop
type TraceRecord = dict[str, object]


# ---------------------------------------------------------------------------
# Tracer seam
# ---------------------------------------------------------------------------


class Tracer(ABC):
    """Sink for trace records, one per loop event. The swappable observability
    seam (JSONL now, OTel later), analogous to SessionStore."""

    @abstractmethod
    async def start(self) -> None:
        """Start any background work after an event loop exists."""
        ...

    @abstractmethod
    def emit(self, record: TraceRecord) -> None:
        """Submit one record without awaiting or performing file I/O."""
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Drain accepted records and release resources."""
        ...

    @property
    @abstractmethod
    def accepted(self) -> int: ...

    @property
    @abstractmethod
    def written(self) -> int: ...

    @property
    @abstractmethod
    def dropped(self) -> int: ...

    @property
    @abstractmethod
    def writer_failures(self) -> int: ...


class NoOpTracer(Tracer):
    """Discards every record. The explicit 'tracing off' tracer; the loop also
    treats `tracer is None` as off, so the hot path stays free of even this call."""

    async def start(self) -> None:
        pass

    def emit(self, record: TraceRecord) -> None:  # noqa: D401 - intentional no-op
        pass

    async def aclose(self) -> None:
        pass

    @property
    def accepted(self) -> int:
        return 0

    @property
    def written(self) -> int:
        return 0

    @property
    def dropped(self) -> int:
        return 0

    @property
    def writer_failures(self) -> int:
        return 0


class _BatchSink(Protocol):
    """Synchronous sink owned exclusively by the background writer task."""

    def write_batch(self, lines: tuple[str, ...]) -> None: ...

    def close(self) -> None: ...


class _JSONLFileSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def write_batch(self, lines: tuple[str, ...]) -> None:
        self._fh.writelines(lines)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class JSONLTracer(Tracer):
    """Bounded, append-only JSONL sink with an off-loop batch writer."""

    def __init__(
        self,
        path: Path | str,
        *,
        _sink_factory: Callable[[Path], _BatchSink] = _JSONLFileSink,
    ) -> None:
        self._path = Path(path)
        self._sink_factory = _sink_factory
        self._queue: asyncio.Queue[_QueueItem] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._startup_ready: asyncio.Event | None = None
        self._startup_error: Exception | None = None
        self._state = _TracerState.NEW
        self._accepting = False
        self._accepted = 0
        self._written = 0
        self._dropped = 0
        self._writer_failures = 0
        self._last_overflow_warning_at: float | None = None

    @property
    def accepted(self) -> int:
        return self._accepted

    @property
    def written(self) -> int:
        return self._written

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def writer_failures(self) -> int:
        return self._writer_failures

    async def start(self) -> None:
        if self._state is _TracerState.RUNNING:
            return
        if self._state is not _TracerState.NEW:
            raise RuntimeError(
                f"cannot start JSONL tracer in state {self._state.value!r}"
            )

        self._state = _TracerState.STARTING
        self._queue = asyncio.Queue(maxsize=TRACE_QUEUE_CAPACITY)
        self._startup_ready = asyncio.Event()
        self._writer_task = asyncio.create_task(
            self._writer_main(), name="jsonl-trace-writer"
        )
        await asyncio.shield(self._startup_ready.wait())
        if self._startup_error is not None:
            raise RuntimeError("failed to open trace sink") from None
        if self._state is not _TracerState.STARTING:
            raise RuntimeError(
                f"JSONL tracer start interrupted in state {self._state.value!r}"
            )
        self._state = _TracerState.RUNNING
        self._accepting = True
        logger.info(
            "tracing enabled: JSONL trace -> %s", _safe_path_display(self._path)
        )

    def emit(self, record: TraceRecord) -> None:
        if not self._accepting or self._queue is None:
            self._dropped += 1
            return

        try:
            line = json.dumps(record, default=_json_default, ensure_ascii=False) + "\n"
        except Exception:  # noqa: BLE001 - tracing is best-effort
            self._dropped += 1
            logger.warning(
                "trace serialization failed for %s",
                _safe_path_display(self._path),
            )
            return

        try:
            self._queue.put_nowait(line)
        except asyncio.QueueFull:
            self._dropped += 1
            self._warn_overflow()
        else:
            self._accepted += 1

    def _warn_overflow(self) -> None:
        now = monotonic()
        previous = self._last_overflow_warning_at
        if (
            previous is None
            or now - previous >= TRACE_OVERFLOW_WARNING_INTERVAL_SECONDS
        ):
            self._last_overflow_warning_at = now
            logger.warning(
                "trace queue full for %s; dropping newest record (dropped=%d)",
                _safe_path_display(self._path),
                self._dropped,
            )

    async def _writer_main(self) -> None:
        assert self._startup_ready is not None
        try:
            sink = await asyncio.to_thread(self._sink_factory, self._path)
        except Exception as exc:  # noqa: BLE001 - surfaced through start().
            self._startup_error = exc
            self._state = _TracerState.FAILED
            self._startup_ready.set()
            return

        self._startup_ready.set()
        try:
            await self._consume(sink)
        finally:
            try:
                await asyncio.to_thread(sink.close)
            except Exception as exc:  # noqa: BLE001 - shutdown is best-effort.
                logger.warning(
                    "trace sink cleanup failed for %s (%s)",
                    type(sink).__name__,
                    type(exc).__name__,
                )

    async def _consume(self, sink: _BatchSink) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()

        while True:
            item = await self._queue.get()
            if item is _STOP:
                return
            assert isinstance(item, str)

            batch = [item]
            deadline = loop.time() + TRACE_FLUSH_INTERVAL_SECONDS
            stopping = False
            while len(batch) < TRACE_BATCH_SIZE:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining
                        )
                    except TimeoutError:
                        break

                if item is _STOP:
                    stopping = True
                    break
                assert isinstance(item, str)
                batch.append(item)

            if not await self._write_batch(sink, batch):
                return
            if stopping:
                return

    async def _write_batch(self, sink: _BatchSink, batch: list[str]) -> bool:
        lines = tuple(batch)
        try:
            await asyncio.to_thread(sink.write_batch, lines)
        except Exception:  # noqa: BLE001 - tracing is permanently fail-safe.
            self._writer_failures += 1
            self._accepting = False
            self._state = _TracerState.FAILED
            self._dropped += len(lines) + self._discard_queued_records()
            logger.warning(
                "trace writer failed for %s; disabling tracing "
                "(accepted=%d written=%d dropped=%d writer_failures=%d)",
                _safe_path_display(self._path),
                self._accepted,
                self._written,
                self._dropped,
                self._writer_failures,
            )
            return False

        self._written += len(lines)
        return True

    def _discard_queued_records(self) -> int:
        assert self._queue is not None
        discarded = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return discarded
            if item is not _STOP:
                discarded += 1

    async def aclose(self) -> None:
        if self._state is _TracerState.CLOSED:
            return
        if self._close_task is None:
            if self._state is _TracerState.NEW:
                self._accepting = False
                self._state = _TracerState.CLOSED
                return
            self._accepting = False
            self._close_task = asyncio.create_task(
                self._finish_close(), name="jsonl-trace-close"
            )

        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                pass
            raise

    async def _finish_close(self) -> None:
        try:
            writer_task = self._writer_task
            queue = self._queue
            if writer_task is not None and queue is not None and not writer_task.done():
                await queue.put(_STOP)
            if writer_task is not None:
                await writer_task
        finally:
            self._state = _TracerState.CLOSED
            logger.info(
                "tracing stopped: path=%s accepted=%d written=%d dropped=%d "
                "writer_failures=%d",
                _safe_path_display(self._path),
                self._accepted,
                self._written,
                self._dropped,
                self._writer_failures,
            )


def build_tracer(*, enabled: bool, path: Path | str) -> Tracer | None:
    """Construct a resource-free tracer, or None when tracing is disabled."""
    if not enabled:
        return None
    try:
        return JSONLTracer(path)
    except Exception:  # noqa: BLE001
        logger.warning(
            "tracing enabled but tracer construction failed; running without traces."
        )
        return None


# ---------------------------------------------------------------------------
# Per-request logging adapter
# ---------------------------------------------------------------------------


class _RunLogAdapter(logging.LoggerAdapter[logging.Logger]):
    """Prefixes every message with the run_id so per-request log lines correlate
    with their trace, regardless of the root formatter's layout."""

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra = self.extra
        assert extra is not None
        return f"[run {extra['run_id']}] {msg}", kwargs


def run_logger(
    base: logging.Logger, run_id: str
) -> logging.LoggerAdapter[logging.Logger]:
    """A logger whose lines are tagged with `run_id`."""
    return _RunLogAdapter(base, {"run_id": run_id})


# ---------------------------------------------------------------------------
# Event -> record mapping
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _json_default(obj: object) -> str:
    """Last-resort JSON encoder for values that aren't natively serializable.

    Tool args/results originate from the model and are normally plain JSON, but
    opaque per-provider state (TextBlock/ToolUseBlock.provider_metadata can hold
    `thought_signature` bytes) may leak `bytes` into a record. Base64-encode
    bytes and stringify anything else rather than letting the serializer crash.
    """
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    return str(obj)


def event_record(event: Event, *, run_id: str | None, step: int) -> TraceRecord:
    """Map one loop event to a JSON-serializable trace record.

    Explicit per-type mapping (never `dataclasses.asdict`, which would walk into
    provider_metadata bytes). Captures the Doc 08 fields where the loop already
    has the data: tool name/args/result/is_error, finish reason, token usage,
    and per-step latency. provider_metadata itself is intentionally omitted — it
    is opaque round-trip state, not trace signal.
    """
    rec: TraceRecord = {
        "run_id": run_id,
        "step": step,
        "ts": _now_iso(),
        "type": event.type,
    }

    if isinstance(event, TextEvent):
        rec["text"] = event.text
    elif isinstance(event, ReasoningEvent):
        rec["reasoning"] = event.text
    elif isinstance(event, ToolCallEvent):
        rec["tool_use_id"] = event.id
        rec["name"] = event.name
        rec["args"] = event.input
    elif isinstance(event, ToolResultEvent):
        rec["tool_use_id"] = event.id
        rec["name"] = event.name
        rec["content"] = event.content
        rec["is_error"] = event.is_error
        rec["latency_ms"] = event.latency_ms
    elif isinstance(event, UsageEvent):
        rec["iteration"] = event.iteration
        rec["input_tokens"] = event.input_tokens
        rec["output_tokens"] = event.output_tokens
        rec["total_tokens"] = event.total_tokens
        rec["thinking_tokens"] = event.thinking_tokens
        rec["cached_tokens"] = event.cached_tokens
        rec["latency_ms"] = event.latency_ms
    elif isinstance(event, OrchestrationDecisionEvent):
        rec["model_id"] = event.model_id
        rec["tools"] = event.tools
        rec["fallback_used"] = event.fallback_used
        rec["fallback_reason"] = event.fallback_reason
        rec["corrections"] = event.corrections
        rec["control_model_id"] = event.control_model_id
        rec["input_tokens"] = event.input_tokens
        rec["output_tokens"] = event.output_tokens
        rec["total_tokens"] = event.total_tokens
        rec["thinking_tokens"] = event.thinking_tokens
        rec["cached_tokens"] = event.cached_tokens
        rec["latency_ms"] = event.latency_ms
        rec["thinking_level"] = event.thinking_level
    elif isinstance(event, DoneEvent):
        rec["reason"] = event.reason
        rec["iterations"] = event.iterations
        rec["total_tokens"] = event.total_tokens
        rec["input_tokens"] = event.input_tokens
        rec["output_tokens"] = event.output_tokens
        rec["thinking_tokens"] = event.thinking_tokens
    elif isinstance(event, ErrorEvent):
        rec["message"] = event.message

    return rec
