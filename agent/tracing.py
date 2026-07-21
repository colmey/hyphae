# agent/tracing.py

"""
Run tracing: serialize the loop's event stream as a durable JSONL trace.

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

import base64
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Tracer seam
# ---------------------------------------------------------------------------


class Tracer(ABC):
    """Sink for trace records, one per loop event. The swappable observability
    seam (JSONL now, OTel later), analogous to SessionStore."""

    @abstractmethod
    def emit(self, record: dict) -> None:
        """Persist one trace record. Must not raise on a routine I/O failure —
        the loop wraps this, but a tracer should degrade quietly on its own."""
        ...

    def close(self) -> None:
        """Release any resources (file handles). No-op by default."""


class NoOpTracer(Tracer):
    """Discards every record. The explicit 'tracing off' tracer; the loop also
    treats `tracer is None` as off, so the hot path stays free of even this call."""

    def emit(self, record: dict) -> None:  # noqa: D401 - intentional no-op
        pass


class JSONLTracer(Tracer):
    """Append-only JSONL sink: one record per line in a single file.

    On the single-threaded event loop each `emit` writes a complete line with no
    `await` in between, so concurrent runs never interleave mid-line; `run_id`
    tags every record, keeping a shared file greppable per run. Writes are
    flushed so a crash still leaves a readable trace. I/O errors are logged and
    swallowed — a broken trace must never break a run.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    def emit(self, record: dict) -> None:
        try:
            self._fh.write(
                json.dumps(record, default=_json_default, ensure_ascii=False)
            )
            self._fh.write("\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001 - tracing is best-effort
            logger.warning("trace write failed for %s", self._path, exc_info=True)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


def build_tracer(*, enabled: bool, path: Path | str) -> Tracer | None:
    """Construct the configured tracer, or None when tracing is off / unbuildable.

    Returns None (rather than NoOpTracer) when disabled so the loop's hot path
    skips emit entirely. Any failure to open the sink degrades to None with a
    warning — tracing must never block startup.
    """
    if not enabled:
        return None
    try:
        tracer = JSONLTracer(path)
        logger.info("tracing enabled: JSONL trace -> %s", path)
        return tracer
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "tracing enabled but tracer init failed: %s; running without traces.", e
        )
        return None


# ---------------------------------------------------------------------------
# Per-request logging adapter
# ---------------------------------------------------------------------------


class _RunLogAdapter(logging.LoggerAdapter):
    """Prefixes every message with the run_id so per-request log lines correlate
    with their trace, regardless of the root formatter's layout."""

    def process(self, msg, kwargs):
        return f"[run {self.extra['run_id']}] {msg}", kwargs


def run_logger(base: logging.Logger, run_id: str) -> logging.LoggerAdapter:
    """A logger whose lines are tagged with `run_id`."""
    return _RunLogAdapter(base, {"run_id": run_id})


# ---------------------------------------------------------------------------
# Event -> record mapping
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _json_default(obj):
    """Last-resort JSON encoder for values that aren't natively serializable.

    Tool args/results originate from the model and are normally plain JSON, but
    opaque per-provider state (TextBlock/ToolUseBlock.provider_metadata can hold
    `thought_signature` bytes) may leak `bytes` into a record. Base64-encode
    bytes and stringify anything else rather than letting the serializer crash.
    """
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    return str(obj)


def event_record(event: Event, *, run_id: str | None, step: int) -> dict:
    """Map one loop event to a JSON-serializable trace record.

    Explicit per-type mapping (never `dataclasses.asdict`, which would walk into
    provider_metadata bytes). Captures the Doc 08 fields where the loop already
    has the data: tool name/args/result/is_error, finish reason, token usage,
    and per-step latency. provider_metadata itself is intentionally omitted — it
    is opaque round-trip state, not trace signal.
    """
    rec: dict = {
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
        rec["system_prompt"] = event.system_prompt
        rec["fallback_used"] = event.fallback_used
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
