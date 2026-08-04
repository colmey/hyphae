"""Run-event JSONL tracing tests."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import agent.tracing as tracing_module
from agent import JSONLTracer, RunContext, RunLimits, Session, build_tracer, run_agent
from agent.events import DoneEvent, ToolCallEvent, ToolResultEvent, UsageEvent
from llm.client import GenerationRequest, LLMClient
from llm.schemas import AssistantMessage, TextBlock, ToolUseBlock, CompletionUsage
from tooling import ToolCallResult


pytestmark = pytest.mark.anyio


def test_done_event_trace_shape_is_byte_for_byte_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing_module, "_now_iso", lambda: "fixed-timestamp")

    record = tracing_module.event_record(
        DoneEvent(
            reason="end_turn",
            iterations=2,
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
            thinking_tokens=5,
        ),
        run_id="run-fixed",
        step=9,
    )

    assert record == {
        "run_id": "run-fixed",
        "step": 9,
        "ts": "fixed-timestamp",
        "type": "done",
        "reason": "end_turn",
        "iterations": 2,
        "total_tokens": 18,
        "input_tokens": 11,
        "output_tokens": 7,
        "thinking_tokens": 5,
    }


class ScriptedLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        if self.calls == 1:
            return AssistantMessage(
                content=[
                    TextBlock(text="Let me look that up."),
                    ToolUseBlock(
                        id="call_1",
                        name="demo__lookup",
                        input={"query": "weather", "blob": b"\x00\xff\xfe"},
                        provider_metadata={"thought_signature": b"\x01\x02\x03"},
                    ),
                ],
                stop_reason="tool_use",
                usage=CompletionUsage(
                    input_tokens=10, output_tokens=5, total_tokens=15
                ),
            )
        return AssistantMessage(
            content=[TextBlock(text="It is sunny.")],
            stop_reason="end_turn",
            usage=CompletionUsage(input_tokens=20, output_tokens=4, total_tokens=24),
        )


class FakeMCP:
    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [{"name": "demo__lookup", "description": "look up", "input_schema": {}}]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        return ToolCallResult(content="sunny, 24C", is_error=False)


async def _drive(tracer, *, run_id: str = "run_test_123") -> list:
    session = Session()
    session.append_user("what's the weather?")
    limits = RunLimits()
    context = RunContext.start(
        max_run_seconds=limits.max_run_seconds,
        base_logger=logging.getLogger(__name__),
        tracer=tracer,
        run_id=run_id,
    )
    return [
        event
        async for event in run_agent(
            session=session,
            llm=ScriptedLLM(),
            mcp=FakeMCP(),
            limits=limits,
            context=context,
        )
    ]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class _RecordingSink:
    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []
        self.wrote = threading.Event()
        self.closed = threading.Event()

    def write_batch(self, lines: tuple[str, ...]) -> None:
        self.batches.append(lines)
        self.wrote.set()

    def close(self) -> None:
        self.closed.set()

    @property
    def lines(self) -> list[str]:
        return [line for batch in self.batches for line in batch]


class _BlockingSink(_RecordingSink):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def write_batch(self, lines: tuple[str, ...]) -> None:
        self.batches.append(lines)
        self.wrote.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release blocked trace writer")


class _FailingSink(_BlockingSink):
    def write_batch(self, lines: tuple[str, ...]) -> None:
        self.batches.append(lines)
        self.wrote.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release failing trace writer")
        raise OSError("disk failed")


class _CloseFailingSink(_RecordingSink):
    def close(self) -> None:
        self.closed.set()
        raise RuntimeError("Authorization: Bearer trace-close-secret")


class _BlockingSinkFactory:
    def __init__(self, sink: _RecordingSink) -> None:
        self.sink = sink
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, _path: Path) -> _RecordingSink:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release blocked trace sink factory")
        return self.sink


async def _wait_for_thread_event(event: threading.Event) -> None:
    assert await asyncio.wait_for(asyncio.to_thread(event.wait), timeout=1.0)


async def _wait_for_writer_failure(tracer: JSONLTracer) -> None:
    async def wait() -> None:
        while tracer.writer_failures == 0:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1.0)


async def test_jsonl_trace_is_exact_serialized_event_log(tmp_path: Path) -> None:
    trace_jsonl_path = tmp_path / "nested" / "trace.jsonl"
    tracer = JSONLTracer(trace_jsonl_path)
    await tracer.start()
    events = await _drive(tracer)
    await tracer.aclose()
    records = _read_jsonl(trace_jsonl_path)

    assert len(records) == len(events)
    assert [record["type"] for record in records] == [event.type for event in events]
    assert all(record["run_id"] == "run_test_123" for record in records)
    assert [record["step"] for record in records] == list(range(1, len(records) + 1))
    for record in records:
        datetime.fromisoformat(record["ts"])

    usage_records = [record for record in records if record["type"] == "usage"]
    tool_records = [record for record in records if record["type"] == "tool_result"]
    assert usage_records and all(
        isinstance(r["latency_ms"], (int, float)) for r in usage_records
    )
    assert tool_records and all(
        isinstance(r["latency_ms"], (int, float)) for r in tool_records
    )
    args = next(record["args"] for record in records if record["type"] == "tool_call")
    assert args["query"] == "weather"
    assert isinstance(args["blob"], str)
    assert any(isinstance(event, UsageEvent) for event in events)
    assert any(isinstance(event, ToolCallEvent) for event in events)
    assert any(isinstance(event, ToolResultEvent) for event in events)
    assert isinstance(events[-1], DoneEvent) and events[-1].reason == "end_turn"


async def test_none_tracer_does_not_change_event_stream_or_write(
    tmp_path: Path,
) -> None:
    traced_path = tmp_path / "trace.jsonl"
    tracer = JSONLTracer(traced_path)
    await tracer.start()
    traced_events = await _drive(tracer)
    await tracer.aclose()

    noop_path = tmp_path / "should_not_exist.jsonl"
    noop_events = await _drive(None)

    assert not noop_path.exists()
    assert [event.type for event in noop_events] == [
        event.type for event in traced_events
    ]


async def test_disabled_and_unstarted_tracing_allocate_no_resources(
    tmp_path: Path,
) -> None:
    disabled_path = tmp_path / "disabled" / "trace.jsonl"
    assert build_tracer(enabled=False, path=disabled_path) is None
    assert not disabled_path.parent.exists()

    unstarted_path = tmp_path / "unstarted" / "trace.jsonl"
    tracer = JSONLTracer(unstarted_path)
    assert tracer._queue is None
    assert tracer._writer_task is None
    assert not unstarted_path.parent.exists()
    await tracer.aclose()
    assert not unstarted_path.parent.exists()


async def test_serialized_line_is_stable_and_detached_from_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = JSONLTracer(path)
    await tracer.start()
    record = {"run_id": "r", "text": "café", "blob": b"\x00\xff"}
    expected = (
        json.dumps(record, default=tracing_module._json_default, ensure_ascii=False)
        + "\n"
    )

    tracer.emit(record)
    record["text"] = "mutated"
    await tracer.aclose()

    assert path.read_text(encoding="utf-8") == expected
    assert tracer.accepted == 1
    assert tracer.written == 1
    assert tracer.dropped == 0
    assert tracer.writer_failures == 0


async def test_batch_size_flushes_exactly_at_one_hundred(tmp_path: Path) -> None:
    sink = _RecordingSink()
    tracer = JSONLTracer(tmp_path / "trace.jsonl", _sink_factory=lambda _path: sink)
    await tracer.start()

    for sequence in range(tracing_module.TRACE_BATCH_SIZE):
        tracer.emit({"sequence": sequence})

    await _wait_for_thread_event(sink.wrote)
    assert [len(batch) for batch in sink.batches] == [100]
    await tracer.aclose()
    assert tracer.accepted == tracer.written == 100


async def test_partial_batch_flushes_after_fixed_interval(tmp_path: Path) -> None:
    sink = _RecordingSink()
    tracer = JSONLTracer(tmp_path / "trace.jsonl", _sink_factory=lambda _path: sink)
    await tracer.start()
    started_at = time.perf_counter()

    tracer.emit({"sequence": 1})
    await _wait_for_thread_event(sink.wrote)
    elapsed = time.perf_counter() - started_at

    assert tracing_module.TRACE_FLUSH_INTERVAL_SECONDS == 0.250
    assert 0.20 <= elapsed < 1.0
    assert [len(batch) for batch in sink.batches] == [1]
    await tracer.aclose()


async def test_blocked_writer_does_not_block_heartbeat_or_turn(tmp_path: Path) -> None:
    sink = _BlockingSink()
    tracer = JSONLTracer(tmp_path / "trace.jsonl", _sink_factory=lambda _path: sink)
    await tracer.start()
    for sequence in range(tracing_module.TRACE_BATCH_SIZE):
        tracer.emit({"sequence": sequence})
    await _wait_for_thread_event(sink.wrote)

    async def heartbeat() -> str:
        await asyncio.sleep(0)
        return "alive"

    try:
        assert await asyncio.wait_for(heartbeat(), timeout=0.1) == "alive"
        events = await asyncio.wait_for(_drive(None), timeout=0.1)
        assert isinstance(events[-1], DoneEvent)
    finally:
        sink.release.set()
        await tracer.aclose()


async def test_overflow_drops_newest_preserves_order_and_rate_limits_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = _BlockingSink()
    tracer = JSONLTracer(tmp_path / "trace.jsonl", _sink_factory=lambda _path: sink)
    await tracer.start()
    for sequence in range(tracing_module.TRACE_BATCH_SIZE):
        tracer.emit({"sequence": sequence})
    await _wait_for_thread_event(sink.wrote)

    for sequence in range(
        tracing_module.TRACE_BATCH_SIZE,
        tracing_module.TRACE_BATCH_SIZE + tracing_module.TRACE_QUEUE_CAPACITY,
    ):
        tracer.emit({"sequence": sequence})

    warning_times = iter((100.0, 110.0, 160.0))
    monkeypatch.setattr(tracing_module, "monotonic", lambda: next(warning_times))
    caplog.set_level(logging.WARNING, logger=tracing_module.__name__)
    for sequence in range(3):
        tracer.emit({"overflow": sequence})

    assert tracer.accepted == 100 + tracing_module.TRACE_QUEUE_CAPACITY
    assert tracer.dropped == 3
    assert sum("trace queue full" in record.message for record in caplog.records) == 2

    sink.release.set()
    await tracer.aclose()
    written_sequences = [json.loads(line)["sequence"] for line in sink.lines]
    assert written_sequences == list(range(tracer.accepted))
    assert tracer.written == tracer.accepted
    assert tracer.writer_failures == 0


async def test_writer_failure_drops_batch_and_backlog_then_disables(
    tmp_path: Path,
) -> None:
    sink = _FailingSink()
    tracer = JSONLTracer(tmp_path / "trace.jsonl", _sink_factory=lambda _path: sink)
    await tracer.start()
    for sequence in range(tracing_module.TRACE_BATCH_SIZE):
        tracer.emit({"sequence": sequence})
    await _wait_for_thread_event(sink.wrote)
    for sequence in range(5):
        tracer.emit({"backlog": sequence})

    sink.release.set()
    await _wait_for_writer_failure(tracer)
    assert tracer.accepted == 105
    assert tracer.written == 0
    assert tracer.dropped == 105
    assert tracer.writer_failures == 1

    tracer.emit({"after": "failure"})
    assert tracer.dropped == 106
    await asyncio.wait_for(tracer.aclose(), timeout=0.5)
    assert len(sink.batches) == 1
    assert sink.closed.is_set()


async def test_sink_cleanup_failure_logs_only_types(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = _CloseFailingSink()
    tracer = JSONLTracer(tmp_path / "trace.jsonl", _sink_factory=lambda _path: sink)
    await tracer.start()
    tracer.emit({"healthy": True})

    await tracer.aclose()

    assert sink.closed.is_set()
    assert "trace-close-secret" not in caplog.text
    assert "_CloseFailingSink" in caplog.text
    assert "RuntimeError" in caplog.text


async def test_emit_lifecycle_rejections_and_repeated_lifecycle(
    tmp_path: Path,
) -> None:
    sink = _RecordingSink()
    tracer = JSONLTracer(tmp_path / "trace.jsonl", _sink_factory=lambda _path: sink)

    tracer.emit({"before": "start"})
    assert tracer.dropped == 1
    await tracer.start()
    await tracer.start()
    tracer.emit({"accepted": True})
    await asyncio.gather(tracer.aclose(), tracer.aclose())
    await tracer.aclose()
    tracer.emit({"after": "close"})

    assert tracer.accepted == 1
    assert tracer.written == 1
    assert tracer.dropped == 2
    with pytest.raises(RuntimeError, match="cannot start"):
        await tracer.start()


async def test_cancelled_start_can_be_closed_without_leaking_sink(
    tmp_path: Path,
) -> None:
    sink = _RecordingSink()
    factory = _BlockingSinkFactory(sink)
    tracer = JSONLTracer(tmp_path / "trace.jsonl", _sink_factory=factory)
    start_task = asyncio.create_task(tracer.start())
    await _wait_for_thread_event(factory.entered)

    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task

    close_task = asyncio.create_task(tracer.aclose())
    factory.release.set()
    await asyncio.wait_for(close_task, timeout=1.0)

    assert sink.closed.is_set()
    assert tracer.accepted == tracer.written == tracer.dropped == 0
    with pytest.raises(RuntimeError, match="cannot start"):
        await tracer.start()


async def test_serialization_failure_is_request_isolated(tmp_path: Path) -> None:
    sink = _RecordingSink()
    tracer = JSONLTracer(tmp_path / "trace.jsonl", _sink_factory=lambda _path: sink)
    await tracer.start()
    circular: dict[str, Any] = {}
    circular["self"] = circular

    tracer.emit(circular)
    tracer.emit({"healthy": True})
    await tracer.aclose()

    assert tracer.accepted == tracer.written == 1
    assert tracer.dropped == 1
    assert json.loads(sink.lines[0]) == {"healthy": True}


async def test_trace_failure_logs_render_paths_without_multiline_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = _RecordingSink()
    tracer = JSONLTracer(
        Path("trace\nsecond-line.jsonl"),
        _sink_factory=lambda _path: sink,
    )
    await tracer.start()
    circular: dict[str, Any] = {}
    circular["self"] = circular

    tracer.emit(circular)
    await tracer.aclose()

    assert "trace second-line.jsonl" in caplog.text
    assert "\nsecond-line" not in caplog.text


async def test_startup_failure_is_terminal_and_cleanup_is_safe(tmp_path: Path) -> None:
    def fail_open(_path: Path) -> _RecordingSink:
        raise OSError("cannot open")

    tracer = JSONLTracer(tmp_path / "trace.jsonl", _sink_factory=fail_open)
    with pytest.raises(RuntimeError, match="failed to open trace sink"):
        await tracer.start()
    tracer.emit({"after": "failed start"})
    await tracer.aclose()
    await tracer.aclose()

    assert tracer.accepted == tracer.written == 0
    assert tracer.dropped == 1
    assert tracer.writer_failures == 0
    with pytest.raises(RuntimeError, match="cannot start"):
        await tracer.start()


async def test_cancellation_during_close_is_re_raised_after_cleanup(
    tmp_path: Path,
) -> None:
    sink = _BlockingSink()
    tracer = JSONLTracer(tmp_path / "trace.jsonl", _sink_factory=lambda _path: sink)
    await tracer.start()
    for sequence in range(tracing_module.TRACE_BATCH_SIZE):
        tracer.emit({"sequence": sequence})
    await _wait_for_thread_event(sink.wrote)

    close_task = asyncio.create_task(tracer.aclose())
    await asyncio.sleep(0)
    close_task.cancel()
    sink.release.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert sink.closed.is_set()
    assert tracer.written == tracer.accepted == 100
    await tracer.aclose()


async def test_concurrent_runs_keep_run_ids_and_monotonic_steps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = JSONLTracer(path)
    await tracer.start()
    first_events, second_events = await asyncio.gather(
        _drive(tracer, run_id="run-one"),
        _drive(tracer, run_id="run-two"),
    )
    await tracer.aclose()

    records = _read_jsonl(path)
    for run_id, events in (
        ("run-one", first_events),
        ("run-two", second_events),
    ):
        run_records = [record for record in records if record["run_id"] == run_id]
        assert [record["step"] for record in run_records] == list(
            range(1, len(events) + 1)
        )
        assert [record["type"] for record in run_records] == [
            event.type for event in events
        ]
