"""
Deterministic agent eval runner.

Hermetic by default:
    ./runscript.sh tests/eval_agent.py

Live evals are skipped unless explicitly enabled:
    EVAL_LIVE=1 ./runscript.sh tests/eval_agent.py

This is deliberately not pytest and not an eval framework. Cases live as data
under tests/eval_data/; adding a hermetic case should only require editing the
dataset.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent import (
    DoneEvent,
    ErrorEvent,
    InMemorySessionStore,
    RunLimits,
    SessionGuard,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    run_agent,
)
from api.turn import OrchestratedRouting, PersistencePolicy, TurnRequest, TurnRunner
from config import get_settings
from llm.client import GenerationRequest, LLMClient, build_llm_client
from llm.schemas import AssistantMessage, TextBlock, ToolUseBlock, CompletionUsage
from tooling import ToolCallResult
from orchestrator.schemas import OrchestrationDecision, OrchestrationProposal

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "tests" / "eval_data" / "agent_eval_v1.yaml"

logging.basicConfig(level=logging.CRITICAL)


class TransientEvalError(Exception):
    """Exception the scripted client classifies as transient."""


class ScriptedLLM(LLMClient):
    """Scripted LLM fake used by hermetic eval cases."""

    def __init__(self, script: list[Any], *, delay: float = 0.0) -> None:
        self._script = list(script)
        self._delay = delay
        self.calls = 0
        self.requests_seen: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        self.requests_seen.append(request)
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def is_transient_error(self, exc: BaseException) -> bool:
        return isinstance(exc, TransientEvalError)


class ScriptedMCP:
    """Small MCPManager stand-in with data-driven tools and results."""

    def __init__(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        results: list[dict[str, Any]] | None = None,
    ) -> None:
        self._tools = tools or [
            {
                "name": "srv__lookup",
                "description": "scripted lookup",
                "input_schema": {},
            },
            {
                "name": "srv__web",
                "description": "scripted web lookup",
                "input_schema": {},
            },
        ]
        self._results = list(results or [])
        self.call_count = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.connected_servers: list[str] = []

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return list(self._tools)

    @asynccontextmanager
    async def open_turn(self, *, timeout_seconds: float | None = None):
        yield self

    def list_tools(self) -> list[tuple[str, dict[str, Any]]]:
        return [(t["name"], t) for t in self._tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.call_count += 1
        self.calls.append((name, arguments))
        if not self._results:
            raise AssertionError(f"ScriptedMCP has no result for call {name!r}")

        item = self._results.pop(0)
        if item.get("delay"):
            await asyncio.sleep(float(item["delay"]))
        if "raises" in item:
            raise RuntimeError(str(item["raises"]))

        content = _result_content(item)
        return ToolCallResult(
            content=content, is_error=bool(item.get("is_error", False))
        )


class EmptyMCP(ScriptedMCP):
    """No-tool MCP used by the live no-tool tier."""

    def __init__(self) -> None:
        super().__init__(tools=[], results=[])


class ScriptedOrchestrator:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    async def decide(
        self, prompt, tools, history=None, timeout=None, log=None
    ) -> OrchestrationDecision:
        proposal = OrchestrationProposal(
            selected_model_id=self._config.get("selected_model_id", "default"),
            selected_tools=list(self._config.get("selected_tools", [])),
            thinking_level=self._config.get("thinking_level", "medium"),
        )
        return OrchestrationDecision(
            result=proposal,
            fallback_used=bool(self._config.get("fallback_used", False)),
            fallback_reason=self._config.get("fallback_reason"),
        )


class ScriptedRegistry:
    def __init__(self, llm: LLMClient, model_ids: list[str] | None = None) -> None:
        self._llm = llm
        self.model_ids = model_ids or ["default"]

    def get_or_default(self, model_id: str | None) -> tuple[str, LLMClient]:
        if model_id in self.model_ids:
            return str(model_id), self._llm
        return self.model_ids[0], self._llm


@dataclass
class CaseResult:
    case_id: str
    tier: str
    passed: bool
    reason: str
    checks: list[str] = field(default_factory=list)


@dataclass
class RunArtifacts:
    events: list[Any]
    answer: str
    done_reason: str | None
    llm: LLMClient
    mcp: ScriptedMCP

    @property
    def done(self) -> DoneEvent | None:
        return next((e for e in self.events if isinstance(e, DoneEvent)), None)

    @property
    def tool_calls(self) -> list[ToolCallEvent]:
        return [e for e in self.events if isinstance(e, ToolCallEvent)]

    @property
    def tool_results(self) -> list[ToolResultEvent]:
        return [e for e in self.events if isinstance(e, ToolResultEvent)]

    @property
    def errors(self) -> list[ErrorEvent]:
        return [e for e in self.events if isinstance(e, ErrorEvent)]


def _usage(raw: dict[str, Any] | None) -> CompletionUsage:
    raw = raw or {}
    return CompletionUsage(
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        total_tokens=int(raw.get("total_tokens", 0)),
        thinking_tokens=int(raw.get("thinking_tokens", 0)),
        cached_tokens=int(raw.get("cached_tokens", 0)),
    )


def _assistant_message(raw: dict[str, Any]) -> AssistantMessage | BaseException:
    if "raises" in raw:
        message = raw.get("message", raw["raises"])
        if raw["raises"] == "transient":
            return TransientEvalError(str(message))
        return ValueError(str(message))

    content: list[Any] = []
    for text in raw.get("text_blocks", []):
        content.append(TextBlock(text=str(text)))
    if "text" in raw:
        content.append(TextBlock(text=str(raw["text"])))
    for call in raw.get("tool_calls", []):
        content.append(
            ToolUseBlock(
                id=str(call["id"]),
                name=str(call["name"]),
                input=dict(call.get("input", {})),
            )
        )
    if "content" in raw and raw["content"] == []:
        content = []

    stop_reason = raw.get("stop_reason")
    if stop_reason is None:
        stop_reason = "tool_use" if raw.get("tool_calls") else "end_turn"
    return AssistantMessage(
        content=content, stop_reason=stop_reason, usage=_usage(raw.get("usage"))
    )


def _build_llm(case: dict[str, Any]) -> ScriptedLLM:
    llm_cfg = case.get("llm") or {}
    script = [_assistant_message(item) for item in llm_cfg.get("script", [])]
    return ScriptedLLM(script, delay=float(llm_cfg.get("delay", 0.0)))


def _result_content(raw: dict[str, Any]) -> str:
    if "content_repeat" in raw:
        repeat = raw["content_repeat"]
        return str(repeat.get("text", "")) * int(repeat.get("count", 0))
    return str(raw.get("content", ""))


def _build_mcp(case: dict[str, Any]) -> ScriptedMCP:
    mcp_cfg = case.get("mcp") or {}
    return ScriptedMCP(tools=mcp_cfg.get("tools"), results=mcp_cfg.get("results"))


def _run_kwargs(case: dict[str, Any]) -> dict[str, Any]:
    run_cfg = dict(case.get("run") or {})
    allowed = {
        "max_iterations",
        "max_tokens",
        "llm_timeout_seconds",
        "tool_timeout_seconds",
        "max_retries",
        "retry_base_delay",
        "tool_result_max_chars",
        "max_run_tokens",
        "max_run_seconds",
        "abort_after_consecutive_tool_failures",
        "thinking_level",
        "context_strategy",
        "context_window",
        "context_safety_margin_tokens",
        "context_recent_messages",
        "context_summary_max_tokens",
    }
    return {k: v for k, v in run_cfg.items() if k in allowed}


async def _run_hermetic(case: dict[str, Any]) -> RunArtifacts:
    llm = _build_llm(case)
    mcp = _build_mcp(case)
    store = InMemorySessionStore()
    session = await store.create()

    events: list[Any] = []
    if case.get("mode") == "turn_runner":
        run_options = _run_kwargs(case)
        run_options.pop("thinking_level", None)
        orch_cfg = case.get("orchestration") or {}
        routing = OrchestratedRouting(
            orchestrator=ScriptedOrchestrator(orch_cfg),
            registry=ScriptedRegistry(llm, model_ids=["default"]),
            agent_system_prompt="trusted agent system",
        )
        runner = TurnRunner(
            routing=routing,
            limits=RunLimits(**run_options),
            mcp=mcp,
            store=store,
            guard=SessionGuard(),
            policy=None,
            tracer=None,
        )
        turn = TurnRequest(
            prompt=case["prompt"],
            session=session,
            persistence=PersistencePolicy.PERSISTENT,
            system_override=(case.get("run") or {}).get("system"),
        )
        async with runner.open(turn) as execution:
            events.extend([event async for event in execution.events])
    else:
        session.append_user(case["prompt"])
        run_options = _run_kwargs(case)
        thinking_level = run_options.pop("thinking_level", None)
        async for event in run_agent(
            session=session,
            llm=llm,
            mcp=mcp,
            store=store,
            system=(case.get("run") or {}).get("system"),
            thinking_level=thinking_level,
            limits=RunLimits(**run_options),
        ):
            events.append(event)

    answer = "".join(e.text for e in events if isinstance(e, TextEvent)).strip()
    done = next((e.reason for e in events if isinstance(e, DoneEvent)), None)
    return RunArtifacts(
        events=events, answer=answer, done_reason=done, llm=llm, mcp=mcp
    )


async def _run_live(case: dict[str, Any]) -> RunArtifacts:
    settings = get_settings()
    llm = build_llm_client(settings)
    mcp = EmptyMCP()
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user(case["prompt"])
    events: list[Any] = []
    async for event in run_agent(
        session=session,
        llm=llm,
        mcp=mcp,
        store=store,
        tools=[],
        limits=RunLimits(
            max_iterations=(case.get("expect") or {}).get("max_iterations", 2),
            max_retries=settings.llm.max_retries,
            retry_base_delay=settings.llm.retry_base_delay,
            llm_timeout_seconds=settings.llm.timeout_seconds,
        ),
    ):
        events.append(event)
    answer = "".join(e.text for e in events if isinstance(e, TextEvent)).strip()
    done = next((e.reason for e in events if isinstance(e, DoneEvent)), None)
    return RunArtifacts(
        events=events, answer=answer, done_reason=done, llm=llm, mcp=mcp
    )


def _check_contains(
    haystack: str, needles: list[str], label: str, failures: list[str]
) -> None:
    for needle in needles:
        if needle not in haystack:
            failures.append(f"{label} missing {needle!r}")


def _check_not_contains(
    haystack: str, needles: list[str], label: str, failures: list[str]
) -> None:
    for needle in needles:
        if needle in haystack:
            failures.append(f"{label} unexpectedly contained {needle!r}")


def _evaluate(case: dict[str, Any], art: RunArtifacts) -> list[str]:
    expect = case.get("expect") or {}
    failures: list[str] = []

    if "done_reason" in expect and art.done_reason != expect["done_reason"]:
        failures.append(
            f"done_reason expected {expect['done_reason']!r}, got {art.done_reason!r}"
        )
    if "answer_exact" in expect and art.answer != expect["answer_exact"]:
        failures.append(
            f"answer_exact expected {expect['answer_exact']!r}, got {art.answer!r}"
        )
    _check_contains(
        art.answer, list(expect.get("answer_contains", [])), "answer", failures
    )

    tool_names = [e.name for e in art.tool_calls]
    for name in expect.get("tool_called", []):
        if name not in tool_names:
            failures.append(f"expected tool call {name!r}, got {tool_names!r}")
    for name in expect.get("tool_not_called", []):
        if name in tool_names:
            failures.append(f"tool {name!r} was called")
    if "tool_call_count" in expect and len(art.tool_calls) != expect["tool_call_count"]:
        failures.append(
            f"tool_call_count expected {expect['tool_call_count']}, got {len(art.tool_calls)}"
        )
    if "mcp_call_count" in expect and art.mcp.call_count != expect["mcp_call_count"]:
        failures.append(
            f"mcp_call_count expected {expect['mcp_call_count']}, got {art.mcp.call_count}"
        )

    result_text = "\n".join(event.content for event in art.tool_results)
    _check_contains(
        result_text,
        list(expect.get("tool_result_contains", [])),
        "tool_result",
        failures,
    )
    _check_not_contains(
        result_text,
        list(expect.get("tool_result_not_contains", [])),
        "tool_result",
        failures,
    )
    if "tool_result_error_count" in expect:
        errors = sum(1 for e in art.tool_results if e.is_error)
        if errors != expect["tool_result_error_count"]:
            failures.append(
                f"tool_result_error_count expected {expect['tool_result_error_count']}, got {errors}"
            )
    if "max_tool_result_chars" in expect:
        longest = max((len(event.content) for event in art.tool_results), default=0)
        if longest > expect["max_tool_result_chars"]:
            failures.append(
                f"max_tool_result_chars expected <= {expect['max_tool_result_chars']}, got {longest}"
            )

    error_text = "\n".join(e.message for e in art.errors)
    _check_contains(
        error_text,
        list(expect.get("error_event_contains", [])),
        "error_event",
        failures,
    )
    if "error_event_count" in expect and len(art.errors) != expect["error_event_count"]:
        failures.append(
            f"error_event_count expected {expect['error_event_count']}, got {len(art.errors)}"
        )

    if "llm_calls" in expect and getattr(art.llm, "calls", None) != expect["llm_calls"]:
        failures.append(
            f"llm_calls expected {expect['llm_calls']}, got {getattr(art.llm, 'calls', None)}"
        )
    if "max_iterations" in expect:
        done = art.done
        if done is None:
            failures.append("missing DoneEvent for max_iterations check")
        elif done.iterations > expect["max_iterations"]:
            failures.append(
                f"iterations expected <= {expect['max_iterations']}, got {done.iterations}"
            )

    for call_num in expect.get("tools_seen_none_on_calls", []):
        index = int(call_num) - 1
        seen = [request.tools for request in art.llm.requests_seen]
        if index >= len(seen) or seen[index] is not None:
            failures.append(
                f"tools on LLM call {call_num} expected None, got {seen[index] if index < len(seen) else '<missing>'!r}"
            )
    for item in expect.get("system_contains_on_calls", []):
        index = int(item["call"]) - 1
        systems = [request.system for request in art.llm.requests_seen]
        system = systems[index] if index < len(systems) else None
        if item["substring"] not in (system or ""):
            failures.append(
                f"system on LLM call {item['call']} missing {item['substring']!r}"
            )
    if "selected_thinking_level" in expect:
        seen = [request.thinking_level for request in art.llm.requests_seen]
        if expect["selected_thinking_level"] not in seen:
            failures.append(
                f"thinking_level {expect['selected_thinking_level']!r} not observed in {seen!r}"
            )

    return failures


async def _run_case(case: dict[str, Any]) -> CaseResult:
    case_id = case.get("id", "<missing id>")
    tier = case.get("tier", "hermetic")
    try:
        art = await (_run_live(case) if tier == "live" else _run_hermetic(case))
        failures = _evaluate(case, art)
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            case_id, tier, False, f"runner error: {type(exc).__name__}: {exc}"
        )
    if failures:
        return CaseResult(case_id, tier, False, "; ".join(failures), failures)
    return CaseResult(case_id, tier, True, "ok")


def _load_dataset(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("dataset root must be a mapping")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset must contain a non-empty cases list")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be a mapping")
        if not case.get("id"):
            raise ValueError("each case needs an id")
        if case["id"] in seen:
            raise ValueError(f"duplicate case id {case['id']!r}")
        seen.add(case["id"])
        if not case.get("prompt"):
            raise ValueError(f"case {case['id']!r} needs a prompt")
        if not isinstance(case.get("expect"), dict):
            raise ValueError(f"case {case['id']!r} needs an expect mapping")
    return data


def _print_table(results: list[CaseResult]) -> None:
    id_width = max([len("case"), *(len(r.case_id) for r in results)])
    tier_width = max([len("tier"), *(len(r.tier) for r in results)])
    print(f"{'status':6}  {'tier':{tier_width}}  {'case':{id_width}}  reason")
    print(f"{'-' * 6}  {'-' * tier_width}  {'-' * id_width}  {'-' * 40}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{status:6}  {r.tier:{tier_width}}  {r.case_id:{id_width}}  {r.reason}")


async def main() -> None:
    dataset_path = Path(os.environ.get("EVAL_DATASET", DEFAULT_DATASET))
    data = _load_dataset(dataset_path)
    live_enabled = os.environ.get("EVAL_LIVE") == "1"

    hermetic = [c for c in data["cases"] if c.get("tier", "hermetic") == "hermetic"]
    live = [c for c in data["cases"] if c.get("tier") == "live"]
    selected = list(hermetic)
    if live_enabled:
        selected.extend(live)

    print(f"Agent eval dataset: {dataset_path}")
    print(f"Dataset version: {data.get('version')}")
    print(f"Hermetic cases: {len(hermetic)}")
    if live_enabled:
        print(f"LIVE EVALS ENABLED: running {len(live)} live case(s).")
    else:
        print(f"LIVE EVALS SKIPPED: {len(live)} live case(s) require EVAL_LIVE=1.")
    print()

    results = [await _run_case(case) for case in selected]
    _print_table(results)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    score = passed / total if total else 0.0
    threshold = float(data.get("threshold", 1.0))
    print()
    print(f"Summary: {passed}/{total} passed ({score:.1%}); threshold {threshold:.1%}")
    if score < threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
