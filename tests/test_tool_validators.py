"""Session 07 coverage for run-scoped compiled tool validators."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

import hyphae.agent.tool_execution as tool_execution_module
from hyphae.agent import DoneEvent, InMemorySessionStore, RunLimits, ToolPolicy, run_agent
from hyphae.agent.tool_policy import PolicyDecision, PolicyVerdict
from hyphae.llm.client import GenerationRequest, LLMClient
from hyphae.llm.schemas import AssistantMessage, TextBlock, ToolUseBlock, CompletionUsage
from hyphae.tooling import ToolCallResult

pytestmark = pytest.mark.anyio


def _tool_call(
    args: dict[str, Any],
    call_id: str,
    name: str = "srv__tool",
) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolUseBlock(
                id=call_id,
                name=name,
                input=args,
            )
        ],
        stop_reason="tool_use",
        usage=CompletionUsage(total_tokens=1),
    )


def _answer() -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock("done")],
        stop_reason="end_turn",
        usage=CompletionUsage(total_tokens=1),
    )


class ScriptedLLM(LLMClient):
    def __init__(self, script: list[AssistantMessage]) -> None:
        self.script = list(script)

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        if not self.script:
            raise AssertionError("LLM script exhausted")
        return self.script.pop(0)


class MutableMCP:
    def __init__(self, schema: Any) -> None:
        self.schema = schema
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "srv__tool",
                "description": "test",
                "input_schema": self.schema,
            }
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.calls.append((name, arguments))
        return ToolCallResult(content="ok", is_error=False)


async def _collect(
    llm: LLMClient,
    mcp: MutableMCP,
    *,
    policy: ToolPolicy | None = None,
) -> list[Any]:
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    return [
        event
        async for event in run_agent(
            session,
            llm,
            mcp,
            store=store,
            policy=policy,
            limits=RunLimits(max_iterations=10),
        )
    ]


def _done(events: list[Any]) -> str:
    return next(
        event.reason for event in reversed(events) if isinstance(event, DoneEvent)
    )


async def test_valid_schema_compiles_once_and_validator_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    real_validator_for = tool_execution_module.validator_for
    counts = {"select": 0, "check": 0, "construct": 0, "validate": 0}

    class CountingValidator:
        @classmethod
        def check_schema(cls, selected_schema: Any) -> None:
            counts["check"] += 1
            real_validator_for(selected_schema).check_schema(selected_schema)

        def __init__(self, selected_schema: Any) -> None:
            counts["construct"] += 1
            self.inner = real_validator_for(selected_schema)(selected_schema)

        def validate(self, instance: Any) -> None:
            counts["validate"] += 1
            self.inner.validate(instance)

    def select_validator(selected_schema: Any) -> type[CountingValidator]:
        counts["select"] += 1
        return CountingValidator

    monkeypatch.setattr(tool_execution_module, "validator_for", select_validator)
    llm = ScriptedLLM(
        [_tool_call({"q": "one"}, "c1"), _tool_call({"q": "two"}, "c2"), _answer()]
    )
    mcp = MutableMCP(schema)

    events = await _collect(llm, mcp)

    assert _done(events) == "end_turn"
    assert counts == {"select": 1, "check": 1, "construct": 1, "validate": 2}
    assert len(mcp.calls) == 2


async def test_duplicate_tool_names_compile_only_the_last_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[Any] = []
    real_validator_for = tool_execution_module.validator_for

    def recording_validator_for(schema: Any) -> Any:
        selected.append(schema)
        return real_validator_for(schema)

    monkeypatch.setattr(
        tool_execution_module,
        "validator_for",
        recording_validator_for,
    )
    tools = [
        {"name": "srv__tool", "input_schema": {"type": "string"}},
        {"name": "srv__tool", "input_schema": {"type": "object"}},
    ]

    validators = tool_execution_module._compile_tool_validators(tools)

    assert len(validators) == 1
    assert selected == [{"type": "object"}]


async def test_malformed_schema_warns_once_and_remains_permissive(caplog) -> None:
    schema = {"type": 123}
    llm = ScriptedLLM(
        [_tool_call({"q": 1}, "c1"), _tool_call({"q": 2}, "c2"), _answer()]
    )
    mcp = MutableMCP(schema)

    with caplog.at_level(logging.WARNING, logger="agent.loop"):
        events = await _collect(llm, mcp)

    warnings = [
        record
        for record in caplog.records
        if "input_schema is invalid" in record.getMessage()
    ]
    assert _done(events) == "end_turn"
    assert len(warnings) == 1
    assert len(mcp.calls) == 2


async def test_runtime_validator_failure_warns_once_then_stays_permissive(
    monkeypatch: pytest.MonkeyPatch,
    caplog,
) -> None:
    validate_calls = 0

    class BrokenValidator:
        @classmethod
        def check_schema(cls, schema: Any) -> None:
            pass

        def __init__(self, schema: Any) -> None:
            pass

        def validate(self, instance: Any) -> None:
            nonlocal validate_calls
            validate_calls += 1
            raise RuntimeError("unresolvable schema reference")

    monkeypatch.setattr(
        tool_execution_module,
        "validator_for",
        lambda schema: BrokenValidator,
    )
    llm = ScriptedLLM(
        [_tool_call({"q": 1}, "c1"), _tool_call({"q": 2}, "c2"), _answer()]
    )
    mcp = MutableMCP({"type": "object"})

    with caplog.at_level(logging.WARNING, logger="agent.loop"):
        events = await _collect(llm, mcp)

    warnings = [
        record
        for record in caplog.records
        if "input validator failed" in record.getMessage()
    ]
    assert _done(events) == "end_turn"
    assert validate_calls == 1
    assert len(warnings) == 1
    assert len(mcp.calls) == 2


async def test_invalid_arguments_keep_stable_path_and_never_reach_mcp() -> None:
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            }
        },
        "required": ["payload"],
    }
    llm = ScriptedLLM([_tool_call({"payload": {"count": "bad"}}, "c1"), _answer()])
    mcp = MutableMCP(schema)

    events = await _collect(llm, mcp)
    result = next(event for event in events if event.type == "tool_result")

    assert result.is_error is True
    assert "field 'payload/count'" in result.content
    assert "'bad' is not of type 'integer'" in result.content
    assert 'Expected shape: {"type": "integer"}' in result.content
    assert mcp.calls == []


class RecordingPolicy(ToolPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, Any]] = []

    def check(self, tool_name: str, args: Any = None) -> PolicyDecision:
        self.calls.append((tool_name, args))
        return PolicyDecision(PolicyVerdict.ALLOW)


async def test_authorization_runs_only_after_successful_validation() -> None:
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    policy = RecordingPolicy()
    llm = ScriptedLLM(
        [_tool_call({"q": 1}, "c1"), _tool_call({"q": "valid"}, "c2"), _answer()]
    )
    mcp = MutableMCP(schema)

    await _collect(llm, mcp, policy=policy)

    assert policy.calls == [("srv__tool", {"q": "valid"})]
    assert mcp.calls == [("srv__tool", {"q": "valid"})]


async def test_different_runs_build_independent_validator_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = 0
    real_validator_for = tool_execution_module.validator_for

    def recording_validator_for(schema: Any) -> Any:
        nonlocal selected
        selected += 1
        return real_validator_for(schema)

    monkeypatch.setattr(
        tool_execution_module,
        "validator_for",
        recording_validator_for,
    )
    schema = {"type": "object"}

    await _collect(ScriptedLLM([_answer()]), MutableMCP(schema))
    await _collect(ScriptedLLM([_answer()]), MutableMCP(schema))

    assert selected == 2


class BlockingFirstLLM(ScriptedLLM):
    def __init__(self, script: list[AssistantMessage]) -> None:
        super().__init__(script)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await self.release.wait()
        return await super().complete(request)


async def test_inventory_refresh_does_not_mutate_in_flight_validators() -> None:
    string_schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    integer_schema = {
        "type": "object",
        "properties": {"q": {"type": "integer"}},
        "required": ["q"],
    }
    mcp = MutableMCP(string_schema)
    first_llm = BlockingFirstLLM([_tool_call({"q": "old"}, "c1"), _answer()])

    first_turn = asyncio.create_task(_collect(first_llm, mcp))
    await first_llm.started.wait()
    mcp.schema = integer_schema
    first_llm.release.set()
    first_events = await first_turn

    second_events = await _collect(
        ScriptedLLM([_tool_call({"q": "old"}, "c2"), _answer()]), mcp
    )

    assert _done(first_events) == _done(second_events) == "end_turn"
    assert mcp.calls == [("srv__tool", {"q": "old"})]
    second_result = next(
        event for event in second_events if event.type == "tool_result"
    )
    assert second_result.is_error is True
    assert "is not of type 'integer'" in second_result.content


async def test_unadvertised_tool_gets_no_validator_authority() -> None:
    policy = ToolPolicy(mode="allow_list", allow=[])
    llm = ScriptedLLM([_tool_call({}, "c1", name="srv__unknown"), _answer()])
    mcp = MutableMCP({"type": "object"})

    events = await _collect(llm, mcp, policy=policy)
    result = next(event for event in events if event.type == "tool_result")

    assert result.is_error is True
    assert "blocked by policy" in result.content
    assert mcp.calls == []
