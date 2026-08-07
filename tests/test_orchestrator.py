"""Hermetic orchestration schema and prompt-construction tests."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import hyphae.orchestrator.orchestrator as orchestrator_module
from hyphae.agent.runtime import ModelLimits
from hyphae.llm.client import GenerationRequest, LLMClient
from hyphae.llm.schemas import AssistantMessage, CompletionUsage, Message, TextBlock
from hyphae.tooling import ToolSnapshot
from hyphae.orchestrator import Orchestrator, ToolPreferences
from hyphae.orchestrator.schemas import OrchestrationProposal


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "medium"),
        ("high", "high"),
        ("HIGH", "high"),
        ("extreme", "medium"),
        ("low", "low"),
    ],
)
def test_thinking_level_is_normalized(value: str | None, expected: str) -> None:
    payload: dict[str, Any] = {
        "selected_model_id": "model",
        "made_up": "ignored",
    }
    if value is not None:
        payload["thinking_level"] = value

    assert OrchestrationProposal.model_validate(payload).thinking_level == expected


def test_routing_prompt_contract_is_selection_only() -> None:
    prompt = (
        Path(__file__).parents[1]
        / "hyphae"
        / "config"
        / "orchestrator_prompt.md"
    ).read_text(encoding="utf-8")

    assert "exactly three fields" in prompt
    assert "generated_system_prompt" not in prompt


class _RegistryStub:
    model_ids = ["model"]

    def default_id(self) -> str:
        return "model"

    def describe_for_prompt(self) -> str:
        return "model: test model"

    def get(self, model_id: str) -> LLMClient:
        raise AssertionError(f"unexpected registry lookup: {model_id}")

    def get_or_default(self, model_id: str | None) -> tuple[str, LLMClient]:
        resolved = model_id or self.default_id()
        return resolved, self.get(resolved)

    def get_entry(self, model_id: str) -> ModelLimits:
        return SimpleNamespace(max_tokens=None, context_window=None)


class _RecordingLLM(LLMClient):
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.requests.append(request)
        return AssistantMessage(
            content=[TextBlock('{"selected_model_id":"model"}')],
            stop_reason="end_turn",
            usage=CompletionUsage(input_tokens=3, output_tokens=2, total_tokens=5),
        )


@pytest.mark.anyio
async def test_orchestration_constructs_structured_generation_request() -> None:
    orchestrator = Orchestrator(
        registry=_RegistryStub(),
        system_prompt="route requests",
    )
    llm = _RecordingLLM()

    raw, usage, latency_ms = await orchestrator._call_orchestrator_llm(
        llm, "choose a model"
    )

    assert raw == '{"selected_model_id":"model"}'
    assert usage == CompletionUsage(input_tokens=3, output_tokens=2, total_tokens=5)
    assert latency_ms >= 0
    request = llm.requests[0]
    assert request.messages == [Message.user("choose a model")]
    assert request.tools is None
    assert request.system == "route requests"
    assert request.response_schema is OrchestrationProposal


def test_prompt_includes_history_only_when_present() -> None:
    orchestrator = Orchestrator(
        registry=_RegistryStub(),
        system_prompt="route requests",
    )
    history = [
        Message.user("List the tables in the customer database."),
        Message.assistant([TextBlock(text="The tables are customers and orders.")]),
    ]

    tools = ToolSnapshot()
    with_history = orchestrator._build_prompt(
        "now do last month", tools, history=history
    )
    without_history = orchestrator._build_prompt("hello", tools, history=None)

    assert "CONVERSATION SO FAR" in with_history
    assert "customer database" in with_history
    assert "CONVERSATION SO FAR" not in without_history


class _PreferenceLLM(LLMClient):
    def __init__(self) -> None:
        self.request: GenerationRequest | None = None

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.request = request
        return AssistantMessage(
            content=[
                TextBlock(
                    '{"selected_model_id":"model","selected_tools":[]}'
                )
            ],
            stop_reason="end_turn",
        )


class _PreferenceRegistry:
    model_ids = ["model"]

    def __init__(self) -> None:
        self.llm = _PreferenceLLM()

    def default_id(self) -> str:
        return "model"

    def describe_for_prompt(self) -> str:
        return "model: test model"

    def get(self, model_id: str) -> LLMClient:
        assert model_id == "model"
        return self.llm

    def get_or_default(self, model_id: str | None) -> tuple[str, LLMClient]:
        resolved = model_id if model_id in self.model_ids else self.default_id()
        return resolved, self.get(resolved)

    def get_entry(self, model_id: str) -> ModelLimits:
        assert model_id == "model"
        return SimpleNamespace(max_tokens=None, context_window=None)


@pytest.mark.anyio
async def test_direct_orchestrator_preferences_remain_supported_and_sanitized() -> None:
    registry = _PreferenceRegistry()
    orchestrator = Orchestrator(
        registry=registry,
        system_prompt="route requests",
    )
    snapshot = ToolSnapshot.from_llm_tools(
        [
            {
                "name": "search__query",
                "description": "Search for a query.",
                "input_schema": {"type": "object"},
            }
        ]
    )
    preferences = ToolPreferences.from_request(
        [
            SimpleNamespace(
                name="search",
                tools={"query": ["q"], "missing": []},
            )
        ]
    )

    decision = await orchestrator.decide(
        "find it",
        snapshot,
        preferences=preferences,
    )

    assert decision.result.selected_tools == ["search__query"]
    assert registry.llm.request is not None
    prompt = registry.llm.request.messages[0].content[0]
    assert isinstance(prompt, TextBlock)
    assert "PREFERRED TOOLS" in prompt.text
    assert "search__query (intended arguments: q)" in prompt.text


class _InvalidOutputLLM(LLMClient):
    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        return AssistantMessage(
            content=[TextBlock("secret router output: do not log")],
            stop_reason="end_turn",
        )


@pytest.mark.anyio
async def test_invalid_router_output_uses_safe_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Registry(_RegistryStub):
        def get(self, model_id: str) -> LLMClient:
            return _InvalidOutputLLM()

    orchestrator = Orchestrator(registry=Registry(), system_prompt="route")
    caplog.set_level(logging.WARNING)

    decision = await orchestrator.decide("hello", ToolSnapshot())

    assert decision.fallback_used is True
    assert decision.fallback_reason == "invalid_control_output"
    assert decision.result.selected_model_id == "model"
    assert decision.result.selected_tools == []
    assert decision.usage.total_tokens > 0
    assert decision.control_model_id == "model"
    assert "secret router output" not in caplog.text
    assert "sha256=" in caplog.text


class _FailingLLM(LLMClient):
    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise RuntimeError("secret control-client failure")


@pytest.mark.anyio
async def test_control_client_failure_uses_safe_diagnostics(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Registry(_RegistryStub):
        def get(self, model_id: str) -> LLMClient:
            return _FailingLLM()

    orchestrator = Orchestrator(registry=Registry(), system_prompt="route")
    caplog.set_level(logging.WARNING)
    ticks = iter((10.0, 11.0, 16.5))
    monkeypatch.setattr(orchestrator_module, "perf_counter", lambda: next(ticks))

    decision = await orchestrator.decide("hello", ToolSnapshot())

    assert decision.fallback_used is True
    assert decision.fallback_reason == "control_call_failed"
    assert decision.result.selected_model_id == "model"
    assert decision.result.selected_tools == []
    assert "secret control-client failure" not in caplog.text
    assert "RuntimeError" in caplog.text
    assert decision.usage == CompletionUsage()
    assert decision.latency_ms == 6_500.0


def test_sanitization_reports_stable_safe_corrections_and_deduplicates() -> None:
    orchestrator = Orchestrator(registry=_RegistryStub(), system_prompt="route")
    tools = ToolSnapshot.from_llm_tools(
        [
            {
                "name": "server__first",
                "description": "first",
                "input_schema": {"type": "object"},
            },
            {
                "name": "server__second",
                "description": "second",
                "input_schema": {"type": "object"},
            },
        ]
    )

    proposal, corrections = orchestrator._sanitize_proposal(
        OrchestrationProposal(
            selected_model_id="unknown",
            selected_tools=["server__first", "server__first", "missing"],
        ),
        tools,
    )

    assert proposal.selected_model_id == "model"
    assert proposal.selected_tools == ["server__first"]
    assert corrections == (
        "unknown_model_id",
        "unknown_tool_id",
        "duplicate_tool_id",
    )


def test_tool_preferences_are_immutable_and_tolerate_missing_tools() -> None:
    preferences = ToolPreferences.from_request(
        [
            SimpleNamespace(name="empty", tools=None),
            SimpleNamespace(name="search", tools={"query": ["q"]}),
        ]
    )

    assert preferences.preferred_tools == ("search__query",)
    assert preferences.tool_arg_hints["search__query"] == ("q",)
    with pytest.raises(TypeError):
        preferences.tool_arg_hints["search__query"] = ()  # type: ignore[index]
