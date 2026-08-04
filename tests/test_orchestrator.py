"""Hermetic orchestration schema and prompt-construction tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent.runtime import ModelLimits
from llm.client import GenerationRequest, LLMClient
from llm.schemas import AssistantMessage, Message, TextBlock
from tooling import ToolSnapshot
from orchestrator import Orchestrator, ToolPreferences
from orchestrator.schemas import OrchestrationProposal


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
        "generated_system_prompt": "system",
        "made_up": "ignored",
    }
    if value is not None:
        payload["thinking_level"] = value

    assert OrchestrationProposal.model_validate(payload).thinking_level == expected


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
        )


@pytest.mark.anyio
async def test_orchestration_constructs_structured_generation_request() -> None:
    orchestrator = Orchestrator(
        registry=_RegistryStub(),
        system_prompt="route requests",
    )
    llm = _RecordingLLM()

    raw = await orchestrator._call_orchestrator_llm(llm, "choose a model")

    assert raw == '{"selected_model_id":"model"}'
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
                    '{"selected_model_id":"model","selected_tools":[],\n'
                    '"generated_system_prompt":"selected system"}'
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
