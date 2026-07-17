"""Hermetic orchestration schema and prompt-construction tests."""

from __future__ import annotations

from typing import Any

import pytest

from llm.client import GenerationRequest, LLMClient
from llm.schemas import AssistantMessage, Message, TextBlock
from mcp_layer import ToolSnapshot
from orchestrator import Orchestrator
from orchestrator.schemas import OrchestrationResult


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

    assert OrchestrationResult.model_validate(payload).thinking_level == expected


class _RegistryStub:
    def default_id(self) -> str:
        return "model"

    def describe_for_prompt(self) -> str:
        return "model: test model"


class _RecordingLLM(LLMClient):
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.requests.append(request)
        return AssistantMessage(
            content=[TextBlock("{\"selected_model_id\":\"model\"}")],
            stop_reason="end_turn",
        )


@pytest.mark.anyio
async def test_orchestration_constructs_structured_generation_request() -> None:
    orchestrator = Orchestrator(
        registry=_RegistryStub(),  # type: ignore[arg-type]
        system_prompt="route requests",
    )
    llm = _RecordingLLM()

    raw = await orchestrator._call_orchestrator_llm(llm, "choose a model")

    assert raw == '{"selected_model_id":"model"}'
    request = llm.requests[0]
    assert request.messages == [Message.user("choose a model")]
    assert request.tools is None
    assert request.system == "route requests"
    assert request.response_schema is OrchestrationResult


def test_prompt_includes_history_only_when_present() -> None:
    orchestrator = Orchestrator(
        registry=_RegistryStub(),  # type: ignore[arg-type]
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
