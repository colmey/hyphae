"""Hermetic orchestration schema and prompt-construction tests."""

from __future__ import annotations

from typing import Any

import pytest

from llm.schemas import Message, TextBlock
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


class _MCPStub:
    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return []


def test_prompt_includes_history_only_when_present() -> None:
    orchestrator = Orchestrator(
        registry=_RegistryStub(),  # type: ignore[arg-type]
        mcp=_MCPStub(),  # type: ignore[arg-type]
        system_prompt="route requests",
    )
    history = [
        Message.user("List the tables in the customer database."),
        Message.assistant([TextBlock(text="The tables are customers and orders.")]),
    ]

    with_history = orchestrator._build_prompt("now do last month", history=history)
    without_history = orchestrator._build_prompt("hello", history=None)

    assert "CONVERSATION SO FAR" in with_history
    assert "customer database" in with_history
    assert "CONVERSATION SO FAR" not in without_history
