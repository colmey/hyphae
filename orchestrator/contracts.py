"""Narrow capabilities consumed by orchestration and turn routing."""

from __future__ import annotations

import logging
from typing import Protocol

from agent.runtime import ModelLimits
from llm.client import LLMClient
from llm.schemas import Message
from tooling import ToolSnapshot

from .schemas import OrchestrationDecision


class ModelRegistry(Protocol):
    """Model inventory and lookup operations used by routing consumers."""

    @property
    def model_ids(self) -> list[str]: ...

    def default_id(self) -> str: ...

    def describe_for_prompt(self) -> str: ...

    def get(self, model_id: str) -> LLMClient: ...

    def get_or_default(self, model_id: str | None) -> tuple[str, LLMClient]: ...

    def get_entry(self, model_id: str) -> ModelLimits: ...


class RoutingService(Protocol):
    """One routing decision operation consumed by the turn boundary."""

    async def decide(
        self,
        user_message: str,
        tools: ToolSnapshot,
        *,
        history: list[Message] | None = None,
        timeout: float | None = None,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    ) -> OrchestrationDecision: ...
