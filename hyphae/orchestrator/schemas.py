# hyphae/orchestrator/schemas.py

"""Structured output and in-process values for orchestration.

Typed file-config (ModelEntry/ModelsConfig) lives in the config package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, Protocol

from pydantic import BaseModel, Field, field_validator

from hyphae.tooling import NAMESPACE_SEP
from hyphae.llm.schemas import CompletionUsage


class _ToolPreferenceInput(Protocol):
    """The direct-caller preference shape accepted by ``from_request``."""

    name: str
    tools: Mapping[str, Iterable[str] | None] | None


class OrchestrationProposal(BaseModel):
    """The structured output the orchestrator LLM must produce.

    Deliberately leaves `extra="forbid"` off because Gemini rejects the
    resulting JSON Schema keyword. Runtime sanitization still validates model
    and tool IDs against live inventory.
    """

    selected_model_id: str = Field(
        ...,
        description="Must match one of the model IDs in models.yaml.",
    )
    selected_tools: list[str] = Field(
        default_factory=list,
        description="Namespaced tool names ({server}__{tool}) to expose to the agent. "
        "Empty list = no tools.",
    )
    thinking_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="How much the downstream model should deliberate: 'low' for "
        "lookups/single-tool calls, 'medium' for the typical case, "
        "'high' for complex multi-step reasoning. Defaults to 'medium' "
        "if the orchestrator omits it.",
    )

    @field_validator("thinking_level", mode="before")
    @classmethod
    def _coerce_thinking_level(cls, v: object) -> str:
        """Clamp stray values to medium rather than failing orchestration."""
        if isinstance(v, str):
            lowered = v.strip().lower()
            if lowered in ("low", "medium", "high"):
                return lowered
        return "medium"


@dataclass(frozen=True, slots=True)
class OrchestrationDecision:
    """Orchestrator output plus safe fallback and control-call telemetry."""

    result: OrchestrationProposal
    fallback_used: bool = False
    fallback_reason: str | None = None
    corrections: tuple[str, ...] = ()
    usage: CompletionUsage = field(default_factory=CompletionUsage)
    latency_ms: float = 0.0
    control_model_id: str = ""


@dataclass(frozen=True, slots=True)
class ToolPreferences:
    """Normalized caller hints about which tools to prioritize.

    This is a soft hint, not a filter; other tools remain available.
    """

    preferred_tools: tuple[str, ...] = ()
    tool_arg_hints: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "preferred_tools", tuple(self.preferred_tools))
        object.__setattr__(
            self,
            "tool_arg_hints",
            MappingProxyType(
                {name: tuple(args) for name, args in self.tool_arg_hints.items()}
            ),
        )

    def __bool__(self) -> bool:
        return bool(self.preferred_tools)

    @classmethod
    def from_request(
        cls, prefs: Iterable[_ToolPreferenceInput] | None
    ) -> "ToolPreferences":
        """Normalize direct-caller server preferences into namespaced tools."""
        preferred: list[str] = []
        arg_hints: dict[str, tuple[str, ...]] = {}
        for server in prefs or ():
            for tool_name, args in (server.tools or {}).items():
                namespaced = f"{server.name}{NAMESPACE_SEP}{tool_name}"
                preferred.append(namespaced)
                arg_hints[namespaced] = tuple(args or ())
        return cls(preferred_tools=tuple(preferred), tool_arg_hints=arg_hints)
