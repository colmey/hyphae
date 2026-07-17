"""Stable tool-inventory values shared by MCP, routing, and the agent loop."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

NAMESPACE_SEP = "__"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class ToolSpec:
    """One provider-neutral tool advertisement."""

    name: str
    description: str
    input_schema: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ToolSpec":
        schema = value.get("input_schema") or {}
        return cls(
            name=str(value["name"]),
            description=str(value.get("description") or ""),
            input_schema=_freeze(schema),
        )

    def as_llm_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _thaw(self.input_schema),
        }


@dataclass(frozen=True)
class ToolSnapshot:
    """Immutable inventory captured exactly once for an accepted turn."""

    tools: tuple[ToolSpec, ...] = ()

    @classmethod
    def from_llm_tools(cls, tools: Iterable[Mapping[str, Any]]) -> "ToolSnapshot":
        return cls(tuple(ToolSpec.from_mapping(tool) for tool in tools))

    @property
    def names(self) -> frozenset[str]:
        return frozenset(tool.name for tool in self.tools)

    def selected(self, names: Iterable[str]) -> "ToolSnapshot":
        selected = set(names)
        return ToolSnapshot(tuple(tool for tool in self.tools if tool.name in selected))

    def as_llm_tools(self) -> list[dict[str, Any]]:
        return [tool.as_llm_dict() for tool in self.tools]

    def __bool__(self) -> bool:
        return bool(self.tools)
