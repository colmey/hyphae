"""Provider-neutral tool contracts and immutable inventory values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

NAMESPACE_SEP = "__"


def _freeze(value: Any) -> Any:
    """Detach JSON-shaped schema data and replace every accepted container."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("tool schema object keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"tool schema values must be JSON-shaped; got {type(value).__name__}"
    )


def _thaw(value: Any) -> Any:
    """Return a detached mutable representation suitable for provider payloads."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Provider-neutral flattened result of one tool call."""

    content: str
    is_error: bool


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One provider-neutral tool advertisement."""

    name: str
    description: str
    input_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("input_schema must be a mapping")
        object.__setattr__(self, "input_schema", _freeze(self.input_schema))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ToolSpec":
        schema = value.get("input_schema")
        if schema is None:
            schema = {}
        return cls(
            name=str(value["name"]),
            description=str(value.get("description") or ""),
            input_schema=schema,
        )

    def as_llm_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _thaw(self.input_schema),
        }


@dataclass(frozen=True, slots=True, init=False)
class ToolSnapshot:
    """Immutable inventory captured exactly once for an accepted turn."""

    tools: tuple[ToolSpec, ...] = ()

    def __init__(self, tools: Iterable[ToolSpec] = ()) -> None:
        object.__setattr__(self, "tools", tuple(tools))

    @classmethod
    def from_llm_tools(cls, tools: Iterable[Mapping[str, Any]]) -> "ToolSnapshot":
        return cls(tuple(ToolSpec.from_mapping(tool) for tool in tools))

    @property
    def names(self) -> frozenset[str]:
        return frozenset(tool.name for tool in self.tools)

    def select(self, names: Iterable[str]) -> "ToolSnapshot":
        selected = set(names)
        return ToolSnapshot(tuple(tool for tool in self.tools if tool.name in selected))

    def as_llm_tools(self) -> list[dict[str, Any]]:
        return [tool.as_llm_dict() for tool in self.tools]

    def __bool__(self) -> bool:
        return bool(self.tools)


class ToolRuntime(Protocol):
    """Neutral inventory and dispatch capability consumed by agent runs."""

    def get_tools_for_llm(self) -> list[dict[str, Any]]: ...

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolCallResult: ...
