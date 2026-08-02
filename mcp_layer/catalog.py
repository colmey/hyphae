"""Immutable MCP catalog values and deterministic tool normalization."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import jsonschema
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for

from .client import Tool
from .contracts import NAMESPACE_SEP, ToolSpec

_MAX_TOOLS_PER_SERVER = 256
_MAX_TOOL_NAME_CHARS = 128
_MAX_TOOL_METADATA_CHARS = 20_000


class CatalogError(ValueError):
    """A health-safe malformed catalog error."""


@dataclass(frozen=True, slots=True)
class ToolRoute:
    """One validated namespaced route in a published catalog."""

    server_name: str
    local_name: str
    spec: ToolSpec
    validator: Validator


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """One atomically replaceable, immutable per-server catalog value."""

    routes: tuple[ToolRoute, ...] = ()
    revision: int = 0
    discovered_at: datetime | None = None
    discovered_monotonic: float | None = None


def normalize_tools(
    server_name: str,
    disabled_tools: Collection[str],
    tools: Sequence[Tool],
) -> tuple[ToolRoute, ...]:
    """Validate, bound, namespace, and freeze one complete server catalog."""
    if len(tools) > _MAX_TOOLS_PER_SERVER:
        raise CatalogError(f"catalog exceeds {_MAX_TOOLS_PER_SERVER} tools per server")

    routes: list[ToolRoute] = []
    seen: set[str] = set()
    disabled = set(disabled_tools)
    for position, tool in enumerate(tools):
        name = tool.name
        if not isinstance(name, str) or not name.strip():
            raise CatalogError(f"tool at position {position} has a blank name")
        if len(name) > _MAX_TOOL_NAME_CHARS:
            raise CatalogError(f"tool name at position {position} is oversized")
        if name in seen:
            raise CatalogError(f"duplicate tool name {name!r}")
        seen.add(name)

        description = tool.description
        if not isinstance(description, str):
            raise CatalogError(f"tool {name!r} has a malformed description")
        if len(description) > _MAX_TOOL_METADATA_CHARS:
            raise CatalogError(f"tool {name!r} has an oversized description")

        schema = tool.input_schema
        if not isinstance(schema, Mapping):
            raise CatalogError(f"tool {name!r} has a non-object input schema")
        try:
            encoded_schema = json.dumps(
                schema,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise CatalogError(f"tool {name!r} has a malformed input schema") from exc
        if len(encoded_schema) > _MAX_TOOL_METADATA_CHARS:
            raise CatalogError(f"tool {name!r} has an oversized input schema")
        normalized_schema = json.loads(encoded_schema)
        try:
            validator_class = validator_for(normalized_schema)
            validator_class.check_schema(normalized_schema)
        except jsonschema.SchemaError as exc:
            raise CatalogError(f"tool {name!r} has an invalid input schema") from exc

        if name in disabled:
            continue

        spec = ToolSpec.from_mapping(
            {
                "name": f"{server_name}{NAMESPACE_SEP}{name}",
                "description": description,
                "input_schema": normalized_schema,
            }
        )
        routes.append(
            ToolRoute(
                server_name=server_name,
                local_name=name,
                spec=spec,
                validator=validator_class(normalized_schema),
            )
        )
    return tuple(routes)


def validation_error(route: ToolRoute, arguments: dict[str, Any]) -> str | None:
    """Return the stable live-schema drift error for one proposed call."""
    try:
        route.validator.validate(arguments)
    except jsonschema.ValidationError as exc:
        path = ".".join(str(item) for item in exc.absolute_path) or "<root>"
        return f"arguments no longer match the refreshed schema at {path}"
    return None
