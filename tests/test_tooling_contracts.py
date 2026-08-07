"""Focused ownership and immutability tests for neutral tool contracts."""

from __future__ import annotations

import ast
from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

import pytest

import hyphae.tooling as tooling
from hyphae.agent import run_agent
from hyphae.mcp_runtime.client import MCPConnection
from hyphae.mcp_runtime.lease import TurnToolRuntime
from hyphae.mcp_runtime.manager import MCPManager
from hyphae.orchestrator.contracts import RoutingService
from hyphae.tooling import ToolCallResult, ToolRuntime, ToolSnapshot, ToolSpec


ROOT = Path(__file__).resolve().parents[1]
OLD_PACKAGE = "_".join(("mcp", "layer"))
CANONICAL_VALUES = {"ToolCallResult", "ToolSnapshot", "ToolSpec"}


def test_tooling_has_deliberate_neutral_exports() -> None:
    assert tooling.__all__ == [
        "NAMESPACE_SEP",
        "ToolCallResult",
        "ToolRuntime",
        "ToolSnapshot",
        "ToolSpec",
    ]
    assert tooling.NAMESPACE_SEP == "__"


def test_neutral_values_are_frozen_slotted_and_detached() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "enum": ["original"],
            }
        },
    }
    spec = ToolSpec("server__lookup", "Lookup", schema)
    result = ToolCallResult("ok", False)
    source_tools = [spec]
    snapshot = ToolSnapshot(source_tools)

    schema["type"] = "array"
    schema["properties"]["query"]["enum"].append("mutated")
    source_tools.clear()

    assert not hasattr(result, "__dict__")
    assert not hasattr(spec, "__dict__")
    assert not hasattr(snapshot, "__dict__")
    assert snapshot.tools == (spec,)
    assert spec.as_llm_dict()["input_schema"] == {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "enum": ["original"],
            }
        },
    }
    with pytest.raises(FrozenInstanceError):
        result.content = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        spec.description = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.tools = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.input_schema["type"] = "array"  # type: ignore[index]


def test_schema_freeze_recurses_through_tuples_and_rejects_non_json_values() -> None:
    nested_list: list[str] = ["original"]
    spec = ToolSpec(
        "server__tuple",
        "Tuple-shaped source",
        {"allOf": ({"enum": nested_list},)},
    )

    nested_list.append("mutated")

    assert spec.as_llm_dict()["input_schema"] == {"allOf": [{"enum": ["original"]}]}
    with pytest.raises(AttributeError):
        spec.input_schema["allOf"][0]["enum"].append("mutated")
    with pytest.raises(TypeError, match="JSON-shaped"):
        ToolSpec("server__set", "Unsupported", {"enum": {"value"}})
    with pytest.raises(TypeError, match="input_schema must be a mapping"):
        ToolSpec.from_mapping({"name": "server__list", "input_schema": []})


def test_schema_thaw_and_snapshot_operations_return_detached_values() -> None:
    snapshot = ToolSnapshot.from_llm_tools(
        [
            {
                "name": "server__first",
                "description": "First",
                "input_schema": {
                    "type": "object",
                    "required": ["query"],
                },
            },
            {
                "name": "server__second",
                "description": "Second",
                "input_schema": {},
            },
        ]
    )

    first_render = snapshot.as_llm_tools()
    first_render[0]["input_schema"]["required"].append("mutated")
    second_render = snapshot.as_llm_tools()

    assert second_render[0]["input_schema"]["required"] == ["query"]
    assert first_render is not second_render
    assert snapshot.names == frozenset({"server__first", "server__second"})
    selected = snapshot.select(["server__second", "unknown"])
    assert tuple(tool.name for tool in selected.tools) == ("server__second",)


def test_tooling_import_is_independent_of_mcp_infrastructure() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import hyphae.tooling; "
                "assert not any(name == 'hyphae.mcp_runtime' or "
                "name.startswith('hyphae.mcp_runtime.') for name in sys.modules); "
                "assert not any(name == 'mcp' or name.startswith('mcp.') "
                "for name in sys.modules)"
            ),
        ],
        check=True,
        cwd=ROOT,
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import mcp; "
                f"root = Path({str(ROOT)!r}).resolve(); "
                "module = Path(mcp.__file__).resolve(); "
                "assert module != root / 'mcp.py'; "
                "assert module != root / 'mcp' / '__init__.py'"
            ),
        ],
        check=True,
        cwd=ROOT,
    )
    assert not (ROOT / "mcp.py").exists()
    assert not (ROOT / "mcp").exists()


def test_public_tool_annotations_resolve_to_neutral_contracts() -> None:
    assert get_type_hints(run_agent)["mcp"] is ToolRuntime
    assert get_type_hints(ToolRuntime.call_tool)["return"] is ToolCallResult
    assert get_type_hints(MCPConnection.call_tool)["return"] is ToolCallResult
    assert get_type_hints(TurnToolRuntime.call_tool)["return"] is ToolCallResult
    assert get_type_hints(RoutingService.decide)["tools"] is ToolSnapshot
    assert get_type_hints(MCPManager.open_turn)["return"] == AsyncIterator[ToolRuntime]


def _imports_from(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_neutral_consumers_do_not_import_mcp_runtime() -> None:
    for package in (ROOT / "hyphae" / "agent", ROOT / "hyphae" / "orchestrator"):
        for path in package.rglob("*.py"):
            imports = _imports_from(path)
            assert not any(
                name == "hyphae.mcp_runtime" or name.startswith("hyphae.mcp_runtime.")
                for name in imports
            ), path


def _implementation_and_configuration_paths() -> list[Path]:
    paths = list((ROOT / "hyphae").rglob("*"))
    paths.extend((ROOT / "tests").rglob("*"))
    paths.extend((ROOT / ".github").rglob("*"))
    paths.append(ROOT / "pyproject.toml")
    return paths


def test_old_package_is_absent_from_implementation_and_configuration() -> None:
    stale = [
        path.relative_to(ROOT)
        for path in _implementation_and_configuration_paths()
        if path.is_file()
        and path.suffix in {".py", ".toml", ".yml", ".yaml"}
        and OLD_PACKAGE in path.read_text(encoding="utf-8")
    ]
    assert stale == []
    assert not (ROOT / OLD_PACKAGE).exists()


def test_neutral_values_have_one_canonical_definition() -> None:
    definitions: dict[str, list[Path]] = {name: [] for name in CANONICAL_VALUES}
    for path in (ROOT / "hyphae").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in definitions:
                definitions[node.name].append(path.relative_to(ROOT))

    assert definitions == {
        name: [Path("hyphae/tooling/contracts.py")] for name in CANONICAL_VALUES
    }
