"""Repository-owned pytest safety policy."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


TESTS_ROOT = Path(__file__).parent


def _pytestmark_value(path: Path) -> ast.expr:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            return node.value
    raise AssertionError(f"{path.name} has no module-level pytestmark")


def _is_pytest_mark(attribute: ast.Attribute) -> bool:
    mark = attribute.value
    return (
        isinstance(mark, ast.Attribute)
        and mark.attr == "mark"
        and isinstance(mark.value, ast.Name)
        and mark.value.id == "pytest"
    )


def test_timeout_plugin_owns_the_hermetic_deadlock_guard(
    pytestconfig: pytest.Config,
) -> None:
    assert pytestconfig.pluginmanager.hasplugin("timeout")
    assert float(pytestconfig.getini("timeout")) == 30.0


def test_live_modules_are_opt_in_and_exempt_from_the_hermetic_timeout() -> None:
    live_modules = sorted(TESTS_ROOT.glob("test_*_live.py"))
    assert live_modules

    for path in live_modules:
        pytestmark = _pytestmark_value(path)
        marker_names = {
            node.attr
            for node in ast.walk(pytestmark)
            if isinstance(node, ast.Attribute) and _is_pytest_mark(node)
        }
        timeout_calls = [
            node
            for node in ast.walk(pytestmark)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _is_pytest_mark(node.func)
            and node.func.attr == "timeout"
        ]

        assert "live" in marker_names, path.name
        assert any(
            len(call.args) == 1
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == 0
            for call in timeout_calls
        ), path.name


def test_test_modules_do_not_configure_global_logging_at_import() -> None:
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            function = node.value.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "basicConfig"
                and isinstance(function.value, ast.Name)
                and function.value.id == "logging"
            ):
                offenders.append(path.name)

    assert offenders == []
