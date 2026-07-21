"""Hermetic configuration validation tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import (
    MCPConfig,
    Settings,
    ToolPolicyConfig,
    load_mcp_config,
    load_mcp_config_from_settings,
)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (ToolPolicyConfig, {"mdoe": "allow_list", "allow": ["x__*"]}),
        (MCPConfig, {"mcpServers": {}, "tool_polciy": {"mode": "allow_all"}}),
        (
            MCPConfig,
            {
                "mcpServers": {
                    "demo": {
                        "transport": "stdio",
                        "command": "demo",
                        "disabled_tool": ["dangerous"],
                    }
                }
            },
        ),
    ],
    ids=["tool-policy-field", "top-level-field", "server-field"],
)
def test_unknown_config_fields_fail_loud(model, payload) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_settings_load_without_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    # The test must exercise declared defaults, independent of a developer's
    # exported harness configuration as well as their .env file.
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_provider
    assert settings.llm_model
    assert settings.llm_max_tokens > 0
    assert settings.max_loop_iterations > 0
    assert settings.mcp_connect_timeout_seconds == 30


@pytest.mark.parametrize("value", [0, -1, 12.5])
def test_mcp_connect_timeout_accepts_disabled_and_positive_values(value: float) -> None:
    settings = Settings(_env_file=None, mcp_connect_timeout_seconds=value)

    assert settings.mcp_connect_timeout_seconds == value


def test_load_mcp_config_and_filter_disabled_servers(tmp_path) -> None:
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        """\
mcpServers:
  enabled:
    transport: stdio
    command: demo
    args: [serve]
  disabled:
    transport: streamable-http
    url: http://example.invalid/mcp
    disabled: true
""",
        encoding="utf-8",
    )

    config = load_mcp_config(config_path)

    assert set(config.mcp_servers) == {"enabled", "disabled"}
    assert set(config.enabled_servers()) == {"enabled"}


def test_settings_owns_dotenv_values_without_mutating_process_environment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "GEMINI_API_KEY=dotenv-gemini",
                "OPENAI_API_KEY=dotenv-openai",
                "FUTURE_API_KEY=dotenv-future",
                "DOTENV_ONLY_API_KEY=dotenv-only",
                "OPEN_WEBSEARCH_URL=http://dotenv.example/mcp",
                "TRACE_ENABLED=false",
            ]
        ),
        encoding="utf-8",
    )
    for name in (
        "GEMINI_API_KEY",
        "DOTENV_ONLY_API_KEY",
        "OPEN_WEBSEARCH_URL",
        "TRACE_ENABLED",
        "CONTEXT_STRATEGY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "process-openai")
    monkeypatch.setenv("FUTURE_API_KEY", "process-future")
    before = dict(os.environ)

    settings = Settings(_env_file=env_path)
    environment = settings.interpolation_environment()

    assert settings.api_key_for_provider("gemini") == "dotenv-gemini"
    assert settings.api_key_for_provider("openai") == "process-openai"
    assert settings.api_key_for_provider("future") == "process-future"
    assert settings.api_key_for_provider("dotenv_only") == "dotenv-only"
    assert environment["OPEN_WEBSEARCH_URL"] == "http://dotenv.example/mcp"
    assert environment["TRACE_ENABLED"] == "false"
    assert "CONTEXT_STRATEGY" not in environment
    assert settings.trace_enabled is False
    assert settings.model_extra is None
    assert not hasattr(settings, "open_websearch_url")
    assert "open_websearch_url" not in settings.model_dump()
    assert os.environ == before


def test_explicit_environment_drives_generic_mcp_interpolation(tmp_path) -> None:
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        """\
mcpServers:
  search:
    transport: streamable-http
    url: ${SEARCH_ENDPOINT}
""",
        encoding="utf-8",
    )

    config = load_mcp_config(
        config_path,
        environment={"SEARCH_ENDPOINT": "http://search.example/mcp"},
    )

    assert config.mcp_servers["search"].url == "http://search.example/mcp"


def test_settings_composition_drives_generic_mcp_interpolation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SEARCH_ENDPOINT", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "SEARCH_ENDPOINT=http://settings.example/mcp\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        """\
mcpServers:
  search:
    transport: streamable-http
    url: ${SEARCH_ENDPOINT}
""",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_path, mcp_config_path=config_path)

    config = load_mcp_config_from_settings(settings)

    assert config.mcp_servers["search"].url == "http://settings.example/mcp"


def test_application_import_does_not_mutate_environment() -> None:
    code = (
        "import os; import config.settings as settings_module; "
        "before = dict(os.environ); assert settings_module._settings_cache is None; "
        "import main; assert settings_module._settings_cache is None; "
        "assert os.environ == before"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
    )


def test_obsolete_config_symbol_is_removed() -> None:
    assert not hasattr(Settings, "required_api_key")
