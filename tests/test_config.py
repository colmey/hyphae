"""Hermetic configuration validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import MCPConfig, Settings, ToolPolicyConfig, load_mcp_config


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
