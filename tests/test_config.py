"""Hermetic configuration validation tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import (
    LLMSettings,
    MCPConfig,
    Settings,
    ToolPolicyConfig,
    load_mcp_config,
    load_mcp_config_from_settings,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

_RENAMED_SETTING_CASES = [
    (
        "OPENAI_COMPAT_BASE_URL",
        "OPENAI_PROVIDER_BASE_URL",
        "openai_compat_base_url",
        "https://canonical.example/v1",
        "https://legacy.example/v1",
    ),
    (
        "LLM_MODEL_NAME",
        "LLM_MODEL",
        "llm.model_name",
        "canonical-model",
        "legacy-model",
    ),
    (
        "LOOP_MAX_ITERATIONS",
        "MAX_LOOP_ITERATIONS",
        "loop_max_iterations",
        "17",
        "12",
    ),
    ("RUN_MAX_TOKENS", "MAX_RUN_TOKENS", "run_max_tokens", "1700", "1200"),
    ("RUN_MAX_SECONDS", "MAX_RUN_SECONDS", "run_max_seconds", "17.5", "12.5"),
    ("SESSION_CAPACITY", "SESSION_MAX_COUNT", "session_capacity", "170", "120"),
    ("HYPHAE_API_KEY", "HARNESS_API_KEY", "hyphae_api_key", "canonical", "legacy"),
    (
        "TRACE_JSONL_PATH",
        "TRACE_PATH",
        "trace_jsonl_path",
        "traces/canonical.jsonl",
        "traces/legacy.jsonl",
    ),
]


def _setting_value(settings: Settings, field_path: str):
    value = settings
    for field_name in field_path.split("."):
        value = getattr(value, field_name)
    return value


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
    for field_name in LLMSettings.model_fields:
        monkeypatch.delenv(f"LLM_{field_name.upper()}", raising=False)
    for _, legacy_name, _, _, _ in _RENAMED_SETTING_CASES:
        monkeypatch.delenv(legacy_name, raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm.provider
    assert settings.llm.model_name
    assert settings.llm.max_tokens > 0
    assert settings.loop_max_iterations > 0
    assert settings.mcp_connect_timeout_seconds == 30
    assert settings.mcp_catalog_ttl_seconds == 300
    assert settings.openai_compat_tool_activity_mode == "reasoning"
    assert settings.openai_compat_tool_activity_max_chars == 2000


@pytest.mark.parametrize(
    ("canonical_name", "legacy_name", "field_name", "_canonical", "legacy"),
    _RENAMED_SETTING_CASES,
)
def test_renamed_setting_legacy_environment_aliases_remain_supported(
    monkeypatch: pytest.MonkeyPatch,
    canonical_name: str,
    legacy_name: str,
    field_name: str,
    _canonical: str,
    legacy: str,
) -> None:
    monkeypatch.delenv(canonical_name, raising=False)
    monkeypatch.setenv(legacy_name, legacy)

    settings = Settings(_env_file=None)

    expected = str(_REPO_ROOT / legacy) if field_name.endswith("_path") else legacy
    assert str(_setting_value(settings, field_name)) == expected
    assert legacy_name.lower() not in settings.model_dump()


@pytest.mark.parametrize(
    ("canonical_name", "legacy_name", "field_name", "canonical", "legacy"),
    _RENAMED_SETTING_CASES,
)
def test_renamed_setting_canonical_names_win_over_legacy_aliases(
    monkeypatch: pytest.MonkeyPatch,
    canonical_name: str,
    legacy_name: str,
    field_name: str,
    canonical: str,
    legacy: str,
) -> None:
    monkeypatch.setenv(canonical_name, canonical)
    monkeypatch.setenv(legacy_name, legacy)

    settings = Settings(_env_file=None)

    expected = (
        str(_REPO_ROOT / canonical) if field_name.endswith("_path") else canonical
    )
    assert str(_setting_value(settings, field_name)) == expected


def test_llm_settings_are_nested_without_flat_runtime_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL_NAME", "nested-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "123")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "4.5")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY", "0.25")

    settings = Settings(_env_file=None)

    assert settings.llm == LLMSettings(
        provider="openai_compatible",
        model_name="nested-model",
        max_tokens=123,
        timeout_seconds=4.5,
        max_retries=2,
        retry_base_delay=0.25,
    )
    assert not hasattr(settings, "llm_provider")
    assert not hasattr(settings, "llm_model_name")


def test_openai_compat_tool_activity_settings_load_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_TOOL_ACTIVITY_MODE", "reasoning_full")
    monkeypatch.setenv("OPENAI_COMPAT_TOOL_ACTIVITY_MAX_CHARS", "1234")

    settings = Settings(_env_file=None)

    assert settings.openai_compat_tool_activity_mode == "reasoning_full"
    assert settings.openai_compat_tool_activity_max_chars == 1234


def test_oldest_openai_base_url_alias_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://oldest.example/v1")

    settings = Settings(_env_file=None)

    assert settings.openai_compat_base_url == "https://oldest.example/v1"


def test_legacy_openai_tool_activity_setting_names_remain_input_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_TOOL_ACTIVITY", "hidden")
    monkeypatch.setenv("OPENAI_TOOL_BLOCK_MAX_CHARS", "987")

    settings = Settings(_env_file=None)

    assert settings.openai_compat_tool_activity_mode == "hidden"
    assert settings.openai_compat_tool_activity_max_chars == 987
    assert "openai_compat_tool_activity" not in settings.model_dump()
    assert "openai_tool_block_max_chars" not in settings.model_dump()


def test_canonical_openai_tool_activity_names_win_over_legacy_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_TOOL_ACTIVITY_MODE", "reasoning")
    monkeypatch.setenv("OPENAI_COMPAT_TOOL_ACTIVITY", "hidden")
    monkeypatch.setenv("OPENAI_COMPAT_TOOL_ACTIVITY_MAX_CHARS", "4321")
    monkeypatch.setenv("OPENAI_TOOL_BLOCK_MAX_CHARS", "987")

    settings = Settings(_env_file=None)

    assert settings.openai_compat_tool_activity_mode == "reasoning"
    assert settings.openai_compat_tool_activity_max_chars == 4321


def test_openai_compat_tool_activity_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_compat_tool_activity_mode="content")


@pytest.mark.parametrize("value", [0, -1, 12.5])
def test_mcp_connect_timeout_accepts_disabled_and_positive_values(value: float) -> None:
    settings = Settings(_env_file=None, mcp_connect_timeout_seconds=value)

    assert settings.mcp_connect_timeout_seconds == value


@pytest.mark.parametrize("value", [0, -1, 45.5])
def test_mcp_catalog_ttl_accepts_disabled_and_positive_values(value: float) -> None:
    settings = Settings(_env_file=None, mcp_catalog_ttl_seconds=value)

    assert settings.mcp_catalog_ttl_seconds == value


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
    assert settings.api_key_for_provider("openai_compatible") == "process-openai"
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
