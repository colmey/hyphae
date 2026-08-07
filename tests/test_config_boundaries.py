"""Strict configuration boundary and security tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import AliasChoices, ValidationError

from hyphae.config import (
    ConfigLoadError,
    ConfigValidationError,
    LLMSettings,
    MCPConfig,
    ModelEntry,
    ModelsConfig,
    SSEServer,
    SamplingParams,
    Settings,
    StdioServer,
    StreamableHTTPServer,
    ToolPolicyConfig,
    load_mcp_config,
    load_mcp_config_from_settings,
    load_models_config,
    load_models_config_from_settings,
    load_agent_prompt,
    load_orchestrator_prompt,
    get_settings,
    reset_settings,
)
from hyphae.config.errors import _SAFE_LOCATION_PARTS, safe_validation_summary

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "text", "duplicate"),
    [
        (
            "top-level",
            "mcpServers: {}\nmcpServers: {}\n",
            "mcpServers",
        ),
        (
            "policy",
            """\
mcpServers: {}
tool_policy:
  mode: allow_all
  mode: allow_list
  allow: [demo__tool]
""",
            "mode",
        ),
        (
            "server",
            """\
mcpServers:
  demo:
    transport: stdio
    command: first
    command: second
""",
            "command",
        ),
        (
            "model",
            """\
models:
  demo:
    provider: gemini
    model: first
    model: second
    description: demo
""",
            "model",
        ),
        (
            "secret-key",
            """\
mcpServers: {}
AuthorizationBearerTopSecret: first
AuthorizationBearerTopSecret: second
""",
            "AuthorizationBearerTopSecret",
        ),
    ],
)
def test_duplicate_yaml_keys_fail_with_source_marks(
    tmp_path: Path,
    name: str,
    text: str,
    duplicate: str,
) -> None:
    path = tmp_path / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    loader = load_models_config if name == "model" else load_mcp_config

    with pytest.raises(ConfigLoadError) as raised:
        loader(path)

    message = str(raised.value)
    assert raised.value.path == path
    assert "duplicate mapping key" in message
    assert f"{duplicate!r}" not in message
    assert "first defined at line" in message
    assert "repeated at line" in message


def test_parser_and_validation_errors_do_not_render_secret_input(
    tmp_path: Path,
) -> None:
    parser_path = tmp_path / "parser.yaml"
    parser_path.write_text(
        "mcpServers:\n  demo: [password=hunter2\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigLoadError) as parser_error:
        load_mcp_config(parser_path)
    assert "hunter2" not in str(parser_error.value)
    assert parser_error.value._cause is not None
    assert not hasattr(parser_error.value, "cause")

    validation_path = tmp_path / "validation.yaml"
    validation_path.write_text(
        """\
mcpServers:
  demo:
    transport: streamable-http
    url: ${SECRET_VALUE}
""",
        encoding="utf-8",
    )
    secret = "ftp://user:password@example.invalid/mcp?token=topsecret"
    with pytest.raises(ConfigValidationError) as validation_error:
        load_mcp_config(validation_path, environment={"SECRET_VALUE": secret})
    rendered = str(validation_error.value)
    assert validation_error.value.path == validation_path
    assert validation_error.value._cause is not None
    for fragment in ("user:password", "topsecret", "input_value", "input_type"):
        assert fragment not in rendered


def test_interpolation_missing_file_and_permission_failures_are_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpolation_path = tmp_path / "interpolation.yaml"
    interpolation_path.write_text(
        """\
mcpServers:
  demo:
    transport: streamable-http
    url: ${MISSING_ENDPOINT}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigLoadError, match="MISSING_ENDPOINT") as interpolation:
        load_mcp_config(interpolation_path, environment={})
    assert interpolation.value.path == interpolation_path

    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError) as not_found:
        load_mcp_config(missing)
    assert Path(not_found.value.filename) == missing

    denied = tmp_path / "denied.yaml"
    denied.write_text("mcpServers: {}\n", encoding="utf-8")
    original_open = Path.open

    def deny_open(path: Path, *args: Any, **kwargs: Any):
        if path == denied:
            raise PermissionError("password=hunter2")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_open)
    with pytest.raises(ConfigLoadError) as permission:
        load_mcp_config(denied)
    assert permission.value.path == denied
    assert "hunter2" not in str(permission.value)


def test_application_paths_are_repository_root_relative_from_any_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        _env_file=None,
        mcp_config_path="deploy/mcp.yaml",
        models_config_path="deploy/models.yaml",
        orchestrator_prompt_path="deploy/prompt.md",
        agent_prompt_path="deploy/agent.md",
        trace_jsonl_path="var/trace.jsonl",
    )

    assert settings.mcp_config_path == _REPO_ROOT / "deploy/mcp.yaml"
    assert settings.models_config_path == _REPO_ROOT / "deploy/models.yaml"
    assert settings.orchestrator_prompt_path == _REPO_ROOT / "deploy/prompt.md"
    assert settings.agent_prompt_path == _REPO_ROOT / "deploy/agent.md"
    assert settings.trace_jsonl_path == _REPO_ROOT / "var/trace.jsonl"


def test_dotenv_relative_paths_use_repository_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "MCP_CONFIG_PATH=custom/mcp.yaml\nTRACE_PATH=custom/trace.jsonl\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=env_path)

    assert settings.mcp_config_path == _REPO_ROOT / "custom/mcp.yaml"
    assert settings.trace_jsonl_path == _REPO_ROOT / "custom/trace.jsonl"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loop_max_iterations", 0),
        ("openai_compat_tool_activity_max_chars", 0),
        ("context_default_window_tokens", 0),
        ("context_recent_messages", 0),
        ("context_summary_max_tokens", 0),
    ],
)
def test_non_disableable_settings_bounds_must_be_positive(
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_retries", -1),
        ("retry_base_delay", -0.1),
    ],
)
def test_retry_settings_must_be_nonnegative(field: str, value: int | float) -> None:
    with pytest.raises(ValidationError):
        LLMSettings(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "tool_timeout_seconds",
        "mcp_connect_timeout_seconds",
        "mcp_catalog_ttl_seconds",
        "tool_result_max_chars",
        "run_max_tokens",
        "run_max_seconds",
        "abort_after_consecutive_tool_failures",
        "session_ttl_seconds",
        "session_capacity",
    ],
)
@pytest.mark.parametrize("value", [-1, 0, 1])
def test_documented_disable_bounds_preserve_nonpositive_values(
    field: str,
    value: int,
) -> None:
    settings = Settings(_env_file=None, **{field: value})
    assert getattr(settings, field) == value


def test_default_context_bounds_must_leave_positive_input_budget() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            llm={"max_tokens": 100},
            context_default_window_tokens=110,
            context_safety_margin_tokens=10,
        )


@pytest.mark.parametrize("value", [True, False, 0, -1])
def test_session_history_max_chars_is_a_positive_integer(value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, session_history_max_chars=value)


@pytest.mark.parametrize(
    "payload",
    [
        {"provider": " ", "model": "model", "description": "demo"},
        {"provider": "gemini", "model": " ", "description": "demo"},
        {"provider": "gemini", "model": "model", "description": " "},
        {
            "provider": "gemini",
            "model": "model",
            "description": "demo",
            "max_tokens": 0,
        },
        {
            "provider": "gemini",
            "model": "model",
            "description": "demo",
            "context_window": 0,
        },
        {
            "provider": "gemini",
            "model": "model",
            "description": "demo",
            "max_tokens": 10,
            "context_window": 10,
        },
    ],
)
def test_model_entry_rejects_invalid_identifiers_and_bounds(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ModelEntry.model_validate(payload)


def test_identifier_and_text_maximum_lengths_are_enforced() -> None:
    valid = ModelEntry(
        provider="p" * 128,
        model="m" * 128,
        description="d" * 20_000,
    )
    assert len(valid.provider) == 128
    with pytest.raises(ValidationError):
        ModelEntry(provider="p" * 129, model="m", description="demo")
    with pytest.raises(ValidationError):
        ModelEntry(provider="p", model="m" * 129, description="demo")
    with pytest.raises(ValidationError):
        ModelEntry(provider="p", model="m", description="d" * 20_001)
    with pytest.raises(ValidationError):
        ToolPolicyConfig(mode="allow_list", allow=["x" * 513])


@pytest.mark.parametrize(
    "server",
    [
        {"transport": "stdio", "command": " "},
        {"transport": "streamable-http", "url": "ftp://internal/mcp"},
        {"transport": "streamable-http", "url": "https:///missing-host"},
        {"transport": "sse", "url": "file://internal/sse"},
    ],
)
def test_mcp_servers_reject_invalid_commands_and_urls(server: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        MCPConfig.model_validate({"mcpServers": {"demo": server}})


@pytest.mark.parametrize(
    "url",
    [
        "http://mcp/mcp",
        "https://mcp.internal/sse",
        "http://localhost:8000/mcp",
        "http://internal./mcp",
        "http://m%C3%BCnchen.internal/mcp",
        "http://[::1]/mcp",
    ],
)
def test_mcp_http_transports_accept_internal_hosts(url: str) -> None:
    config = MCPConfig.model_validate(
        {"mcpServers": {"demo": {"transport": "streamable-http", "url": url}}}
    )
    assert config.mcp_servers["demo"].url == url


@pytest.mark.parametrize(
    "url",
    [
        "http://bad host/mcp",
        "http://host:bad/mcp",
        "http://./mcp",
        "http://-invalid.internal/mcp",
        "http://999.999.999.999/mcp",
        "http://bad%20host/mcp",
        "http://%2E/mcp",
        "http://%ZZ/mcp",
        "http://bad%2Fhost/mcp",
        "http://bad%3Ahost/mcp",
        "http://%40/mcp",
        "http://invalid-.internal/mcp",
        "http://a..b/mcp",
    ],
)
def test_mcp_http_transports_reject_malformed_authorities(url: str) -> None:
    with pytest.raises(ValidationError):
        MCPConfig.model_validate(
            {"mcpServers": {"demo": {"transport": "streamable-http", "url": url}}}
        )


def test_model_default_rules_cover_zero_one_and_multiple_entries() -> None:
    with pytest.raises(ValidationError):
        ModelsConfig(models={})

    single = ModelsConfig(
        models={
            "only": ModelEntry(
                provider="gemini",
                model="gemini-model",
                description="only model",
            )
        }
    )
    assert single.default_id() == "only"

    entries = {
        name: ModelEntry(
            provider="gemini",
            model=f"model-{name}",
            description=name,
            default=default,
        )
        for name, default in (("one", False), ("two", False))
    }
    with pytest.raises(ValidationError):
        ModelsConfig(models=entries)
    entries["one"] = entries["one"].model_copy(update={"default": True})
    assert ModelsConfig(models=entries).default_id() == "one"
    entries["two"] = entries["two"].model_copy(update={"default": True})
    with pytest.raises(ValidationError):
        ModelsConfig(models=entries)


def test_model_is_the_canonical_entry_field() -> None:
    entry = ModelEntry(provider="gemini", model="provider-model", description="demo")

    assert entry.model == "provider-model"
    assert entry.model_dump()["model"] == "provider-model"
    assert not hasattr(entry, "model_name")


def test_loaded_configuration_snapshot_is_deeply_immutable(tmp_path: Path) -> None:
    path = tmp_path / "mcp.yaml"
    path.write_text(
        """\
mcpServers:
  demo:
    transport: stdio
    command: demo
    args: [serve]
    env: {TOKEN: value}
    disabled_tools: [hidden]
tool_policy:
  mode: allow_list
  allow: [demo__*]
""",
        encoding="utf-8",
    )
    config = load_mcp_config(path)
    server = config.mcp_servers["demo"]

    assert isinstance(config.mcp_servers, MappingProxyType)
    assert server.args == ("serve",)
    assert server.disabled_tools == ("hidden",)
    assert isinstance(server.env, MappingProxyType)
    assert config.tool_policy.allow == ("demo__*",)
    dumped = config.model_dump(by_alias=True)
    assert isinstance(dumped["mcpServers"], dict)
    dumped["mcpServers"]["demo"]["env"]["TOKEN"] = "detached"
    assert server.env["TOKEN"] == "value"
    assert "mcpServers" in config.model_dump_json(by_alias=True)
    servers: Any = config.mcp_servers
    with pytest.raises(TypeError):
        servers["other"] = server
    server_env: Any = server.env
    with pytest.raises(TypeError):
        server_env["TOKEN"] = "changed"
    with pytest.raises(ValidationError):
        config.tool_policy.mode = "allow_all"


def test_settings_snapshot_is_frozen() -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        settings.loop_max_iterations = 20
    with pytest.raises(ValidationError):
        settings.llm.max_tokens = 20


def test_settings_interpolation_snapshot_is_frozen_and_time_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROTATING_SECRET", "captured")
    settings = Settings(_env_file=None)

    monkeypatch.setenv("ROTATING_SECRET", "changed")

    assert settings.interpolation_environment()["ROTATING_SECRET"] == "captured"
    with pytest.raises(AttributeError, match="snapshot is immutable"):
        settings._interpolation_values = MappingProxyType(
            {"ROTATING_SECRET": "changed"}
        )


def test_config_error_path_is_bounded_single_line_and_cause_is_private() -> None:
    source = Path("segment\nAuthorization: Bearer hidden").joinpath(*(["x" * 100] * 8))
    error = ConfigLoadError(source, "invalid config", cause=ValueError("secret"))

    assert "\n" not in str(error)
    assert len(str(error)) <= 1_600
    assert error.path == source.resolve(strict=False)
    assert error._cause is not None
    assert not hasattr(error, "cause")


def test_invalid_source_paths_fail_through_safe_typed_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError) as nul_error:
        load_mcp_config(Path("embedded\x00secret.yaml"))
    assert "\x00" not in str(nul_error.value)

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.symlink_to(second)
    second.symlink_to(first)

    with pytest.raises(ConfigLoadError) as loop_error:
        load_mcp_config(first / "mcp.yaml")

    assert "\n" not in str(loop_error.value)
    assert isinstance(loop_error.value._cause, RuntimeError)


def test_settings_path_resolution_failure_is_structured(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.symlink_to(second)
    second.symlink_to(first)

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, mcp_config_path=first / "mcp.yaml")

    assert error.value.errors(include_input=False)[0]["type"] == "path_resolution"


def test_settings_composed_mcp_loader_resolves_only_referenced_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mcp.yaml"
    path.write_text(
        """\
mcpServers:
  search:
    transport: streamable-http
    url: ${SEARCH_ENDPOINT}
""",
        encoding="utf-8",
    )
    requested: list[str] = []

    class _InterpolationSettings:
        mcp_config_path = path

        def interpolation_value(self, name: str) -> str:
            requested.append(name)
            values = {
                "SEARCH_ENDPOINT": "http://search.internal/mcp",
                "UNRELATED_SECRET": "must-not-be-requested",
            }
            return values[name]

    config = load_mcp_config_from_settings(_InterpolationSettings())

    assert config.mcp_servers["search"].url == "http://search.internal/mcp"
    assert requested == ["SEARCH_ENDPOINT"]


def test_settings_composed_models_loader_validates_effective_bounds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        """\
models:
  only:
    provider: gemini
    model: provider-model
    description: demo
""",
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        models_config_path=path,
        llm={"max_tokens": 100},
        context_default_window_tokens=110,
        context_safety_margin_tokens=9,
    )
    assert (
        load_models_config_from_settings(
            settings,
            known_providers={"gemini"},
        ).default_id()
        == "only"
    )

    invalid = settings.model_copy(update={"context_safety_margin_tokens": 10})
    with pytest.raises(ConfigValidationError):
        load_models_config_from_settings(invalid, known_providers={"gemini"})


def test_prompt_loader_uses_root_paths_and_safe_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text(" \n", encoding="utf-8")
    with pytest.raises(ConfigValidationError) as error:
        load_orchestrator_prompt(empty)
    assert error.value.path == empty
    with pytest.raises(ConfigValidationError) as error:
        load_agent_prompt(empty)
    assert error.value.path == empty


def test_agent_prompt_loader_reports_unreadable_files_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = tmp_path / "agent.md"
    prompt.write_text("trusted", encoding="utf-8")

    def deny_read_text(self: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError("secret prompt text")

    monkeypatch.setattr(Path, "read_text", deny_read_text)
    with pytest.raises(ConfigLoadError) as error:
        load_agent_prompt(prompt)
    assert error.value.path == prompt
    assert "secret prompt text" not in str(error.value)


def test_application_settings_error_does_not_render_rejected_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "user:password@example.invalid?token=topsecret"
    monkeypatch.setenv("LLM_MAX_TOKENS", secret)
    reset_settings()
    try:
        with pytest.raises(ConfigValidationError) as error:
            get_settings()
    finally:
        reset_settings()
    rendered = str(error.value)
    assert error.value._cause is not None
    for fragment in ("user:password", "topsecret", "input_value", "input_type"):
        assert fragment not in rendered


def test_application_settings_source_error_is_safely_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM", "password=hunter2")
    reset_settings()
    try:
        with pytest.raises(
            ConfigLoadError, match="could not load application settings"
        ) as error:
            get_settings()
    finally:
        reset_settings()

    assert "hunter2" not in str(error.value)
    assert error.value._cause is not None


def test_executable_samples_load_from_alternate_process_cwd(tmp_path: Path) -> None:
    code = f"""
from pathlib import Path
from hyphae.config import (
    Settings,
    load_mcp_config_from_settings,
    load_models_config_from_settings,
    load_orchestrator_prompt,
    load_agent_prompt,
)
root = Path({str(_REPO_ROOT)!r})
settings = Settings(_env_file=root / '.env.example')
assert settings.mcp_config_path == root / 'hyphae/config/mcp_config.yaml'
assert settings.models_config_path == root / 'hyphae/config/models.yaml'
assert settings.orchestrator_prompt_path == root / 'hyphae/config/orchestrator_prompt.md'
assert settings.agent_prompt_path == root / 'hyphae/config/agent_prompt.md'
assert settings.trace_jsonl_path == root / 'traces/harness.jsonl'
assert load_mcp_config_from_settings(settings).mcp_servers
assert load_models_config_from_settings(
    settings,
    known_providers={{'gemini', 'openai', 'openai_compatible'}},
).default_id()
assert load_orchestrator_prompt(settings.orchestrator_prompt_path)
assert load_agent_prompt(settings.agent_prompt_path)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={"PYTHONPATH": str(_REPO_ROOT)},
        check=True,
    )


def test_remaining_numeric_boundaries() -> None:
    assert LLMSettings(max_tokens=1, timeout_seconds=-1).timeout_seconds == -1
    with pytest.raises(ValidationError):
        LLMSettings(max_tokens=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, context_safety_margin_tokens=-1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field",
    [
        "tool_timeout_seconds",
        "mcp_connect_timeout_seconds",
        "mcp_catalog_ttl_seconds",
        "run_max_seconds",
    ],
)
def test_settings_float_controls_reject_nonfinite_values(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize("field", ["timeout_seconds", "retry_base_delay"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_llm_float_controls_reject_nonfinite_values(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        LLMSettings(**{field: value})


@pytest.mark.parametrize("value", [True, False, 1.0])
def test_integer_configuration_rejects_boolean_and_float_coercion(value: Any) -> None:
    with pytest.raises(ValidationError):
        LLMSettings(max_tokens=value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, loop_max_iterations=value)
    with pytest.raises(ValidationError):
        ModelEntry(
            provider="gemini",
            model="model",
            description="demo",
            context_window=value,
        )


def test_validation_summaries_use_safe_actionable_codes(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        """\
models:
  one:
    provider: gemini
    model: one
    description: first
  two:
    provider: gemini
    model: two
    description: second
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="missing_default"):
        load_models_config(path)


def test_validation_locations_do_not_render_unknown_mapping_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mcp.yaml"
    path.write_text(
        """\
mcpServers:
  demo:
    transport: stdio
    command: demo
    AuthorizationBearerSecret: forbidden
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError) as error:
        load_mcp_config(path)

    assert "AuthorizationBearerSecret" not in str(error.value)


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"context_strategy": "broken"}, "context_strategy"),
        ({"max_loop_iterations": 0}, "max_loop_iterations"),
        ({"max_run_seconds": float("nan")}, "max_run_seconds"),
        ({"session_max_count": True}, "session_max_count"),
    ],
)
def test_safe_validation_summaries_retain_static_alias_names(
    payload: dict[str, Any],
    field: str,
) -> None:
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, **payload)

    assert safe_validation_summary(error.value).startswith(f"{field}:")


def test_safe_validation_location_allowlist_matches_schema_fields_and_aliases() -> None:
    model_types = (
        LLMSettings,
        Settings,
        MCPConfig,
        StreamableHTTPServer,
        SSEServer,
        StdioServer,
        ToolPolicyConfig,
        SamplingParams,
        ModelEntry,
        ModelsConfig,
    )
    expected = {"<root>"}
    for model_type in model_types:
        for name, field in model_type.model_fields.items():
            expected.add(name)
            if isinstance(field.alias, str):
                expected.add(field.alias)
            validation_alias = field.validation_alias
            if isinstance(validation_alias, str):
                expected.add(validation_alias)
            elif isinstance(validation_alias, AliasChoices):
                expected.update(
                    choice
                    for choice in validation_alias.choices
                    if isinstance(choice, str)
                )

    assert _SAFE_LOCATION_PARTS == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"mcpServers": {" ": {"transport": "stdio", "command": "demo"}}},
        {"mcpServers": {"x" * 129: {"transport": "stdio", "command": "demo"}}},
        {
            "mcpServers": {
                "demo": {
                    "transport": "stdio",
                    "command": "demo",
                    "disabled_tools": [" "],
                }
            }
        },
        {
            "mcpServers": {
                "demo": {
                    "transport": "stdio",
                    "command": "demo",
                    "disabled_tools": ["x" * 129],
                }
            }
        },
    ],
)
def test_server_and_configured_tool_identifiers_are_bounded(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        MCPConfig.model_validate(payload)


def test_models_snapshot_is_deeply_immutable() -> None:
    config = ModelsConfig(
        models={
            "only": ModelEntry(
                provider="gemini",
                model="model",
                description="demo",
            )
        }
    )
    assert isinstance(config.models, MappingProxyType)
    models: Any = config.models
    with pytest.raises(TypeError):
        models["other"] = config.models["only"]
    with pytest.raises(ValidationError):
        config.models["only"].description = "changed"
