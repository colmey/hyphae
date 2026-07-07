"""
Smoke test for step 1 of the build.

Demonstrates the in-process bootstrap pattern:
  1. Bootstrap runs first: load_secrets() loads the .env file into os.environ.
  2. Then harness code calls get_settings() / load_mcp_config().

Run from the project root:
    ./runscript.sh tests/smoke_test_config.py
"""

# ---------------------------------------------------------------------------
# 1. Bootstrap. load_secrets() loads the project's .env file into os.environ.
# ---------------------------------------------------------------------------
from bootstrap import load_secrets
load_secrets()


# ---------------------------------------------------------------------------
# 2. Now import and use the harness config. These imports happen AFTER
#    os.environ has been populated, just as in main.py.
# ---------------------------------------------------------------------------
from pydantic import ValidationError

from harness_config import MCPConfig, ToolPolicyConfig, get_settings, load_mcp_config


def _check_fail_loud_on_typos() -> None:
    """A config typo must raise at load, not silently disable the tool policy."""
    print("=" * 60)
    print("Config hardening: unknown fields fail loud")
    print("=" * 60)

    # Typo in a ToolPolicyConfig field (e.g. `mdoe`) must not drop to allow_all.
    try:
        ToolPolicyConfig.model_validate({"mdoe": "allow_list", "allow": ["x__*"]})
        raise SystemExit("  FAILED: ToolPolicyConfig accepted an unknown field ('mdoe')")
    except ValidationError:
        print("  [PASS] ToolPolicyConfig rejects an unknown field ('mdoe')")

    # Typo in the top-level block name (e.g. `tool_polciy`) must not be dropped.
    try:
        MCPConfig.model_validate({"mcpServers": {}, "tool_polciy": {"mode": "allow_all"}})
        raise SystemExit("  FAILED: MCPConfig accepted an unknown top-level key ('tool_polciy')")
    except ValidationError:
        print("  [PASS] MCPConfig rejects an unknown top-level key ('tool_polciy')")

    # Typo in a server field (e.g. `disabled_tool`) must not silently re-enable it.
    try:
        MCPConfig.model_validate({
            "mcpServers": {
                "demo": {
                    "transport": "stdio",
                    "command": "demo",
                    "disabled_tool": ["dangerous"],
                }
            }
        })
        raise SystemExit("  FAILED: MCPConfig accepted an unknown server key ('disabled_tool')")
    except ValidationError:
        print("  [PASS] MCPConfig rejects an unknown server key ('disabled_tool')")
    print()


def main() -> None:
    settings = get_settings()
    print("=" * 60)
    print("Settings loaded")
    print("=" * 60)
    print(f"  provider:            {settings.llm_provider}")
    print(f"  model:               {settings.llm_model}")
    print(f"  max_tokens:          {settings.llm_max_tokens}")
    print(f"  mcp_config_path:     {settings.mcp_config_path}")
    print(f"  max_loop_iterations: {settings.max_loop_iterations}")
    print(f"  anthropic_api_key:   {'<set>' if settings.anthropic_api_key else '<not set>'}")
    print(f"  gemini_api_key:      {'<set>' if settings.gemini_api_key else '<not set>'}")
    print(f"  openai_api_key:      {'<set>' if settings.openai_api_key else '<not set>'}")

    # Confirm the configured provider actually has a key.
    try:
        _ = settings.required_api_key()
        print(f"  required key check:  ok ({settings.llm_provider})")
    except RuntimeError as e:
        print(f"  required key check:  FAILED -- {e}")

    print()

    mcp = load_mcp_config(settings.mcp_config_path)
    print("=" * 60)
    print(
        f"MCP config loaded ({len(mcp.mcp_servers)} servers, "
        f"{len(mcp.enabled_servers())} enabled)"
    )
    print("=" * 60)
    for name, server in mcp.mcp_servers.items():
        status = "disabled" if server.disabled else "enabled"
        line = f"  [{status:8}] {name:20} transport={server.transport}"
        if server.transport in ("streamable-http", "sse"):
            line += f"  url={server.url}"
        elif server.transport == "stdio":
            line += f"  command={server.command} {' '.join(server.args)}"
        print(line)
        if server.disabled_tools:
            print(f"               disabled_tools: {server.disabled_tools}")

    print()
    _check_fail_loud_on_typos()


if __name__ == "__main__":
    main()
