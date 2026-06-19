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
from harness_config import get_settings, load_mcp_config


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


if __name__ == "__main__":
    main()