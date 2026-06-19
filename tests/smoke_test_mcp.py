"""
Smoke test for step 2 of the build.

Connects to all enabled MCP servers, lists their tools, optionally calls one
tool, then shuts down cleanly. No LLM, no HTTP server.

Run from the project root (with your bootstrap script having populated env):
    ./runscript.sh smoke_test_mcp.py
or
    python smoke_test_mcp.py
"""

from __future__ import annotations

import asyncio
import logging

from harness_config import get_settings, load_mcp_config
from mcp_layer import MCPManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)


async def main() -> None:
    settings = get_settings()
    mcp_config = load_mcp_config(settings.mcp_config_path)

    manager = MCPManager(mcp_config)
    await manager.startup()

    try:
        print()
        print("=" * 70)
        print(f"Connected servers: {manager.connected_servers}")
        print("=" * 70)

        tools = manager.list_tools()
        print(f"Total tools available: {len(tools)}")
        print()

        # Group by server for readability.
        by_server: dict[str, list[tuple[str, str]]] = {}
        for namespaced, tool in tools:
            server = namespaced.split("__", 1)[0]
            by_server.setdefault(server, []).append((namespaced, tool.description))

        for server, entries in by_server.items():
            print(f"--- {server} ({len(entries)} tools) ---")
            for namespaced, desc in entries:
                first_line = desc.strip().splitlines()[0] if desc.strip() else "(no description)"
                # Truncate long descriptions for readable output.
                if len(first_line) > 80:
                    first_line = first_line[:77] + "..."
                print(f"  {namespaced}")
                print(f"      {first_line}")
            print()

        # Show what gets sent to the LLM. Just the first one to keep output short.
        llm_tools = manager.get_tools_for_llm()
        if llm_tools:
            print("=" * 70)
            print("Sample tool schema as it would be sent to the LLM:")
            print("=" * 70)
            import json
            print(json.dumps(llm_tools[0], indent=2)[:800])
            print()

    finally:
        await manager.shutdown()
        print("shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())