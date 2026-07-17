"""Live pytest coverage for the configured LLM provider.

Exercises the configured provider (currently Gemini) end-to-end through
build_llm_client(settings) -> the provider registry, in three scenarios:

  1. Tool-less completion — verifies auth, model name, and basic round-trip.
  2. Completion with MCP tool list attached — verifies tool schema
     translation and that the model can produce a function call.
  3. Manual tool-result round-trip — simulates what the agent loop will do:
     send tools, receive tool_use, execute the tool via MCPManager, send the
     result back, get a final answer.

Run explicitly with ``./runscript.sh -m pytest -m "live and model and mcp"``.
"""

from __future__ import annotations

import config
import logging

import pytest

from config import get_settings, load_mcp_config, reset_settings
from llm import (
    GenerationRequest,
    LLMClient,
    Message,
    ToolResultBlock,
    ToolUseBlock,
    build_llm_client,
)
from llm.schemas import TextBlock
from mcp_layer import MCPManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
pytestmark = [pytest.mark.live, pytest.mark.model, pytest.mark.mcp, pytest.mark.anyio]


def _print_response_summary(label: str, msg) -> None:
    print(f"--- {label} ---")
    print(f"  stop_reason: {msg.stop_reason}")
    print(f"  model:       {msg.model}")
    print(f"  blocks:      {len(msg.content)}")
    for i, block in enumerate(msg.content):
        if isinstance(block, TextBlock):
            preview = block.text.strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:197] + "..."
            print(f"    [{i}] text: {preview}")
        elif isinstance(block, ToolUseBlock):
            print(f"    [{i}] tool_use: {block.name} id={block.id}")
            print(f"          input: {block.input}")
    print()


async def scenario_1_no_tools(llm: LLMClient) -> None:
    print("=" * 70)
    print("Scenario 1: tool-less completion")
    print("=" * 70)
    response = await llm.complete(
        GenerationRequest(
            messages=[Message.user("What is 2 + 2? Answer with just the number.")]
        )
    )
    _print_response_summary("response", response)


async def scenario_2_with_tools(llm: LLMClient, mcp: MCPManager) -> None:
    print("=" * 70)
    print("Scenario 2: completion with MCP tools attached")
    print("=" * 70)
    tools = mcp.get_tools_for_llm()
    print(f"  attached {len(tools)} tools from MCP manager")

    # Pick a prompt that should clearly steer the model toward a tool call.
    # We don't know exactly which tools your servers expose, but listing
    # database tables is a common, low-risk capability for a database toolbox.
    response = await llm.complete(
        GenerationRequest(
            messages=[
                Message.user(
                    "List the tables available in the customer database. "
                    "Use the appropriate tool."
                )
            ],
            tools=tools,
            system=(
                "You are a database assistant with access to tools. "
                "When the user asks about data, use the available tools to fetch it."
            ),
        )
    )
    _print_response_summary("response", response)
    return response


async def scenario_3_full_roundtrip(llm: LLMClient, mcp: MCPManager) -> None:
    print("=" * 70)
    print("Scenario 3: manual tool-result round-trip")
    print("=" * 70)
    tools = mcp.get_tools_for_llm()

    history: list[Message] = [
        Message.user(
            "List the tables available in the customer database. "
            "Use the appropriate tool."
        ),
    ]
    system = (
        "You are a database assistant with access to tools. "
        "Use them when needed, then summarize results clearly."
    )

    # Turn 1: model decides to call a tool (hopefully).
    first = await llm.complete(
        GenerationRequest(messages=history, tools=tools, system=system)
    )
    _print_response_summary("turn 1", first)
    history.append(first.to_message())

    tool_uses = first.tool_uses()
    if not tool_uses:
        print("  model did not call a tool; skipping round-trip.")
        return

    # Execute each tool via the MCP manager and build tool_result blocks.
    results: list[ToolResultBlock] = []
    for tu in tool_uses:
        print(f"  executing tool: {tu.name} args={tu.input}")
        result = await mcp.call_tool(tu.name, tu.input)
        # Truncate noisy output for the smoke test.
        preview = (
            result.content
            if len(result.content) < 400
            else result.content[:397] + "..."
        )
        print(f"    result (is_error={result.is_error}): {preview}")
        results.append(
            ToolResultBlock(
                tool_use_id=tu.id,
                name=tu.name,
                content=result.content,
                is_error=result.is_error,
            )
        )
    history.append(Message.tool_results(results))

    # Turn 2: model produces a final answer using the tool output.
    second = await llm.complete(
        GenerationRequest(messages=history, tools=tools, system=system)
    )
    _print_response_summary("turn 2 (final)", second)


async def test_configured_llm_scenarios() -> None:
    config.load_secrets()
    reset_settings()
    settings = get_settings()
    print(f"using provider={settings.llm_provider} model={settings.llm_model}")
    print()

    llm = build_llm_client(settings)
    mcp_config = load_mcp_config(settings.mcp_config_path)
    mcp = MCPManager(mcp_config)
    await mcp.startup()

    try:
        await scenario_1_no_tools(llm)
        await scenario_2_with_tools(llm, mcp)
        await scenario_3_full_roundtrip(llm, mcp)
    finally:
        await mcp.shutdown()
        print("shutdown complete")
