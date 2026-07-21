"""Live pytest coverage for an OpenAI-compatible provider (real or local
server reached through its OpenAI-compatible /v1 endpoint).

Exercises OpenAILLMClient end-to-end in three scenarios, mirroring
smoke_test_llm.py:

  1. Tool-less completion — verifies auth, base_url, model name, round-trip.
  2. Completion with the MCP tool list attached — verifies tool schema
     translation and that the model can produce a function call (needs a
     tools-capable model).
  3. Manual tool-result round-trip — simulates the agent loop: send tools,
     receive tool_use, execute via MCPManager, send the result back, get a
     final answer.

Configure via the process environment or project ``.env`` before running:
    export OPENAI_BASE_URL=http://localhost:11434/v1   # for local Ollama
    export OPENAI_API_KEY=<key>                         # any non-empty value
    export OPENAI_MODEL=qwen3.6-35b-a3b                 # an `ollama list` tag

Run explicitly with ``./runscript.sh -m pytest -m "live and model and mcp"``.
"""

from __future__ import annotations

import logging

import pytest

from config import get_settings, load_mcp_config_from_settings, reset_settings
from llm import (
    GenerationRequest,
    LLMClient,
    Message,
    ToolResultBlock,
    ToolUseBlock,
)
from llm.schemas import TextBlock
from llm.providers.openai import OpenAILLMClient
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

    first = await llm.complete(
        GenerationRequest(messages=history, tools=tools, system=system)
    )
    _print_response_summary("turn 1", first)
    history.append(first.to_message())

    tool_uses = first.tool_uses()
    if not tool_uses:
        print("  model did not call a tool; skipping round-trip.")
        return

    results: list[ToolResultBlock] = []
    for tu in tool_uses:
        print(f"  executing tool: {tu.name} args={tu.input}")
        result = await mcp.call_tool(tu.name, tu.input)
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

    second = await llm.complete(
        GenerationRequest(messages=history, tools=tools, system=system)
    )
    _print_response_summary("turn 2 (final)", second)


async def test_configured_openai_provider_scenarios() -> None:
    reset_settings()
    settings = get_settings()
    environment = settings.interpolation_environment()

    # Construct the OpenAI-compatible client directly from env so this test
    # works regardless of settings.llm_provider. OPENAI_MODEL picks the model
    # (default is a placeholder; set it to a real `ollama list` tag).
    base_url = settings.openai_base_url or environment.get("OPENAI_BASE_URL", "")
    api_key = settings.openai_api_key or environment.get("OPENAI_API_KEY", "")
    model = environment.get("OPENAI_MODEL", "qwen3.6-35b-a3b")

    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Set it (any non-empty value for Ollama) "
            "and OPENAI_BASE_URL before running this test."
        )

    print(
        f"using openai-compatible base_url={base_url or '<real OpenAI>'} model={model}"
    )
    print()

    llm = OpenAILLMClient(
        api_key=api_key,
        model=model,
        default_max_tokens=settings.llm_max_tokens,
        base_url=base_url or None,
    )
    mcp_config = load_mcp_config_from_settings(settings)
    mcp = MCPManager(mcp_config)
    await mcp.startup()

    try:
        await scenario_1_no_tools(llm)
        await scenario_2_with_tools(llm, mcp)
        await scenario_3_full_roundtrip(llm, mcp)
    finally:
        await mcp.shutdown()
        print("shutdown complete")
