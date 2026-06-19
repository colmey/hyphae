"""
Smoke test for step 5: the agent loop.

This is the first true end-to-end test. Real LLM, real MCP, real session.
The only thing missing is the FastAPI surface (step 6) and HTTP request
parsing.

Three scenarios:
  1. Trivial prompt with no tool need ("what's 2+2?") -> single TextEvent
     then DoneEvent(end_turn) in one iteration.
  2. A prompt that requires one tool call -> ToolCall + ToolResult + Text
     + Done across two iterations.
  3. A multi-step prompt that should chain tool calls ("list tables, then
     describe one of them") -> several iterations.

We run each scenario with both the streaming-style consumer (iterating the
generator and reacting to events) and verify the final session state.

Run from the project root:
    ./runscript.sh smoke_test_agent.py
"""

from __future__ import annotations

from bootstrap import load_secrets
load_secrets()

import asyncio
import logging
from typing import Any

from agent import (
    DoneEvent,
    ErrorEvent,
    InMemorySessionStore,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    run_agent,
)
from harness_config import get_settings, load_mcp_config
from llm import build_llm_client
from mcp_layer import MCPManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)


SYSTEM_PROMPT = (
    "You are a helpful database assistant with access to tools. "
    "When the user asks about data, use the available tools to look it up. "
    "Be concise in your final answer."
)


async def run_scenario(
    label: str,
    user_message: str,
    llm: Any,
    mcp: MCPManager,
    store: InMemorySessionStore,
    max_iterations: int = 10,
) -> None:
    print("=" * 72)
    print(f"Scenario: {label}")
    print("=" * 72)
    print(f"prompt: {user_message}")
    print()

    session = await store.create()
    session.append_user(user_message)

    # Counters and last-seen values, for the summary at the end.
    text_chunks: list[str] = []
    tool_calls = 0
    tool_results = 0
    errors = 0
    done_reason: str | None = None
    done_iterations: int = 0

    async for event in run_agent(
        session=session,
        llm=llm,
        mcp=mcp,
        store=store,
        system=SYSTEM_PROMPT,
        max_iterations=max_iterations,
    ):
        if isinstance(event, TextEvent):
            text_chunks.append(event.text)
            preview = event.text.strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"  [text]         {preview}")
        elif isinstance(event, ToolCallEvent):
            tool_calls += 1
            print(f"  [tool_call]    {event.name} id={event.id}")
            print(f"                 args={event.input}")
        elif isinstance(event, ToolResultEvent):
            tool_results += 1
            preview = event.content.replace("\n", " ")
            if len(preview) > 100:
                preview = preview[:97] + "..."
            marker = "ERR" if event.is_error else "ok "
            print(f"  [tool_result]  [{marker}] id={event.id} -> {preview}")
        elif isinstance(event, ErrorEvent):
            errors += 1
            print(f"  [error]        {event.message}")
        elif isinstance(event, DoneEvent):
            done_reason = event.reason
            done_iterations = event.iterations
            print(f"  [done]         reason={event.reason} iterations={event.iterations}")

    print()
    print(f"  summary: {tool_calls} tool call(s), {tool_results} result(s), "
          f"{errors} error(s); finished in {done_iterations} iter(s) "
          f"with reason={done_reason}")
    print(f"  session {session.session_id}: {len(session.messages)} messages total")
    print()


async def main() -> None:
    settings = get_settings()
    print(f"using provider={settings.llm_provider} model={settings.llm_model}")
    print()

    llm = build_llm_client(settings)
    mcp_config = load_mcp_config(settings.mcp_config_path)
    mcp = MCPManager(mcp_config)
    store = InMemorySessionStore()

    await mcp.startup()
    try:
        # 1. No tools needed.
        await run_scenario(
            label="trivial prompt, no tool calls expected",
            user_message="What is 2 + 2? Just give the number.",
            llm=llm, mcp=mcp, store=store,
        )

        # 2. Exactly one tool call expected.
        await run_scenario(
            label="single tool call",
            user_message="List the tables in the customer database.",
            llm=llm, mcp=mcp, store=store,
        )

        # 3. Multi-step: list tables, then describe one. Should be at least
        #    two tool calls across two iterations.
        await run_scenario(
            label="multi-step tool chaining",
            user_message=(
                "First, list the tables in the customer database. "
                "Then pick one that looks like it stores customer data "
                "and describe its columns."
            ),
            llm=llm, mcp=mcp, store=store,
            max_iterations=8,
        )

        print(f"store holds {len(store)} session(s); step 5 smoke test complete.")
    finally:
        await mcp.shutdown()


if __name__ == "__main__":
    asyncio.run(main())