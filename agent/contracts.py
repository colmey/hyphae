"""Narrow runtime capabilities consumed by the agent layer."""

from __future__ import annotations

from typing import Any, Protocol

from mcp_layer import ToolCallResult


class ToolRuntime(Protocol):
    """Tool inventory and dispatch operations required by the agent loop."""

    def get_tools_for_llm(self) -> list[dict[str, Any]]: ...

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolCallResult: ...
