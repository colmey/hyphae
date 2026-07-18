"""MCP integration layer for the harness."""

from .client import MCPClient, Tool, ToolCallResult
from .contracts import NAMESPACE_SEP, ToolSnapshot, ToolSpec
from .manager import MCPManager, MCPServerState, MCPServerStatus

__all__ = [
    "MCPClient",
    "MCPManager",
    "MCPServerState",
    "MCPServerStatus",
    "NAMESPACE_SEP",
    "Tool",
    "ToolCallResult",
    "ToolSnapshot",
    "ToolSpec",
]
