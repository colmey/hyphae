"""MCP integration layer for the harness."""
from .client import MCPClient, Tool, ToolCallResult
from .contracts import NAMESPACE_SEP, ToolSnapshot, ToolSpec
from .manager import MCPManager

__all__ = [
    "MCPClient",
    "MCPManager",
    "NAMESPACE_SEP",
    "Tool",
    "ToolCallResult",
    "ToolSnapshot",
    "ToolSpec",
]
