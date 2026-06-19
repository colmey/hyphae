"""MCP integration layer for the harness."""
from .client import MCPClient, Tool, ToolCallResult
from .manager import MCPManager

__all__ = ["MCPClient", "MCPManager", "Tool", "ToolCallResult"]