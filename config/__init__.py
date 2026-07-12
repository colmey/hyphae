# config/__init__.py

"""The harness configuration layer: one seam for env, settings, and file config.

Import everything from this package root (`from config import ...`), never
from submodules -- tests patch `config.load_secrets` at this boundary.
"""

from .env import load_secrets
from .loaders import load_mcp_config, load_models_config, load_orchestrator_prompt
from .schemas import (
    MCPConfig,
    MCPServerConfig,
    ModelEntry,
    ModelsConfig,
    SamplingParams,
    SSEServer,
    StdioServer,
    StreamableHTTPServer,
    ToolPolicyConfig,
)
from .settings import Settings, get_settings, reset_settings

__all__ = [
    "MCPConfig",
    "MCPServerConfig",
    "ModelEntry",
    "ModelsConfig",
    "SSEServer",
    "SamplingParams",
    "Settings",
    "StdioServer",
    "StreamableHTTPServer",
    "ToolPolicyConfig",
    "get_settings",
    "load_mcp_config",
    "load_models_config",
    "load_orchestrator_prompt",
    "load_secrets",
    "reset_settings",
]
