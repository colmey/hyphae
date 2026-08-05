"""Orchestration layer: routes incoming requests to a model + tool subset.

Public API:
  - Orchestrator: the LLM-backed router (call .decide() per request).
  - LLMRegistry: configured catalog plus structurally-ready LLMClient identities.
  - OrchestrationDecision: per-call return value from Orchestrator.decide()
                           (wraps selection, safe fallback facts, and telemetry).
  - OrchestrationProposal: the structured-output schema the LLM produces.

Config types and loaders (ModelEntry, ModelsConfig, load_models_config,
load_orchestrator_prompt, load_agent_prompt) live in the config package.
"""

from .orchestrator import Orchestrator
from .registry import LLMRegistry, ModelUnavailableError
from .schemas import OrchestrationDecision, OrchestrationProposal, ToolPreferences

__all__ = [
    "LLMRegistry",
    "ModelUnavailableError",
    "Orchestrator",
    "OrchestrationDecision",
    "OrchestrationProposal",
    "ToolPreferences",
]
