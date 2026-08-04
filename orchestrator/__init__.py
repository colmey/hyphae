"""Orchestration layer: routes incoming requests to a model + tool subset.

Public API:
  - Orchestrator: the LLM-backed router (call .decide() per request).
  - LLMRegistry: lazy cache of LLMClient instances keyed by model_id.
  - OrchestrationDecision: per-call return value from Orchestrator.decide()
                           (wraps result + fallback_used + fallback_reason).
  - OrchestrationProposal: the structured-output schema the LLM produces.

Config types and loaders (ModelEntry, ModelsConfig, load_models_config,
load_orchestrator_prompt, load_agent_prompt) live in the config package.
"""

from .orchestrator import Orchestrator
from .registry import LLMRegistry
from .schemas import OrchestrationDecision, OrchestrationProposal, ToolPreferences

__all__ = [
    "LLMRegistry",
    "Orchestrator",
    "OrchestrationDecision",
    "OrchestrationProposal",
    "ToolPreferences",
]
