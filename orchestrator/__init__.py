"""Orchestration layer: routes incoming requests to a model + tool subset + system prompt.

Public API:
  - Orchestrator: the LLM-backed router (call .decide() per request).
  - LLMRegistry: lazy cache of LLMClient instances keyed by model_id.
  - OrchestrationDecision: per-call return value from Orchestrator.decide()
                           (wraps result + fallback_used + fallback_reason).
  - OrchestrationResult: the structured-output schema the LLM produces.

Config types and loaders (ModelEntry, ModelsConfig, load_models_config,
load_orchestrator_prompt) live in the config package.
"""

from .orchestrator import Orchestrator
from .registry import LLMRegistry
from .schemas import OrchestrationDecision, OrchestrationResult, ToolPreferences

__all__ = [
    "LLMRegistry",
    "Orchestrator",
    "OrchestrationDecision",
    "OrchestrationResult",
    "ToolPreferences",
]
