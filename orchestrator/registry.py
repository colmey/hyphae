# orchestrator/registry.py

"""
LLMRegistry: lazy-loading store of LLMClient instances keyed by model_id.

Responsibilities:
  - Map a model_id (from models.yaml) onto a constructed LLMClient.
  - Cache constructed clients so we don't rebuild a client for every request.
  - Expose the inventory of models for the orchestrator prompt.

The registry is intentionally narrow. It does NOT decide which model to
use (that's the Orchestrator's job) and does NOT know about routes or
sessions. One responsibility: model_id -> LLMClient.

Concurrency: clients are built on first use and stashed in a dict. We
don't lock around the dict because client construction is idempotent --
a rare double-build wastes a few cycles but cannot produce wrong behavior.
"""

from __future__ import annotations

import logging
from typing import Any

from llm.client import LLMClient, build_llm_client_from_entry

from .schemas import ModelEntry, ModelsConfig

logger = logging.getLogger(__name__)


class LLMRegistry:
    """Holds the model inventory and lazily builds LLMClient instances."""

    def __init__(self, models_config: ModelsConfig, settings: Any) -> None:
        """
        Args:
          models_config: the parsed models.yaml.
          settings: the runtime Settings (duck-typed); used by the per-entry
                    factory to pick up API keys and default max_tokens.
        """
        self._config = models_config
        self._settings = settings
        self._clients: dict[str, LLMClient] = {}

    # ----- inventory -----

    @property
    def model_ids(self) -> list[str]:
        return list(self._config.models.keys())

    def get_entry(self, model_id: str) -> ModelEntry:
        """Return the ModelEntry for `model_id` or raise KeyError."""
        try:
            return self._config.models[model_id]
        except KeyError as e:
            raise KeyError(
                f"unknown model_id {model_id!r}; known: {self.model_ids!r}"
            ) from e

    def default_id(self) -> str:
        """The model_id marked default in models.yaml."""
        return self._config.default_id()

    def describe_for_prompt(self) -> str:
        """Format the model inventory as a block for the orchestrator prompt.

        Plain text, one entry per stanza. The orchestrator LLM consumes this
        verbatim, so prefer clarity over compactness.
        """
        lines: list[str] = []
        for mid, entry in self._config.models.items():
            default_marker = " (default)" if entry.default else ""
            desc = " ".join(entry.description.split())  # collapse whitespace
            lines.append(f"- {mid}{default_marker}\n    {desc}")
        return "\n".join(lines)

    # ----- client construction -----

    def get(self, model_id: str) -> LLMClient:
        """Return the LLMClient for `model_id`, building it on first call.

        Raises KeyError if the model_id is not registered.
        """
        if model_id in self._clients:
            return self._clients[model_id]

        entry = self.get_entry(model_id)
        client = build_llm_client_from_entry(entry, self._settings)
        self._clients[model_id] = client
        logger.info("built LLM client for model_id=%s (provider=%s model=%s)",
                    model_id, entry.provider, entry.model)
        return client

    def get_or_default(self, model_id: str | None) -> tuple[str, LLMClient]:
        """Return (resolved_id, client). Falls back to default on unknown id.

        Used by the route when the orchestrator output references a model
        that doesn't exist -- we log and degrade rather than 500-ing.
        """
        if model_id and model_id in self._config.models:
            return model_id, self.get(model_id)
        fallback = self.default_id()
        if model_id and model_id != fallback:
            logger.warning(
                "model_id %r not in registry; falling back to default %r",
                model_id, fallback,
            )
        return fallback, self.get(fallback)