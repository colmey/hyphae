# orchestrator/registry.py

"""
LLMRegistry: structurally-ready store of LLMClient instances keyed by model_id.

Responsibilities:
  - Map a model_id (from models.yaml) onto a constructed LLMClient.
  - Cache constructed clients so we don't rebuild a client for every request.
  - Preflight local construction of configured clients at application startup.
  - Expose the inventory of models for the orchestrator prompt.

The registry is intentionally narrow. It does NOT decide which model to
use (that's the Orchestrator's job) and does NOT know about routes or
sessions. One responsibility: model_id -> LLMClient.

Concurrency: clients are built on first use and stashed in a dict. We
don't lock around the dict because client construction is idempotent --
a rare double-build wastes a few cycles but cannot produce wrong behavior.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from itertools import chain

from llm.client import (
    LLMClient,
    _DefaultClientSettings,
    build_llm_client_from_entry,
)

from config import ModelEntry, ModelsConfig

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """A configured model is unavailable to the current routing runtime."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"configured model is unavailable: {model_id}")


class LLMRegistry:
    """Holds configured catalog data and structurally-ready client identities."""

    def __init__(
        self, models_config: ModelsConfig, settings: _DefaultClientSettings
    ) -> None:
        """
        Args:
          models_config: the parsed models.yaml.
          settings: the runtime Settings (duck-typed); used by the per-entry
                    factory to pick up API keys and default max_tokens.
        """
        self._config = models_config
        self._settings = settings
        self._clients: dict[str, LLMClient] = {}
        self._unavailable: dict[str, str] = {}

    # ----- inventory -----

    @property
    def model_ids(self) -> list[str]:
        """The ready-only model inventory exposed to runtime consumers."""
        return [model_id for model_id in self._config.models if model_id in self._clients]

    def is_configured(self, model_id: str) -> bool:
        """Whether an ID is present in models.yaml, regardless of readiness."""
        return model_id in self._config.models

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
        default_id = self.default_id()
        for mid in self.model_ids:
            entry = self._config.models[mid]
            default_marker = " (default)" if mid == default_id else ""
            desc = " ".join(entry.description.split())  # collapse whitespace
            lines.append(f"- {mid}{default_marker}\n    {desc}")
        return "\n".join(lines)

    # ----- client construction -----

    def get(self, model_id: str) -> LLMClient:
        """Return the ready LLMClient for ``model_id``.

        Unknown IDs retain ``KeyError`` behavior. Configured IDs that failed
        local construction raise the typed unavailable outcome.
        """
        if model_id in self._clients:
            return self._clients[model_id]

        if model_id in self._unavailable:
            raise ModelUnavailableError(model_id)

        entry = self.get_entry(model_id)
        client = build_llm_client_from_entry(entry, self._settings)
        self._clients[model_id] = client
        logger.info(
            "built LLM client for model_id=%s (provider=%s model=%s)",
            model_id,
            entry.provider,
            entry.model,
        )
        return client

    def preflight(self) -> None:
        """Construct every configured client without invoking provider operations."""
        for model_id in self._config.models:
            if model_id in self._clients or model_id in self._unavailable:
                continue
            try:
                self.get(model_id)
            except Exception as exc:  # noqa: BLE001 -- readiness is per model.
                self._unavailable[model_id] = type(exc).__name__
                logger.warning(
                    "configured model is structurally unavailable: model_id=%s cause=%s",
                    model_id,
                    type(exc).__name__,
                )

    def get_or_default(self, model_id: str | None) -> tuple[str, LLMClient]:
        """Return (resolved_id, client). Falls back to default on unknown id.

        Used by the route when the orchestrator output references a model
        that doesn't exist -- we log and degrade rather than 500-ing.
        """
        if model_id and model_id in self._clients:
            return model_id, self.get(model_id)
        fallback = self.default_id()
        if model_id and model_id != fallback:
            logger.warning(
                "model_id %r is not structurally available; falling back to default %r",
                model_id,
                fallback,
            )
        return fallback, self.get(fallback)

    async def aclose(self, *, additional_clients: Iterable[LLMClient] = ()) -> None:
        """Detach the cache and close each cached/additional identity once.

        The application lifespan supplies its default client through
        ``additional_clients`` so aliases across both ownership paths are
        deduplicated in the same snapshot.
        """
        cached = tuple(self._clients.values())
        self._clients.clear()

        clients: list[LLMClient] = []
        seen: set[int] = set()
        for client in chain(additional_clients, cached):
            identity = id(client)
            if identity in seen:
                continue
            seen.add(identity)
            clients.append(client)

        if not clients:
            return
        await asyncio.gather(
            *(_close_client(client) for client in clients),
        )


async def _close_client(client: LLMClient) -> None:
    """Best-effort close for one registry-owned client."""
    try:
        await client.aclose()
    except asyncio.CancelledError as exc:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise
        logger.warning(
            "LLM client cleanup was cancelled for %s (%s)",
            type(client).__name__,
            type(exc).__name__,
        )
    except Exception as exc:  # noqa: BLE001 -- continue closing sibling clients.
        logger.warning(
            "failed to close LLM client %s (%s)",
            type(client).__name__,
            type(exc).__name__,
        )
