# llm/client.py
"""
LLM client: the provider-agnostic abstraction and the provider registry.

The abstract `LLMClient` exposes a single async method (`complete`) that takes
provider-agnostic messages and tools and returns a provider-agnostic
`AssistantMessage`. Each provider implementation (in `llm/providers/`) is
responsible for two translations:

  1. Internal message/tool types -> provider SDK request shape.
  2. Provider SDK response -> internal AssistantMessage.

The agent loop, session store, and MCP manager never see provider types.
This is the "one bridge between worlds" principle in the architecture spec.

Provider registry:
  `_PROVIDERS` maps a provider name onto a builder function. Builders import
  their provider module *lazily* (inside the function body), so importing this
  module -- e.g. just to get the `LLMClient` ABC -- never pulls in a provider
  SDK. This is the single source of truth for which providers exist; config
  validators key off `supported_providers()`. Adding a provider is two steps:
  drop a file in `llm/providers/`, then add one entry here.

Structured output (response_schema):
  `complete()` accepts an optional `response_schema` (a Pydantic model class)
  for providers that support structured output. Providers without native
  support may ignore the kwarg; callers should be prepared to parse JSON out of
  the text response either way.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable

from .schemas import AssistantMessage, Message, ModelProfile, StreamChunk, StreamEnd, TextBlock, TextDelta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Provider-agnostic LLM client interface."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        response_schema: type | None = None,
        thinking_level: str | None = None,
    ) -> AssistantMessage:
        """Run one completion turn and return the assistant's response.

        `tools` is the generic shape from MCPManager.get_tools_for_llm():
        [{name, description, input_schema}]. The implementation reshapes
        it for its provider.

        `response_schema`, if provided, is a Pydantic model class that the
        response should conform to. Providers that support structured output
        (Gemini, OpenAI) honor this; others may ignore it. The orchestration
        layer uses this kwarg; the agent loop does not.

        `thinking_level` ("low"|"medium"|"high"), if provided, asks the model
        to deliberate more or less for this call -- the orchestrator's
        per-request "intelligence on demand" knob. None leaves the model's
        default. Providers without a thinking control may ignore it.
        """
        ...

    def is_transient_error(self, exc: BaseException) -> bool:
        """Whether `exc` from complete() is worth retrying.

        Classifying provider errors needs provider knowledge, so this lives on
        the client (the agent loop owns the retry *policy* but asks the client
        whether a given failure is transient). The base implementation only
        recognizes provider-agnostic transient conditions; concrete clients
        override to add SDK-specific cases (e.g. HTTP 429/5xx).
        """
        return isinstance(exc, (TimeoutError, ConnectionError))

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        thinking_level: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream one completion turn.

        Default implementation for complete-only providers: run `complete()`,
        emit any text blocks coarsely, then return the assembled message.
        Structured-output calls intentionally stay on `complete()`.
        """
        msg = await self.complete(
            messages=messages,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
            thinking_level=thinking_level,
        )
        for block in msg.content:
            if isinstance(block, TextBlock) and block.text:
                yield TextDelta(text=block.text)
        yield StreamEnd(message=msg)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
# Each builder lazily imports its provider module so importing this file never
# drags in a provider SDK. Builders take the duck-typed `settings` and pull
# whatever they need themselves (an API key, a base_url, nothing) -- this keeps
# the registry contract stable across very different providers.


def _build_gemini(
    *, model: str, max_tokens: int, settings: Any, profile: Any = None
) -> LLMClient:
    from llm.providers.gemini import GeminiLLMClient

    return GeminiLLMClient(
        api_key=settings.api_key_for_provider("gemini"),
        model=model,
        default_max_tokens=max_tokens,
        profile=profile,
    )


def _build_openai(
    *, model: str, max_tokens: int, settings: Any, profile: Any = None
) -> LLMClient:
    from llm.providers.openai import OpenAILLMClient

    # OpenAI-compatible: base_url empty -> real OpenAI; set it (e.g. Ollama's
    # /v1) to target a local/compatible server. The key is required by the SDK
    # even when the server ignores it.
    return OpenAILLMClient(
        api_key=settings.api_key_for_provider("openai"),
        model=model,
        default_max_tokens=max_tokens,
        base_url=settings.openai_base_url or None,
        profile=profile,
    )


# THE one place a provider is declared. Adding a provider = one file in
# llm/providers/ + one entry here. Nothing else in the harness enumerates
# providers (config validators key off supported_providers()).
_PROVIDERS: dict[str, Callable[..., LLMClient]] = {
    "gemini": _build_gemini,
    # `openai` is OpenAI-compatible: point Settings.openai_base_url at an
    # alternate /v1 endpoint (e.g. local Ollama) to reuse the same client.
    "openai": _build_openai,
    # "anthropic": _build_anthropic,
}


def supported_providers() -> frozenset[str]:
    """Provider names the harness can build. Config validators key off this."""
    return frozenset(_PROVIDERS)


def _build_client(
    provider: str, *, model: str, max_tokens: int, settings: Any, profile: Any = None
) -> LLMClient:
    """Dispatch to the registered builder for `provider`.

    Raises NotImplementedError for an unknown/unregistered provider, preserving
    the harness's previous behavior for unimplemented providers.
    """
    try:
        build = _PROVIDERS[provider]
    except KeyError:
        raise NotImplementedError(f"LLM provider {provider!r} is not implemented yet")
    return build(model=model, max_tokens=max_tokens, settings=settings, profile=profile)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def build_llm_client(settings: Any) -> LLMClient:
    """Construct the LLM client matching settings.llm_provider.

    Takes Settings duck-typed to avoid a circular import with harness_config.

    Preserved for the legacy/default path. The orchestration layer uses
    `build_llm_client_from_entry` instead, which is parameterized by a
    ModelEntry rather than the global Settings.
    """
    return _build_client(
        settings.llm_provider,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        settings=settings,
    )


def profile_from_entry(entry: Any) -> ModelProfile:
    """Build the ModelProfile declared by one models.yaml entry.

    `entry` is duck-typed (config.ModelEntry shape: supports_native_tools,
    thinking, sampling) so this module never imports the config layer.
    """
    s = entry.sampling
    return ModelProfile(
        supports_native_tools=entry.supports_native_tools,
        thinking=entry.thinking,
        temperature=s.temperature if s else None,
        top_p=s.top_p if s else None,
        top_k=s.top_k if s else None,
    )


def build_llm_client_from_entry(entry: Any, settings: Any) -> LLMClient:
    """Construct an LLM client for one ModelEntry from models.yaml.

    This is the multi-model variant of build_llm_client. The builder pulls the
    API key by provider name (not by settings.llm_provider) so that a single
    process can hold clients for multiple providers simultaneously.

    `entry` is duck-typed (must expose `.provider`, `.model`, `.max_tokens`)
    to avoid an import-time dependency on the config layer.
    """
    profile = profile_from_entry(entry)
    client = _build_client(
        entry.provider,
        model=entry.model,
        max_tokens=entry.max_tokens or settings.llm_max_tokens,
        settings=settings,
        profile=profile,
    )
    if profile.supports_native_tools is False:
        from llm.prompted_tools import PromptedToolLLMClient

        client = PromptedToolLLMClient(client, model=entry.model)
    return client
