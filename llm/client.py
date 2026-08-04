# llm/client.py
"""
LLM client: the provider-agnostic abstraction and the provider registry.

The abstract `LLMClient` consumes one provider-neutral `GenerationRequest` and
returns provider-neutral outcomes. Each provider implementation (in
`llm/providers/`) is responsible for two translations:

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
  add a module or package in `llm/providers/`, then add one entry here.

Generation inputs versus execution controls:
  `GenerationRequest` contains only values that shape provider generation.
  Retry, timeout, deadline, cancellation, tracing, persistence, and run state
  stay with their existing execution owners. Request containers are borrowed;
  clients and decorators must not mutate them.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

from .schemas import (
    AssistantMessage,
    Message,
    ModelProfile,
    ReasoningDelta,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
)

logger = logging.getLogger(__name__)


class _ProviderConstructionSettings(Protocol):
    """Credentials and endpoint configuration used by provider builders."""

    @property
    def openai_compat_base_url(self) -> str: ...

    def api_key_for_provider(self, provider: str) -> str: ...


class _LLMDefaults(Protocol):
    """Nested LLM defaults used by client factories."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def max_tokens(self) -> int: ...


class _DefaultClientSettings(_ProviderConstructionSettings, Protocol):
    """Global defaults used when a model entry omits a value."""

    @property
    def llm(self) -> _LLMDefaults: ...


class _UnorchestratedClientSettings(_DefaultClientSettings, Protocol):
    """Selection defaults required by the unorchestrated client factory."""


class _SamplingConfig(Protocol):
    """Optional sampling values declared by one model entry."""

    @property
    def temperature(self) -> float | None: ...

    @property
    def top_p(self) -> float | None: ...

    @property
    def top_k(self) -> int | None: ...


class _ModelEntry(Protocol):
    """Model configuration required by provider construction."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def max_tokens(self) -> int | None: ...

    @property
    def supports_native_tools(self) -> bool: ...

    @property
    def thinking(self) -> str: ...

    @property
    def sampling(self) -> _SamplingConfig | None: ...


@dataclass(frozen=True, slots=True)
class _ProviderBuildSpec:
    """Provider-neutral values needed to construct one client."""

    model: str
    max_tokens: int
    profile: ModelProfile | None = None


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Shallow-immutable, provider-neutral inputs for one generation."""

    messages: Sequence[Message]
    tools: Sequence[Mapping[str, Any]] | None = None
    system: str | None = None
    max_tokens: int | None = None
    response_schema: type | None = None
    thinking_level: str | None = None


class LLMClient(ABC):
    """Provider-agnostic LLM client interface."""

    async def aclose(self) -> None:
        """Release client-owned resources. Resource-free clients need no override."""

    @abstractmethod
    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        """Run one completion turn and return the assistant's response.

        `request.tools` is the generic shape from MCPManager.get_tools_for_llm():
        [{name, description, input_schema}]. The implementation reshapes
        it for its provider.

        `request.response_schema`, if provided, is a Pydantic model class that
        the response should conform to. Providers that support structured
        output (Gemini, OpenAI) honor this; others may ignore it. The
        orchestration layer uses this field; the agent loop does not.

        `request.thinking_level` ("low"|"medium"|"high"), if provided, asks
        the model to deliberate more or less for this call -- the
        orchestrator's per-request "intelligence on demand" knob. None leaves
        the model's default. Providers without a thinking control may ignore
        it.
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

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Stream one completion turn.

        Default implementation for complete-only providers: run `complete()`,
        emit optional reasoning and text blocks coarsely, then return the
        assembled message.
        Structured-output calls intentionally stay on `complete()` and are
        rejected explicitly here rather than silently losing their schema.
        """
        if request.response_schema is not None:
            raise ValueError("response_schema is not supported for streaming")
        msg = await self.complete(request)
        if msg.reasoning:
            yield ReasoningDelta(text=msg.reasoning)
        for block in msg.content:
            if isinstance(block, TextBlock) and block.text:
                yield TextDelta(text=block.text)
        yield StreamEnd(message=msg)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
# Each builder lazily imports its provider module so importing this file never
# drags in a provider SDK. Builders receive one immutable construction spec plus
# the narrow settings capability used to resolve credentials and endpoint
# configuration.


class _ProviderBuilder(Protocol):
    """Construction callable stored in the provider registry."""

    def __call__(
        self,
        *,
        spec: _ProviderBuildSpec,
        settings: _ProviderConstructionSettings,
    ) -> LLMClient: ...


def _build_gemini(
    *,
    spec: _ProviderBuildSpec,
    settings: _ProviderConstructionSettings,
) -> LLMClient:
    from llm.providers.gemini import GeminiLLMClient

    return GeminiLLMClient(
        api_key=settings.api_key_for_provider("gemini"),
        model=spec.model,
        default_max_tokens=spec.max_tokens,
        profile=spec.profile,
    )


def _build_openai_compatible(
    *,
    spec: _ProviderBuildSpec,
    settings: _ProviderConstructionSettings,
) -> LLMClient:
    from llm.providers.openai_compatible import OpenAICompatibleLLMClient

    # OpenAI-compatible: base_url empty -> real OpenAI; set it (e.g. Ollama's
    # /v1) to target a local/compatible server. The key is required by the SDK
    # even when the server ignores it.
    return OpenAICompatibleLLMClient(
        api_key=settings.api_key_for_provider("openai_compatible"),
        model=spec.model,
        default_max_tokens=spec.max_tokens,
        base_url=settings.openai_compat_base_url or None,
        profile=spec.profile,
    )


# THE one place a provider is declared. Adding a provider = one file in
# llm/providers/ + one entry here. Nothing else in the harness enumerates
# providers (config validators key off supported_providers()).
_PROVIDERS: dict[str, _ProviderBuilder] = {
    "gemini": _build_gemini,
    # `openai` remains a compatibility alias for existing configs.
    "openai_compatible": _build_openai_compatible,
    "openai": _build_openai_compatible,
    # "anthropic": _build_anthropic,
}


def supported_providers() -> frozenset[str]:
    """Provider names the harness can build. Config validators key off this."""
    return frozenset(_PROVIDERS)


def _build_client(
    provider: str,
    *,
    spec: _ProviderBuildSpec,
    settings: _ProviderConstructionSettings,
) -> LLMClient:
    """Dispatch to the registered builder for `provider`.

    Raises NotImplementedError for an unknown/unregistered provider, preserving
    the harness's previous behavior for unimplemented providers.
    """
    try:
        build = _PROVIDERS[provider]
    except KeyError:
        raise NotImplementedError(f"LLM provider {provider!r} is not implemented yet")
    return build(spec=spec, settings=settings)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_llm_client(settings: _UnorchestratedClientSettings) -> LLMClient:
    """Construct the LLM client matching settings.llm.provider.

    Takes Settings duck-typed to avoid a circular import with the config layer.

    Preserved for the unorchestrated path. The orchestration layer uses
    `build_llm_client_from_entry` instead, which is parameterized by a
    ModelEntry rather than the global Settings.
    """
    return _build_client(
        settings.llm.provider,
        spec=_ProviderBuildSpec(
            model=settings.llm.model,
            max_tokens=settings.llm.max_tokens,
        ),
        settings=settings,
    )


def profile_from_entry(entry: _ModelEntry) -> ModelProfile:
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


def build_llm_client_from_entry(
    entry: _ModelEntry, settings: _DefaultClientSettings
) -> LLMClient:
    """Construct an LLM client for one ModelEntry from models.yaml.

    This is the multi-model variant of build_llm_client. The builder pulls the
    API key by provider name (not by settings.llm.provider) so that a single
    process can hold clients for multiple providers simultaneously.

    `entry` is duck-typed (must expose `.provider`, `.model`, `.max_tokens`)
    to avoid an import-time dependency on the config layer.
    """
    profile = profile_from_entry(entry)
    client = _build_client(
        entry.provider,
        spec=_ProviderBuildSpec(
            model=entry.model,
            max_tokens=entry.max_tokens or settings.llm.max_tokens,
            profile=profile,
        ),
        settings=settings,
    )
    if profile.supports_native_tools is False:
        from llm.tool_prompt_protocol import PromptedToolLLMClient

        client = PromptedToolLLMClient(client, model=entry.model)
    return client
