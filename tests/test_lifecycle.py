"""LLM client, registry, and application lifecycle ownership contracts."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from google.genai.client import AsyncClient as GeminiAsyncClient
from openai import AsyncOpenAI, AsyncStream

import main as main_module
import orchestrator.registry as registry_module
from config import MCPConfig, ModelEntry, ModelsConfig
from llm.client import GenerationRequest, LLMClient
from llm.prompted_tools import PromptedToolLLMClient
from llm.providers.gemini import GeminiLLMClient
from llm.providers.openai import OpenAILLMClient
from llm.schemas import AssistantMessage
from main import _close_application_resources, lifespan
from orchestrator import LLMRegistry

pytestmark = pytest.mark.anyio


class _ClosingClient(LLMClient):
    def __init__(
        self,
        *,
        failure: BaseException | None = None,
        on_close: Any = None,
    ) -> None:
        self.close_calls = 0
        self.failure = failure
        self.on_close = on_close

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("completion is not part of lifecycle tests")

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.on_close is not None:
            self.on_close()
        if self.failure is not None:
            raise self.failure


class _ResourceFreeClient(LLMClient):
    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        raise AssertionError("completion is not part of lifecycle tests")


class _IdempotentClosingClient(_ClosingClient):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        await super().aclose()


class _SDKAsyncClose:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _SDKAioClose:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _MCP:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.shutdown_calls = 0
        self.failure = failure

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.failure is not None:
            raise self.failure


class _Tracer:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.close_calls = 0
        self.failure = failure

    def close(self) -> None:
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure


def _models_config() -> ModelsConfig:
    return ModelsConfig(
        models={
            "test": ModelEntry(
                provider="test",
                model="test-model",
                description="Lifecycle test model.",
                default=True,
            )
        }
    )


def _registry() -> LLMRegistry:
    return LLMRegistry(_models_config(), SimpleNamespace())


async def test_inherited_llm_close_is_a_safe_noop() -> None:
    client = _ResourceFreeClient()
    await client.aclose()
    await client.aclose()


async def test_provider_and_prompted_clients_delegate_close_once() -> None:
    openai_sdk = _SDKAsyncClose()
    openai_client = object.__new__(OpenAILLMClient)
    openai_client._client = openai_sdk
    openai_client._closed = False

    gemini_aio = _SDKAioClose()
    gemini_client = object.__new__(GeminiLLMClient)
    gemini_client._client = SimpleNamespace(aio=gemini_aio)
    gemini_client._closed = False

    inner = _ClosingClient()
    prompted = PromptedToolLLMClient(inner)

    for client in (openai_client, gemini_client, prompted):
        await client.aclose()
        await client.aclose()

    assert openai_sdk.close_calls == 1
    assert gemini_aio.close_calls == 1
    assert inner.close_calls == 1


async def test_installed_sdk_close_surfaces_are_the_exercised_async_methods() -> None:
    assert inspect.iscoroutinefunction(AsyncOpenAI.close)
    assert inspect.iscoroutinefunction(GeminiAsyncClient.aclose)
    assert inspect.iscoroutinefunction(AsyncStream.close)
    assert not hasattr(AsyncStream, "aclose")


async def test_registry_snapshots_clears_deduplicates_and_is_repeatable() -> None:
    registry = _registry()
    cache_was_clear: list[bool] = []
    duplicate = _ClosingClient(
        on_close=lambda: cache_was_clear.append(not registry._clients)
    )
    other = _ClosingClient()
    registry._clients.update(
        {
            "first": duplicate,
            "alias": duplicate,
            "other": other,
        }
    )

    await registry.aclose(additional_clients=(duplicate, other))
    await registry.aclose()

    assert registry._clients == {}
    assert duplicate.close_calls == 1
    assert other.close_calls == 1
    assert cache_was_clear == [True]


async def test_registry_is_best_effort_for_failures_and_child_cancellation() -> None:
    registry = _registry()
    failed = _ClosingClient(failure=RuntimeError("close failed"))
    cancelled = _ClosingClient(failure=asyncio.CancelledError())
    healthy = _ClosingClient()
    registry._clients.update(
        {"failed": failed, "cancelled": cancelled, "healthy": healthy}
    )

    await registry.aclose()

    assert registry._clients == {}
    assert failed.close_calls == 1
    assert cancelled.close_calls == 1
    assert healthy.close_calls == 1


async def test_registry_handles_failed_and_never_started_lazy_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls = 0

    def fail_build(entry: Any, settings: Any) -> LLMClient:
        nonlocal build_calls
        build_calls += 1
        raise RuntimeError("construction failed")

    monkeypatch.setattr(registry_module, "build_llm_client_from_entry", fail_build)
    never_used = _registry()
    await never_used.aclose()
    assert build_calls == 0

    failed = _registry()
    with pytest.raises(RuntimeError, match="construction failed"):
        failed.get("test")
    await failed.aclose()

    assert build_calls == 1
    assert failed._clients == {}


async def test_application_cleanup_deduplicates_default_and_registry_alias() -> None:
    default = _ClosingClient()
    registry = _registry()
    registry._clients.update({"default": default, "alias": default})
    mcp = _MCP()
    tracer = _Tracer()

    await _close_application_resources(
        llm=default,
        registry=registry,
        mcp=mcp,
        tracer=tracer,
    )

    assert default.close_calls == 1
    assert mcp.shutdown_calls == 1
    assert tracer.close_calls == 1


async def test_application_cleanup_isolates_llm_mcp_and_tracer_failures() -> None:
    default = _ClosingClient(failure=RuntimeError("LLM close failed"))
    sibling = _ClosingClient()
    registry = _registry()
    registry._clients["sibling"] = sibling
    mcp = _MCP(failure=RuntimeError("MCP close failed"))
    tracer = _Tracer(failure=RuntimeError("tracer close failed"))

    await _close_application_resources(
        llm=default,
        registry=registry,
        mcp=mcp,
        tracer=tracer,
    )

    assert default.close_calls == 1
    assert sibling.close_calls == 1
    assert mcp.shutdown_calls == 1
    assert tracer.close_calls == 1


async def test_repeated_application_cleanup_is_safe() -> None:
    default = _IdempotentClosingClient()
    registry = _registry()
    registry._clients["default"] = default
    mcp = _MCP()
    tracer = _Tracer()

    for _ in range(2):
        await _close_application_resources(
            llm=default,
            registry=registry,
            mcp=mcp,
            tracer=tracer,
        )

    assert default.close_calls == 1
    assert mcp.shutdown_calls == 2
    assert tracer.close_calls == 2


async def test_active_application_cancellation_survives_remaining_cleanup() -> None:
    started = asyncio.Event()
    blocker = asyncio.Event()

    class _BlockingClient(_ClosingClient):
        async def aclose(self) -> None:
            self.close_calls += 1
            started.set()
            await blocker.wait()

    default = _BlockingClient()
    mcp = _MCP()
    tracer = _Tracer()
    task = asyncio.create_task(
        _close_application_resources(
            llm=default,
            registry=None,
            mcp=mcp,
            tracer=tracer,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert default.close_calls == 1
    assert mcp.shutdown_calls == 1
    assert tracer.close_calls == 1


async def test_lifespan_closes_default_llm_after_partial_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = _ClosingClient()
    settings = SimpleNamespace(
        log_level="INFO",
        llm_provider="test",
        llm_model="test-model",
        mcp_config_path="missing-mcp.yaml",
    )

    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "build_llm_client", lambda value: default)

    def fail_config(settings: Any) -> Any:
        raise RuntimeError("startup failed")

    monkeypatch.setattr(main_module, "load_mcp_config_from_settings", fail_config)

    with pytest.raises(RuntimeError, match="startup failed"):
        async with lifespan(FastAPI()):
            raise AssertionError("startup failure must prevent lifespan entry")

    assert default.close_calls == 1


async def test_lifespan_injects_mcp_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    default = _ClosingClient()
    captured: dict[str, float] = {}

    class _Manager:
        connected_servers: list[str] = []

        def __init__(self, config: MCPConfig, *, connect_timeout_seconds: float) -> None:
            captured["timeout"] = connect_timeout_seconds

        async def startup(self) -> None:
            pass

        async def shutdown(self) -> None:
            pass

        def list_tools(self) -> list[Any]:
            return []

        def status_snapshot(self) -> tuple[Any, ...]:
            return ()

    settings = SimpleNamespace(
        log_level="INFO",
        llm_provider="test",
        llm_model="test-model",
        mcp_config_path="mcp.yaml",
        mcp_connect_timeout_seconds=17.5,
        orchestration_enabled=False,
        session_ttl_seconds=0,
        session_max_count=0,
        trace_enabled=False,
        trace_path=tmp_path / "trace.jsonl",
    )
    mcp_config = MCPConfig.model_validate({"mcpServers": {}})
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "build_llm_client", lambda value: default)
    monkeypatch.setattr(
        main_module,
        "load_mcp_config_from_settings",
        lambda settings: mcp_config,
    )
    monkeypatch.setattr(main_module, "MCPManager", _Manager)

    async with lifespan(FastAPI()):
        pass

    assert captured == {"timeout": 17.5}
    assert default.close_calls == 1
