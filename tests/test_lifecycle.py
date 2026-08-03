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
from config import ConfigLoadError, MCPConfig, ModelEntry, ModelsConfig
from config.errors import CredentialUnavailableError
from llm.client import GenerationRequest, LLMClient
from llm.tool_prompt_protocol import PromptedToolLLMClient
from llm.providers.gemini import GeminiLLMClient
from llm.providers.openai_compatible import OpenAICompatibleLLMClient
from llm.schemas import AssistantMessage
from main import _close_application_resources, _start_optional_tracer, lifespan
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

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure


class _AppMCPManager:
    connected_servers: list[str] = []
    instances: list[_AppMCPManager] = []

    def __init__(
        self,
        config: MCPConfig,
        *,
        connect_timeout_seconds: float,
        catalog_ttl_seconds: float,
    ) -> None:
        self.connect_timeout_seconds = connect_timeout_seconds
        self.catalog_ttl_seconds = catalog_ttl_seconds
        self.shutdown_calls = 0
        self.instances.append(self)

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    def list_tools(self) -> list[Any]:
        return []

    def status_snapshot(self) -> tuple[Any, ...]:
        return ()


class _RegistrySettings:
    llm = SimpleNamespace(max_tokens=4096)
    openai_compat_base_url = ""

    def api_key_for_provider(self, provider: str) -> str:
        raise AssertionError(f"unexpected provider construction: {provider}")


def _wire_lifespan_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    orchestration_enabled: bool = False,
    trace_enabled: bool = False,
) -> SimpleNamespace:
    settings = SimpleNamespace(
        log_level="INFO",
        llm=SimpleNamespace(
            provider="test",
            model="test-model",
            max_tokens=4096,
        ),
        mcp_config_path="mcp.yaml",
        mcp_connect_timeout_seconds=17.5,
        mcp_catalog_ttl_seconds=123.0,
        orchestration_enabled=orchestration_enabled,
        session_ttl_seconds=0,
        session_capacity=0,
        trace_enabled=trace_enabled,
        trace_jsonl_path=tmp_path / "trace.jsonl",
    )
    mcp_config = MCPConfig.model_validate({"mcpServers": {}})
    _AppMCPManager.instances.clear()
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        main_module,
        "load_mcp_config_from_settings",
        lambda value: mcp_config,
    )
    monkeypatch.setattr(main_module, "MCPManager", _AppMCPManager)
    return settings


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
    return LLMRegistry(_models_config(), _RegistrySettings())


async def test_inherited_llm_close_is_a_safe_noop() -> None:
    client = _ResourceFreeClient()
    await client.aclose()
    await client.aclose()


async def test_provider_and_prompted_clients_delegate_close_once() -> None:
    openai_sdk = _SDKAsyncClose()
    openai_client = object.__new__(OpenAICompatibleLLMClient)
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
        unorchestrated_llm=default,
        registry=registry,
        mcp=mcp,
        tracer=tracer,
    )

    assert default.close_calls == 1
    assert mcp.shutdown_calls == 1
    assert tracer.close_calls == 1


async def test_application_cleanup_supports_registry_without_unorchestrated_client() -> (
    None
):
    registry = _registry()
    cached = _ClosingClient()
    registry._clients["cached"] = cached
    mcp = _MCP()
    tracer = _Tracer()

    for _ in range(2):
        await _close_application_resources(
            unorchestrated_llm=None,
            registry=registry,
            mcp=mcp,
            tracer=tracer,
        )

    assert cached.close_calls == 1
    assert mcp.shutdown_calls == 2
    assert tracer.close_calls == 2


async def test_application_cleanup_isolates_llm_mcp_and_tracer_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    default = _ClosingClient(
        failure=RuntimeError("LLM close failed: Authorization Bearer llm-secret")
    )
    sibling = _ClosingClient()
    registry = _registry()
    registry._clients["sibling"] = sibling
    mcp = _MCP(
        failure=RuntimeError(
            "MCP close failed at https://user:pass.invalid/mcp?token=hidden"
        )
    )
    tracer = _Tracer(
        failure=RuntimeError("tracer close failed: Authorization Bearer tracer-secret")
    )

    await _close_application_resources(
        unorchestrated_llm=default,
        registry=registry,
        mcp=mcp,
        tracer=tracer,
    )

    assert default.close_calls == 1
    assert sibling.close_calls == 1
    assert mcp.shutdown_calls == 1
    assert tracer.close_calls == 1
    assert "user:pass" not in caplog.text
    assert "token=hidden" not in caplog.text
    assert "tracer-secret" not in caplog.text
    assert "llm-secret" not in caplog.text


async def test_unorchestrated_llm_cleanup_does_not_log_raw_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _ClosingClient(
        failure=RuntimeError("Authorization: Bearer standalone-llm-secret")
    )

    await _close_application_resources(
        unorchestrated_llm=client,
        registry=None,
        mcp=None,
        tracer=None,
    )

    assert client.close_calls == 1
    assert "standalone-llm-secret" not in caplog.text
    assert "RuntimeError" in caplog.text


async def test_repeated_application_cleanup_is_safe() -> None:
    unorchestrated_only = _IdempotentClosingClient()
    for _ in range(2):
        await _close_application_resources(
            unorchestrated_llm=unorchestrated_only,
            registry=None,
            mcp=None,
            tracer=None,
        )

    aliased = _IdempotentClosingClient()
    registry = _registry()
    registry._clients.update({"default": aliased, "alias": aliased})
    mcp = _MCP()
    tracer = _Tracer()

    for _ in range(2):
        await _close_application_resources(
            unorchestrated_llm=aliased,
            registry=registry,
            mcp=mcp,
            tracer=tracer,
        )

    assert unorchestrated_only.close_calls == 1
    assert aliased.close_calls == 1
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
            unorchestrated_llm=default,
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


async def test_cancelled_tracer_start_closes_once_and_reraises() -> None:
    started = asyncio.Event()

    class _BlockingStartupTracer(_Tracer):
        async def start(self) -> None:
            started.set()
            await asyncio.Event().wait()

    tracer = _BlockingStartupTracer()
    task = asyncio.create_task(_start_optional_tracer(tracer))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert tracer.close_calls == 1


async def test_lifespan_does_not_build_unorchestrated_llm_before_it_is_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = _ClosingClient()
    settings = SimpleNamespace(
        log_level="INFO",
        llm=SimpleNamespace(
            provider="test",
            model="test-model",
            max_tokens=4096,
        ),
        mcp_config_path="missing-mcp.yaml",
    )

    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    build_calls = 0

    def build_unorchestrated(value: Any) -> LLMClient:
        nonlocal build_calls
        build_calls += 1
        return default

    monkeypatch.setattr(main_module, "build_llm_client", build_unorchestrated)

    def fail_config(settings: Any) -> Any:
        raise RuntimeError("startup failed")

    monkeypatch.setattr(main_module, "load_mcp_config_from_settings", fail_config)

    with pytest.raises(RuntimeError, match="startup failed"):
        async with lifespan(FastAPI()):
            raise AssertionError("startup failure must prevent lifespan entry")

    assert build_calls == 0
    assert default.close_calls == 0


async def test_orchestrated_lifespan_skips_unorchestrated_client_and_closes_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    app = FastAPI()
    control = _ClosingClient()
    registry = _registry()
    registry._clients["control"] = control
    _wire_lifespan_dependencies(
        monkeypatch,
        tmp_path,
        orchestration_enabled=True,
    )
    monkeypatch.setattr(
        main_module,
        "build_llm_client",
        lambda value: pytest.fail(
            "orchestrated startup must not build an unorchestrated LLM"
        ),
    )
    monkeypatch.setattr(
        main_module,
        "_try_build_orchestration",
        lambda value: (registry, object()),
    )

    async with lifespan(app):
        assert app.state.unorchestrated_llm is None

    assert control.close_calls == 1
    assert _AppMCPManager.instances[0].shutdown_calls == 1


async def test_orchestration_setup_failure_builds_one_unorchestrated_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    default = _ClosingClient()
    build_calls = 0

    def build_unorchestrated(value: Any) -> LLMClient:
        nonlocal build_calls
        build_calls += 1
        return default

    _wire_lifespan_dependencies(
        monkeypatch,
        tmp_path,
        orchestration_enabled=True,
    )
    monkeypatch.setattr(main_module, "build_llm_client", build_unorchestrated)
    monkeypatch.setattr(
        main_module,
        "_try_build_orchestration",
        lambda value: (None, None),
    )

    async with lifespan(FastAPI()) as _:
        pass

    assert build_calls == 1
    assert default.close_calls == 1


def test_present_invalid_orchestration_config_is_fatal_before_provider_build(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        """\
models:
  one:
    provider: gemini
    model: first
    model: second
    description: invalid duplicate
""",
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("route requests", encoding="utf-8")
    settings = SimpleNamespace(
        orchestration_enabled=True,
        models_config_path=models_path,
        orchestrator_prompt_path=prompt_path,
        orchestrator_model_id="",
        llm=SimpleNamespace(max_tokens=128),
        context_default_window_tokens=4096,
        context_safety_margin_tokens=128,
    )
    monkeypatch.setattr(
        main_module,
        "build_llm_client",
        lambda value: pytest.fail("invalid config must fail before provider build"),
    )

    with pytest.raises(ConfigLoadError, match="invalid models config"):
        main_module._try_build_orchestration(settings)


def _wire_failing_orchestrator_registry(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> SimpleNamespace:
    class _FailingRegistry:
        model_ids = ["control"]

        def __init__(self, models_config: Any, settings: Any) -> None:
            pass

        def default_id(self) -> str:
            return "control"

        def get_entry(self, model_id: str) -> SimpleNamespace:
            return SimpleNamespace(supports_native_tools=True)

        def get(self, model_id: str) -> None:
            raise failure

    monkeypatch.setattr(main_module, "LLMRegistry", _FailingRegistry)
    monkeypatch.setattr(
        main_module,
        "load_models_config_from_settings",
        lambda settings, known_providers: object(),
    )
    monkeypatch.setattr(
        main_module,
        "load_orchestrator_prompt",
        lambda path: "route requests",
    )
    return SimpleNamespace(
        orchestration_enabled=True,
        models_config_path="models.yaml",
        orchestrator_prompt_path="prompt.md",
        orchestrator_model_id="",
    )


def test_missing_mandatory_provider_dependency_aborts_startup(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _wire_failing_orchestrator_registry(
        monkeypatch,
        ModuleNotFoundError("No module named 'provider_sdk'"),
    )

    with pytest.raises(ModuleNotFoundError, match="provider_sdk"):
        main_module._try_build_orchestration(settings)

    assert "provider credential unavailable" not in caplog.text


def test_missing_provider_credential_degrades_without_raw_error_logging(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "credential token=hidden"
    settings = _wire_failing_orchestrator_registry(
        monkeypatch,
        CredentialUnavailableError(secret),
    )

    assert main_module._try_build_orchestration(settings) == (None, None)
    assert "provider credential unavailable" in caplog.text
    assert secret not in caplog.text


async def test_tracer_startup_failure_does_not_log_raw_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "Authorization: Bearer tracer-secret"

    class _FailingTracer(_Tracer):
        async def start(self) -> None:
            raise RuntimeError(secret)

    tracer = _FailingTracer()

    assert await _start_optional_tracer(tracer) is None
    assert "tracing startup failed" in caplog.text
    assert secret not in caplog.text
    assert "_FailingTracer" in caplog.text
    assert "RuntimeError" in caplog.text


def test_unexpected_provider_construction_failure_aborts_startup_without_logging(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "https://user:pass.invalid/mcp?token=hidden"
    settings = _wire_failing_orchestrator_registry(
        monkeypatch,
        RuntimeError(secret),
    )

    with pytest.raises(RuntimeError, match="token=hidden"):
        main_module._try_build_orchestration(settings)

    assert secret not in caplog.text


def test_single_implicit_default_is_marked_in_prompt_inventory() -> None:
    config = ModelsConfig(
        models={
            "only": ModelEntry(
                provider="gemini",
                model="model",
                description="Only model.",
            )
        }
    )

    assert "- only (default)" in LLMRegistry(config, _RegistrySettings()).describe_for_prompt()


async def test_disabled_orchestration_builds_one_unorchestrated_client_and_injects_mcp_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    default = _ClosingClient()
    build_calls = 0

    def build_unorchestrated(value: Any) -> LLMClient:
        nonlocal build_calls
        build_calls += 1
        return default

    _wire_lifespan_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(main_module, "build_llm_client", build_unorchestrated)

    async with lifespan(FastAPI()):
        pass

    assert _AppMCPManager.instances[0].connect_timeout_seconds == 17.5
    assert _AppMCPManager.instances[0].catalog_ttl_seconds == 123.0
    assert build_calls == 1
    assert default.close_calls == 1


@pytest.mark.parametrize("fail_start", [False, True])
async def test_lifespan_starts_tracer_before_publish_and_degrades_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    fail_start: bool,
) -> None:
    default = _ClosingClient()
    app = FastAPI()

    class _StartupTracer(_Tracer):
        def __init__(self) -> None:
            super().__init__()
            self.start_calls = 0

        async def start(self) -> None:
            self.start_calls += 1
            assert not hasattr(app.state, "tracer")
            if fail_start:
                raise OSError("trace open failed")

    tracer = _StartupTracer()
    _wire_lifespan_dependencies(monkeypatch, tmp_path, trace_enabled=True)
    monkeypatch.setattr(main_module, "build_llm_client", lambda value: default)
    monkeypatch.setattr(main_module, "build_tracer", lambda **kwargs: tracer)

    async with lifespan(app):
        assert app.state.tracer is (None if fail_start else tracer)

    assert tracer.start_calls == 1
    assert tracer.close_calls == 1
    assert default.close_calls == 1


@pytest.mark.parametrize(
    "orchestrated", [False, True], ids=["unorchestrated", "registry"]
)
async def test_lifespan_cancellation_during_tracer_start_closes_all_owners(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    orchestrated: bool,
) -> None:
    app = FastAPI()
    owned_client = _ClosingClient()
    tracer_started = asyncio.Event()

    class _BlockingStartupTracer(_Tracer):
        async def start(self) -> None:
            tracer_started.set()
            await asyncio.Event().wait()

    tracer = _BlockingStartupTracer()
    _wire_lifespan_dependencies(
        monkeypatch,
        tmp_path,
        orchestration_enabled=orchestrated,
        trace_enabled=True,
    )
    if orchestrated:
        registry = _registry()
        registry._clients["control"] = owned_client
        monkeypatch.setattr(
            main_module,
            "_try_build_orchestration",
            lambda value: (registry, object()),
        )
        monkeypatch.setattr(
            main_module,
            "build_llm_client",
            lambda value: pytest.fail(
                "orchestrated startup must not build an unorchestrated LLM"
            ),
        )
    else:
        monkeypatch.setattr(
            main_module,
            "build_llm_client",
            lambda value: owned_client,
        )
    monkeypatch.setattr(main_module, "build_tracer", lambda **kwargs: tracer)

    async def run_lifespan() -> None:
        async with lifespan(app):
            raise AssertionError("cancelled startup must not enter lifespan")

    task = asyncio.create_task(run_lifespan())
    await tracer_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert tracer.close_calls == 1
    assert owned_client.close_calls == 1
    assert _AppMCPManager.instances[0].shutdown_calls == 1
