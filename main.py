# main.py

"""FastAPI entry point and lifespan wiring for the harness."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent import (
    InMemorySessionStore,
    SessionGuard,
    Tracer,
    build_tool_policy,
    build_tracer,
)
from api import router
from api.openai_compatible import openai_auth_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from llm import LLMClient, build_llm_client
from mcp_layer import MCPManager
from config import (
    get_settings,
    load_mcp_config_from_settings,
    load_models_config,
    load_orchestrator_prompt,
)
from llm.client import supported_providers
from orchestrator import LLMRegistry, Orchestrator

logger = logging.getLogger(__name__)


async def _close_application_resources(
    *,
    unorchestrated_llm: LLMClient | None,
    registry: LLMRegistry | None,
    mcp: MCPManager | None,
    tracer: Tracer | None,
) -> None:
    """Close process-owned resources without masking the active outcome."""
    active_cancellation: asyncio.CancelledError | None = None

    async def _run_async_cleanup(
        label: str, cleanup: Callable[[], Awaitable[None]]
    ) -> None:
        nonlocal active_cancellation
        try:
            await cleanup()
        except asyncio.CancelledError as exc:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                active_cancellation = active_cancellation or exc
            else:
                logger.warning("%s cleanup was cancelled", label, exc_info=True)
        except Exception:  # noqa: BLE001 -- shutdown is best-effort.
            logger.warning("%s cleanup failed", label, exc_info=True)

    if registry is not None:
        additional_clients = (
            (unorchestrated_llm,) if unorchestrated_llm is not None else ()
        )
        await _run_async_cleanup(
            "LLM",
            lambda: registry.aclose(additional_clients=additional_clients),
        )
    elif unorchestrated_llm is not None:
        await _run_async_cleanup("LLM", unorchestrated_llm.aclose)

    if mcp is not None:
        await _run_async_cleanup("MCP", mcp.shutdown)

    if tracer is not None:
        await _run_async_cleanup("tracer", tracer.aclose)

    if active_cancellation is not None:
        raise active_cancellation


async def _start_optional_tracer(tracer: Tracer | None) -> Tracer | None:
    """Start optional tracing without turning an observability failure fatal."""
    if tracer is None:
        return None
    try:
        await tracer.start()
    except asyncio.CancelledError:
        try:
            await tracer.aclose()
        except BaseException:  # noqa: BLE001 -- preserve startup cancellation.
            logger.warning(
                "tracer cleanup after cancelled startup failed", exc_info=True
            )
        raise
    except Exception:  # noqa: BLE001 -- tracing is an optional layer.
        logger.warning("tracing startup failed; running without traces", exc_info=True)
        try:
            await tracer.aclose()
        except Exception:  # noqa: BLE001 -- startup remains best-effort.
            logger.warning("failed tracer cleanup failed", exc_info=True)
        return None
    return tracer


def _try_build_orchestration(
    settings,
) -> tuple[LLMRegistry | None, Orchestrator | None]:
    """Attempt to construct (LLMRegistry, Orchestrator). Returns (None, None) on any failure.

    Orchestration is optional; load/build errors degrade to no-orchestration mode.
    """
    if not settings.orchestration_enabled:
        logger.info("orchestration disabled by Settings.orchestration_enabled=False")
        return None, None

    try:
        models_config = load_models_config(
            settings.models_config_path, known_providers=supported_providers()
        )
    except FileNotFoundError:
        logger.warning(
            "orchestration enabled but models config not found at %s; "
            "running in unorchestrated mode.",
            settings.models_config_path,
        )
        return None, None
    except Exception as e:
        logger.warning(
            "orchestration enabled but models config failed to load: %s; "
            "running in unorchestrated mode.",
            e,
        )
        return None, None

    try:
        orch_prompt = load_orchestrator_prompt(settings.orchestrator_prompt_path)
    except FileNotFoundError:
        logger.warning(
            "orchestrator prompt not found at %s; running in unorchestrated mode.",
            settings.orchestrator_prompt_path,
        )
        return None, None
    except Exception as e:
        logger.warning(
            "orchestrator prompt failed to load: %s; unorchestrated mode.", e
        )
        return None, None

    registry = LLMRegistry(models_config, settings)

    # Catch a bad orchestrator_model_id at startup instead of first request.
    orch_model_id = settings.orchestrator_model_id or registry.default_id()
    try:
        orch_entry = registry.get_entry(orch_model_id)
        if orch_entry.supports_native_tools is False:
            logger.warning(
                "orchestrator_model_id=%r declares supports_native_tools:false; "
                "prompted-tool models are not supported as orchestrator control "
                "models. running in unorchestrated mode.",
                orch_model_id,
            )
            return None, None
        registry.get(orch_model_id)
    except Exception as e:
        logger.warning(
            "could not build LLM client for orchestrator_model_id=%r: %s; "
            "running in unorchestrated mode.",
            orch_model_id,
            e,
        )
        return None, None

    orchestrator = Orchestrator(
        registry=registry,
        system_prompt=orch_prompt,
        model_id=orch_model_id,
    )
    logger.info(
        "orchestration enabled: %d models registered, orchestrator_model_id=%s",
        len(registry.model_ids),
        orch_model_id,
    )
    return registry, orchestrator


def _log_ready_summary(
    *,
    settings,
    mcp: MCPManager,
    registry: LLMRegistry | None,
) -> None:
    """Emit one greppable startup summary."""
    if registry is not None:
        orch_part = f"orchestration=on (models={len(registry.model_ids)})"
    else:
        orch_part = "orchestration=off"

    logger.info(
        "harness ready: provider=%s default_model=%s | %s | "
        "mcp=%d/%d servers healthy (%d tools)",
        settings.llm.provider,
        settings.llm.model_name,
        orch_part,
        len(mcp.connected_servers),
        len(mcp.status_snapshot()),
        len(mcp.list_tools()),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Settings fail loud if env/config is broken. The unorchestrated LLM is
    # constructed only when orchestration is disabled or unavailable.
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )
    logger.info(
        "starting harness: provider=%s model=%s",
        settings.llm.provider,
        settings.llm.model_name,
    )

    unorchestrated_llm: LLMClient | None = None
    mcp: MCPManager | None = None
    registry: LLMRegistry | None = None
    tracer: Tracer | None = None
    try:
        mcp_config = load_mcp_config_from_settings(settings)
        mcp = MCPManager(
            mcp_config,
            connect_timeout_seconds=settings.mcp_connect_timeout_seconds,
            catalog_ttl_seconds=settings.mcp_catalog_ttl_seconds,
        )
        await mcp.startup()

        # Dispatch policy is what may run; orchestration is only what the model sees.
        policy = build_tool_policy(mcp_config.tool_policy)

        # The guard rejects concurrent requests for the same session_id.
        store = InMemorySessionStore(
            ttl_seconds=settings.session_ttl_seconds,
            max_count=settings.session_capacity,
        )
        guard = SessionGuard()

        registry, orchestrator = _try_build_orchestration(settings)
        if registry is None or orchestrator is None:
            unorchestrated_llm = build_llm_client(settings)

        tracer = await _start_optional_tracer(
            build_tracer(enabled=settings.trace_enabled, path=settings.trace_jsonl_path)
        )

        app.state.settings = settings
        app.state.unorchestrated_llm = unorchestrated_llm
        app.state.mcp = mcp
        app.state.store = store
        app.state.guard = guard
        app.state.registry = registry
        app.state.orchestrator = orchestrator
        app.state.policy = policy
        app.state.tracer = tracer

        _log_ready_summary(
            settings=settings,
            mcp=mcp,
            registry=registry,
        )

        yield
    finally:
        logger.info("shutting down harness")
        await _close_application_resources(
            unorchestrated_llm=unorchestrated_llm,
            registry=registry,
            mcp=mcp,
            tracer=tracer,
        )


app = FastAPI(
    title="hyphae",
    description="Minimal extendable AI harness with MCP tool support and orchestration layer.",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(router)
# Reshape /v1 401s into the OpenAI error envelope (native /chat 401s pass through).
app.add_exception_handler(StarletteHTTPException, openai_auth_exception_handler)
