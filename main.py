# main.py

"""FastAPI entry point and lifespan wiring for the harness."""

from __future__ import annotations

from config import load_secrets
load_secrets()

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent import InMemorySessionStore, SessionGuard, build_tool_policy, build_tracer
from api import router
from api.openai_compatible import openai_auth_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from llm import build_llm_client
from mcp_layer import MCPManager
from config import get_settings, load_mcp_config, load_models_config, load_orchestrator_prompt
from llm.client import supported_providers
from orchestrator import LLMRegistry, Orchestrator

logger = logging.getLogger(__name__)


def _try_build_orchestration(settings, mcp: MCPManager) -> tuple[LLMRegistry | None, Orchestrator | None]:
    """Attempt to construct (LLMRegistry, Orchestrator). Returns (None, None) on any failure.

    Orchestration is optional; load/build errors degrade to no-orchestration mode.
    """
    if not settings.orchestration_enabled:
        logger.info("orchestration disabled by Settings.orchestration_enabled=False")
        return None, None

    try:
        models_config = load_models_config(settings.models_config_path, known_providers=supported_providers())
    except FileNotFoundError:
        logger.warning(
            "orchestration enabled but models config not found at %s; "
            "running in legacy (no-orchestration) mode.",
            settings.models_config_path,
        )
        return None, None
    except Exception as e:
        logger.warning(
            "orchestration enabled but models config failed to load: %s; "
            "running in legacy mode.", e,
        )
        return None, None

    try:
        orch_prompt = load_orchestrator_prompt(settings.orchestrator_prompt_path)
    except FileNotFoundError:
        logger.warning(
            "orchestrator prompt not found at %s; running in legacy mode.",
            settings.orchestrator_prompt_path,
        )
        return None, None
    except Exception as e:
        logger.warning("orchestrator prompt failed to load: %s; legacy mode.", e)
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
                "models. running in legacy mode.",
                orch_model_id,
            )
            return None, None
        registry.get(orch_model_id)
    except Exception as e:
        logger.warning(
            "could not build LLM client for orchestrator_model_id=%r: %s; "
            "running in legacy mode.", orch_model_id, e,
        )
        return None, None

    orchestrator = Orchestrator(
        registry=registry,
        system_prompt=orch_prompt,
        model_id=orch_model_id,
    )
    logger.info(
        "orchestration enabled: %d models registered, orchestrator_model_id=%s",
        len(registry.model_ids), orch_model_id,
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
        "mcp=%d servers connected (%d tools)",
        settings.llm_provider,
        settings.llm_model,
        orch_part,
        len(mcp.connected_servers),
        len(mcp.list_tools()),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Settings and default LLM fail loud if env/config is broken.
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )
    logger.info("starting harness: provider=%s model=%s",
                settings.llm_provider, settings.llm_model)

    llm = build_llm_client(settings)

    mcp_config = load_mcp_config(settings.mcp_config_path)
    mcp = MCPManager(mcp_config)
    await mcp.startup()

    # Dispatch policy is what may run; orchestration is only what the model sees.
    policy = build_tool_policy(mcp_config.tool_policy)

    # The guard rejects concurrent requests for the same session_id.
    store = InMemorySessionStore(
        ttl_seconds=settings.session_ttl_seconds,
        max_count=settings.session_max_count,
    )
    guard = SessionGuard()

    registry, orchestrator = _try_build_orchestration(settings, mcp)

    tracer = build_tracer(enabled=settings.trace_enabled, path=settings.trace_path)

    app.state.settings = settings
    app.state.llm = llm
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

    try:
        yield
    finally:
        logger.info("shutting down harness")
        await mcp.shutdown()
        if tracer is not None:
            tracer.close()


app = FastAPI(
    title="hyphae",
    description="Minimal extendable AI harness with MCP tool support and orchestration layer.",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(router)
# Reshape /v1 401s into the OpenAI error envelope (native /chat 401s pass through).
app.add_exception_handler(StarletteHTTPException, openai_auth_exception_handler)
