# main.py

"""
FastAPI entry point for the AI harness.

Run with:
    ./runscript.sh -m uvicorn main:app --host 0.0.0.0 --port 8000

The bootstrap script must populate os.environ BEFORE importing this module
(or anything from the harness). Once env vars are set, lifespan() builds:

    1. Settings        (via get_settings())
    2. LLMClient       (via build_llm_client(settings)) -- default/fallback
    3. MCPConfig       (load_mcp_config from YAML)
    4. MCPManager      (started in parallel)
    5. SessionStore    (in-memory, bounded by TTL + max-size)
    6. SessionGuard    (reject-if-busy guard for same-session concurrency)
    7. LLMRegistry     (when orchestration_enabled and models.yaml loads)
    8. Orchestrator    (when LLMRegistry was built)

All are stashed on app.state for the duration of the process.
Orchestration is optional: if models.yaml is missing or unparseable, the
harness logs a warning and runs without it (routes fall back to the default
LLM and the full MCP tool inventory).

On shutdown, MCP connections are closed cleanly. The session store, LLM
clients, and orchestrator don't need explicit teardown today; if a future
provider's client holds resources, add it here.
"""

from __future__ import annotations

from bootstrap import load_secrets
load_secrets()

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent import InMemorySessionStore, SessionGuard, build_tracer
from api import router
from harness_config import get_settings, load_mcp_config
from llm import build_llm_client
from mcp_layer import MCPManager
from orchestrator import (
    LLMRegistry,
    Orchestrator,
    load_models_config,
    load_orchestrator_prompt,
)

logger = logging.getLogger(__name__)


def _try_build_orchestration(settings, mcp: MCPManager) -> tuple[LLMRegistry | None, Orchestrator | None]:
    """Attempt to construct (LLMRegistry, Orchestrator). Returns (None, None) on any failure.

    Orchestration is optional and must never block harness startup. Any
    error -- missing file, schema violation, bad model_id override -- is
    logged and degrades the harness to legacy (no-orchestration) mode.
    """
    if not settings.orchestration_enabled:
        logger.info("orchestration disabled by Settings.orchestration_enabled=False")
        return None, None

    try:
        models_config = load_models_config(settings.models_config_path)
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

    # Eagerly construct the orchestrator's own client so a bad
    # orchestrator_model_id setting is caught at startup instead of on the
    # first request.
    orch_model_id = settings.orchestrator_model_id or registry.default_id()
    try:
        registry.get(orch_model_id)
    except Exception as e:
        logger.warning(
            "could not build LLM client for orchestrator_model_id=%r: %s; "
            "running in legacy mode.", orch_model_id, e,
        )
        return None, None

    orchestrator = Orchestrator(
        registry=registry,
        mcp=mcp,
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
    """Emit a single consolidated status line at the end of startup.

    The individual lifecycle steps already log their own INFO lines; this
    one pulls the headline facts into one greppable place so operators
    can confirm "the harness came up correctly" without scrolling.
    """
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
    """Startup/shutdown lifecycle. See module docstring for the bootstrap contract."""
    # Settings + LLM client. Both will raise loudly if bootstrap didn't run
    # (missing API key, unknown provider). That's the right behavior — the
    # app should not start up half-broken.
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )
    logger.info("starting harness: provider=%s model=%s",
                settings.llm_provider, settings.llm_model)

    llm = build_llm_client(settings)

    # MCP. Graceful degradation: a bad server is logged but doesn't kill startup.
    mcp_config = load_mcp_config(settings.mcp_config_path)
    mcp = MCPManager(mcp_config)
    await mcp.startup()

    # Session store. In-memory and bounded (TTL + max-size) so it can't grow
    # without limit under concurrent load. Swap to a durable backend later by
    # changing this one line. The guard rejects a second concurrent request on
    # the same session_id (distinct sessions are already isolated).
    store = InMemorySessionStore(
        ttl_seconds=settings.session_ttl_seconds,
        max_count=settings.session_max_count,
    )
    guard = SessionGuard()

    # Orchestration. Built only when configured AND files load successfully.
    # When the registry/orchestrator are None, api/routes.py falls back to
    # the default LLM + all tools -- the pre-orchestrator behavior.
    registry, orchestrator = _try_build_orchestration(settings, mcp)

    # Run tracer. Optional and best-effort: None when disabled or unbuildable,
    # so the loop's hot path is untouched and a bad sink never blocks startup.
    tracer = build_tracer(enabled=settings.trace_enabled, path=settings.trace_path)

    # Publish to app.state for the dependency providers in api/.
    app.state.settings = settings
    app.state.llm = llm
    app.state.mcp = mcp
    app.state.store = store
    app.state.guard = guard
    app.state.registry = registry
    app.state.orchestrator = orchestrator
    app.state.tracer = tracer

    # One consolidated ready line at the end of startup.
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
    title="PyAiHarness",
    description="Minimal extendable AI harness with MCP tool support and orchestration layer.",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(router)