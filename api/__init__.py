"""HTTP API layer for the harness.

Exposes one combined router: the native plain-text `/chat` + `/health`
(api/routes.py) and the OpenAI-compatible `/v1` adapter (api/openai_compat.py),
both thin shells over the same shared core.
"""
from fastapi import APIRouter

from .openai_compat import router as _openai_router
from .routes import router as _routes_router

router = APIRouter()
router.include_router(_routes_router)
router.include_router(_openai_router)

__all__ = ["router"]
