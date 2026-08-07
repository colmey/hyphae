"""Optional API-key authentication tests."""

from __future__ import annotations

import pytest

from hyphae.llm.client import GenerationRequest, LLMClient
from hyphae.llm.schemas import AssistantMessage, TextBlock, CompletionUsage
from hyphae.api.request_body import MAX_REQUEST_BODY_BYTES
from tests._app_support import wired_app


pytestmark = pytest.mark.anyio
API_KEY = "s3cret-test-key"


class FakeLLM(LLMClient):
    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        return AssistantMessage(
            content=[TextBlock(text="ok")],
            stop_reason="end_turn",
            model="fake",
            usage=CompletionUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        )


async def test_auth_disabled_leaves_protected_routes_open(asgi_client) -> None:
    with wired_app(FakeLLM(), settings_overrides={"hyphae_api_key": ""}) as (
        app,
        _settings,
    ):
        client = asgi_client(app)
        assert (await client.post("/chat", content="hello")).status_code == 200
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        assert (await client.get("/health")).status_code == 200


@pytest.mark.parametrize(
    ("route", "headers"),
    [
        ("/chat", {}),
        ("/chat", {"X-API-Key": "wrong"}),
        ("/chat", {"X-API-Key": "café".encode("latin-1")}),
        ("/v1/chat/completions", {"Authorization": "Bearer wrong"}),
    ],
    ids=["missing", "wrong-x-api-key", "non-ascii", "wrong-bearer"],
)
async def test_auth_enabled_rejects_invalid_credentials(
    asgi_client, route, headers
) -> None:
    with wired_app(FakeLLM(), settings_overrides={"hyphae_api_key": API_KEY}) as (
        app,
        _settings,
    ):
        client = asgi_client(app)
        if route.startswith("/v1"):
            response = await client.post(
                route,
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers=headers,
            )
            body = response.json()
            assert isinstance(body.get("error"), dict)
            assert isinstance(body["error"].get("message"), str)
        else:
            response = await client.post(route, content="hello", headers=headers)
        assert response.status_code == 401
        if route.startswith("/v1"):
            assert response.json() == {
                "error": {
                    "message": "Invalid or missing API key.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "authentication_failed",
                }
            }
        else:
            assert response.json() == {
                "code": "authentication_failed",
                "message": "Invalid or missing API key.",
            }


@pytest.mark.parametrize(
    ("route", "headers"),
    [
        ("/chat", {"X-API-Key": API_KEY}),
        ("/chat", {"Authorization": f"Bearer {API_KEY}"}),
        ("/v1/chat/completions", {"Authorization": f"Bearer {API_KEY}"}),
    ],
)
async def test_auth_enabled_accepts_valid_credentials(
    asgi_client, route, headers
) -> None:
    with wired_app(FakeLLM(), settings_overrides={"hyphae_api_key": API_KEY}) as (
        app,
        _settings,
    ):
        client = asgi_client(app)
        if route.startswith("/v1"):
            response = await client.post(
                route,
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers=headers,
            )
        else:
            response = await client.post(route, content="hello", headers=headers)
        assert response.status_code == 200


async def test_health_remains_open_when_auth_enabled(asgi_client) -> None:
    with wired_app(FakeLLM(), settings_overrides={"hyphae_api_key": API_KEY}) as (
        app,
        _settings,
    ):
        assert (await asgi_client(app).get("/health")).status_code == 200


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [("get", "/v1/not-a-route", 404), ("post", "/v1/models", 405)],
)
async def test_unrelated_openai_http_errors_preserve_their_status(
    asgi_client, method: str, path: str, status: int
) -> None:
    with wired_app(FakeLLM()) as (app, _settings):
        response = await getattr(asgi_client(app), method)(path)

    assert response.status_code == status


@pytest.mark.parametrize("route", ["/chat", "/chat/stream", "/v1/chat/completions"])
async def test_auth_rejection_precedes_body_materialization(asgi_client, route: str) -> None:
    with wired_app(FakeLLM(), settings_overrides={"hyphae_api_key": API_KEY}) as (
        app,
        _settings,
    ):
        response = await asgi_client(app).post(
            route,
            content=b"x" * (MAX_REQUEST_BODY_BYTES + 1),
        )

    assert response.status_code == 401
