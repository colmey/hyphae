"""Shared pytest support for hermetic and explicitly selected live tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx
import pytest
import config
from agent import InMemorySessionStore, Session
from config import Settings, reset_settings

from tests.fakes import ScriptedLLM, ScriptedMCP, collect_agent_events


_REAL_LOAD_SECRETS = config.load_secrets


def _ignore_dotenv(*args: Any, **kwargs: Any) -> None:
    """Collection-safe replacement for the process-global dotenv loading."""


# conftest is imported before test modules. Neutralize dotenv at that boundary,
# not merely once fixtures begin, because application modules may be imported
# while pytest is collecting tests.
config.load_secrets = _ignore_dotenv


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """The harness is asyncio-native; do not run every async test under Trio."""
    return "asyncio"


@pytest.fixture
def isolated_settings() -> Settings:
    """Settings instance that never reads the developer's dotenv file."""
    return Settings(_env_file=None)


@pytest.fixture
async def agent_session() -> tuple[InMemorySessionStore, Session]:
    """Fresh store/session pair for tests that need to inspect persisted state."""
    store = InMemorySessionStore()
    session = await store.create()
    return store, session


@pytest.fixture
def scripted_llm_factory() -> type[ScriptedLLM]:
    return ScriptedLLM


@pytest.fixture
def scripted_mcp_factory() -> type[ScriptedMCP]:
    return ScriptedMCP


@pytest.fixture
def agent_event_collector() -> Callable[..., Any]:
    return collect_agent_events


@pytest.fixture(autouse=True)
def isolate_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Isolate settings and prevent hermetic tests from loading developer secrets."""
    loader = (
        _REAL_LOAD_SECRETS
        if request.node.get_closest_marker("live")
        else _ignore_dotenv
    )
    monkeypatch.setattr(config, "load_secrets", loader)
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def parse_sse() -> Callable[[str], list[Any]]:
    """Parse data-only SSE bodies, preserving the OpenAI ``[DONE]`` sentinel."""

    def parse(body: str) -> list[Any]:
        events: list[Any] = []
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            events.append(payload if payload == "[DONE]" else json.loads(payload))
        return events

    return parse


@pytest.fixture
async def asgi_client() -> AsyncIterator[Callable[[Any], httpx.AsyncClient]]:
    """Create in-process HTTP clients without starting a network listener."""
    clients: list[httpx.AsyncClient] = []

    def build(app: Any) -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://harness.test",
            timeout=30.0,
        )
        clients.append(client)
        return client

    yield build

    for client in clients:
        await client.aclose()
