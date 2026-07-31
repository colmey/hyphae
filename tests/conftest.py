"""Shared pytest support for hermetic and explicitly selected live tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx
import pytest
from agent import InMemorySessionStore, Session
from config import Settings, reset_settings

from tests.fakes import ScriptedLLM, ScriptedMCP, collect_agent_events


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
def isolate_settings_cache() -> Iterator[None]:
    """Prevent cached Settings instances from crossing test boundaries."""
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
