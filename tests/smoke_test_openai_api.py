"""
Smoke test for the OpenAI-compatible adapter (POST /v1/chat/completions, GET /v1/models).

Hermetic by design: it verifies wire-format translation, not a live model. The
app is driven in-process via httpx + ASGITransport with app.state populated by
hand (a scripted fake LLM + an empty fake MCP, orchestration off), so the test
is deterministic and needs no real LLM/MCP backend. End-to-end behavior against
a real model is covered by pointing OpenWebUI's "OpenAI API" connection at /v1.

Scenarios:
  1. GET  /v1/models                       -> OpenAI list shape from the registry
  2. POST /v1/chat/completions (no stream) -> chat.completion shape + usage
  3. POST /v1/chat/completions (stream)    -> chat.completion.chunk deltas + [DONE]
  4. POST /v1/chat/completions, no messages -> OpenAI-style 400 error JSON

Run from the project root:
    ./runscript.sh tests/smoke_test_openai_api.py
"""

from __future__ import annotations

from bootstrap import load_secrets
load_secrets()

import asyncio
import json
from typing import Any, AsyncIterator

import httpx
from httpx import ASGITransport

from agent import InMemorySessionStore, SessionGuard
from harness_config import get_settings
from llm.client import LLMClient
from llm.schemas import AssistantMessage, StreamChunk, StreamEnd, TextBlock, TextDelta, Usage
from main import app
from orchestrator import LLMRegistry, load_models_config


# --- scripted fakes --------------------------------------------------------

class FakeLLM(LLMClient):
    """Returns a fixed two-block answer, no tool calls -> a clean end_turn."""

    def __init__(self) -> None:
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        self.complete_calls += 1
        return AssistantMessage(
            content=[TextBlock(text="Hello"), TextBlock(text=", world")],
            stop_reason="end_turn",
            model="fake",
            usage=Usage(input_tokens=11, output_tokens=3, total_tokens=14),
        )

    async def stream(self, messages, tools=None, system=None, max_tokens=None,
                     thinking_level=None) -> AsyncIterator[StreamChunk]:
        self.stream_calls += 1
        yield TextDelta(text="Hello")
        yield TextDelta(text=", ")
        yield TextDelta(text="world")
        yield StreamEnd(AssistantMessage(
            content=[TextBlock(text="Hello, world")],
            stop_reason="end_turn",
            model="fake",
            usage=Usage(input_tokens=11, output_tokens=3, total_tokens=14),
        ))


class FakeMCP:
    """Empty MCP: the fake model never calls a tool, so an empty inventory is fine."""
    connected_servers: list[str] = []

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return []

    def list_tools(self) -> list[Any]:
        return []


def _wire_state() -> None:
    """Populate app.state by hand (no lifespan): fakes + a real registry, orchestration off."""
    settings = get_settings()
    app.state.settings = settings
    app.state.llm = FakeLLM()
    app.state.mcp = FakeMCP()
    app.state.store = InMemorySessionStore(
        ttl_seconds=settings.session_ttl_seconds,
        max_count=settings.session_max_count,
    )
    app.state.guard = SessionGuard()
    # Registry present so /v1/models lists it; orchestrator None keeps routing in
    # legacy mode (the fake LLM), and model_ids never builds a client.
    app.state.registry = LLMRegistry(load_models_config(settings.models_config_path), settings)
    app.state.orchestrator = None


def _parse_sse(body: str) -> list[str]:
    """Return the payloads of every `data:` line in an SSE response."""
    return [line[len("data:"):].strip() for line in body.splitlines() if line.startswith("data:")]


async def main() -> None:
    _wire_state()
    expected_models = app.state.registry.model_ids
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://harness.local", timeout=30.0) as client:

        # ----- /v1/models -----
        print("=" * 72)
        print("Scenario 1: GET /v1/models")
        print("=" * 72)
        r = await client.get("/v1/models")
        r.raise_for_status()
        body = r.json()
        print(json.dumps(body, indent=2))
        assert body["object"] == "list"
        listed = [m["id"] for m in body["data"]]
        assert listed == expected_models, f"expected {expected_models}, got {listed}"
        assert all(m["object"] == "model" for m in body["data"])
        print()

        # ----- non-stream completion -----
        print("=" * 72)
        print("Scenario 2: POST /v1/chat/completions (non-stream)")
        print("=" * 72)
        r = await client.post("/v1/chat/completions", json={
            "model": "whatever",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Hi there"},
            ],
        })
        r.raise_for_status()
        body = r.json()
        print(json.dumps(body, indent=2))
        assert body["object"] == "chat.completion"
        assert body["model"] == "whatever"
        choice = body["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Hello, world"
        assert choice["finish_reason"] == "stop"
        usage = body["usage"]
        assert usage["prompt_tokens"] == 11 and usage["completion_tokens"] == 3
        assert usage["total_tokens"] == 14
        print()

        # ----- streaming completion -----
        print("=" * 72)
        print("Scenario 3: POST /v1/chat/completions (stream=true)")
        print("=" * 72)
        async with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "temperature": 0.7,  # unknown-but-tolerated field
        }) as resp:
            assert resp.status_code == 200
            raw = "".join([chunk async for chunk in resp.aiter_text()])
        payloads = _parse_sse(raw)
        print("\n".join(payloads))
        assert payloads[-1] == "[DONE]", "stream must terminate with [DONE]"
        frames = [json.loads(p) for p in payloads if p != "[DONE]"]
        assert all(f["object"] == "chat.completion.chunk" for f in frames)
        assert frames[0]["choices"][0]["delta"].get("role") == "assistant"
        deltas = [f["choices"][0]["delta"].get("content", "") for f in frames]
        assert deltas.count("Hello") == 1 and ", " in deltas and "world" in deltas, deltas
        assert "".join(deltas) == "Hello, world", f"reassembled deltas wrong: {deltas}"
        assert frames[-1]["choices"][0]["finish_reason"] == "stop"
        assert app.state.llm.stream_calls == 1, "stream:true must route through LLMClient.stream()"
        print()

        # ----- graceful error -----
        print("=" * 72)
        print("Scenario 4: POST /v1/chat/completions with no messages -> 400 error JSON")
        print("=" * 72)
        r = await client.post("/v1/chat/completions", json={"messages": []})
        print(f"  status: {r.status_code}")
        print(json.dumps(r.json(), indent=2))
        assert r.status_code == 400
        assert "error" in r.json() and r.json()["error"]["type"] == "invalid_request_error"
        print()

        print("openai-compat smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
