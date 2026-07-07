"""
Smoke test for the optional API-key auth (api/dependencies.require_api_key).

Hermetic: the app is driven in-process via httpx + ASGITransport with app.state
wired by hand (fake LLM + empty fake MCP, orchestration off), mirroring
smoke_test_openai_api.py. Auth is toggled by setting app.state.settings.harness_api_key.

Scenarios:
  1. auth off (empty key)  - /chat and /v1/chat/completions behave as today (200).
  2. auth on, no/wrong key - 401 on the protected routes.
  3. auth on, correct key  - 200 via X-API-Key and via Authorization: Bearer.
  4. /health always open    - 200 regardless of the key.

Run from the project root:
    ./runscript.sh tests/smoke_test_auth.py
"""

from __future__ import annotations

from bootstrap import load_secrets
load_secrets()

import asyncio
from typing import Any

import httpx
from httpx import ASGITransport

from agent import InMemorySessionStore, SessionGuard
from harness_config import get_settings
from llm.client import LLMClient
from llm.schemas import AssistantMessage, TextBlock, Usage
from main import app
from orchestrator import LLMRegistry, load_models_config

_API_KEY = "s3cret-test-key"


class FakeLLM(LLMClient):
    async def complete(self, messages, tools=None, system=None, max_tokens=None,
                       response_schema=None, thinking_level=None) -> AssistantMessage:
        return AssistantMessage(
            content=[TextBlock(text="ok")], stop_reason="end_turn",
            model="fake", usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )


class FakeMCP:
    connected_servers: list[str] = []

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return []

    def list_tools(self) -> list[Any]:
        return []


def _wire_state() -> None:
    settings = get_settings()
    settings.harness_api_key = ""  # start with auth off
    app.state.settings = settings
    app.state.llm = FakeLLM()
    app.state.mcp = FakeMCP()
    app.state.store = InMemorySessionStore(
        ttl_seconds=settings.session_ttl_seconds,
        max_count=settings.session_max_count,
    )
    app.state.guard = SessionGuard()
    app.state.registry = LLMRegistry(load_models_config(settings.models_config_path), settings)
    app.state.orchestrator = None
    app.state.policy = None  # allow-all


_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


async def main() -> None:
    _wire_state()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://harness.local", timeout=30.0) as client:

        # 1. auth off: everything works without a key.
        print("--- auth off (empty key) ---")
        app.state.settings.harness_api_key = ""
        r = await client.post("/chat", content="hello")
        check(r.status_code == 200, f"/chat 200 with auth off (got {r.status_code})")
        r = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        check(r.status_code == 200, f"/v1 200 with auth off (got {r.status_code})")
        r = await client.get("/health")
        check(r.status_code == 200, f"/health 200 with auth off (got {r.status_code})")

        # 2. auth on: missing/wrong credentials are rejected.
        print("--- auth on: reject missing/wrong ---")
        app.state.settings.harness_api_key = _API_KEY
        r = await client.post("/chat", content="hello")
        check(r.status_code == 401, f"/chat 401 without a key (got {r.status_code})")
        r = await client.post("/chat", content="hello", headers={"X-API-Key": "wrong"})
        check(r.status_code == 401, f"/chat 401 with a wrong key (got {r.status_code})")
        # Non-ASCII key: sent as latin-1 bytes so Starlette decodes it to a
        # non-ASCII str on the wire (as a real client would). The bytes compare
        # must yield a clean 401, not a 500 (str compare_digest raises TypeError).
        r = await client.post("/chat", content="hello",
                              headers={"X-API-Key": "café".encode("latin-1")})
        check(r.status_code == 401, f"/chat 401 with a non-ASCII key (got {r.status_code})")
        r = await client.post("/v1/chat/completions",
                              json={"messages": [{"role": "user", "content": "hi"}]},
                              headers={"Authorization": "Bearer wrong"})
        check(r.status_code == 401, f"/v1 401 with a wrong bearer (got {r.status_code})")
        # /v1 auth failures must use the OpenAI error envelope, not FastAPI's {"detail":...}.
        body = r.json()
        check(isinstance(body.get("error"), dict) and isinstance(body["error"].get("message"), str),
              f"/v1 401 body is the OpenAI error envelope (got {body})")

        # 3. auth on: correct credentials pass, both header forms.
        print("--- auth on: accept correct key ---")
        r = await client.post("/chat", content="hello", headers={"X-API-Key": _API_KEY})
        check(r.status_code == 200, f"/chat 200 with X-API-Key (got {r.status_code})")
        r = await client.post("/chat", content="hello", headers={"Authorization": f"Bearer {_API_KEY}"})
        check(r.status_code == 200, f"/chat 200 with Bearer (got {r.status_code})")
        r = await client.post("/v1/chat/completions",
                              json={"messages": [{"role": "user", "content": "hi"}]},
                              headers={"Authorization": f"Bearer {_API_KEY}"})
        check(r.status_code == 200, f"/v1 200 with Bearer (OpenWebUI path) (got {r.status_code})")

        # 4. /health stays open even with auth on.
        print("--- /health open under auth ---")
        r = await client.get("/health")
        check(r.status_code == 200, f"/health 200 with auth on, no key (got {r.status_code})")

    print()
    if _failures:
        print(f"auth smoke test FAILED: {len(_failures)} check(s) failed.")
        raise SystemExit(1)
    print("auth smoke test complete: all checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
