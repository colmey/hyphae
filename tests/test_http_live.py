"""Live pytest coverage for the configured HTTP surface.

Spins up the FastAPI app in-process via httpx.AsyncClient + ASGITransport,
with asgi_lifespan.LifespanManager driving the lifespan protocol (httpx
does not run lifespan events on its own).

The chat endpoint is plain-text in / plain-text out: the request body is the
prompt, the response body is the answer, and session continuation rides on the
`X-Session-Id` header (echoed back, with `X-Done-Reason`). There is no JSON
request/response schema to validate -- tool selection is covered by
smoke_test_orchestrator.py.

Scenarios:
  1. GET /health (still JSON)
  2. POST /chat, no session -> creates one; plain-text answer + X-Session-Id
  3. POST /chat, continue via X-Session-Id; a prompt that needs a tool
  4. POST /chat with an unknown session header -> 404
  5. POST /chat with an empty body -> 400

Run explicitly with ``./runscript.sh -m pytest -m "live and http_server"``.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport

from config import reset_settings
from main import app


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
pytestmark = [
    pytest.mark.live,
    pytest.mark.model,
    pytest.mark.mcp,
    pytest.mark.http_server,
    pytest.mark.anyio,
]


def _print_chat(label: str, r: httpx.Response) -> None:
    answer = r.text.replace("\n", " ")
    if len(answer) > 200:
        answer = answer[:197] + "..."
    print(f"--- {label} ---")
    print(f"  session:     {r.headers.get('X-Session-Id')}")
    print(f"  done_reason: {r.headers.get('X-Done-Reason')}")
    print(f"  answer:      {answer}\n")


async def test_configured_http_surface() -> None:
    reset_settings()
    transport = ASGITransport(app=app)
    base_url = "http://harness.local"

    # httpx does NOT trigger ASGI lifespan events on its own; LifespanManager
    # runs the FastAPI lifespan so app.state is populated before any request.
    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=transport, base_url=base_url, timeout=120.0
        ) as client:
            # ----- /health -----
            print("=" * 72)
            print("Scenario 1: GET /health")
            print("=" * 72)
            r = await client.get("/health")
            r.raise_for_status()
            health = r.json()
            print(json.dumps(health, indent=2))
            assert health["status"] == "ok"
            assert health["tool_count"] > 0, "expected at least one MCP tool"
            print()

            # ----- /chat (no session) -----
            print("=" * 72)
            print("Scenario 2: POST /chat, no session, trivial prompt")
            print("=" * 72)
            r = await client.post("/chat", content="What is 5 + 7? Just the number.")
            r.raise_for_status()
            _print_chat("trivial", r)
            assert r.headers["content-type"].startswith("text/plain")
            session_id = r.headers.get("X-Session-Id", "")
            assert session_id.startswith("sess_"), "expected an X-Session-Id header"
            assert r.headers.get("X-Done-Reason") == "end_turn"
            assert "12" in r.text, f"expected '12' in the answer, got {r.text!r}"

            # ----- /chat (continue session, requires tool) -----
            print("=" * 72)
            print("Scenario 3: POST /chat, continue session, prompt requires a tool")
            print("=" * 72)
            r = await client.post(
                "/chat",
                content="Now list the tables in the customer database.",
                headers={"X-Session-Id": session_id},
            )
            r.raise_for_status()
            _print_chat("tool-using continuation", r)
            assert r.headers.get("X-Session-Id") == session_id, "session must continue"
            assert r.text.strip(), "expected a non-empty answer"

            # ----- 404 on unknown session -----
            print("=" * 72)
            print("Scenario 4: POST /chat with unknown session -> 404")
            print("=" * 72)
            r = await client.post(
                "/chat", content="hi", headers={"X-Session-Id": "sess_doesnotexist"}
            )
            print(f"  status: {r.status_code}")
            assert r.status_code == 404
            print()

            # ----- 400 on empty body -----
            print("=" * 72)
            print("Scenario 5: POST /chat with an empty body -> 400")
            print("=" * 72)
            r = await client.post("/chat", content="")
            print(f"  status: {r.status_code}")
            assert r.status_code == 400, "expected 400 for an empty prompt body"
            print()

            print("http smoke test passed.")
