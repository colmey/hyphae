"""Live pytest coverage for configured-request concurrency hardening.

Drives the live FastAPI app in-process (like smoke_test_http.py) to prove the
two guarantees the concurrency work added:

Scenarios:
  1. Many simultaneous /chat requests on DISTINCT sessions stay isolated:
     every response gets its own session_id and its own short history -- no
     crossover between concurrent requests.
  2. Concurrent /chat requests on the SAME session_id are guarded: exactly
     one runs (200) and the overlapping ones are rejected (409) by
     SessionGuard rather than racing on one message history.

Makes real LLM + MCP calls, so Settings loads the configured project ``.env``
when lifespan starts.

Run explicitly with ``./runscript.sh -m pytest -m "live and http_server"``.
"""

from __future__ import annotations

import asyncio
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


# Tool-free prompts: each resolves in a single turn (user + assistant), so an
# isolated native session ends with exactly 2 messages.
DISTINCT_PROMPTS = [
    "What is 11 + 11? Reply with just the number.",
    "What is 22 + 22? Reply with just the number.",
    "What is 33 + 33? Reply with just the number.",
    "What is 44 + 44? Reply with just the number.",
]

# How many overlapping requests to fire at one session. With several inflight
# at once, the claim winner holds the guard across its loop, so the rest are
# rejected.
SAME_SESSION_FANOUT = 4


async def test_configured_request_concurrency() -> None:
    reset_settings()
    transport = ASGITransport(app=app)
    base_url = "http://harness.local"

    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=transport, base_url=base_url, timeout=120.0
        ) as client:
            # ----- Scenario 1: distinct sessions stay isolated -----
            print("=" * 72)
            print("Scenario 1: concurrent /chat on DISTINCT sessions -> isolated")
            print("=" * 72)
            responses = await asyncio.gather(
                *(client.post("/chat", content=p) for p in DISTINCT_PROMPTS)
            )
            for r in responses:
                r.raise_for_status()

            session_ids = [r.headers["X-Session-Id"] for r in responses]
            assert len(set(session_ids)) == len(session_ids), (
                f"expected all-distinct session_ids, got {session_ids}"
            )
            for session_id, response in zip(session_ids, responses):
                # 2 == user + assistant. Anything larger would mean another
                # request's turns bled into this session.
                session = await app.state.runtime.store.get(session_id)
                assert len(session.messages) == 2, (
                    f"session {session_id} has {len(session.messages)} messages; "
                    "expected 2 (history bled across concurrent requests?)"
                )
                assert response.headers["X-Done-Reason"] == "end_turn"
                assert response.text.strip()
                ans = response.text.replace("\n", " ")[:40]
                print(f"  {session_id}: msgs={len(session.messages)} -> {ans!r}")
            print(f"  {len(responses)} concurrent requests, all isolated.\n")

            # ----- Scenario 2: same session_id is guarded (409) -----
            print("=" * 72)
            print("Scenario 2: concurrent /chat on the SAME session_id -> 409 guard")
            print("=" * 72)
            seed = await client.post(
                "/chat", content="What is 1 + 1? Reply with just the number."
            )
            seed.raise_for_status()
            session_id = seed.headers["X-Session-Id"]
            print(f"  seeded session: {session_id}")

            overlapping = await asyncio.gather(
                *(
                    client.post(
                        "/chat",
                        content=f"What is {i} + {i}? Reply with just the number.",
                        headers={"X-Session-Id": session_id},
                    )
                    for i in range(SAME_SESSION_FANOUT)
                )
            )
            statuses = [r.status_code for r in overlapping]
            n_ok = statuses.count(200)
            n_busy = statuses.count(409)
            print(f"  statuses: {sorted(statuses)} (200x{n_ok}, 409x{n_busy})")

            # The guard must reject at least one overlapping request and let at
            # least one through. Every response is either accepted or a clean
            # 409 -- never a 5xx or a raced result.
            assert set(statuses) <= {200, 409}, (
                f"unexpected status among {statuses}; guard should only 200/409"
            )
            assert n_busy >= 1, (
                "expected at least one 409; SessionGuard did not reject the "
                "concurrent same-session request"
            )
            assert n_ok >= 1, "expected at least one 200 to win the claim"
            for r in overlapping:
                if r.status_code == 409:
                    assert r.json() == {
                        "code": "session_busy",
                        "message": "Session is processing another request.",
                    }
                else:
                    assert r.headers["X-Session-Id"] == session_id
                    assert r.headers["X-Done-Reason"] == "end_turn"
                    assert r.text.strip()
            print("  same-session overlap correctly rejected with 409.\n")

            print("concurrency smoke test passed.")
