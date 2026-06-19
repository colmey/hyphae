"""
Smoke test for concurrency hardening.

Drives the live FastAPI app in-process (like smoke_test_http.py) to prove the
two guarantees the concurrency work added:

Scenarios:
  1. Many simultaneous /chat requests on DISTINCT sessions stay isolated:
     every response gets its own session_id and its own short history -- no
     crossover between concurrent requests.
  2. Concurrent /chat requests on the SAME session_id are guarded: exactly
     one runs (200) and the overlapping ones are rejected (409) by
     SessionGuard rather than racing on one message history.

Makes real LLM + MCP calls, so it needs the same bootstrap as the other
HTTP smoke tests.

Run from the project root:
    ./runscript.sh smoke_test_concurrency.py
"""

from __future__ import annotations

from bootstrap import load_secrets
load_secrets()

import asyncio
import logging

import httpx
from asgi_lifespan import LifespanManager
from httpx import ASGITransport

from main import app


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)


# Tool-free prompts: each resolves in a single turn (user + assistant), so an
# isolated session ends with exactly 2 messages.
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


async def main() -> None:
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
            responses = await asyncio.gather(*(
                client.post("/chat", json={"prompt": p})
                for p in DISTINCT_PROMPTS
            ))
            bodies = []
            for r in responses:
                r.raise_for_status()
                bodies.append(r.json())

            session_ids = [b["session_id"] for b in bodies]
            assert len(set(session_ids)) == len(session_ids), (
                f"expected all-distinct session_ids, got {session_ids}"
            )
            for b in bodies:
                # 2 == user + assistant. Anything larger would mean another
                # request's turns bled into this session.
                assert b["message_count"] == 2, (
                    f"session {b['session_id']} has {b['message_count']} messages; "
                    "expected 2 (history bled across concurrent requests?)"
                )
                assert b["done_reason"] == "end_turn", b["done_reason"]
            for sid, b in zip(session_ids, bodies):
                ans = b["response"].replace("\n", " ")[:40]
                print(f"  {sid}: msgs={b['message_count']} -> {ans!r}")
            print(f"  {len(bodies)} concurrent requests, all isolated.\n")

            # ----- Scenario 2: same session_id is guarded (409) -----
            print("=" * 72)
            print("Scenario 2: concurrent /chat on the SAME session_id -> 409 guard")
            print("=" * 72)
            seed = await client.post(
                "/chat", json={"prompt": "What is 1 + 1? Reply with just the number."}
            )
            seed.raise_for_status()
            session_id = seed.json()["session_id"]
            print(f"  seeded session: {session_id}")

            overlapping = await asyncio.gather(*(
                client.post("/chat", json={
                    "prompt": f"What is {i} + {i}? Reply with just the number.",
                    "commands": {"session": session_id},
                })
                for i in range(SAME_SESSION_FANOUT)
            ))
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
                    assert "processing another request" in r.json()["detail"]
            print("  same-session overlap correctly rejected with 409.\n")

            print("concurrency smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
