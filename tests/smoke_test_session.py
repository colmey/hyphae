"""
Smoke test for step 4 of the build.

Exercises the in-memory session store: create, append messages of every
kind, save, re-fetch, verify content survives the round-trip. No LLM, no
MCP, no network.

Run from the project root:
    ./runscript.sh smoke_test_session.py
or just
    python smoke_test_session.py
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from agent import InMemorySessionStore, Session, SessionBusyError, SessionGuard
from agent.session import SessionNotFoundError
from llm.schemas import (
    AssistantMessage,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def _print_session(label: str, session: Session) -> None:
    print(f"--- {label} ---")
    print(f"  session_id:  {session.session_id}")
    print(f"  created_at:  {session.created_at.isoformat()}")
    print(f"  updated_at:  {session.updated_at.isoformat()}")
    print(f"  metadata:    {session.metadata}")
    print(f"  turn_count:  {session.turn_count()}")
    print(f"  messages:    {len(session.messages)}")
    for i, msg in enumerate(session.messages):
        block_summary = []
        for b in msg.content:
            if isinstance(b, TextBlock):
                preview = b.text.replace("\n", " ")
                if len(preview) > 60:
                    preview = preview[:57] + "..."
                block_summary.append(f"text({preview!r})")
            elif isinstance(b, ToolUseBlock):
                block_summary.append(f"tool_use(name={b.name}, id={b.id})")
            elif isinstance(b, ToolResultBlock):
                block_summary.append(
                    f"tool_result(id={b.tool_use_id}, name={b.name}, "
                    f"is_error={b.is_error})"
                )
        print(f"    [{i}] role={msg.role.value:10s} {' '.join(block_summary)}")
    print()


async def main() -> None:
    store = InMemorySessionStore()

    # ----- create -----
    print("=" * 70)
    print("Create + populate a session")
    print("=" * 70)
    session = await store.create(metadata={"user": "smoke-test"})
    _print_session("after create", session)
    assert session.session_id.startswith("sess_")
    assert session.turn_count() == 0

    # ----- append a user message -----
    session.append_user("List the tables available in the customer database.")

    # ----- append an assistant turn with a tool call -----
    tool_use = ToolUseBlock(
        id="call_demo_1",
        name="my-toolbox__list-tables",
        input={},
        provider_metadata={"thought_signature": b"<fake-signature-bytes>"},
    )
    session.append_assistant(AssistantMessage(
        content=[tool_use],
        stop_reason="tool_use",
        model="gemini-3-flash-preview",
    ))

    # ----- introspection: last tool uses -----
    pending = session.last_assistant_tool_uses()
    assert len(pending) == 1, f"expected 1 pending tool use, got {len(pending)}"
    assert pending[0].id == "call_demo_1"
    assert pending[0].provider_metadata.get("thought_signature") == b"<fake-signature-bytes>", (
        "provider_metadata did not round-trip through the session"
    )
    print(f"detected {len(pending)} pending tool use(s); signature preserved.\n")

    # ----- append tool result -----
    session.append_tool_results([
        ToolResultBlock(
            tool_use_id=pending[0].id,
            name=pending[0].name,
            content='{"TABLE_NAME":"customers"}\n{"TABLE_NAME":"orders"}',
            is_error=False,
        )
    ])

    # ----- append final assistant text turn -----
    session.append_assistant(AssistantMessage(
        content=[TextBlock(text="The customer database contains tables including customers and orders.")],
        stop_reason="end_turn",
        model="gemini-3-flash-preview",
    ))

    _print_session("after full turn", session)

    # ----- save + fetch -----
    print("=" * 70)
    print("Save and round-trip via the store")
    print("=" * 70)
    await store.save(session)
    fetched = await store.get(session.session_id)
    assert fetched is session, "in-memory store should return the same instance"
    assert len(fetched.messages) == 4
    assert fetched.turn_count() == 1
    print(f"store now holds {len(store)} session(s): {store.ids()}")
    print(f"fetched session matches original: {fetched.session_id == session.session_id}")
    print()

    # ----- unknown session -----
    print("=" * 70)
    print("Negative test: unknown session_id raises SessionNotFoundError")
    print("=" * 70)
    try:
        await store.get("sess_doesnotexist")
    except SessionNotFoundError as e:
        print(f"  raised as expected: {type(e).__name__}({e!s})")
    else:
        raise AssertionError("expected SessionNotFoundError")
    print()

    # ----- second independent session -----
    print("=" * 70)
    print("Second session is isolated from the first")
    print("=" * 70)
    second = await store.create(metadata={"user": "someone-else"})
    second.append_user("hi")
    assert second.session_id != session.session_id
    assert len(second.messages) == 1
    assert len(session.messages) == 4
    print(f"  first  : {session.session_id} ({len(session.messages)} msgs)")
    print(f"  second : {second.session_id} ({len(second.messages)} msgs)")
    print(f"  store  : {len(store)} sessions total")
    print()

    await _test_eviction()
    await _test_guard()

    print("step 4 smoke test passed.")


async def _test_eviction() -> None:
    """InMemorySessionStore stays bounded by max-size and TTL."""
    print("=" * 70)
    print("Eviction: TTL + max-size keep the in-memory store bounded")
    print("=" * 70)

    # ----- max-size (TTL disabled) -----
    store = InMemorySessionStore(ttl_seconds=0, max_count=3)
    created = [await store.create() for _ in range(5)]
    assert len(store) <= 3, f"expected <= 3 sessions, got {len(store)}"
    # Oldest-created (oldest updated_at) must have been evicted first.
    try:
        await store.get(created[0].session_id)
    except SessionNotFoundError:
        pass
    else:
        raise AssertionError("oldest session should have been evicted by max_count")
    # Newest must still be present.
    assert await store.get(created[-1].session_id) is created[-1]
    print(f"  max_count=3 honored: store holds {len(store)} after creating 5")

    # ----- TTL (max-size disabled) -----
    store = InMemorySessionStore(ttl_seconds=60, max_count=0)
    stale = await store.create()
    # Backdate past the TTL window; the next create() sweeps it.
    stale.updated_at = stale.updated_at - timedelta(seconds=120)
    fresh = await store.create()
    try:
        await store.get(stale.session_id)
    except SessionNotFoundError:
        pass
    else:
        raise AssertionError("stale session should have been evicted by TTL")
    assert await store.get(fresh.session_id) is fresh
    print("  ttl honored: stale session evicted on next create()")
    print()


async def _test_guard() -> None:
    """SessionGuard rejects a second concurrent claim of the same id."""
    print("=" * 70)
    print("SessionGuard: reject-if-busy for the same session_id")
    print("=" * 70)

    guard = SessionGuard()
    async with guard.claim("sess_x"):
        # A second claim of the SAME id is rejected.
        try:
            async with guard.claim("sess_x"):
                raise AssertionError("second claim of same id should raise")
        except SessionBusyError:
            pass
        # A DIFFERENT id is unaffected — distinct sessions run freely.
        async with guard.claim("sess_y"):
            assert guard.in_flight() == {"sess_x", "sess_y"}

    # Released on exit, and re-claimable afterwards.
    assert guard.in_flight() == set()
    async with guard.claim("sess_x"):
        pass
    print("  same-id rejected, distinct ids allowed, claim released on exit")
    print()


if __name__ == "__main__":
    asyncio.run(main())