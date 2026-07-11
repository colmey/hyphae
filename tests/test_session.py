"""In-memory session and concurrency-guard tests."""

from __future__ import annotations

from datetime import timedelta

import pytest

from agent import InMemorySessionStore, SessionBusyError, SessionGuard
from agent.session import SessionNotFoundError
from llm.schemas import AssistantMessage, TextBlock, ToolResultBlock, ToolUseBlock


pytestmark = pytest.mark.anyio


async def test_session_round_trip_preserves_all_message_kinds() -> None:
    store = InMemorySessionStore()
    session = await store.create(metadata={"user": "pytest"})
    assert session.session_id.startswith("sess_")
    assert session.turn_count() == 0

    session.append_user("List the tables available in the customer database.")
    session.append_assistant(
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id="call_demo_1",
                    name="my-toolbox__list-tables",
                    input={},
                    provider_metadata={"thought_signature": b"<fake-signature-bytes>"},
                )
            ],
            stop_reason="tool_use",
            model="fake",
        )
    )
    pending = session.last_assistant_tool_uses()
    assert len(pending) == 1
    assert pending[0].id == "call_demo_1"
    assert pending[0].provider_metadata["thought_signature"] == b"<fake-signature-bytes>"

    session.append_tool_results(
        [
            ToolResultBlock(
                tool_use_id=pending[0].id,
                name=pending[0].name,
                content='{"TABLE_NAME":"customers"}\n{"TABLE_NAME":"orders"}',
                is_error=False,
            )
        ]
    )
    session.append_assistant(
        AssistantMessage(
            content=[TextBlock(text="The database contains customers and orders.")],
            stop_reason="end_turn",
            model="fake",
        )
    )

    await store.save(session)
    fetched = await store.get(session.session_id)
    assert fetched is session
    assert len(fetched.messages) == 4
    assert fetched.turn_count() == 1
    assert len(store) == 1


async def test_unknown_session_raises() -> None:
    with pytest.raises(SessionNotFoundError):
        await InMemorySessionStore().get("sess_doesnotexist")


async def test_sessions_are_isolated() -> None:
    store = InMemorySessionStore()
    first = await store.create(metadata={"user": "one"})
    second = await store.create(metadata={"user": "two"})
    first.append_user("first")
    second.append_user("second")

    assert first.session_id != second.session_id
    assert first.messages[0].content[0].text == "first"
    assert second.messages[0].content[0].text == "second"


async def test_store_evicts_oldest_session_at_max_count() -> None:
    store = InMemorySessionStore(ttl_seconds=0, max_count=3)
    created = [await store.create() for _ in range(5)]

    assert len(store) <= 3
    with pytest.raises(SessionNotFoundError):
        await store.get(created[0].session_id)
    assert await store.get(created[-1].session_id) is created[-1]


async def test_store_evicts_stale_session_by_ttl() -> None:
    store = InMemorySessionStore(ttl_seconds=60, max_count=0)
    stale = await store.create()
    stale.updated_at -= timedelta(seconds=120)
    fresh = await store.create()

    with pytest.raises(SessionNotFoundError):
        await store.get(stale.session_id)
    assert await store.get(fresh.session_id) is fresh


async def test_session_guard_rejects_same_id_and_releases_claim() -> None:
    guard = SessionGuard()
    async with guard.claim("sess_x"):
        with pytest.raises(SessionBusyError):
            async with guard.claim("sess_x"):
                pass
        async with guard.claim("sess_y"):
            assert guard.in_flight() == {"sess_x", "sess_y"}

    assert guard.in_flight() == set()
    async with guard.claim("sess_x"):
        assert guard.in_flight() == {"sess_x"}
