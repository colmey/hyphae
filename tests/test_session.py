"""In-memory session and concurrency-guard tests."""

from __future__ import annotations

from datetime import timedelta

import pytest

from agent import (
    InMemorySessionStore,
    Session,
    SessionBusyError,
    SessionCapacityError,
    SessionGuard,
)
from agent.session import (
    SessionHistoryLimitExceeded,
    SessionNotFoundError,
    session_history_chars,
    validate_transcript,
)
from llm.schemas import (
    AssistantMessage,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


pytestmark = pytest.mark.anyio


async def test_append_assistant_text_owns_canonical_replay_shape() -> None:
    session = await InMemorySessionStore().create()

    message = session.append_assistant_text("replayed answer")

    assert message.role is Role.ASSISTANT
    assert len(message.content) == 1
    assert isinstance(message.content[0], TextBlock)
    assert message.content[0].text == "replayed answer"
    assert session.messages == [message]


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
    assert (
        pending[0].provider_metadata["thought_signature"] == b"<fake-signature-bytes>"
    )

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
    assert fetched is not session
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
    assert await store.get(created[-1].session_id) == created[-1]


async def test_store_evicts_stale_session_by_ttl() -> None:
    store = InMemorySessionStore(ttl_seconds=60, max_count=0)
    stale = await store.create()
    store._sessions[stale.session_id].updated_at -= timedelta(seconds=120)
    fresh = await store.create()

    with pytest.raises(SessionNotFoundError):
        await store.get(stale.session_id)
    assert await store.get(fresh.session_id) == fresh


async def test_store_capacity_keeps_claimed_session_and_evicts_unclaimed_lru() -> None:
    guard = SessionGuard()
    store = InMemorySessionStore(max_count=2)
    claimed = await store.create()
    unclaimed = await store.create()
    store._sessions[claimed.session_id].updated_at -= timedelta(seconds=2)
    store._sessions[unclaimed.session_id].updated_at -= timedelta(seconds=1)

    async with guard.claim(claimed.session_id):
        admitted = await store.create(protected_session_ids=guard.claimed_session_ids)

    assert set(store.ids()) == {claimed.session_id, admitted.session_id}
    with pytest.raises(SessionNotFoundError):
        await store.get(unclaimed.session_id)


async def test_store_ttl_and_capacity_keep_claimed_stale_session() -> None:
    guard = SessionGuard()
    store = InMemorySessionStore(ttl_seconds=1, max_count=2)
    claimed = await store.create()
    unclaimed = await store.create()
    store._sessions[claimed.session_id].updated_at -= timedelta(seconds=2)

    async with guard.claim(claimed.session_id):
        admitted = await store.create(protected_session_ids=guard.claimed_session_ids)

    assert set(store.ids()) == {claimed.session_id, admitted.session_id}
    with pytest.raises(SessionNotFoundError):
        await store.get(unclaimed.session_id)


async def test_existing_save_leaves_unrelated_stale_session_until_admission() -> None:
    store = InMemorySessionStore(ttl_seconds=1)
    existing = await store.create()
    stale = await store.create()
    store._sessions[stale.session_id].updated_at -= timedelta(seconds=2)

    await store.save(existing)

    assert set(store.ids()) == {existing.session_id, stale.session_id}


async def test_admission_supplier_failure_is_atomic() -> None:
    store = InMemorySessionStore(ttl_seconds=1, max_count=1)
    existing = await store.create()
    store._sessions[existing.session_id].updated_at -= timedelta(seconds=2)

    def unavailable_snapshot() -> frozenset[str]:
        raise RuntimeError("claim snapshot unavailable")

    with pytest.raises(RuntimeError, match="claim snapshot unavailable"):
        await store.create(protected_session_ids=unavailable_snapshot)

    assert store.ids() == [existing.session_id]


async def test_store_all_claimed_capacity_failure_is_atomic() -> None:
    guard = SessionGuard()
    store = InMemorySessionStore(max_count=2)
    first = await store.create()
    second = await store.create()

    async with guard.claim(first.session_id), guard.claim(second.session_id):
        with pytest.raises(SessionCapacityError, match="capacity temporarily unavailable"):
            await store.create(protected_session_ids=guard.claimed_session_ids)

    assert store.ids() == [first.session_id, second.session_id]
    assert len(store) == 2


async def test_missing_save_at_capacity_only_admits_when_an_unclaimed_victim_exists() -> None:
    guard = SessionGuard()
    store = InMemorySessionStore(max_count=2)
    claimed = await store.create()
    unclaimed = await store.create()
    replacement = Session()

    async with guard.claim(claimed.session_id), guard.claim(unclaimed.session_id):
        with pytest.raises(SessionCapacityError):
            await store.save(replacement, protected_session_ids=guard.claimed_session_ids)

    assert store.ids() == [claimed.session_id, unclaimed.session_id]
    async with guard.claim(claimed.session_id):
        await store.save(replacement, protected_session_ids=guard.claimed_session_ids)

    assert set(store.ids()) == {claimed.session_id, replacement.session_id}
    assert len(store) == 2


def test_transcript_validator_accepts_ordinary_and_complete_multi_tool_history() -> (
    None
):
    messages = [
        Message.user("go"),
        Message.assistant(
            [
                ToolUseBlock(id="one", name="same-tool", input={}),
                ToolUseBlock(id="two", name="same-tool", input={}),
            ]
        ),
        Message.tool_results(
            [
                ToolResultBlock(tool_use_id="one", name="same-tool", content="1"),
                ToolResultBlock(tool_use_id="two", name="same-tool", content="2"),
            ]
        ),
        Message.assistant([TextBlock("done")]),
        Message.user("thanks"),
    ]

    validate_transcript(messages)

    messages[2] = Message.tool_results(list(reversed(messages[2].content)))  # type: ignore[arg-type]
    validate_transcript(messages)


@pytest.mark.parametrize(
    "messages",
    [
        [Message.tool_results([ToolResultBlock("orphan", "tool", "x")])],
        [Message(role=Role.USER, content=[ToolUseBlock("call", "tool", {})])],
        [
            Message(
                role=Role.ASSISTANT,
                content=[ToolResultBlock("call", "tool", "result")],
            )
        ],
        [Message.assistant([ToolUseBlock("one", "tool", {})])],
        [
            Message.assistant(
                [ToolUseBlock("same", "one", {}), ToolUseBlock("same", "two", {})]
            ),
            Message.tool_results(
                [
                    ToolResultBlock("same", "one", "1"),
                    ToolResultBlock("same", "two", "2"),
                ]
            ),
        ],
        [
            Message.assistant([ToolUseBlock("one", "tool", {})]),
            Message.tool_results(
                [
                    ToolResultBlock("one", "tool", "1"),
                    ToolResultBlock("one", "tool", "2"),
                ]
            ),
        ],
        [
            Message.assistant([ToolUseBlock("one", "tool", {})]),
            Message.tool_results([ToolResultBlock("other", "tool", "1")]),
        ],
        [
            Message.assistant([ToolUseBlock("one", "tool", {})]),
            Message.tool_results([ToolResultBlock("one", "other", "1")]),
        ],
    ],
    ids=[
        "orphan",
        "tool-use-in-user",
        "tool-result-in-assistant",
        "missing",
        "duplicate-call-id",
        "duplicate-result",
        "mismatched-id",
        "mismatched-name",
    ],
)
def test_transcript_validator_rejects_malformed_tool_protocol(
    messages: list[Message],
) -> None:
    with pytest.raises(ValueError, match="transcript"):
        validate_transcript(messages)


async def test_store_create_get_and_save_are_deeply_detached() -> None:
    store = InMemorySessionStore()
    created = await store.create(metadata={"nested": {"values": [1]}})
    created.metadata["nested"]["values"].append(2)
    assert (await store.get(created.session_id)).metadata == {"nested": {"values": [1]}}

    created.append_assistant(
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id="call",
                    name="tool",
                    input={"nested": ["original"]},
                    provider_metadata={"signature": b"opaque", "nested": {"x": [1]}},
                )
            ]
        )
    )
    created.append_tool_results([ToolResultBlock("call", "tool", "ok")])
    await store.save(created)
    created.messages[0].content[0].input["nested"].append("caller")  # type: ignore[union-attr]
    created.messages[0].content[0].provider_metadata["nested"]["x"].append(2)  # type: ignore[union-attr]

    first = await store.get(created.session_id)
    block = first.messages[0].content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.input == {"nested": ["original"]}
    assert block.provider_metadata == {
        "signature": b"opaque",
        "nested": {"x": [1]},
    }
    first.messages[0].content[0].provider_metadata["nested"]["x"].append(3)  # type: ignore[union-attr]
    second = await store.get(created.session_id)
    assert second.messages[0].content[0].provider_metadata["nested"] == {"x": [1]}  # type: ignore[union-attr]


async def test_store_rejects_malformed_saved_and_restored_transcripts() -> None:
    store = InMemorySessionStore()
    session = await store.create()
    session.messages.append(
        Message.tool_results([ToolResultBlock("orphan", "tool", "result")])
    )
    with pytest.raises(ValueError, match="orphan"):
        await store.save(session)

    store._sessions[session.session_id].messages.append(
        Message.tool_results([ToolResultBlock("orphan", "tool", "result")])
    )
    with pytest.raises(ValueError, match="orphan"):
        await store.get(session.session_id)


async def test_store_rejects_restored_transcript_over_history_limit() -> None:
    store = InMemorySessionStore(session_history_max_chars=2)
    session = await store.create()
    store._sessions[session.session_id].append_user("oversized restored history")

    with pytest.raises(SessionHistoryLimitExceeded, match="session history limit"):
        await store.get(session.session_id)


async def test_store_accepts_exact_history_limit_and_rejects_one_character_over() -> (
    None
):
    candidate = Session()
    candidate.append_user("bounded")
    exact = session_history_chars(candidate.messages)
    store = InMemorySessionStore(session_history_max_chars=exact)
    stored = await store.create()
    candidate.session_id = stored.session_id

    await store.save(candidate)
    candidate.messages[0].content[0].text += "x"  # type: ignore[union-attr]
    with pytest.raises(SessionHistoryLimitExceeded, match="session history limit"):
        await store.save(candidate)
    assert (await store.get(stored.session_id)).messages[0].content[0].text == "bounded"  # type: ignore[union-attr]


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
        assert guard.claimed_session_ids() == frozenset({"sess_x"})
