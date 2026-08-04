# agent/session.py

"""
Session: conversation state for one agentic interaction.

A Session is the thing the agent loop reads from and appends to across
iterations. It holds the message history in canonical internal types
(see llm.schemas) plus a stable session_id and a creation timestamp.

`SessionStore` is the ABC that persistence implementations satisfy. The
in-memory implementation is `InMemorySessionStore` — a bounded dict. The
harness keeps conversation context client-side (LibreChat re-feeds it), so
sessions are short-lived scratchpads and durable persistence isn't needed;
the ABC is retained purely as the seam for swapping in SQLite/Postgres/Redis
later (one file, one line in `main.py`'s lifespan).

Concurrency: distinct sessions are fully isolated. Each accepted persistent
request resolves the latest stored checkpoint by ID and stages a copy before
appending its prompt. `SessionGuard` rejects (HTTP 409) a second in-flight
request for the same session ID.

Design notes:
  - The store interface is async even though the in-memory implementation
    doesn't need it. This keeps the call sites unchanged when we swap to a
    real DB.
  - `InMemorySessionStore` evicts on TTL + max-size and rejects retained
    transcripts over its positive character bound. Eviction is lazy (swept
    on create), not a background task.
  - `SessionGuard` is an in-process reject-if-busy guard. It is lock-free: on
    the single-threaded asyncio loop a check-then-add with no `await` between
    is atomic. Switching to wait-semantics later means swapping the in-flight
    `set` for a `dict[str, asyncio.Lock]`.
  - `save()` is explicit rather than auto-on-mutate. The loop publishes only
    protocol-safe checkpoints: a complete non-tool assistant response, or a
    tool-use message followed by exactly one result per call. After an
    intermediate checkpoint it continues on another staged copy. Store
    create/get/save boundaries also deep-copy canonical nested values, so
    caller mutation cannot alter a retained checkpoint.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

from llm.schemas import (
    AssistantMessage,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def _new_session_id() -> str:
    """Generate a fresh session ID. Short UUID; collisions are not a concern at our scale."""
    return f"sess_{uuid.uuid4().hex[:16]}"


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class SessionHistoryLimitExceeded(ValueError):
    """The complete retained transcript cannot fit within its configured bound."""

    def __init__(self) -> None:
        super().__init__("session history limit reached; start a new session")


class TranscriptValidationError(ValueError):
    """Canonical retained history violates the tool-call/result protocol."""


def session_history_chars(messages: Sequence[Message]) -> int:
    """Measure the deterministic in-memory canonical representation.

    Canonical messages contain dataclasses, JSON-shaped collections, and opaque
    provider bytes. Their Python representation includes every retained field
    without introducing a second serialization format or changing byte values.
    """
    return len(repr(list(messages)))


def validate_transcript(
    messages: Sequence[Message],
    *,
    allow_pending: bool = False,
) -> None:
    """Validate canonical assistant tool-call/result protocol units."""
    pending: tuple[ToolUseBlock, ...] | None = None
    for message in messages:
        tool_uses_in_message = tuple(
            block for block in message.content if isinstance(block, ToolUseBlock)
        )
        tool_results_in_message = tuple(
            block for block in message.content if isinstance(block, ToolResultBlock)
        )
        if tool_uses_in_message and message.role is not Role.ASSISTANT:
            raise TranscriptValidationError(
                "invalid transcript: tool calls may appear only in assistant messages"
            )
        if tool_results_in_message and message.role is not Role.TOOL:
            raise TranscriptValidationError(
                "invalid transcript: tool results may appear only in tool messages"
            )

        if pending is not None:
            if message.role is not Role.TOOL:
                raise TranscriptValidationError(
                    "invalid transcript: assistant tool calls require one following result message"
                )
            if not all(isinstance(block, ToolResultBlock) for block in message.content):
                raise TranscriptValidationError(
                    "invalid transcript: tool messages may contain only tool results"
                )
            results = tuple(
                block for block in message.content if isinstance(block, ToolResultBlock)
            )
            if len(results) != len(pending):
                raise TranscriptValidationError(
                    "invalid transcript: tool calls require exactly one result each"
                )
            expected = {tool_use.id: tool_use.name for tool_use in pending}
            actual_ids = [result.tool_use_id for result in results]
            if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(
                expected
            ):
                raise TranscriptValidationError(
                    "invalid transcript: tool result IDs and names must match their calls"
                )
            if any(expected[result.tool_use_id] != result.name for result in results):
                raise TranscriptValidationError(
                    "invalid transcript: tool result IDs and names must match their calls"
                )
            pending = None
            continue

        if message.role is Role.TOOL:
            raise TranscriptValidationError(
                "invalid transcript: orphan tool result message"
            )
        if message.role is not Role.ASSISTANT:
            continue

        tool_uses = tool_uses_in_message
        if not tool_uses:
            continue
        ids = [tool_use.id for tool_use in tool_uses]
        if len(ids) != len(set(ids)):
            raise TranscriptValidationError(
                "invalid transcript: assistant tool-call IDs must be unique"
            )
        pending = tool_uses

    if pending is not None and not allow_pending:
        raise TranscriptValidationError(
            "invalid transcript: assistant tool calls require one following result message"
        )


@dataclass
class Session:
    """One conversation. Owned by the agent loop while a request is in flight."""

    session_id: str = field(default_factory=_new_session_id)
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    # Free-form bag for whatever the loop or API surface wants to attach
    # (e.g. user_id, system prompt overrides, tags). The store persists it
    # opaquely.
    metadata: dict[str, Any] = field(default_factory=dict)

    def staged_copy(self) -> "Session":
        """Return a copy-on-write view suitable for uncommitted work.

        Canonical messages and their content blocks are treated as immutable,
        so only the message list and mutable metadata bag are copied. A saved
        checkpoint must never be mutated again; callers that continue after a
        save stage another copy first.
        """
        return Session(
            session_id=self.session_id,
            messages=list(self.messages),
            created_at=self.created_at,
            updated_at=self.updated_at,
            metadata=dict(self.metadata),
        )

    # ----- mutation helpers -----
    #
    # The agent loop should call these rather than poking `.messages`
    # directly. That gives us one place to update `updated_at` and one place
    # to enforce shape rules later (e.g. validating that tool_results match
    # the most recent assistant turn's tool_uses).

    def append_user(self, text: str) -> Message:
        msg = Message.user(text)
        self.messages.append(msg)
        self.updated_at = _utc_now()
        return msg

    def append_assistant(self, response: AssistantMessage) -> Message:
        """Append an AssistantMessage's content as a single assistant Message."""
        msg = response.to_message()
        self.messages.append(msg)
        self.updated_at = _utc_now()
        return msg

    def append_assistant_text(self, text: str) -> Message:
        """Append replayed assistant text in the canonical session shape."""
        msg = Message.assistant([TextBlock(text=text)])
        self.messages.append(msg)
        self.updated_at = _utc_now()
        return msg

    def append_tool_results(self, results: list[ToolResultBlock]) -> Message:
        if not results:
            raise ValueError("append_tool_results requires at least one result")
        msg = Message.tool_results(results)
        self.messages.append(msg)
        self.updated_at = _utc_now()
        return msg

    # ----- inspection -----

    def last_assistant_tool_uses(self) -> list[ToolUseBlock]:
        """Return the tool_use blocks from the most recent assistant turn, if any."""
        for msg in reversed(self.messages):
            if msg.role == Role.ASSISTANT:
                return [b for b in msg.content if isinstance(b, ToolUseBlock)]
            # If the most recent assistant turn is further back than the
            # latest tool/user message, walk past those silently.
        return []

    def turn_count(self) -> int:
        """How many user/assistant exchanges this session contains (rough metric)."""
        return sum(1 for m in self.messages if m.role == Role.USER)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SessionNotFoundError(KeyError):
    """Raised when get() is called with an unknown session_id."""


class SessionStore(ABC):
    """Persistence interface for Session objects.

    Async even though the in-memory backend doesn't need to be — keeps the
    call sites unchanged when we swap in SQLite/Postgres/Redis.
    """

    @abstractmethod
    async def create(self, metadata: dict[str, Any] | None = None) -> Session:
        """Create and persist a new Session, return it."""
        ...

    @abstractmethod
    async def get(self, session_id: str) -> Session:
        """Return the session with this id, or raise SessionNotFoundError."""
        ...

    @abstractmethod
    async def save(self, session: Session) -> None:
        """Persist (or re-persist) the given session."""
        ...

    @property
    @abstractmethod
    def session_history_max_chars(self) -> int:
        """Maximum complete retained transcript size for persistent sessions."""
        ...


class InMemorySessionStore(SessionStore):
    """Dict-backed store, bounded by TTL and max-size. Process-lifetime only.

    Because the harness keeps no durable copy, an unbounded dict would leak
    under many users. Eviction keeps memory flat:
      - ttl_seconds: sessions idle longer than this are dropped. Active
        sessions stay young (save() bumps updated_at every turn), so an
        in-flight session is never TTL-evicted.
      - max_count: if over the cap, the least-recently-updated sessions are
        dropped first. In-flight sessions are recently updated, so they are
        not realistic eviction victims.

    Eviction is lazy: it runs on create() (the only operation that grows the
    store), avoiding a background sweeper and the lifecycle that comes with it.
    ttl_seconds <= 0 or max_count <= 0 (the defaults) disables that dimension;
    production bounds are injected from Settings in main.py.
    """

    def __init__(
        self,
        ttl_seconds: int = 0,
        max_count: int = 0,
        session_history_max_chars: int = 256_000,
    ) -> None:
        if session_history_max_chars <= 0:
            raise ValueError("session_history_max_chars must be positive")
        self._sessions: dict[str, Session] = {}
        self._ttl_seconds = ttl_seconds
        self._max_count = max_count
        self._session_history_max_chars = session_history_max_chars

    @property
    def session_history_max_chars(self) -> int:
        return self._session_history_max_chars

    def _validated_snapshot(self, session: Session) -> Session:
        validate_transcript(session.messages)
        if session_history_chars(session.messages) > self._session_history_max_chars:
            raise SessionHistoryLimitExceeded
        return deepcopy(session)

    def _evict(self) -> None:
        """Drop expired then surplus sessions. Cheap; called on create()."""
        if self._ttl_seconds > 0:
            cutoff = _utc_now() - timedelta(seconds=self._ttl_seconds)
            expired = [
                sid for sid, s in self._sessions.items() if s.updated_at < cutoff
            ]
            for sid in expired:
                del self._sessions[sid]

        if self._max_count > 0 and len(self._sessions) >= self._max_count:
            # Oldest-updated first; trim down to (max_count - 1) to leave room
            # for the session about to be created.
            ordered = sorted(self._sessions.values(), key=lambda s: s.updated_at)
            surplus = len(self._sessions) - (self._max_count - 1)
            for s in ordered[:surplus]:
                del self._sessions[s.session_id]

    async def create(self, metadata: dict[str, Any] | None = None) -> Session:
        self._evict()
        session = Session(metadata=dict(metadata) if metadata else {})
        self._sessions[session.session_id] = self._validated_snapshot(session)
        return session

    async def get(self, session_id: str) -> Session:
        try:
            stored = self._sessions[session_id]
        except KeyError as e:
            raise SessionNotFoundError(session_id) from e
        return self._validated_snapshot(stored)

    async def save(self, session: Session) -> None:
        snapshot = self._validated_snapshot(session)
        snapshot.updated_at = _utc_now()
        self._sessions[session.session_id] = snapshot

    # ----- convenience for tests / debugging -----

    def __len__(self) -> int:
        return len(self._sessions)

    def ids(self) -> list[str]:
        return list(self._sessions.keys())


# ---------------------------------------------------------------------------
# Concurrency guard
# ---------------------------------------------------------------------------


class SessionBusyError(Exception):
    """Raised when a session already has a request in flight."""


class SessionGuard:
    """Reject-if-busy guard for same-session concurrency.

    Distinct sessions are already isolated (each request owns its own Session).
    This closes the one crossover vector: two concurrent requests on the same
    session_id sharing one .messages list. The second request is rejected.

    Lock-free by design: on the single-threaded asyncio event loop, the
    membership check and the add below execute with no `await` between them, so
    they are atomic relative to other tasks. To switch to wait-semantics later,
    replace the set with a dict[str, asyncio.Lock] and acquire it here.
    """

    def __init__(self) -> None:
        self._in_flight: set[str] = set()

    @asynccontextmanager
    async def claim(self, session_id: str) -> AsyncIterator[None]:
        if session_id in self._in_flight:
            raise SessionBusyError(session_id)
        self._in_flight.add(session_id)
        try:
            yield
        finally:
            self._in_flight.discard(session_id)

    # ----- convenience for tests / debugging -----

    def in_flight(self) -> set[str]:
        return set(self._in_flight)
