"""Provider-neutral reusable fakes for hermetic agent-loop tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from agent import InMemorySessionStore, RunLimits, run_agent
from llm.client import GenerationRequest, LLMClient
from llm.schemas import AssistantMessage
from tooling import ToolCallResult


class ScriptedLLM(LLMClient):
    """Replay complete responses or exceptions while recording each request."""

    def __init__(
        self,
        script: list[AssistantMessage | BaseException],
        *,
        transient_types: tuple[type[BaseException], ...] = (),
    ) -> None:
        self._script = list(script)
        self._transient_types = transient_types
        self.calls = 0
        self.requests_seen: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        self.requests_seen.append(request)
        if not self._script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def is_transient_error(self, exc: BaseException) -> bool:
        return isinstance(exc, self._transient_types)


ToolHandler = Callable[
    [str, dict[str, Any]], ToolCallResult | Awaitable[ToolCallResult]
]


class ScriptedMCP:
    """Configurable MCP stand-in with request recording and optional delay."""

    def __init__(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        content: str = "tool ok",
        is_error: bool = False,
        delay: float = 0,
        handler: ToolHandler | None = None,
    ) -> None:
        self._tools = (
            tools
            if tools is not None
            else [{"name": "srv__tool", "description": "test tool", "input_schema": {}}]
        )
        self._content = content
        self._is_error = is_error
        self._delay = delay
        self._handler = handler
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return self._tools

    @asynccontextmanager
    async def open_turn(self, *, timeout_seconds: float | None = None):
        yield self

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.calls.append((name, arguments))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._handler is not None:
            result = self._handler(name, arguments)
            if isinstance(result, Awaitable):
                return await result
            return result
        return ToolCallResult(content=self._content, is_error=self._is_error)


async def collect_agent_events(
    *,
    llm: LLMClient,
    mcp: Any,
    prompt: str = "go",
    **run_options: Any,
) -> list[Any]:
    """Create an isolated session and collect one complete agent event stream."""
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user(prompt)
    stream = bool(run_options.pop("stream", False))
    return [
        event
        async for event in run_agent(
            session=session,
            llm=llm,
            mcp=mcp,
            store=store,
            limits=RunLimits(**run_options),
            stream=stream,
        )
    ]
