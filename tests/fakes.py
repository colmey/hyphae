"""Provider-neutral reusable fakes for hermetic agent-loop tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from hyphae.agent import Event, InMemorySessionStore, RunLimits, run_agent
from hyphae.llm.client import GenerationRequest, LLMClient
from hyphae.llm.schemas import AssistantMessage
from hyphae.tooling import ToolCallResult, ToolRuntime


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
    async def open_turn(
        self, *, timeout_seconds: float | None = None
    ) -> AsyncIterator[ToolRuntime]:
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
    mcp: ToolRuntime,
    prompt: str = "go",
    stream: bool = False,
    max_iterations: int = 10,
    max_tokens: int | None = None,
    llm_timeout_seconds: float | None = None,
    tool_timeout_seconds: float | None = None,
    max_retries: int = 0,
    retry_base_delay: float = 0.5,
    tool_result_max_chars: int | None = None,
    max_run_tokens: int | None = None,
    max_run_seconds: float | None = None,
    abort_after_consecutive_tool_failures: int | None = None,
    context_strategy: str = "naive",
    context_window: int | None = None,
    context_safety_margin_tokens: int = 1024,
    context_recent_messages: int = 6,
    context_summary_max_tokens: int = 512,
) -> list[Event]:
    """Create an isolated session and collect one complete agent event stream."""
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user(prompt)
    return [
        event
        async for event in run_agent(
            session=session,
            llm=llm,
            mcp=mcp,
            store=store,
            limits=RunLimits(
                max_iterations=max_iterations,
                max_tokens=max_tokens,
                llm_timeout_seconds=llm_timeout_seconds,
                tool_timeout_seconds=tool_timeout_seconds,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
                tool_result_max_chars=tool_result_max_chars,
                max_run_tokens=max_run_tokens,
                max_run_seconds=max_run_seconds,
                abort_after_consecutive_tool_failures=(
                    abort_after_consecutive_tool_failures
                ),
                context_strategy=context_strategy,
                context_window=context_window,
                context_safety_margin_tokens=context_safety_margin_tokens,
                context_recent_messages=context_recent_messages,
                context_summary_max_tokens=context_summary_max_tokens,
            ),
            stream=stream,
        )
    ]
