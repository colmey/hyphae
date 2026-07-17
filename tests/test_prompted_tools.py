"""Hermetic pytest coverage for prompted-tool adapter behavior.

It exercises the prompted-tool renderer/parser,
the LLMClient wrapper, factory selection, and the drop-in proof through the
unchanged agent loop.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent import (
    DoneEvent,
    InMemorySessionStore,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    run_agent,
)
from llm.client import LLMClient, build_llm_client_from_entry
from llm.prompted_tools import (
    PromptedToolLLMClient,
    parse_prompted_action,
    render_prompted_tools,
)
from llm.schemas import (
    AssistantMessage,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from mcp_layer.client import ToolCallResult
from config import ModelEntry


TOOL = {
    "name": "srv__lookup",
    "description": "Lookup a value",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
pytestmark = pytest.mark.anyio


def check(cond: bool, msg: str) -> None:
    assert cond, msg


def text_response(text: str, *, usage: Usage | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        model="scripted-prose",
        usage=usage,
        reasoning="kept private",
    )


class ScriptedLLM(LLMClient):
    def __init__(self, script: list[AssistantMessage]) -> None:
        self._script = list(script)
        self.calls = 0
        self.messages_seen: list[list[Message]] = []
        self.tools_seen: list[list[dict[str, Any]] | None] = []
        self.systems_seen: list[str | None] = []
        self.response_schemas_seen: list[type | None] = []

    async def complete(
        self,
        messages,
        tools=None,
        system=None,
        max_tokens=None,
        response_schema=None,
        thinking_level=None,
    ) -> AssistantMessage:
        self.calls += 1
        self.messages_seen.append(list(messages))
        self.tools_seen.append(tools)
        self.systems_seen.append(system)
        self.response_schemas_seen.append(response_schema)
        if not self._script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._script.pop(0)


class FakeMCP:
    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return [TOOL]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.call_count += 1
        self.calls.append((name, arguments))
        return ToolCallResult(
            content=f"lookup result for {arguments.get('query')}", is_error=False
        )


def action_text(query: str = "example") -> str:
    return f'```json\n{{"tool":"srv__lookup","arguments":{{"query":"{query}"}}}}\n```'


def test_render() -> None:
    print("--- render prompted tools ---")
    rendered = render_prompted_tools([TOOL])
    check("Available tools:" in rendered, "render includes available tools heading")
    check("srv__lookup" in rendered, "render includes tool name")
    check("Lookup a value" in rendered, "render includes one-line description")
    check('"required":["query"]' in rendered, "render includes compact sorted schema")
    check("Call at most one tool." in rendered, "render includes action rules")
    check(render_prompted_tools([]) == "", "no tools render as empty string")


def test_parser() -> None:
    print("--- parse prompted actions ---")
    allowed = {"srv__lookup"}

    parsed = parse_prompted_action(action_text("alpha"), allowed)
    check(parsed.action is not None, "fenced JSON parses as an action")
    check(
        parsed.action is not None and parsed.action.name == "srv__lookup",
        "action name parsed",
    )
    check(
        parsed.action is not None and parsed.action.arguments == {"query": "alpha"},
        "action arguments parsed",
    )

    parsed = parse_prompted_action(
        '{"name":"srv__lookup","input":{"query":"beta"}}', allowed
    )
    check(
        parsed.action is not None and parsed.action.arguments == {"query": "beta"},
        "unfenced whole-response JSON parses",
    )

    parsed = parse_prompted_action("Here is the final answer.", allowed)
    check(
        parsed.action is None and parsed.error is None, "ordinary prose is final text"
    )
    check(parsed.text == "Here is the final answer.", "final prose is preserved")

    parsed = parse_prompted_action(
        '```json\n{"tool":"srv__lookup","arguments":}\n```', allowed
    )
    check(
        parsed.action is None and parsed.error is not None, "malformed JSON is an error"
    )

    parsed = parse_prompted_action('{"tool":"srv__missing","arguments":{}}', allowed)
    check(
        parsed.action is None and parsed.error is not None, "unknown tool is an error"
    )
    check(parsed.error_tool_name == "srv__missing", "unknown tool name is preserved")

    parsed = parse_prompted_action('{"tool":"srv__lookup","arguments":"nope"}', allowed)
    check(
        parsed.action is None and parsed.error is not None, "non-dict args are an error"
    )
    check(parsed.error_tool_name == "srv__lookup", "bad-args tool name is preserved")


async def test_wrapper_basic() -> None:
    print("--- wrapper sends prompted tools, not native tools ---")
    inner = ScriptedLLM([text_response(action_text("gamma"))])
    wrapper = PromptedToolLLMClient(inner, model="wrapped-model")
    response = await wrapper.complete(
        messages=[Message.user("lookup gamma")],
        tools=[TOOL],
        system="base system",
    )

    check(inner.tools_seen == [None], "inner client saw tools=None")
    check(
        "base system" in (inner.systems_seen[0] or ""), "original system is preserved"
    )
    check(
        "Available tools:" in (inner.systems_seen[0] or ""),
        "prompted tool instructions are in system",
    )
    check(
        len(response.content) == 1 and isinstance(response.content[0], ToolUseBlock),
        "wrapper returns one ToolUseBlock",
    )
    tool_use = response.content[0]
    check(
        isinstance(tool_use, ToolUseBlock) and tool_use.name == "srv__lookup",
        "ToolUseBlock name normalized",
    )
    check(
        isinstance(tool_use, ToolUseBlock) and tool_use.input == {"query": "gamma"},
        "ToolUseBlock input normalized",
    )
    check(response.reasoning == "kept private", "reasoning metadata preserved")


async def test_passthroughs() -> None:
    print("--- wrapper passthrough cases ---")
    schema = dict
    inner = ScriptedLLM([text_response("structured")])
    wrapper = PromptedToolLLMClient(inner)
    response = await wrapper.complete(
        messages=[Message.user("structured please")],
        tools=[TOOL],
        response_schema=schema,
    )
    check(
        response.text_blocks()[0].text == "structured",
        "response_schema call passes through",
    )
    check(
        inner.tools_seen == [[TOOL]],
        "response_schema passthrough keeps native tools argument",
    )
    check(
        inner.response_schemas_seen == [schema], "response_schema reaches inner client"
    )

    history = [
        Message.user("lookup"),
        Message.assistant(
            [
                ToolUseBlock(
                    id="prompted_old", name="srv__lookup", input={"query": "old"}
                )
            ]
        ),
        Message.tool_results(
            [
                ToolResultBlock(
                    tool_use_id="prompted_old",
                    name="srv__lookup",
                    content="old result",
                )
            ]
        ),
    ]
    inner = ScriptedLLM([text_response("final from history")])
    wrapper = PromptedToolLLMClient(inner)
    response = await wrapper.complete(messages=history, tools=[TOOL])
    check(
        response.text_blocks()[0].text == "final from history",
        "final prose passes through",
    )
    check(
        inner.messages_seen[0][-1].role.value == "tool",
        "tool result history is passed through",
    )


async def test_repair() -> None:
    print("--- repair ladder ---")
    inner = ScriptedLLM(
        [
            text_response(
                '```json\n{"tool":"srv__lookup","arguments":}\n```',
                usage=Usage(total_tokens=2),
            ),
            text_response(action_text("repaired"), usage=Usage(total_tokens=3)),
        ]
    )
    wrapper = PromptedToolLLMClient(inner)
    response = await wrapper.complete(
        messages=[Message.user("lookup repaired")], tools=[TOOL]
    )
    check(inner.calls == 2, "malformed JSON triggers one repair call")
    repair_text = "".join(
        block.text
        for block in inner.messages_seen[1][-1].content
        if isinstance(block, TextBlock)
    )
    check(
        "invalid JSON tool action" in repair_text, "repair prompt includes parse error"
    )
    check(
        response.tool_uses()[0].input == {"query": "repaired"},
        "repair action becomes ToolUseBlock",
    )
    check(
        response.usage is not None and response.usage.total_tokens == 5,
        "usage from repair calls is combined",
    )

    inner = ScriptedLLM(
        [
            text_response('```json\n{"tool":"srv__lookup","arguments":}\n```'),
            text_response('```json\n{"tool":"srv__lookup","arguments":}\n```'),
        ]
    )
    wrapper = PromptedToolLLMClient(inner)
    response = await wrapper.complete(
        messages=[Message.user("lookup fail")], tools=[TOOL]
    )
    check(inner.calls == 2, "repair failure stops after one retry")
    check(not response.tool_uses(), "repair failure returns no ToolUseBlock")
    check(
        "I could not parse a valid tool action" in response.text_blocks()[0].text,
        "repair failure is visible text",
    )


async def test_semantic_errors() -> None:
    print("--- parsed semantic errors become parse_error tool uses ---")
    inner = ScriptedLLM([text_response('{"tool":"srv__lookup","arguments":"bad"}')])
    wrapper = PromptedToolLLMClient(inner)
    response = await wrapper.complete(
        messages=[Message.user("lookup bad")], tools=[TOOL]
    )
    tool_uses = response.tool_uses()
    check(len(tool_uses) == 1, "bad arguments still return a ToolUseBlock")
    check(tool_uses[0].input == {}, "bad arguments do not execute with model args")
    check(tool_uses[0].parse_error is not None, "bad arguments carry parse_error")


async def test_drop_in_loop() -> None:
    print("--- drop-in proof through run_agent ---")
    inner = ScriptedLLM(
        [
            text_response(action_text("phase6")),
            text_response("Final answer from lookup result."),
        ]
    )
    llm = PromptedToolLLMClient(inner)
    mcp = FakeMCP()
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("Use the lookup tool.")

    events: list[Any] = []
    async for event in run_agent(session=session, llm=llm, mcp=mcp, store=store):
        events.append(event)

    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    text_events = [e for e in events if isinstance(e, TextEvent)]
    done = next((e for e in events if isinstance(e, DoneEvent)), None)

    check(len(tool_calls) == 1, "loop emitted one ToolCallEvent")
    check(tool_calls[0].name == "srv__lookup", "loop saw prompted tool call name")
    check(
        mcp.calls == [("srv__lookup", {"query": "phase6"})],
        "MCP dispatched parsed args",
    )
    check(
        len(tool_results) == 1 and not tool_results[0].is_error,
        "loop emitted successful ToolResultEvent",
    )
    check(
        "Final answer" in "".join(e.text for e in text_events),
        "loop emitted final TextEvent",
    )
    check(done is not None and done.reason == "end_turn", "loop completed end_turn")
    check(inner.tools_seen == [None, None], "inner prose model never saw native tools")


async def test_factory_selection() -> None:
    print("--- factory selection by profile ---")
    import llm.client as client_module

    def build_fake(*, model, max_tokens, settings, profile=None):
        return ScriptedLLM([])

    original = dict(client_module._PROVIDERS)
    try:
        client_module._PROVIDERS["fake"] = build_fake
        settings = SimpleNamespace(llm_max_tokens=111)

        native_entry = ModelEntry(
            provider="fake",
            model="native",
            description="native model",
        )
        native = build_llm_client_from_entry(native_entry, settings)
        check(
            not isinstance(native, PromptedToolLLMClient),
            "omitted supports_native_tools stays native",
        )

        true_entry = ModelEntry(
            provider="fake",
            model="native-true",
            description="native model",
            supports_native_tools=True,
        )
        native_true = build_llm_client_from_entry(true_entry, settings)
        check(
            not isinstance(native_true, PromptedToolLLMClient),
            "true supports_native_tools stays native",
        )

        prompted_entry = ModelEntry(
            provider="fake",
            model="prompted",
            description="prompted model",
            supports_native_tools=False,
        )
        prompted = build_llm_client_from_entry(prompted_entry, settings)
        check(
            isinstance(prompted, PromptedToolLLMClient),
            "false supports_native_tools wraps client",
        )
        check(
            isinstance(prompted.inner, ScriptedLLM), "wrapper contains base fake client"
        )
    finally:
        client_module._PROVIDERS.clear()
        client_module._PROVIDERS.update(original)


def test_orchestrator_guardrail(tmp_path: Path) -> None:
    print("--- orchestrator prompted-only guardrail ---")
    from main import _try_build_orchestration

    models_path = tmp_path / "models.yaml"
    prompt_path = tmp_path / "orchestrator_prompt.md"
    models_path.write_text(
        """
models:
  prompted-control:
    provider: openai
    model: prompted-control
    description: Prompted-only control model.
    supports_native_tools: false
    default: true
""".lstrip(),
        encoding="utf-8",
    )
    prompt_path.write_text("You route requests.", encoding="utf-8")
    settings = SimpleNamespace(
        orchestration_enabled=True,
        models_config_path=str(models_path),
        orchestrator_prompt_path=str(prompt_path),
        orchestrator_model_id="",
        llm_max_tokens=128,
    )
    registry, orchestrator = _try_build_orchestration(settings, FakeMCP())

    check(
        registry is None and orchestrator is None,
        "prompted-only orchestrator disables orchestration",
    )
