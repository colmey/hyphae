"""Hermetic pytest coverage for the prompted-tool protocol adapter.

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
from llm.client import GenerationRequest, LLMClient, build_llm_client_from_entry
from llm.tool_prompt_protocol import (
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
    CompletionUsage,
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


def text_response(text: str, *, usage: CompletionUsage | None = None) -> AssistantMessage:
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
        self.requests_seen: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.calls += 1
        self.requests_seen.append(request)
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
    original = GenerationRequest(
        messages=[Message.user("lookup gamma")], tools=[TOOL], system="base system"
    )
    response = await wrapper.complete(original)

    check(inner.requests_seen[0].tools is None, "inner client saw tools=None")
    check(
        "base system" in (inner.requests_seen[0].system or ""),
        "original system is preserved",
    )
    check(
        "Available tools:" in (inner.requests_seen[0].system or ""),
        "prompted tool instructions are in system",
    )
    check(inner.requests_seen[0] is not original, "prompted request is derived")
    check(original.tools == [TOOL], "original request was not mutated")
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


@pytest.mark.parametrize(
    "stop_reason",
    ["max_tokens", "empty", "content_filter", "refusal", "provider_error"],
)
async def test_wrapper_does_not_launder_non_normal_provider_outcomes(
    stop_reason: str,
) -> None:
    source = AssistantMessage(
        content=[TextBlock(text=action_text("must-not-run"))],
        stop_reason=stop_reason,
        raw_stop_reason="provider-raw",
        usage=CompletionUsage(total_tokens=3),
    )
    inner = ScriptedLLM([source])

    response = await PromptedToolLLMClient(inner).complete(
        GenerationRequest(messages=[Message.user("lookup")], tools=[TOOL])
    )

    check(response is source, "non-normal outcome passes through unchanged")
    check(response.stop_reason == stop_reason, "canonical reason is preserved")
    check(response.raw_stop_reason == "provider-raw", "raw reason is preserved")
    check(inner.calls == 1, "abnormal outcome is not repaired")


async def test_passthroughs() -> None:
    print("--- wrapper passthrough cases ---")
    schema = dict
    inner = ScriptedLLM([text_response("structured")])
    wrapper = PromptedToolLLMClient(inner)
    original = GenerationRequest(
        messages=[Message.user("structured please")],
        tools=[TOOL],
        response_schema=schema,
    )
    response = await wrapper.complete(original)
    check(
        response.text_blocks()[0].text == "structured",
        "response_schema call passes through",
    )
    check(
        inner.requests_seen[0].tools == [TOOL],
        "response_schema passthrough keeps native tools argument",
    )
    check(
        inner.requests_seen[0].response_schema is schema,
        "response_schema reaches inner client",
    )
    check(inner.requests_seen[0] is original, "unchanged request preserves identity")

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
    response = await wrapper.complete(GenerationRequest(messages=history, tools=[TOOL]))
    check(
        response.text_blocks()[0].text == "final from history",
        "final prose passes through",
    )
    check(
        inner.requests_seen[0].messages[-1].role.value == "tool",
        "tool result history is passed through",
    )


async def test_repair() -> None:
    print("--- repair ladder ---")
    inner = ScriptedLLM(
        [
            text_response(
                '```json\n{"tool":"srv__lookup","arguments":}\n```',
                usage=CompletionUsage(total_tokens=2),
            ),
            text_response(action_text("repaired"), usage=CompletionUsage(total_tokens=3)),
        ]
    )
    wrapper = PromptedToolLLMClient(inner)
    original = GenerationRequest(
        messages=[Message.user("lookup repaired")],
        tools=[TOOL],
        max_tokens=321,
        thinking_level="high",
    )
    response = await wrapper.complete(original)
    check(inner.calls == 2, "malformed JSON triggers one repair call")
    repair_text = "".join(
        block.text
        for block in inner.requests_seen[1].messages[-1].content
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
    check(
        all(request.max_tokens == 321 for request in inner.requests_seen),
        "prompted and repair requests preserve max_tokens",
    )
    check(
        all(request.thinking_level == "high" for request in inner.requests_seen),
        "prompted and repair requests preserve thinking_level",
    )
    check(
        all(
            request.tools is None and request.response_schema is None
            for request in inner.requests_seen
        ),
        "prompted and repair requests clear native-only fields",
    )
    check(original.tools == [TOOL], "repair path does not mutate original request")

    inner = ScriptedLLM(
        [
            text_response('```json\n{"tool":"srv__lookup","arguments":}\n```'),
            text_response('```json\n{"tool":"srv__lookup","arguments":}\n```'),
        ]
    )
    wrapper = PromptedToolLLMClient(inner)
    response = await wrapper.complete(
        GenerationRequest(messages=[Message.user("lookup fail")], tools=[TOOL])
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
        GenerationRequest(messages=[Message.user("lookup bad")], tools=[TOOL])
    )
    tool_uses = response.tool_uses()
    check(len(tool_uses) == 1, "bad arguments still return a ToolUseBlock")
    check(
        tool_uses[0].input == {},
        "bad arguments do not execute with model arguments",
    )
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
    check(
        [request.tools for request in inner.requests_seen] == [None, None],
        "inner prose model never saw native tools",
    )


async def test_factory_selection() -> None:
    print("--- factory selection by profile ---")
    import llm.client as client_module

    specs = []

    def build_fake(*, spec, settings):
        specs.append(spec)
        return ScriptedLLM([])

    class FactorySettings:
        llm = SimpleNamespace(max_tokens=111)
        openai_compat_base_url = ""

        def api_key_for_provider(self, provider: str) -> str:
            raise AssertionError(f"unexpected provider construction: {provider}")

    original = dict(client_module._PROVIDERS)
    try:
        client_module._PROVIDERS["fake"] = build_fake
        settings = FactorySettings()

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
        check(
            [spec.model for spec in specs] == ["native", "native-true", "prompted"],
            "builder receives each model through the typed construction spec",
        )
        check(
            all(spec.max_tokens == 111 for spec in specs),
            "builder spec retains the settings token fallback",
        )
        check(
            [spec.profile.supports_native_tools for spec in specs]
            == [True, True, False],
            "builder spec carries the resolved model profile",
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
    provider: openai_compatible
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
        llm=SimpleNamespace(max_tokens=128),
    )
    registry, orchestrator = _try_build_orchestration(settings)

    check(
        registry is None and orchestrator is None,
        "prompted-only orchestrator disables orchestration",
    )
