"""
Smoke test for Phase 5 capability profiles and adapter-seam behavior.

Hermetic by default:
    ./runscript.sh tests/smoke_test_capabilities.py

This is deliberately not pytest. It exercises the intended Phase 5 interfaces:

  1. ModelEntry defaults resolve to a default ModelProfile.
  2. Bad capability profile config fails loudly with model id + field.
  3. OpenAI sampling and reasoning_effort reach the request payload.
  4. OpenAI <think> routing produces trace-only reasoning.
  5. Malformed tool-call JSON becomes a model-facing is_error result via run_agent.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from agent.events import ReasoningEvent, ToolResultEvent
from agent.loop import run_agent
from agent.session import InMemorySessionStore
from agent.tracing import event_record
from llm.client import LLMClient
from llm.providers.openai import OpenAILLMClient
from llm.schemas import AssistantMessage, Message, ModelProfile, TextBlock, ToolUseBlock
from orchestrator.schemas import ModelsConfig


ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.WARNING, format="%(levelname)-5s %(name)s: %(message)s")


def _load_eval_fakes():
    """Load eval fakes from the repo path without relying on tests as a package."""
    path = ROOT / "tests" / "eval_agent.py"
    spec = importlib.util.spec_from_file_location("phase5_eval_agent_fakes", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import fakes from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ScriptedLLM, module.ScriptedMCP


ScriptedLLM, ScriptedMCP = _load_eval_fakes()


_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"    [{status}] {msg}")
    if not cond:
        _failures.append(msg)


def text_of(msg: AssistantMessage) -> str:
    return "".join(block.text for block in msg.content if isinstance(block, TextBlock))


def tool_results(events: list[Any]) -> list[ToolResultEvent]:
    return [e for e in events if isinstance(e, ToolResultEvent)]


def reasoning_events(events: list[Any]) -> list[ReasoningEvent]:
    return [e for e in events if isinstance(e, ReasoningEvent)]


def fake_openai_response(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    model: str = "phase5-model",
) -> Any:
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(finish_reason=finish_reason, message=message)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def fake_tool_call(call_id: str, name: str, arguments: str) -> Any:
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, function=fn)


class CaptureCreate:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **request: Any) -> Any:
        self.requests.append(dict(request))
        return self.response


def install_capture(client: OpenAILLMClient, response: Any) -> CaptureCreate:
    capture = CaptureCreate(response)
    client._client = SimpleNamespace(  # type: ignore[attr-defined]
        chat=SimpleNamespace(completions=SimpleNamespace(create=capture))
    )
    return capture


class CapturingScriptedLLM(ScriptedLLM):
    """ScriptedLLM that also records the messages each complete() call saw."""

    def __init__(self, script: list[Any]) -> None:
        super().__init__(script)
        self.messages_seen: list[list[Message]] = []

    async def complete(
        self,
        messages,
        tools=None,
        system=None,
        max_tokens=None,
        response_schema=None,
        thinking_level=None,
    ) -> AssistantMessage:
        self.messages_seen.append(list(messages))
        return await super().complete(
            messages,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
            response_schema=response_schema,
            thinking_level=thinking_level,
        )


def scenario_model_config_defaults() -> None:
    print("--- model config default profile parse ---")
    raw = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    cfg = ModelsConfig.model_validate(raw)
    model_id = cfg.default_id()
    profile = cfg.models[model_id].to_profile()
    check(isinstance(profile, ModelProfile), "default model resolves to ModelProfile")
    check(profile == ModelProfile.default(), "unprofiled current config keeps default profile")


def scenario_bad_profile_errors() -> None:
    print("--- bad profile errors name model id and field ---")
    unknown = {
        "models": {
            "bad-openai": {
                "provider": "openai",
                "model": "phase5",
                "description": "bad unknown field",
                "surprise": True,
            }
        }
    }
    try:
        ModelsConfig.model_validate(unknown)
    except Exception as exc:  # pydantic version can vary; assert on text.
        text = str(exc)
        check("bad-openai" in text and "surprise" in text, "unknown field error names model id + field")
    else:
        check(False, "unknown field rejected")

    too_hot = {
        "models": {
            "bad-openai": {
                "provider": "openai",
                "model": "phase5",
                "description": "bad sampling",
                "sampling": {"temperature": 9},
            }
        }
    }
    try:
        ModelsConfig.model_validate(too_hot)
    except Exception as exc:
        text = str(exc)
        check("bad-openai" in text and "temperature" in text, "temperature range error names model id + field")
    else:
        check(False, "temperature=9 rejected")


async def scenario_openai_request_capture() -> None:
    print("--- OpenAI request captures sampling and reasoning_effort ---")
    profile = ModelProfile(
        thinking="hint-param",
        temperature=0.7,
        top_p=0.91,
        top_k=42,
    )
    client = OpenAILLMClient(
        api_key="test-key",
        model="phase5-model",
        default_max_tokens=123,
        base_url="http://127.0.0.1:9/v1",
        profile=profile,
    )
    capture = install_capture(client, fake_openai_response(content="ok"))

    await client.complete(
        messages=[Message.user("hello")],
        thinking_level="high",
    )
    request = capture.requests[-1]
    check(request.get("temperature") == 0.7, "temperature reaches OpenAI request")
    check(request.get("top_p") == 0.91, "top_p reaches OpenAI request")
    check(request.get("top_k") == 42, "top_k reaches OpenAI-compatible request")
    check(request.get("reasoning_effort") == "high", "hint-param thinking maps to reasoning_effort")

    none_profile = ModelProfile(thinking="none")
    inert = OpenAILLMClient(
        api_key="test-key",
        model="no-thinking-model",
        default_max_tokens=123,
        base_url="http://127.0.0.1:9/v1",
        profile=none_profile,
    )
    inert_capture = install_capture(inert, fake_openai_response(content="ok"))
    await inert.complete(messages=[Message.user("hello")], thinking_level="high")
    check(
        "reasoning_effort" not in inert_capture.requests[-1],
        "thinking:none does not send reasoning_effort",
    )


async def scenario_reasoning_routing() -> None:
    print("--- think-tag reasoning routing and trace record ---")
    client = OpenAILLMClient(
        api_key="test-key",
        model="think-tags-model",
        default_max_tokens=123,
        base_url="http://127.0.0.1:9/v1",
        profile=ModelProfile(thinking="think-tags"),
    )
    response = client._from_openai_response(fake_openai_response(
        content="<think>private chain</think>\nvisible answer",
        model="think-tags-model",
    ))

    check(response.reasoning == "private chain", "AssistantMessage.reasoning carries think text")
    check(text_of(response).strip() == "visible answer", "visible text excludes think block")
    replay = response.to_message()
    replay_text = "".join(
        block.text for block in replay.content if isinstance(block, TextBlock)
    )
    check("<think>" not in replay_text and "private chain" not in replay_text, "replay content excludes reasoning")

    llm = CapturingScriptedLLM([response])
    mcp = ScriptedMCP(tools=[], results=[])
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("go")
    events: list[Any] = []
    async for event in run_agent(session=session, llm=llm, mcp=mcp, store=store):
        events.append(event)

    rs = reasoning_events(events)
    check(len(rs) == 1 and rs[0].text == "private chain", "run_agent emits ReasoningEvent")
    record = event_record(rs[0], run_id="phase5", step=1) if rs else {}
    check(record.get("type") == "reasoning" and record.get("reasoning") == "private chain", "trace record carries reasoning")


async def scenario_malformed_args_error_signal() -> None:
    print("--- malformed JSON parse_error becomes model-facing is_error ---")
    client = OpenAILLMClient(
        api_key="test-key",
        model="bad-json-model",
        default_max_tokens=123,
        base_url="http://127.0.0.1:9/v1",
        profile=ModelProfile.default(),
    )
    bad_tool_response = client._from_openai_response(fake_openai_response(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[fake_tool_call("call_bad", "srv__lookup", '{"query": ')],
        model="bad-json-model",
    ))
    tool_uses = [b for b in bad_tool_response.content if isinstance(b, ToolUseBlock)]
    check(len(tool_uses) == 1, "malformed OpenAI tool call still becomes ToolUseBlock")
    check(tool_uses[0].input == {}, "malformed arguments are kept as empty input")
    check(bool(tool_uses[0].parse_error), "ToolUseBlock.parse_error records JSON failure")

    final = AssistantMessage(content=[TextBlock(text="recovered")], stop_reason="end_turn")
    llm = CapturingScriptedLLM([bad_tool_response, final])
    mcp = ScriptedMCP(
        tools=[{"name": "srv__lookup", "description": "lookup", "input_schema": {}}],
        results=[],
    )
    store = InMemorySessionStore()
    session = await store.create()
    session.append_user("look up this thing")
    events: list[Any] = []
    async for event in run_agent(session=session, llm=llm, mcp=mcp, store=store):
        events.append(event)

    trs = tool_results(events)
    check(len(trs) == 1 and trs[0].is_error, "parse_error turns into ToolResultEvent(is_error=True)")
    check("valid JSON" in trs[0].content or "not valid JSON" in trs[0].content, "tool result teaches JSON repair")
    check(mcp.call_count == 0, "malformed tool call is not dispatched to MCP")

    second_seen_tool_results = [
        block
        for msg in llm.messages_seen[1]
        for block in msg.content
        if getattr(block, "type", None) == "tool_result"
    ] if len(llm.messages_seen) > 1 else []
    check(
        len(second_seen_tool_results) == 1 and second_seen_tool_results[0].is_error,
        "next model turn sees model-facing is_error tool result",
    )


async def main() -> None:
    scenario_model_config_defaults()
    scenario_bad_profile_errors()
    await scenario_openai_request_capture()
    await scenario_reasoning_routing()
    await scenario_malformed_args_error_signal()

    print()
    if _failures:
        print(f"CAPABILITIES SMOKE TEST FAILED: {len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("CAPABILITIES SMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
