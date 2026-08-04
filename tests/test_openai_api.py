"""OpenAI-compatible HTTP and SSE wire-format tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

from agent import InMemorySessionStore, Session
from api import openai_compatible
from api.turn import UnorchestratedRouting
from llm.client import GenerationRequest, LLMClient
from llm.schemas import (
    AssistantMessage,
    Role,
    StreamChunk,
    StreamEnd,
    TextBlock,
    TextDelta,
    CompletionUsage,
)
from config import load_models_config
from orchestrator import LLMRegistry
from orchestrator.schemas import OrchestrationDecision, OrchestrationProposal
from tests._app_support import replace_routing, runtime_of, wired_app


pytestmark = pytest.mark.anyio


class FakeLLM(LLMClient):
    def __init__(self) -> None:
        self.complete_calls = 0
        self.stream_calls = 0
        self.requests_seen: list[GenerationRequest] = []

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        self.complete_calls += 1
        self.requests_seen.append(request)
        return AssistantMessage(
            content=[TextBlock(text="Hello"), TextBlock(text=", world")],
            stop_reason="end_turn",
            model="fake",
            usage=CompletionUsage(input_tokens=11, output_tokens=3, total_tokens=14),
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        self.stream_calls += 1
        self.requests_seen.append(request)
        yield TextDelta(text="Hello")
        yield TextDelta(text=", ")
        yield TextDelta(text="world")
        yield StreamEnd(
            AssistantMessage(
                content=[TextBlock(text="Hello, world")],
                stop_reason="end_turn",
                model="fake",
                usage=CompletionUsage(
                    input_tokens=11, output_tokens=3, total_tokens=14
                ),
            )
        )


class OutcomeLLM(LLMClient):
    def __init__(self, stop_reason: str) -> None:
        self.stop_reason = stop_reason

    async def complete(self, request: GenerationRequest) -> AssistantMessage:
        return AssistantMessage(
            content=[
                TextBlock(
                    text="provider-visible",
                    provider_metadata={"thought_signature": b"opaque-signature"},
                )
            ],
            stop_reason=self.stop_reason,
            raw_stop_reason="raw-provider-reason",
            reasoning="private-chain-of-thought",
            model="provider-model",
            usage=CompletionUsage(input_tokens=2, output_tokens=1, total_tokens=3),
        )


class CountingMCP:
    connected_servers: list[str] = []

    def __init__(self) -> None:
        self.inventory_reads = 0

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        self.inventory_reads += 1
        return []

    def list_tools(self) -> list[Any]:
        return []


class RegistryStub:
    model_ids = ["selected", "requested"]

    def __init__(self, clients: dict[str, LLMClient]) -> None:
        self.clients = clients

    def get_or_default(self, model_id: str | None) -> tuple[str, LLMClient]:
        resolved = model_id if model_id in self.model_ids else "selected"
        return resolved, self.clients[resolved]

    def get(self, model_id: str) -> LLMClient:
        return self.clients[model_id]

    def get_entry(self, _model_id: str) -> Any:
        return SimpleNamespace(max_tokens=256, context_window=4096)


class BrokenRegistry:
    @property
    def model_ids(self) -> list[str]:
        raise RuntimeError("private registry failure")


class SelectingOrchestrator:
    def __init__(self, selected_model_id: str = "selected") -> None:
        self.selected_model_id = selected_model_id
        self.calls = 0

    async def decide(
        self, prompt, tools, history=None, timeout=None, log=None
    ) -> OrchestrationDecision:
        self.calls += 1
        return OrchestrationDecision(
            result=OrchestrationProposal(
                selected_model_id=self.selected_model_id,
                selected_tools=[],
            )
        )


def _registry(settings) -> LLMRegistry:
    return LLMRegistry(load_models_config(settings.models_config_path), settings)


async def test_models_returns_registry_in_openai_list_shape(asgi_client) -> None:
    llm = FakeLLM()
    with wired_app(llm) as (app, settings):
        registry = _registry(settings)
        replace_routing(
            app,
            UnorchestratedRouting(
                llm=llm,
                model_id=settings.llm.model,
                inventory=registry,
            ),
        )
        response = await asgi_client(app).get("/v1/models")

    response.raise_for_status()
    body = response.json()
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"]] == registry.model_ids
    assert all(model["object"] == "model" for model in body["data"])


@pytest.mark.parametrize("path", ["/v1/models", "/v1/chat/completions"])
async def test_model_inventory_failure_is_sanitized(asgi_client, path: str) -> None:
    with wired_app(FakeLLM(), registry=BrokenRegistry()) as (app, _settings):
        client = asgi_client(app)
        if path.endswith("completions"):
            response = await client.post(
                path,
                json={
                    "model": "requested",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        else:
            response = await client.get(path)

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "message": "model inventory unavailable",
            "type": "server_error",
            "param": None,
            "code": None,
        }
    }
    assert "private registry failure" not in response.text


async def test_non_stream_completion_shape_and_usage(asgi_client) -> None:
    llm = FakeLLM()
    with wired_app(llm) as (app, settings):
        response = await asgi_client(app).post(
            "/v1/chat/completions",
            json={
                "model": settings.llm.model,
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Hi there"},
                ],
            },
        )

    response.raise_for_status()
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == settings.llm.model
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hello, world"}
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }
    assert llm.complete_calls == 1


@pytest.mark.parametrize("stop_reason", ["content_filter", "refusal"])
@pytest.mark.parametrize("stream", [False, True])
async def test_policy_outcomes_map_to_openai_content_filter(
    asgi_client, parse_sse, stop_reason: str, stream: bool
) -> None:
    with wired_app(OutcomeLLM(stop_reason)) as (app, _settings):
        response = await asgi_client(app).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
        )

    assert response.status_code == 200
    if stream:
        payloads = parse_sse(response.text)
        assert payloads[-1] == "[DONE]"
        frames = [payload for payload in payloads if payload != "[DONE]"]
        assert not any("error" in frame for frame in frames)
        assert frames[-1]["choices"][0]["finish_reason"] == "content_filter"
        rendered = response.text
    else:
        body = response.json()
        assert body["choices"][0]["finish_reason"] == "content_filter"
        rendered = response.text
    if stream:
        assert "private-chain-of-thought" in rendered
        assert "<think>" not in rendered
        assert "</think>" not in rendered
    else:
        assert "private-chain-of-thought" not in rendered
    assert "opaque-signature" not in rendered


@pytest.mark.parametrize("stop_reason", ["provider_error", "incomplete_stream"])
@pytest.mark.parametrize("stream", [False, True])
async def test_provider_and_incomplete_stream_errors_fail_closed(
    asgi_client, parse_sse, stop_reason: str, stream: bool
) -> None:
    with wired_app(OutcomeLLM(stop_reason)) as (app, _settings):
        response = await asgi_client(app).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
        )

    if stream:
        assert response.status_code == 200
        payloads = parse_sse(response.text)
        assert payloads[-1] == "[DONE]"
        frames = [payload for payload in payloads if payload != "[DONE]"]
        assert sum("error" in frame for frame in frames) == 1
        assert not any(
            frame.get("choices") and frame["choices"][0]["finish_reason"] is not None
            for frame in frames
        )
    else:
        assert response.status_code == 500
        assert response.json()["error"]["type"] == "server_error"


async def test_stream_completion_emits_deltas_finish_and_done(
    asgi_client, parse_sse
) -> None:
    llm = FakeLLM()
    with wired_app(llm) as (app, _settings):
        async with asgi_client(app).stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "temperature": 0.7,
            },
        ) as response:
            assert response.status_code == 200
            payloads = parse_sse(
                "".join([chunk async for chunk in response.aiter_text()])
            )

    assert payloads[-1] == "[DONE]"
    frames = [payload for payload in payloads if payload != "[DONE]"]
    assert all(frame["object"] == "chat.completion.chunk" for frame in frames)
    assert all(frame["model"] == _settings.llm.model for frame in frames)
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
    deltas = [frame["choices"][0]["delta"].get("content", "") for frame in frames]
    assert deltas.count("Hello") == 1
    assert "".join(deltas) == "Hello, world"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert all("usage" not in frame for frame in frames)
    assert llm.stream_calls == 1


async def test_empty_messages_returns_openai_error(asgi_client) -> None:
    with wired_app(FakeLLM()) as (app, _settings):
        response = await asgi_client(app).post(
            "/v1/chat/completions", json={"messages": []}
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("stream", [False, True])
async def test_repeated_v1_calls_never_publish_sessions(
    asgi_client, parse_sse, stream: bool
) -> None:
    llm = FakeLLM()
    with wired_app(llm) as (app, _settings):
        store = runtime_of(app).store
        initial_ids = store.ids()
        client = asgi_client(app)
        for _ in range(4):
            payload = {
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": stream,
            }
            if stream:
                async with client.stream(
                    "POST", "/v1/chat/completions", json=payload
                ) as response:
                    assert response.status_code == 200
                    events = parse_sse(
                        "".join([chunk async for chunk in response.aiter_text()])
                    )
                    assert events[-1] == "[DONE]"
            else:
                response = await client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()

        assert len(store) == len(initial_ids)
        assert store.ids() == initial_ids


async def test_v1_traffic_cannot_evict_native_session_at_capacity(asgi_client) -> None:
    llm = FakeLLM()
    store = InMemorySessionStore(max_count=1)
    with wired_app(llm, store=store) as (app, _settings):
        client = asgi_client(app)
        native = await client.post("/chat", content="native first turn")
        native.raise_for_status()
        session_id = native.headers["X-Session-Id"]

        for _ in range(5):
            response = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "ephemeral"}]},
            )
            response.raise_for_status()

        assert store.ids() == [session_id]
        continued = await client.post(
            "/chat",
            content="native follow-up",
            headers={"X-Session-Id": session_id},
        )
        continued.raise_for_status()
        assert continued.headers["X-Session-Id"] == session_id
        assert len((await store.get(session_id)).messages) == 4


@pytest.mark.parametrize(
    ("requested_model", "expected_model"),
    [("requested", "requested"), (None, "selected")],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_reports_the_model_that_orchestration_actually_executes(
    asgi_client,
    parse_sse,
    requested_model: str | None,
    expected_model: str,
    stream: bool,
) -> None:
    selected = FakeLLM()
    requested = FakeLLM()
    registry = RegistryStub({"selected": selected, "requested": requested})
    orchestrator = SelectingOrchestrator()
    with wired_app(
        selected,
        registry=registry,
        orchestrator=orchestrator,
    ) as (app, _settings):
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": "route me"}],
            "stream": stream,
        }
        if requested_model is not None:
            payload["model"] = requested_model
        response = await asgi_client(app).post("/v1/chat/completions", json=payload)

    response.raise_for_status()
    if stream:
        frames = [event for event in parse_sse(response.text) if event != "[DONE]"]
        assert frames
        assert all(frame["model"] == expected_model for frame in frames)
    else:
        assert response.json()["model"] == expected_model
    assert orchestrator.calls == 1
    executed = selected if expected_model == "selected" else requested
    unused = requested if expected_model == "selected" else selected
    assert executed.complete_calls + executed.stream_calls == 1
    assert unused.complete_calls + unused.stream_calls == 0


@pytest.mark.parametrize("orchestrated", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_unknown_model_is_rejected_before_session_inventory_or_execution(
    asgi_client,
    monkeypatch: pytest.MonkeyPatch,
    orchestrated: bool,
    stream: bool,
) -> None:
    llm = FakeLLM()
    mcp = CountingMCP()
    created_sessions = 0
    real_session = Session

    def counting_session() -> Session:
        nonlocal created_sessions
        created_sessions += 1
        return real_session()

    monkeypatch.setattr(openai_compatible, "Session", counting_session)
    registry = (
        RegistryStub({"selected": llm, "requested": llm}) if orchestrated else None
    )
    orchestrator = SelectingOrchestrator() if orchestrated else None
    with wired_app(
        llm,
        mcp=mcp,
        registry=registry,
        orchestrator=orchestrator,
    ) as (app, settings):
        store = runtime_of(app).store
        initial_ids = store.ids()
        response = await asgi_client(app).post(
            "/v1/chat/completions",
            json={
                "model": "bogus",
                "messages": [{"role": "user", "content": "must not run"}],
                "stream": stream,
            },
        )

    available = "selected, requested" if orchestrated else settings.llm.model
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": f"invalid model 'bogus'; available model IDs: {available}",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }
    assert created_sessions == 0
    assert store.ids() == initial_ids
    assert mcp.inventory_reads == 0
    assert orchestrator is None or orchestrator.calls == 0
    assert llm.complete_calls == 0
    assert llm.stream_calls == 0


async def test_ordered_history_replays_without_altering_user_content(
    asgi_client,
) -> None:
    llm = FakeLLM()
    tool_block = (
        "\n<details>\n<summary>🔧 web__search ✅</summary>\ntool output\n</details>\n"
    )
    user_text = f"user-authored block stays{tool_block}after"
    with wired_app(llm) as (app, _settings):
        response = await asgi_client(app).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "system one"},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": f"checked{tool_block}done"},
                    {"role": "tool", "content": "ignored unsupported role"},
                    {"role": "system", "content": "system two"},
                    {"role": "user", "content": "active prompt"},
                ]
            },
        )

    response.raise_for_status()
    seen = llm.requests_seen[0].messages
    assert [message.role for message in seen] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert seen[0].content[0].text == user_text
    assert seen[1].content[0].text == "checkeddone"
    assert seen[2].content[0].text == "active prompt"
    assert llm.requests_seen[0].system == "system one\n\nsystem two"


async def test_trailing_assistant_is_rejected_instead_of_reordered(asgi_client) -> None:
    llm = FakeLLM()
    with wired_app(llm) as (app, _settings):
        store = runtime_of(app).store
        response = await asgi_client(app).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "prefill"},
                ],
                "stream": True,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "the final conversational message must have role 'user'"
    )
    assert llm.complete_calls == llm.stream_calls == 0
    assert len(store) == 0


async def test_assistant_only_request_preserves_no_user_error(asgi_client) -> None:
    with wired_app(FakeLLM()) as (app, _settings):
        response = await asgi_client(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "assistant", "content": "hello"}]},
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "no user message found in 'messages'"
