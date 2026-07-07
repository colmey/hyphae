# Phase 6 Build Guide - Prompted-Tool Encoder + Adapter Drop-In Proof

> **For the executing agent.** This is a complete, self-contained implementation
> guide for Phase 6. Read
> [EXECUTION_PLAN/phase-6-prompted-tool-adapter-dropin.md](EXECUTION_PLAN/phase-6-prompted-tool-adapter-dropin.md)
> first for canonical scope, then
> [EXECUTION_PLAN/shared/design-principles.md](EXECUTION_PLAN/shared/design-principles.md)
> and [EXECUTION_PLAN/shared/verification-and-dod.md](EXECUTION_PLAN/shared/verification-and-dod.md).
> Line numbers are anchors only; always read the surrounding code before editing.

---

## 1. Why (the problem you are fixing)

Phase 5 made model capabilities declarative. Phase 6 must prove the model-interface
boundary is real.

Today a model with poor or absent native tool calling simply never emits native
`tool_calls`, so the harness cannot degrade gracefully for weak/local models. The
missing path is a prompted-tool encoder: render the selected tool schemas into the
prompt, ask the model for one structured action in prose, parse that action, and return
the same normalized `ToolUseBlock` the loop already understands.

This phase is also the architecture test from the design reference: prompted tool use is
a genuinely different algorithm from native function calling. If it can drop in without
changing `agent/loop.py`, the adapter seam is in the right place. If the loop needs a
branch, the abstraction leaked; fix the boundary instead of patching the loop.

One sentence: **implement prompted tool calling as an `LLMClient` wrapper selected by
`ModelProfile.supports_native_tools == false`, while keeping the agent loop byte-identical.**

---

## 2. Confirmed design decisions (do not re-litigate)

1. **Use a wrapper/decorator `LLMClient` by default.** Build the normal provider client
   first, then wrap it for prompted-tool models. Do not add provider-specific branches.
2. **Selection is profile-driven.** `supports_native_tools: false` activates the wrapper.
   Omitted or `true` means native behavior is unchanged.
3. **The loop stays unchanged.** `agent/loop.py` must have an empty diff at the end of
   the phase. Treat any required loop edit as an adapter-boundary failure.
4. **One action per turn.** Prompted mode supports either a final text answer or one tool
   action. No parallel calls, no multi-action arrays, no streaming integration.
5. **Prompted action format is boring JSON.** Prefer one fenced JSON object:

```json
{
  "tool": "server__tool_name",
  "arguments": {"arg": "value"}
}
```

6. **No new public settings.** This is selected by the existing `models.yaml`
   `supports_native_tools` capability.

### Doctrine that constrains you

- **Algorithm -> adapter. Value -> config.** Prompted parsing is an algorithm, so it lives
  in an adapter/wrapper, not as flags inside providers or the loop.
- **Same normalized response.** The wrapper returns `AssistantMessage` with normal
  `TextBlock` and `ToolUseBlock` objects. The loop must not know whether a tool call came
  from native API fields or a parsed prose action.
- **No silent `{}`.** Bad or non-dict action arguments become `parse_error` on a
  `ToolUseBlock`, or a final visible error after bounded repair. Never execute an empty
  call because parsing failed.
- **Dispatch on declared capability fields.** Do not branch on provider name, model name,
  or local-model guesses.
- **Keep provider SDK imports lazy.** `llm/client.py` stays SDK-import-free at module
  import time.
- **Do not duplicate retry policy.** If wrapper-level retry needs shared mechanics, first
  extract a stable helper; do not copy `agent/loop.py` retry logic and do not call loop
  internals.

---

## 3. Architecture you are touching (orientation)

Request flow remains:

```text
POST /chat or /v1/chat/completions
  -> api/turn.py TurnRunner
  -> orchestrator decides model + tools + system + thinking_level
  -> agent/loop.py run_agent()
  -> llm.client.LLMClient.complete(messages, tools, system, ...)
  -> MCP dispatch happens only after normalized ToolUseBlock is returned
```

Phase 6 inserts a wrapper at client construction time:

```text
orchestrator/registry.py
  -> llm/client.py build_llm_client_from_entry(entry, settings)
       profile = entry.to_profile()
       base = _build_client(...)
       if not profile.supports_native_tools:
           base = PromptedToolLLMClient(base, profile=profile, model=entry.model)
       return base
```

Native model path:

```text
run_agent tools=[...]
  -> OpenAILLMClient/GeminiLLMClient complete(..., tools=[...])
  -> provider native tool_calls
  -> AssistantMessage(content=[ToolUseBlock(...)]
```

Prompted model path:

```text
run_agent tools=[...]
  -> PromptedToolLLMClient complete(..., tools=[...])
       render tools into system prompt
       call inner.complete(..., tools=None)
       parse inner visible text as either action JSON or final answer
       return AssistantMessage(content=[ToolUseBlock(...)] or [TextBlock(...)])
  -> loop sees the same normalized shape
```

Important current code facts:

- `llm/schemas.py` already has `ModelProfile.supports_native_tools`,
  `ToolUseBlock.parse_error`, and `AssistantMessage.reasoning`.
- `llm/client.py` already threads `profile` through provider builders.
- `agent/loop.py` already turns `ToolUseBlock.parse_error` into a teaching
  `is_error` result. Do not change that.
- `config/models.yaml` already documents `supports_native_tools`; Phase 6 should add a
  prompted-tools example row, not mutate live model entries.

---

## 4. Target action format

Use exactly one primary action object. Keep the prompt short and deterministic.

### Final answer

If the model has enough information, it writes normal final prose. The wrapper returns it
as visible `TextBlock` content.

### Tool action

If the model needs a tool, it writes a single JSON object, preferably fenced:

````markdown
```json
{"tool": "srv__lookup", "arguments": {"query": "example"}}
```
````

Accepted aliases for tolerance:

- `tool` or `name` for the tool name.
- `arguments`, `args`, or `input` for the argument object.

Required normalized result:

```python
ToolUseBlock(
    id="prompted_<stable_suffix>",
    name="srv__lookup",
    input={"query": "example"},
)
```

Invalid action handling:

- Unknown tool name -> `ToolUseBlock(..., input={}, parse_error="unknown prompted tool ...")`
  so the loop returns a model-facing error without MCP dispatch.
- Non-dict arguments -> `ToolUseBlock(..., input={}, parse_error="arguments must be a JSON object ...")`.
- Invalid JSON -> bounded repair re-ask once; if still invalid, return visible text explaining
  the parse failure rather than a tool call.

---

## 5. Task-by-task implementation

### Task A - Add pure prompted-tool helpers

**File: `llm/prompted_tools.py`** (new)

Create a small module that contains only provider-agnostic prompting/parsing plus the
wrapper. Start with pure helpers so tests can pin behavior before integration.

Suggested constants:

```python
_ACTION_INSTRUCTIONS = """\
You can call one tool by returning exactly one JSON object in a ```json fence:
{"tool":"server__tool_name","arguments":{"arg":"value"}}

Rules:
- Call at most one tool.
- Use only a listed tool name.
- "arguments" must be a JSON object matching that tool's schema.
- If no tool is needed, answer normally and do not include action JSON.
"""

_REPAIR_INSTRUCTIONS = """\
Your previous tool action could not be parsed:
{error}

Return exactly one corrected JSON action object in a ```json fence, or answer normally
if no tool is needed.
"""
```

Implement:

```python
def render_prompted_tools(tools: list[dict[str, Any]]) -> str:
    """Return compact tool instructions for the model-visible system prompt."""
```

Rendering requirements:

- Include only `name`, one-line `description`, and compact JSON schema.
- Use `json.dumps(..., separators=(",", ":"), ensure_ascii=False, sort_keys=True)` for schemas.
- If there are no tools, return an empty string.
- Do not include MCP internals or execution policy details.

Suggested render shape:

```text
Available tools:
- srv__lookup: Lookup a value
  schema: {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
```

Implement:

```python
@dataclass(frozen=True)
class PromptedAction:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class PromptParseResult:
    action: PromptedAction | None = None
    text: str = ""
    error: str | None = None


def parse_prompted_action(text: str, allowed_tools: set[str]) -> PromptParseResult:
    """Parse a single prompted action from model text, or return final text."""
```

Parsing requirements:

- Prefer the first fenced `json` block if present.
- Otherwise accept the whole response if it appears to be a single JSON object.
- Treat ordinary prose as final text, not an error.
- If there are multiple JSON-looking blocks, parse the first and ignore the rest; record
  this in tests as intentionally simple v1 behavior.
- Validate tool name is present and allowed.
- Validate arguments are a dict.
- Preserve final answer text when no action is present.

Suggested extraction helpers:

```python
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

def _extract_action_candidate(text: str) -> str | None:
    ...
```

Do **not** use ad hoc brace matching beyond a simple single-object fallback unless tests
prove it is needed. This phase is a narrow v1.

### Task B - Implement `PromptedToolLLMClient`

**File: `llm/prompted_tools.py`**

Implement a wrapper over any existing `LLMClient`:

```python
class PromptedToolLLMClient(LLMClient):
    def __init__(
        self,
        inner: LLMClient,
        *,
        model: str | None = None,
        max_repairs: int = 1,
    ) -> None:
        self._inner = inner
        self._model = model
        self._max_repairs = max_repairs

    def is_transient_error(self, exc: BaseException) -> bool:
        return self._inner.is_transient_error(exc)

    async def complete(...same signature...) -> AssistantMessage:
        ...
```

Core behavior:

1. If `tools` is empty or `response_schema is not None`, call `inner.complete(...)` unchanged.
2. If tools are present, append prompted-tool instructions to `system`.
3. Call `inner.complete(..., tools=None, system=rendered_system, ...)`.
4. Preserve `usage`, `model`, and `reasoning` from the inner response.
5. Parse visible text from the inner response.
6. If final prose, return the original response or a normalized `AssistantMessage` with text.
7. If valid action, return `AssistantMessage(content=[ToolUseBlock(...)], ...)`.
8. If parse error, perform one repair re-ask by appending a synthetic user message containing
   the repair instructions. If repair fails, return a visible `TextBlock` explaining the failure.

Important: The wrapper should not execute tools. It only returns `ToolUseBlock`s.

Suggested text extraction:

```python
def _visible_text(message: AssistantMessage) -> str:
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))
```

Suggested call shape:

```python
tool_prompt = render_prompted_tools(tools or [])
effective_system = _join_system(system, tool_prompt)
inner = await self._inner.complete(
    messages=messages,
    tools=None,
    system=effective_system,
    max_tokens=max_tokens,
    response_schema=response_schema,
    thinking_level=thinking_level,
)
```

Repair call:

```python
repair_messages = list(messages)
repair_messages.append(Message.assistant(inner.content))
repair_messages.append(Message.user(_REPAIR_INSTRUCTIONS.format(error=parse.error)))
repair = await self._inner.complete(
    messages=repair_messages,
    tools=None,
    system=effective_system,
    max_tokens=max_tokens,
    response_schema=None,
    thinking_level=thinking_level,
)
```

Failure after repair:

```python
return AssistantMessage(
    content=[TextBlock(text=f"I could not parse a valid tool action: {parse.error}")],
    stop_reason="end_turn",
    model=inner.model or self._model,
    usage=combined_usage,
    reasoning=inner.reasoning,
)
```

Usage note: If both first and repair calls report `Usage`, sum them with `Usage.__add__`.
If either is `None`, keep the non-None usage or `None`; do not invent token counts in the wrapper.

Tool-call id:

```python
def _new_tool_id() -> str:
    return f"prompted_{uuid.uuid4().hex[:12]}"
```

Stable-enough unique IDs are sufficient. Do not derive IDs from args; repeated-call detection
already uses `(name, args)`.

### Task C - Wire profile selection in `llm/client.py`

**File: `llm/client.py`**

Only `build_llm_client_from_entry()` should wrap prompted clients. The legacy
`build_llm_client(settings)` path has no model entry and must remain unchanged.

Sketch:

```python
def build_llm_client_from_entry(entry: Any, settings: Any) -> LLMClient:
    profile = entry.to_profile()
    client = _build_client(
        entry.provider,
        model=entry.model,
        max_tokens=entry.max_tokens or settings.llm_max_tokens,
        settings=settings,
        profile=profile,
    )
    if not profile.supports_native_tools:
        from llm.prompted_tools import PromptedToolLLMClient
        client = PromptedToolLLMClient(client, model=entry.model)
    return client
```

Constraints:

- Import `PromptedToolLLMClient` lazily inside the function.
- Do not change `_PROVIDERS`.
- Do not add SDK imports to `llm/client.py`.
- Do not wrap native models.

### Task D - Add startup/config guardrails

The orchestrator itself uses `response_schema=OrchestrationResult` and must not be routed
through prompted-tool behavior if that behavior would weaken structured output. The safest
v1 rule: **the orchestrator model must declare `supports_native_tools: true`**. This is not
because the orchestrator uses tools; it is because `supports_native_tools:false` means this
client is a weak/prose-oriented adapter target, not the structured-output control model.

**File: `main.py` or `orchestrator/registry.py`**

Where `_try_build_orchestration()` chooses `orch_model_id`, add a startup check after
`registry.get_entry(orch_model_id)` is available and before `registry.get(orch_model_id)`.

Sketch:

```python
orch_entry = registry.get_entry(orch_model_id)
if not orch_entry.supports_native_tools:
    logger.warning(
        "orchestrator_model_id=%r declares supports_native_tools:false; "
        "prompted-tool models are not supported as orchestrator control models. "
        "running in legacy mode.",
        orch_model_id,
    )
    return None, None
```

Document this as a v1 guardrail. If the user later wants prompted/prose orchestrators, make
that a separate phase with dedicated structured-output parsing.

### Task E - Hermetic tests

**File: `tests/smoke_test_prompted_tools.py`** (new standalone script, no pytest)

Follow the style of `tests/smoke_test_capabilities.py`:

- `main()` runs scenarios.
- `check(cond, msg)` records failures and prints `[PASS]` / `[FAIL]`.
- No network, no live backend.
- Run with `./runscript.sh tests/smoke_test_prompted_tools.py`.

Required scenarios:

1. **Render is compact and complete**
   - Given one tool, output includes name, description, compact schema, and action rules.
   - Given no tools, render returns empty string.

2. **Parser: valid fenced JSON**
   - Fenced action parses into `PromptedAction(name, arguments)`.

3. **Parser: unfenced JSON object**
   - Whole-response JSON object parses the same way.

4. **Parser: final prose**
   - Ordinary answer text returns `action=None`, `error=None`, and preserves text.

5. **Parser: malformed JSON**
   - Fenced malformed JSON returns `error` and no action.

6. **Parser: unknown tool and non-dict args**
   - Unknown tool is an error.
   - `arguments: "not an object"` is an error.

7. **Wrapper sends prompted tools, not native tools**
   - A scripted inner `LLMClient` records calls.
   - Wrapper call receives `tools=[...]`.
   - Inner call sees `tools=None`.
   - Inner system contains the rendered prompted-tool instructions.

8. **Wrapper returns normalized ToolUseBlock**
   - Inner prose response is fenced action JSON.
   - Wrapper response contains one `ToolUseBlock` with name/input and no visible text.

9. **Tool result history is visible to prompted model**
   - On a second wrapper call with a prior `ToolResultBlock` in messages, assert the inner
     messages/system still give the model enough context to answer. Minimal v1 can pass
     existing messages through unchanged if provider translation renders tool messages as text;
     if it does not, add wrapper-side rendering and test it explicitly.

10. **Repair re-ask**
    - First inner response has malformed action JSON.
    - Wrapper makes exactly one repair call containing the parse error.
    - Repair response with valid action returns `ToolUseBlock`.

11. **Repair failure is visible, not a silent tool call**
    - First and repair responses malformed.
    - Wrapper returns visible `TextBlock` explaining parse failure and no `ToolUseBlock`.

12. **Drop-in proof through unchanged loop**
    - Use `run_agent()` with wrapped scripted prose-only model and fake MCP.
    - Script:
      - first model response: prompted JSON action for `srv__lookup`
      - fake MCP returns content
      - second model response: final prose answer using the tool result
    - Assert one `ToolCallEvent`, one `ToolResultEvent`, final `TextEvent`, done `end_turn`.

13. **Native path unaffected**
    - Build a `ModelEntry` with omitted/true `supports_native_tools`; assert
      `build_llm_client_from_entry()` returns the base provider client, not the wrapper.
      If direct provider construction would need SDK/client setup, use monkeypatch-style
      replacement of `_PROVIDERS` inside the script and restore it afterward.

14. **Profile false wraps**
    - With `supports_native_tools:false`, assert factory returns `PromptedToolLLMClient`
      around the base fake client.

Do not use pytest monkeypatch. If you temporarily mutate `llm.client._PROVIDERS`, restore it in
`try/finally`.

### Task F - Documentation and config example

Update docs in the same session:

- `config/models.yaml`: add a commented prompted-tools example row only. Do not change live
  model entries.
- `docs/architecture.md`: document prompted-tool wrapper as the second dialect proof.
- `docs/configuration.md`: document `supports_native_tools:false` behavior and orchestrator
  guardrail.
- `docs/operations.md`: add weak-model onboarding runbook and `smoke_test_prompted_tools.py`.
- `EXECUTION_PLAN/phase-6-prompted-tool-adapter-dropin.md`: fill Session Notes.
- `EXECUTION_PLAN/README.md`: flip Phase 6 status only after verification.

Suggested commented example:

```yaml
  # weak-local-prompted:
  #   provider: openai
  #   model: small-local-model
  #   description: >
  #     Local prose-only model with no reliable native tool calling. Uses the
  #     prompted-tool adapter selected by supports_native_tools: false.
  #   supports_native_tools: false
  #   thinking: none
  #   context_window: 32768
  #   max_tokens: 4096
```

---

## 6. Tests (hermetic; each must fail without the implementation)

New test:

```text
./runscript.sh tests/smoke_test_prompted_tools.py
```

Required existing verification:

```text
./runscript.sh -c "import main"
./runscript.sh tests/eval_agent.py
./runscript.sh tests/smoke_test_prompted_tools.py
./runscript.sh tests/smoke_test_capabilities.py
./runscript.sh tests/smoke_test_reliability.py
./runscript.sh tests/smoke_test_context_assembly.py
./runscript.sh tests/smoke_test_openai_api.py
git diff -- agent/loop.py
git diff --check
```

Expected evidence:

- `eval_agent.py` remains at the current hermetic pass count.
- `smoke_test_prompted_tools.py` prints pass/fail checks and exits non-zero on failure.
- `git diff -- agent/loop.py` prints no diff. If it prints anything, Phase 6 is not done.

Optional live check, record as not verifiable if unavailable:

```text
TRACE_ENABLED=true TRACE_PATH=traces/phase6-live.jsonl ./runscript.sh -m uvicorn main:app --host 127.0.0.1 --port 8000
curl -sS http://127.0.0.1:8000/chat \
  -H "Content-Type: text/plain" \
  --data-binary "Use the available lookup/search tool to answer a simple question."
```

Use a `models.yaml` entry with `supports_native_tools:false` and a weak/prose model for the live
check. Inspect traces for normal `tool_call` / `tool_result` events; there should be no new
loop-specific prompted event type.

---

## 7. Acceptance criteria (all must hold)

- [ ] A scripted prose-only model completes a multi-step tool task through `run_agent()` with
      no edits to `agent/loop.py`.
- [ ] `supports_native_tools:false` wraps a model client; omitted/true does not.
- [ ] Prompted action JSON parses into normalized `ToolUseBlock`s with JSON-able dict args.
- [ ] Malformed action JSON triggers one repair re-ask, then a visible error if repair fails.
- [ ] Native model paths are unaffected: eval suite and Phase 5 capability smoke stay green.
- [ ] A prompted-tools model can be onboarded as a commented `models.yaml` row only.
- [ ] Docs and Phase 6 Session Notes are updated with real verification evidence.

---

## 8. Out of scope (resist)

- Editing `agent/loop.py`.
- Creating an `adapters/` directory or broad adapter framework.
- Harmony channel parsing.
- Multi-action-per-turn or parallel prompted tool calls.
- Streaming/token-level prompted-tool behavior.
- Grammar-constrained decoding.
- Prompted orchestrator structured-output support.
- New environment variables or `Settings` fields.

---

## 9. Definition of done (finish the session)

- `PHASE_6_BUILD_GUIDE.md` has been followed or deviations are recorded.
- `tests/smoke_test_prompted_tools.py` exists and passes.
- Required verification commands are run through `./runscript.sh` where applicable.
- `git diff -- agent/loop.py` is empty and pasted/summarized in Session Notes.
- Docs/config examples are updated.
- `EXECUTION_PLAN/phase-6-prompted-tool-adapter-dropin.md` Session Notes are filled:
  Done / Verified / Not-verifiable / Deviations / Follow-ups / Pointer.
- `EXECUTION_PLAN/README.md` marks Phase 6 done only after verification.
- Do not commit or push unless the user explicitly asks.
