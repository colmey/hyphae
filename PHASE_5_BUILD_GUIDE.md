# Phase 5 Build Guide — Capability Profiles + Adapter-Seam Foundations

> **For the executing agent.** This is a complete, self-contained spec. You should not
> need to re-derive the architecture — paths, line numbers, and code sketches are inline.
> Read [EXECUTION_PLAN/phase-5-capability-profiles-and-adapter-seam.md](EXECUTION_PLAN/phase-5-capability-profiles-and-adapter-seam.md)
> for the canonical scope, [EXECUTION_PLAN/shared/design-principles.md](EXECUTION_PLAN/shared/design-principles.md)
> and [shared/verification-and-dod.md](EXECUTION_PLAN/shared/verification-and-dod.md) for doctrine + definition of done.
> Recommended model: Claude Opus 4.8 (`claude-opus-4-8`, effort `xhigh`).
> Line numbers are current-as-of-writing anchors — confirm by reading before editing.

---

## 1. Why (the problem you are fixing)

The harness *assumes* capabilities per provider instead of *declaring* them per model.
Four concrete leaks, all in `llm/providers/openai.py`:

| Leak | Location | Symptom |
|---|---|---|
| `thinking_level` dropped | `openai.py:102-103` | Orchestrator's "intelligence on demand" knob is a no-op on the default local model. |
| `<think>` reasoning discarded | `openai.py:46-50`, `:244` | Reasoning is stripped and thrown away, not traceable/debuggable. |
| Malformed tool args → `{}` | `openai.py:251-255` | Hallucinated args silently become an empty call — no error signal (the weak-local-model trap). |
| No home for sampling | — | Per-model temperature/top_p/top_k ("never greedy-decode Qwen") has nowhere to live. |

**Fix additively**: config rows *describe* models; the normalized response *carries*
reasoning; parse failures *become signals* the loop already knows how to handle. This is
**NOT** an `adapters/` rewrite — the existing `LLMClient`/`AssistantMessage` seam already
is the adapter interface. Restructure only if Phase 6 proves it can't hold a second dialect.

---

## 2. Confirmed design decisions (do not re-litigate)

1. **One `ModelProfile` value object**, not per-capability kwargs. Bundle capabilities into
   a single immutable object threaded as ONE new kwarg through the build chain.
2. **Dedicated `ReasoningEvent`** carries reasoning to the trace (trace-only).
3. **Wire `reasoning_effort` now** for `thinking: hint-param` models (config-only value pass-through).

### Doctrine that constrains you

- **Additive**: every new `ModelEntry`/`AssistantMessage`/`ToolUseBlock` field has a default
  that reproduces today's behavior. Existing `models.yaml` files parse unchanged.
- **Never strip `provider_metadata`.** `reasoning` is a new *sibling* field, not a replacement.
- **Config errors fail loud at startup**, naming the model id + field — not mid-request.
- **Dispatch on declared fields.** No `if provider == …`, no model-name string matching
  outside config.
- **Defaults live in ONE place** (the pydantic `ModelEntry`). Do NOT add parallel
  capability/default fallbacks in `api/turn.py` or provider code (Phase-3's cleanup lesson).
- `llm/client.py` stays **SDK-import-free at module level**; `_PROVIDERS` stays the single
  source of truth for providers.
- The loop consumes only the normalized `AssistantMessage` — it must **never learn what
  `<think>` is** (transport/dialect separation stays in `llm/providers/*`).

---

## 3. Architecture you're touching (orientation)

Request flow: `POST /chat` (or `/v1/chat/completions`) → `TurnRunner` (`api/turn.py`) →
`Orchestrator.decide()` → `run_agent()` (`agent/loop.py`) → LLM ⇄ MCP.

**The build chain sampling/capabilities must cross:**
```
orchestrator/registry.py  builds clients via
  llm/client.py: build_llm_client_from_entry(entry, settings)   # :175
    -> _build_client(provider, model=, max_tokens=, settings=)   # :139
      -> _build_openai(...) / _build_gemini(...)                 # :108 / :98
        -> OpenAILLMClient.__init__(api_key, model, default_max_tokens, base_url)  # openai.py:56
```
`build_llm_client(settings)` (`:158`) is the **legacy** path — no entry, so it must keep
working with `profile=None`.

**The trace only sees events.** `agent/loop.py` yields `Event`s; `agent/tracing.py`
`event_record()` (`:166`) maps each event type to a JSON record. `AssistantMessage.reasoning`
is NOT an event today — that's why decision #2 adds `ReasoningEvent`.

**Renderer event dispatch (verified):**
- `/v1/_stream` (`api/openai_compatible.py:216`) and `/chat` plaintext dispatch by
  `isinstance` / collect-text → they **ignore** unknown events. A new `ReasoningEvent` is
  invisible there. ✅ (satisfies "user-facing output unchanged" + the `/v1` acceptance criterion.)
- `/chat/stream` (`api/routes.py:109`) serializes **every** event via `event_record` → the
  `ReasoningEvent` WILL appear there. This is **by design** — that endpoint is the raw
  event-stream/debug view, the same audience as the trace. Record it as a deliberate decision.
- Replayed history: `to_message()` only carries `.content`; `reasoning` is a separate field
  → structurally never replayed. No stripping code needed.

---

## 4. Task-by-task implementation

### Task A — `ModelProfile` value object + `AssistantMessage.reasoning` + `ToolUseBlock.parse_error`
**File: `llm/schemas.py`** (canonical, provider-agnostic model-interface types)

Add a frozen dataclass (near `Usage`, keep dataclass style — this module is plain dataclasses):
```python
@dataclass(frozen=True)
class ModelProfile:
    """Declared model-interface capabilities, resolved from a models.yaml row.

    Provider-agnostic and immutable. Built by ModelEntry.to_profile(); the legacy
    (entry-less) client path uses default() to reproduce pre-Phase-5 behavior.
    """
    supports_native_tools: bool = True
    thinking: str = "none"                 # "none" | "hint-param" | "think-tags"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    @classmethod
    def default(cls) -> "ModelProfile":
        return cls()
```

Add `reasoning` to `AssistantMessage` (`:142-155`) — sibling field, keep docstring note:
```python
    usage: Usage | None = None
    reasoning: str | None = None   # provider-extracted chain-of-thought; trace-only,
                                    # never replayed to the model or sent on the wire.
```

Add `parse_error` to `ToolUseBlock` (`:48-64`):
```python
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None  # set by a provider when the model's tool-call
                                     # arguments were unparseable; the loop turns this
                                     # into a teaching is_error result (input stays {}).
    type: Literal["tool_use"] = "tool_use"
```

### Task B — capability fields + validation on `ModelEntry`
**File: `orchestrator/schemas.py`**

Add a nested sampling model above `ModelEntry`:
```python
class SamplingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
```

Extend `ModelEntry` (after `default`, `:50-55`) — flat siblings matching the existing
`context_window` pattern:
```python
    supports_native_tools: bool = Field(
        default=True,
        description="Whether the served endpoint supports native tool/function calling. "
                    "Declared only in Phase 5; prompted-tool fallback is Phase 6.",
    )
    thinking: Literal["none", "hint-param", "think-tags"] = Field(
        default="none",
        description="How this model exposes a thinking control. 'none': no knob "
                    "(default, reproduces prior behavior). 'hint-param': a request "
                    "field like reasoning_effort. 'think-tags': self-emits <think> inline.",
    )
    sampling: SamplingParams | None = Field(
        default=None,
        description="Optional per-model sampling passed through to the provider request.",
    )

    def to_profile(self) -> "ModelProfile":
        from llm.schemas import ModelProfile
        s = self.sampling
        return ModelProfile(
            supports_native_tools=self.supports_native_tools,
            thinking=self.thinking,
            temperature=s.temperature if s else None,
            top_p=s.top_p if s else None,
            top_k=s.top_k if s else None,
        )
```
`extra="forbid"` + `Field(ge/le)` already give "fail loud at startup naming the field" via
the existing `ModelsConfig` parse — no new validator needed unless a cross-field rule appears.
(Import `ModelProfile` lazily inside `to_profile` to avoid import-order coupling; `llm.schemas`
has no orchestrator dependency, so a top-level import is also fine — prefer whichever keeps
the module import graph clean.)

### Task C — thread the profile through the build chain
**File: `llm/client.py`**

- `build_llm_client_from_entry` (`:175`): `profile = entry.to_profile()`, pass
  `profile=profile` into `_build_client`.
- `_build_client` (`:139`): add `profile: Any = None` param, forward to the builder.
- `_build_openai` / `_build_gemini` (`:108` / `:98`): accept `profile=None`, forward to
  the constructor.
- `build_llm_client` (legacy, `:158`): leave call site as-is → `profile` stays `None`.

Keep the duck-typed `Any` typing already used for `entry`/`settings` so `llm/client.py`
does not import `ModelProfile` at module top if that would tangle imports — passing it
through opaquely is fine (the constructor is what reads fields).

### Task D — OpenAI provider behavior
**File: `llm/providers/openai.py`**

`__init__` (`:56-68`): add `profile: ModelProfile | None = None`; store
`self._profile = profile or ModelProfile.default()` and `self._warned_inert_thinking = False`.
Import `ModelProfile` from `llm.schemas` (already imports from there).

`complete()` request build (`:82-103`) — after the `max_tokens` line, merge sampling:
```python
        p = self._profile
        if p.temperature is not None: request["temperature"] = p.temperature
        if p.top_p is not None:       request["top_p"] = p.top_p
        if p.top_k is not None:       request["top_k"] = p.top_k
```
Replace the thinking no-op (`:102-103`) with capability-aware handling:
```python
        if thinking_level is not None:
            if p.thinking == "hint-param":
                request["reasoning_effort"] = thinking_level        # low|medium|high align
            elif p.thinking == "none":
                if not self._warned_inert_thinking:
                    logger.info(
                        "thinking_level=%r requested but model %s declares thinking:none; "
                        "the knob is inert for this model", thinking_level, self._model,
                    )
                    self._warned_inert_thinking = True
            # "think-tags": no request knob; the model self-emits <think>.
```

Reasoning capture — replace `_strip_reasoning` (`:46-50`) with a split helper reusing the
existing `_THINK_BLOCK` regex (`:31`):
```python
def _split_reasoning(text: str | None) -> tuple[str | None, str]:
    """Return (reasoning, visible). Pulls a leading <think>...</think> block out of
    content: the block's inner text becomes reasoning, the remainder is visible."""
    if not text:
        return None, ""
    m = _THINK_BLOCK.match(text)
    if not m:
        return None, text
    reasoning = re.sub(r"</?think>", "", m.group(0)).strip() or None
    return reasoning, text[m.end():]
```
In `_from_openai_response` (`:229-269`): call `reasoning, content = _split_reasoning(getattr(message, "content", None))`,
build the visible `TextBlock` from `content`, and set `AssistantMessage(..., reasoning=reasoning)`.
(Also set `reasoning=None` on the early no-choices return at `:235`.)

Malformed args (`:251-255`) — set `parse_error` instead of swallowing:
```python
            try:
                args = json.loads(raw_args) if raw_args else {}
                parse_error = None
            except (json.JSONDecodeError, TypeError):
                logger.warning("could not parse tool arguments: %r", raw_args)
                args = {}
                parse_error = f"arguments were not valid JSON: {raw_args!r}"
            blocks.append(ToolUseBlock(
                id=getattr(tc, "id", "") or "",
                name=getattr(fn, "name", "") if fn else "",
                input=args,
                parse_error=parse_error,
            ))
```

### Task E — Gemini provider (minimal, don't assume)
**File: `llm/providers/gemini.py`**

- `__init__` (`:43`): accept + store `profile` (default `ModelProfile.default()`).
- `complete()` `config_kwargs` (`:60-68`): add `temperature/top_p/top_k` from the profile
  when set (valid `GenerateContentConfig` fields). Leave the existing
  `thinking_level → ThinkingConfig` path unchanged (Gemini already honors it).
- `_from_genai_response`: leave `reasoning=None` unless the SDK trivially exposes a thought
  part — **detect, don't assume**; `None` is acceptable this phase.

### Task F — `ReasoningEvent` + loop wiring
**Files: `agent/events.py`, `agent/loop.py`, `agent/tracing.py`**

`agent/events.py`: add `ReasoningEvent` mirroring `TextEvent`:
```python
@dataclass
class ReasoningEvent:
    text: str
    type: Literal["reasoning"] = "reasoning"
```
(Match the existing event dataclass style/`type` literal convention in that file.)

`agent/loop.py`: import `ReasoningEvent`. After `session.append_assistant(response)` (`:424`)
and the `UsageEvent` emit (`:434-442`) — and BEFORE the `TextEvent` loop — add:
```python
        if response.reasoning:
            yield await _emit(ReasoningEvent(text=response.reasoning))
```
Do **not** append reasoning to session content (replay stays clean).

Malformed-args wiring — loop dispatch (`:527-536`). Before `_validate_tool_args`:
```python
                if tu.parse_error is not None:
                    validation_error = (
                        f"tool call arguments were not valid JSON ({tu.parse_error}); "
                        "return the arguments as a JSON object matching the tool schema."
                    )
                else:
                    validation_error = _validate_tool_args(tool_schemas.get(tu.name), tu.input)
```
Everything downstream (the `is_error` result, `_FAILURE_NUDGE`, `no_progress` abort, trace)
is reused unchanged.

`agent/tracing.py` `event_record` (`:182-217`): add a branch:
```python
    elif isinstance(event, ReasoningEvent):
        rec["reasoning"] = event.text
```
(Add `ReasoningEvent` to the imports at `:40-49`.)

### Task G — config example + docs
- `config/models.yaml`: flesh out the commented `qwen3.6` example (`:46-51`) with a **full
  profile** (`thinking: think-tags`, a `sampling:` block, `supports_native_tools: true`) and
  extend the per-entry field header comment (`:10-23`). **Leave the user's live uncommitted
  model additions untouched** — do not restyle or reorder existing entries.
- `docs/architecture.md`: model-interface section — reasoning as routable data (content/field
  separation), capability dispatch via `ModelProfile`, the `ReasoningEvent` trace path.
- `docs/configuration.md`: `models.yaml` capability-profile reference (fields, ranges, defaults).
- `docs/operations.md`: limitations (Gemini reasoning `None`; `supports_native_tools`
  unconsumed until Phase 6; reasoning visible in `/chat/stream` raw view by design).
- **No new `Settings`/`.env` vars** — all config is per-model in `models.yaml`.

---

## 5. Tests (hermetic; each must FAIL without your change)

New `tests/smoke_test_capabilities.py` — standalone runnable script (NOT pytest), reuse
`ScriptedLLM`/`ScriptedMCP` from `tests/eval_agent.py` and fakes in
`tests/smoke_test_reliability.py`:

1. **No-profile parity** — current `models.yaml` parses, builds a client, behaves identically.
2. **Bad profile fails startup** — `temperature: 9` or an unknown field → `ModelsConfig`
   parse raises with a message naming model id + field.
3. **Sampling reaches the request** — construct `OpenAILLMClient` with a profile, monkeypatch
   `self._client.chat.completions.create` to capture the request dict, assert
   `temperature/top_p/top_k` present (no network).
4. **Reasoning routing** — scripted `<think>…</think>` response → `AssistantMessage.reasoning`
   populated; visible `TextBlock` + `to_message()` content contain no think text; a
   `ReasoningEvent` is emitted and its trace record carries `reasoning`.
5. **Malformed args → visible error** — scripted tool call with invalid-JSON arguments →
   `ToolUseBlock.parse_error` set; driven through `run_agent`, the model sees a teaching
   `is_error` result (not a silent `{}` execution) and the loop recovers.
6. **Thinking honesty** — `hint-param` profile + `thinking_level` → `reasoning_effort` in the
   request; `none` profile → inert-knob info log + no `reasoning_effort`.

---

## 6. Verification (evidence, not assertions)

Canonical launcher: `./runscript.sh` (activates `.venv`, sets `PYTHONPATH`).

```
./runscript.sh -c "import main"                       # import OK, before and after
./runscript.sh tests/eval_agent.py                    # identical to baseline (25/25) — regression gate
./runscript.sh tests/smoke_test_capabilities.py       # new checks pass
./runscript.sh tests/smoke_test_reliability.py
./runscript.sh tests/smoke_test_context_assembly.py
./runscript.sh tests/smoke_test_openai_api.py
./runscript.sh tests/smoke_test_config.py
```

**Not verifiable here** (record exact commands in the phase-5 Session Notes for a human): a
live `/v1` round-trip against the local backend confirming a `<think>`-emitting model returns
clean visible text while the trace file shows a `reasoning` record; `reasoning_effort`
reaching the served endpoint.

---

## 7. Acceptance criteria (from the phase file — all must hold)

- [ ] Existing `models.yaml` (no profiles) boots and behaves identically; eval suite unchanged.
- [ ] A profile with bad values (unknown field, `temperature: 9`) fails startup naming the model id + field.
- [ ] Scripted `<think>…</think>`: reasoning lands in `AssistantMessage.reasoning` + trace record;
      user-visible text, session history, and `/v1` payloads contain no `<think>` content.
- [ ] Scripted malformed-arguments tool call → visible error signal (not silent `{}`); loop recovers.
- [ ] `sampling` values from a model entry demonstrably reach the provider request (hermetic assert).
- [ ] Definition of done per `EXECUTION_PLAN/shared/verification-and-dod.md`.

---

## 8. Out of scope (resist)

- `adapters/` restructure, `format_request`/`parse_response` extraction, Harmony channel
  parsing → Phase 6.
- Prompted-tool fallback (`supports_native_tools=false` behavior) → Phase 6. Here the flag
  only exists/validates/is visible.
- Real tokenizer, streaming, per-task thinking toggles in the orchestrator prompt.

## 9. Definition of done (finish the session)

- Fill the phase-5 file's **Session Notes** (Done / Verified / Not-verifiable / Deviations /
  Follow-ups / Pointer for next session).
- Flip the README status-table entry for Phase 5.
- Update the docs listed in Task G in the same session.
- Do **not** commit without confirming with the user first.
