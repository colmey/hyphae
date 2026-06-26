# STREAMING-NEXT-STEPS.md

**Action plan: real token-level streaming for the agent loop.**

Audience: a fresh Claude Code session. Read this top-to-bottom before touching
code. It assumes the repo conventions in `CLAUDE.md` and the design principles
in `AGENTIC-HARNESS-RESEARCH/`.

---

## 1. Problem statement

The OpenAI-compatible adapter already speaks SSE: `POST /v1/chat/completions`
with `stream: true` returns `chat.completion.chunk` frames terminated by
`[DONE]` ([api/openai_compat.py:185](api/openai_compat.py#L185),
[:265](api/openai_compat.py#L265)). **But the harness does not actually stream
tokens.** The whole answer arrives in one burst at the end. The reason is two
layers down, not at the wire:

1. The agent loop awaits the *entire* completion before emitting any text:
   `response = await _complete_with_retry(...)`
   ([agent/loop.py:367](agent/loop.py#L367)), then replays the finished blocks
   as `TextEvent`s ([agent/loop.py:407-410](agent/loop.py#L407-L410)).
2. The LLM client is non-streaming by contract. The ABC exposes only
   `complete()` ([llm/client.py:49-75](llm/client.py#L49-L75)); the OpenAI
   provider calls `chat.completions.create(**request)` **without** `stream=True`
   and awaits the full object ([llm/providers/openai.py:152](llm/providers/openai.py#L152)).

So `_stream` in the adapter faithfully forwards `TextEvent`s — there just aren't
any until generation finishes.

**Goal:** emit `TextEvent`s incrementally as the model generates, end to end,
without violating any harness invariant.

---

## 2. Goals and non-goals

**Goals**
- True incremental `TextEvent` emission from `run_agent` when a caller opts in.
- Implemented for the OpenAI-compatible provider (covers the user's local
  Ollama/qwen3.6 backend).
- **Model-agnostic:** providers that don't implement streaming keep working,
  transparently, via a fallback. No provider is *forced* to implement it.
- Minimally invasive: the legacy non-streaming path (`/chat`, smoke tests, the
  orchestrator's structured call) is byte-for-byte unchanged.

**Non-goals (explicitly out of scope for v1)**
- Streaming the **orchestrator** decision. It is a single structured-output
  (`response_schema`) call and must stay non-streaming — run it once before the
  stream opens, exactly as today ([docs/operations.md:126-127](docs/operations.md#L126-L127)).
- Streaming **tool-call deltas** to the client as they assemble. Tool calls must
  be fully assembled before execution (sequential-execution invariant), so we
  *accumulate* tool deltas internally and only stream **text**. This matches the
  research caveat that local servers stream tool-call deltas unreliably
  ([AGENTIC-HARNESS-RESEARCH/03_model_interface.md:75-77](AGENTIC-HARNESS-RESEARCH/03_model_interface.md#L75-L77)).
- Mid-stream retry/resume. Retry only applies *before the first delta*; once
  bytes are on the wire we cannot un-send them. See §6.
- A new env var / kill-switch. Streaming is gated by the existing per-request
  `stream` flag. (Optional toggle discussed in §9, not required.)

---

## 3. Design principles to honor (from the codebase)

These are hard constraints. Re-read them before designing:

- **`agent/loop.py` is the only bridge** between `LLMClient` and `MCPManager`
  (`CLAUDE.md`). The streaming protocol must stay inside the loop ⇄ provider
  seam; the MCP layer and the API layer must not learn about stream chunks.
- **"Keep streaming behind the interface so the loop can ignore it"**
  ([03_model_interface.md:77](AGENTIC-HARNESS-RESEARCH/03_model_interface.md#L77)).
  Translation here: the *loop* opts into streaming, but a provider with no
  native streaming still satisfies the interface — so the **ABC ships a default
  `stream()` that wraps `complete()`**.
- **Never strip `provider_metadata`** from `TextBlock`/`ToolUseBlock`
  (`CLAUDE.md`). The assembled `AssistantMessage` from a stream must carry the
  same fields a non-streamed one would.
- **Sequential tool execution; no parallel** (`CLAUDE.md`). Unaffected — we
  assemble tool calls fully, then run the existing executor.
- **Orchestration must never break a request** and **degrades to a safe
  default** (`CLAUDE.md`). The streaming path must degrade to non-streaming on
  any provider that lacks it.
- **`llm/client.py` must not import a provider SDK at module level** (`CLAUDE.md`).
  The streaming chunk types live in `llm/schemas.py` (already SDK-free), not in a
  provider module.

---

## 4. The shape of the change

One new concept: a tiny provider-agnostic **stream chunk** protocol that the
loop consumes. Everything else is plumbing a `stream: bool` flag through.

```
/v1 (stream:true) ── stream=True ─┐
                                  ├─ _turn_events(stream=…) ─ run_agent(stream=…)
/chat ───────────── stream=False ─┘                               │
/v1 (stream:false) ─ stream=False ─ _run_turn ───────────────────┘
                                                                  │
                          ┌───────────────────────────────────────┘
                          ▼
            stream=True → llm.stream(...)  → TextDelta* , StreamEnd(AssistantMessage)
            stream=False → _complete_with_retry(...) → AssistantMessage  (UNCHANGED)
                          │
                          ▼  (shared tail, unchanged)
            append_assistant → UsageEvent → tool extraction → sequential exec → loop
```

The provider's `stream()` yields `TextDelta`s **and** a terminal
`StreamEnd(message=AssistantMessage)`. The loop streams the deltas as
`TextEvent`s and then feeds the assembled `AssistantMessage` into the *existing*
tail of the loop (append, usage, tool handling) — so the two paths converge
after the LLM call and share all tool logic.

---

## 5. File-by-file implementation plan

### 5.1 `llm/schemas.py` — add the stream-chunk types

SDK-free dataclasses next to `AssistantMessage`. Keep the existing dataclass
style (no Pydantic).

```python
@dataclass
class TextDelta:
    """A fragment of assistant text produced mid-generation."""
    text: str
    type: Literal["text_delta"] = "text_delta"


@dataclass
class StreamEnd:
    """Terminal chunk of a stream: the fully assembled assistant turn.

    `message` is identical in shape to what `complete()` would have returned
    (text + tool_use blocks, stop_reason, model, usage) so the loop's tail is
    provider- and path-agnostic. provider_metadata is preserved on its blocks.
    """
    message: AssistantMessage
    type: Literal["stream_end"] = "stream_end"


StreamChunk = Union[TextDelta, StreamEnd]
```

Export them from `llm/__init__.py` alongside the other schema re-exports
(check what's re-exported there and match it).

### 5.2 `llm/client.py` — default `stream()` on the ABC (the model-agnostic hinge)

Add a **concrete** (not `@abstractmethod`) async-generator method to `LLMClient`.
This is what keeps the feature model-agnostic: any provider that implements only
`complete()` (Gemini today, the test `FakeLLM`) gets correct — if coarse —
streaming for free.

```python
async def stream(
    self,
    messages: list[Message],
    tools: list[dict[str, Any]] | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    thinking_level: str | None = None,
) -> AsyncIterator[StreamChunk]:
    """Token stream for one completion turn.

    Default implementation: no native streaming. Run complete() and emit its
    text as a single TextDelta per text block, then the StreamEnd. Providers
    override this for true token-level streaming. Streaming stays *behind* the
    interface — the loop consumes StreamChunks identically regardless of whether
    the provider streams natively (research doc 03 §3).

    Note: response_schema is intentionally absent — streaming never carries
    structured output; the orchestrator's structured call uses complete().
    """
    msg = await self.complete(
        messages=messages, tools=tools, system=system,
        max_tokens=max_tokens, thinking_level=thinking_level,
    )
    for block in msg.content:
        if isinstance(block, TextBlock) and block.text:
            yield TextDelta(text=block.text)
    yield StreamEnd(message=msg)
```

Imports: add `AsyncIterator` (typing), and `TextBlock`, `TextDelta`, `StreamEnd`,
`StreamChunk` from `.schemas`. These are SDK-free — the module-level "no provider
SDK" rule is preserved.

### 5.3 `llm/providers/openai.py` — native streaming override

This is the substantive provider work. Three pieces:

**(a) DRY the request shaping.** `complete()` and `stream()` build the same
request dict. Extract a private `_build_request(messages, tools, system,
max_tokens, response_schema, thinking_level) -> dict` from the body of
`complete()` ([openai.py:116-150](llm/providers/openai.py#L116-L150)) and call it
from both. Avoids drift. `complete()` then adds nothing; `stream()` adds
`stream=True` and `stream_options`.

**(b) The `stream()` method:**

```python
async def stream(self, messages, tools=None, system=None,
                 max_tokens=None, thinking_level=None):
    request = self._build_request(messages, tools, system, max_tokens,
                                  response_schema=None, thinking_level=thinking_level)
    request["stream"] = True
    # Ask for a final usage chunk. Real OpenAI honors this; many local servers
    # omit it — we degrade to Usage() zeros (research 03 §3: never depend on usage).
    request["stream_options"] = {"include_usage": True}

    stripper = _ReasoningStreamStripper()   # see (c)
    text_parts: list[str] = []
    tool_accs: dict[int, dict[str, str]] = {}   # index -> {id,name,args}
    finish_reason: Any = None
    raw_usage: Any = None
    model = self._model

    sdk_stream = await self._client.chat.completions.create(**request)
    async for chunk in sdk_stream:
        if getattr(chunk, "usage", None):
            raw_usage = chunk.usage
        for choice in getattr(chunk, "choices", None) or []:
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            piece = getattr(delta, "content", None)
            if piece:
                visible = stripper.feed(piece)
                if visible:
                    text_parts.append(visible)
                    yield TextDelta(text=visible)
            for tc in getattr(delta, "tool_calls", None) or []:
                acc = tool_accs.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if getattr(tc, "id", None):
                    acc["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn and getattr(fn, "name", None):
                    acc["name"] = fn.name
                if fn and getattr(fn, "arguments", None):
                    acc["args"] += fn.arguments

    # ----- assemble the terminal AssistantMessage -----
    blocks: list[Any] = []
    full_text = "".join(text_parts)        # already reasoning-stripped, incrementally
    if full_text:
        blocks.append(TextBlock(text=full_text))
    for idx in sorted(tool_accs):
        acc = tool_accs[idx]
        try:
            args = json.loads(acc["args"]) if acc["args"] else {}
        except (json.JSONDecodeError, TypeError):
            logger.warning("could not parse streamed tool arguments: %r", acc["args"])
            args = {}
        blocks.append(ToolUseBlock(id=acc["id"] or "", name=acc["name"] or "", input=args))

    msg = AssistantMessage(
        content=blocks,
        stop_reason=_canonical_stop_reason(finish_reason) if blocks else "empty",
        model=model,
        usage=self._usage_from_stream(raw_usage),
    )
    yield StreamEnd(message=msg)
```

Add `_usage_from_stream(raw_usage)` mirroring `_usage_from_response`
([openai.py:335-353](llm/providers/openai.py#L335-L353)) — same field mapping,
just reading the streamed `usage` object; return `Usage()` when `None`.

**(c) Stateful reasoning stripper.** Critical for the user's model: qwen3.6
inlines chain-of-thought as a leading `<think>...</think>` block
([openai.py:55-58](llm/providers/openai.py#L55-L58),
[:80-84](llm/providers/openai.py#L80-L84)). The existing `_strip_reasoning`
regexes the *whole* string — impossible across deltas. Without a streaming-aware
stripper the user would watch raw `<think>` reasoning stream into OpenWebUI.

Implement a small state machine whose concatenated `feed()` output equals
`_strip_reasoning(full_content)` for the leading-block case:

```python
class _ReasoningStreamStripper:
    """Suppress a single leading <think>...</think> block across streamed deltas.

    States: HEAD (undecided — buffer until we know if a <think> opens),
    IN_THINK (drop until </think>), PASS (emit everything). The open/close tags
    can be split across deltas, so we buffer in HEAD until we have enough chars
    to decide, and buffer a small tail in IN_THINK to catch a split </think>.
    """
    _OPEN, _CLOSE = "<think>", "</think>"
    def __init__(self) -> None:
        self._buf = ""
        self._state = "HEAD"
    def feed(self, piece: str) -> str:
        # HEAD: accumulate; if buffer (lstripped) starts with a prefix of
        #   "<think>", keep buffering; once it definitively is "<think>…" switch
        #   to IN_THINK; once it definitively is NOT, flush buffer to PASS.
        # IN_THINK: accumulate; when "</think>" appears, drop through it, emit
        #   the remainder, switch to PASS. Retain a 7-char tail between feeds to
        #   catch a tag split across deltas.
        # PASS: return piece unchanged.
        ...
```

Write thorough unit coverage for this in the smoke test (§7): tag in one delta;
tag split mid-token (`"<thi"`+`"nk>"`); no think block at all; `</think>` split
across deltas; content that merely contains the substring later (must not strip).

> Keep `complete()` and `_strip_reasoning` exactly as they are — the
> non-streaming path is untouched.

### 5.4 `agent/loop.py` — consume the stream

**Signature:** add `stream: bool = False` to `run_agent`
([loop.py:233-251](agent/loop.py#L233-L251)). Default `False` preserves every
existing caller (smoke tests, `/chat`).

**New helper** `_stream_with_retry(...)` next to `_complete_with_retry`
([loop.py:151](agent/loop.py#L151)). It is an async generator yielding
`StreamChunk`s with **pre-first-delta** retry only:

```python
async def _stream_with_retry(llm, *, messages, tools, system, max_tokens,
                             timeout, max_retries, base_delay, thinking_level):
    """Drive llm.stream() with bounded retry that applies ONLY before the first
    TextDelta is produced. Once any delta is emitted the stream is committed:
    a later failure propagates (the loop turns it into ErrorEvent + done).

    timeout (when set) caps time-to-first-chunk via asyncio.timeout around the
    create()+first-iteration; it is NOT an overall cap on a long stream. See §6.
    """
    attempts = max(0, max_retries) + 1
    for attempt in range(attempts):
        produced = False
        try:
            agen = llm.stream(messages=messages, tools=tools, system=system,
                              max_tokens=max_tokens, thinking_level=thinking_level)
            async for chunk in agen:           # apply timeout to first chunk only
                produced = produced or isinstance(chunk, TextDelta)
                yield chunk
            return
        except Exception as exc:
            transient = isinstance(exc, TimeoutError) or llm.is_transient_error(exc)
            if produced or not transient or attempt == attempts - 1:
                raise
            await asyncio.sleep(_backoff_delay(base_delay, attempt))
```

> Implementer note: applying `asyncio.timeout` to *only* the first chunk while
> letting the rest stream unbounded needs care (you can't wrap the whole
> `async for` in one timeout). Simplest correct approach: manually pull the first
> chunk under `asyncio.timeout`, then iterate the remainder without it. Decide
> and document; see §6 and §9.

**Branch the LLM-call section** ([loop.py:364-410](agent/loop.py#L364-L410)):

```python
llm_started = time.perf_counter()
response: AssistantMessage | None = None
try:
    if stream:
        async for chunk in _stream_with_retry(llm, messages=session.messages,
                tools=effective_tools or None, system=effective_system,
                max_tokens=max_tokens, timeout=llm_timeout_seconds,
                max_retries=max_retries, base_delay=retry_base_delay,
                thinking_level=thinking_level):
            if isinstance(chunk, TextDelta):
                if chunk.text:
                    yield await _emit(TextEvent(text=chunk.text))
            elif isinstance(chunk, StreamEnd):
                response = chunk.message
        if response is None:  # defensive: stream ended with no StreamEnd
            response = AssistantMessage(content=[], stop_reason="empty")
    else:
        response = await _complete_with_retry(llm, messages=session.messages,
                tools=effective_tools or None, system=effective_system,
                max_tokens=max_tokens, timeout=llm_timeout_seconds,
                max_retries=max_retries, base_delay=retry_base_delay,
                thinking_level=thinking_level)
except Exception as e:
    logger.exception("LLM completion failed on iteration %d", iteration)
    yield await _emit(ErrorEvent(message=f"LLM call failed: {e}"))
    yield await _emit(_done(reason="llm_error"))
    return
llm_latency_ms = round((time.perf_counter() - llm_started) * 1000, 2)

session.append_assistant(response)
# ... UsageEvent + store.save UNCHANGED ...

# Emit text blocks ONLY in non-stream mode (already streamed above otherwise).
if not stream:
    for block in response.content:
        if isinstance(block, TextBlock) and block.text:
            yield await _emit(TextEvent(text=block.text))

# ... tool extraction + sequential execution UNCHANGED ...
```

The `if not stream:` guard is the whole trick that prevents double-emitting text.
Everything from `tool_uses = [...]` onward is **unchanged** — tool calls
assembled by the provider flow through the existing executor.

**Docstring:** update the module algorithm (step 3, "Stream out the model's text
blocks") and `run_agent`'s docstring to describe the `stream` parameter and that
text is emitted incrementally when set. Add `TextDelta`/`StreamEnd` to imports.

### 5.5 `api/routes.py` — thread the flag

- Add `stream: bool = False` to `_turn_events`
  ([api/routes.py:142](api/routes.py#L142)) and pass it into `run_agent(...,
  stream=stream)` ([api/routes.py:199](api/routes.py#L199)).
- `_run_turn` already forwards `**kwargs` to `_turn_events`
  ([routes.py:229-249](api/routes.py#L229-L249)) — it needs no change beyond
  callers optionally including `stream`.
- `/chat` ([routes.py:279](api/routes.py#L279)) passes nothing → stays `False`.
  Native `/chat` remains non-streaming (out of scope, consistent with
  [docs/operations.md:122](docs/operations.md#L122)).

### 5.6 `api/openai_compat.py` — opt the SSE path in

In `chat_completions`, add `stream=req.stream` to `turn_kwargs`
([openai_compat.py:249-263](api/openai_compat.py#L249-L263)). Because:

- The **SSE branch** `_stream(...)` iterates `_turn_events(**turn_kwargs)`
  ([openai_compat.py:196](api/openai_compat.py#L196)) → now `stream=True` →
  real token streaming. `_stream`'s existing per-`TextEvent` → `delta.content`
  mapping ([openai_compat.py:197-198](api/openai_compat.py#L197-L198)) needs **no
  change** — it already forwards each `TextEvent`; there will simply be many
  small ones instead of one big one.
- The **non-stream JSON branch** calls `_run_turn(**turn_kwargs)` only when
  `req.stream` is `False` ([openai_compat.py:265-274](api/openai_compat.py#L265-L274)),
  so `stream=False` there → unchanged collect-then-respond.

That's the entire wiring. No change to `_chunk`, `_completion_body`, or the
`[DONE]` sentinel.

---

## 6. Edge cases and how to handle them

| Case | Handling |
|---|---|
| **qwen `<think>` leading block** | Stateful `_ReasoningStreamStripper` (§5.3c). Must match `_strip_reasoning` output. Highest-risk correctness item for the user's setup. |
| **Usage missing in stream** | `stream_options={"include_usage": True}`; if the server omits it, `_usage_from_stream` returns `Usage()` zeros. Never block on usage (research 03 §3). |
| **Tool-only turn (no text)** | No `TextDelta`s emitted; `StreamEnd.message` carries `ToolUseBlock`s; existing executor runs. `if not stream` guard emits nothing extra. |
| **Tool deltas split across chunks** | Accumulate by `tc.index`; concatenate `function.arguments` fragments; parse JSON once at `StreamEnd`. |
| **Error before first delta** | `_stream_with_retry` retries per budget (transient only). |
| **Error after first delta** | Propagates → loop emits `ErrorEvent` + `DoneEvent("llm_error")`. On SSE this becomes a trailing error frame ([openai_compat.py:201-204](api/openai_compat.py#L201-L204)) after partial text. Document as a known limitation. |
| **`llm_timeout_seconds` vs long stream** | v1: cap **time-to-first-chunk** only, not total stream duration (a long legitimate answer must not be killed). Note the tradeoff; a per-chunk inactivity timeout is a future improvement (§9). |
| **Empty stream** | `StreamEnd` with no blocks → `stop_reason="empty"`; loop's existing no-tool-calls branch reports `end_turn`/`empty`. (Pre-first-delta empty retry is optional; non-streaming has it via `_complete_with_retry`.) |
| **provider_metadata** | Preserved: assembled `TextBlock`/`ToolUseBlock` are built the same way as in `_from_openai_response`. OpenAI-compatible servers don't populate it today, but keep the construction identical so the invariant holds. |
| **Orchestrator** | Never streams. It calls `complete()` with `response_schema`; `stream` flag never reaches it. Verify it still runs once before the first frame. |

---

## 7. Test plan

All tests are standalone runnable scripts in `tests/` (no `pytest`, per
`CLAUDE.md`). Launch with `./runscript.sh tests/<file>.py`.

1. **`tests/smoke_test_streaming.py` (new, hermetic).** A `FakeStreamingLLM`
   that overrides `stream()` to yield several `TextDelta`s then `StreamEnd`.
   Drive `run_agent(..., stream=True)` and assert:
   - multiple `TextEvent`s arrive (count > 1) and concatenate to the full text;
   - exactly one terminal `DoneEvent`;
   - no duplicate text (the `if not stream` guard works);
   - a tool-call scenario: `StreamEnd` carrying a `ToolUseBlock` triggers the
     existing executor (reuse `FakeMCP` style from
     [tests/smoke_test_openai_api.py:55](tests/smoke_test_openai_api.py#L55)).
   - **Fallback case:** a `FakeLLM` implementing *only* `complete()` driven with
     `stream=True` still produces `TextEvent`(s) via the ABC default. This is the
     model-agnosticism guarantee.

2. **`_ReasoningStreamStripper` unit cases** (inside the same script or a small
   dedicated one): tag in one piece; tag split (`"<thi"`+`"nk>"`); `</think>`
   split; no-think passthrough; later-substring must-not-strip. Assert each
   stream's joined output equals `_strip_reasoning(full)`.

3. **Extend `tests/smoke_test_openai_api.py`.** Its existing `FakeLLM` implements
   only `complete()` ([smoke_test_openai_api.py:42-53](tests/smoke_test_openai_api.py#L42-L53)).
   After the change the SSE scenario routes through `run_agent(stream=True)` and
   exercises the **fallback** — the existing assertions (chunk deltas + `[DONE]`)
   must still pass unchanged. This is your regression guard that the wiring
   didn't break the non-streaming-provider path. Optionally add a
   `FakeStreamingLLM` variant asserting multiple deltas.

4. **`tests/smoke_test_openai.py` (real backend, env-gated).** Add a streaming
   scenario hitting the user's Ollama/qwen via `OPENAI_BASE_URL`. Assert
   incremental deltas and that no `<think>` content leaks. This is the only test
   that needs the live server; keep it skippable when env is unset, matching the
   file's existing pattern ([smoke_test_openai.py:24-26](tests/smoke_test_openai.py#L24-L26)).

5. **Regression sweep:** run `smoke_test_agent.py`, `smoke_test_loop_intelligence.py`,
   `smoke_test_reliability.py` — all call `run_agent` without `stream`, so they
   must pass untouched. Confirms the default-`False` path is inert.

---

## 8. Documentation updates (required by `CLAUDE.md`)

- **`docs/operations.md`** Streaming section ([:111-127](docs/operations.md#L111-L127))
  and the "known limitations" note ([:357-360](docs/operations.md#L357-L360)):
  change from "the loop is ready for it" to "the loop streams token deltas when
  `stream=True`; the OpenAI provider streams natively, other providers fall back
  to non-streaming via the ABC default." Note the partial-stream error and
  time-to-first-chunk-timeout limitations.
- **`docs/architecture.md`**: update the loop description / the
  "non-streaming endpoint collects all events" note
  ([:738-739](docs/architecture.md#L738-L739)) to describe the
  `complete()` vs `stream()` split and the `StreamChunk` protocol.
- **`docs/api.md`**: wire format is unchanged (still `chat.completion.chunk`
  frames + `[DONE]`); add a sentence that deltas are now genuinely incremental.
- **`AGENTIC-HARNESS-RESEARCH/03_model_interface.md`** is research, not harness
  docs — **do not edit it.** It already prescribes this design; cite it, don't
  change it.
- No `.env.example` / `docs/configuration.md` change unless you add the optional
  toggle in §9.

---

## 9. Open decisions (resolve before/while implementing)

1. **First-chunk timeout vs inactivity timeout.** v1 recommends capping
   time-to-first-chunk only. A per-chunk inactivity timeout (reset on each
   delta) is more robust for hung mid-streams but more code. Pick one; document
   it. *(Recommendation: first-chunk cap for v1, inactivity timeout as a
   follow-up.)*
2. **Optional `STREAMING_ENABLED` safety toggle.** Could add a `Settings` flag
   that forces the fallback even when `stream:true` is requested (operator kill
   switch). The research's "behind the interface" stance makes this optional.
   *(Recommendation: skip for v1 — the per-request `stream` flag plus the
   provider fallback already give safe degradation; adding env surface is the
   opposite of minimally invasive.)*
3. **Native `/chat/stream` route.** Out of scope, but the design makes it a thin
   addition later ([docs/operations.md:122-127](docs/operations.md#L122-L127)).

---

## 10. Suggested commit sequence

Small, reviewable, each independently green:

1. `schemas + ABC default stream()` — add `TextDelta`/`StreamEnd`/`StreamChunk`
   and the fallback `stream()`. Existing tests still pass (nothing calls it yet).
2. `loop: stream param + _stream_with_retry` — add the branch, default `False`.
   Add `smoke_test_streaming.py` (fallback + fake-streaming). Regression sweep.
3. `openai provider: native stream() + reasoning stripper` — `_build_request`
   refactor, `stream()`, `_usage_from_stream`, `_ReasoningStreamStripper` + its
   unit cases.
4. `api wiring` — thread `stream` through `_turn_events`/`run_agent`; set
   `stream=req.stream` in the adapter. Extend `smoke_test_openai_api.py`.
5. `docs` — operations/architecture/api updates.

After (4), verify live against the user's setup: OpenWebUI → harness (`/v1`,
`stream:true`) → Ollama/qwen3.6, and confirm tokens render incrementally with no
`<think>` leakage.

---

## 11. Definition of done

- OpenWebUI shows tokens appearing progressively, not all-at-once.
- No `<think>` reasoning visible in the streamed output for qwen3.6.
- `/chat`, non-stream `/v1`, the orchestrator, and all pre-existing smoke tests
  behave identically to before (verified by the regression sweep).
- Gemini (or any `complete()`-only provider) still works under `stream:true` via
  the ABC fallback.
- Docs in `docs/` reflect real streaming; `AGENTIC-HARNESS-RESEARCH/` untouched.
- No provider SDK imported at module level in `llm/client.py`; sequential tool
  execution and `provider_metadata` preservation intact.
