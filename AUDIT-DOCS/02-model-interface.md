# 02 — Model Interface (Provider- & Model-Agnostic)

**Research basis:** Doc 03 (Model Interface), Doc 10 (Reference Architecture — Model seam).
**Verdict:** ✅ **Strong** — the single most important portability recommendation (OpenAI-compatible client) is *implemented and is the default*. The gaps are weak-model degradation features the research ties specifically to running local models.

---

## What the research wants (Doc 03)

Run against many providers and models — hosted and local — **without code changes**. Achieve it by (1) targeting the **OpenAI-compatible wire format** as the lingua franca (swap `base_url` to reach Ollama/vLLM/LM Studio/OpenRouter), (2) keeping a **narrow `ModelClient` interface** behind which all provider glue hides, (3) **detecting capabilities, not assuming them** (native tool-calling, JSON mode, grammars, context size), and (4) degrading gracefully for weak models (prompted-tool fallback, tolerant parsing + repair, token-estimate fallback when `usage` is missing). Transport robustness (timeouts, backoff+jitter, error normalization) belongs in the client, not the loop.

---

## What the code does

### ✅ A narrow, provider-agnostic interface
`LLMClient` is an ABC with a single `complete()` surface ([llm/client.py:46-75](../llm/client.py#L46-L75)). The loop, session store, and MCP manager never see provider types — the docstring states this is the "one bridge between worlds." Tools cross the boundary as the generic `[{name, description, input_schema}]` shape. This is exactly Doc 03's "one narrow `ModelClient` interface; providers behind adapters."

### ✅ OpenAI-compatible path — implemented and default
`OpenAILLMClient` is backed by `AsyncOpenAI` and serves **real OpenAI *and* any OpenAI-compatible server** by pointing `base_url` at the alternate `/v1` endpoint ([llm/providers/openai.py:87-103](../llm/providers/openai.py#L87-L103)). `config/models.yaml` uses exactly this to run a **local Qwen3.6** as the default model. This is the research's #1 recommendation — "depend on the wire format, not a vendor SDK" — *live in production config*. The provider registry comment even documents the Ollama target ([llm/client.py:108-119](../llm/client.py#L108-L119)).

### ✅ Provider registry as single source of truth
`_PROVIDERS` is "THE one place a provider is declared" ([llm/client.py:122-131](../llm/client.py#L122-L131)); `supported_providers()` keys config validators off it ([llm/client.py:134-136](../llm/client.py#L134-L136)); builders import their SDK lazily so importing the LLM layer never drags in a provider SDK. Adding a provider is genuinely two steps. This is the Model seam from Doc 10, done right.

### ✅ Config-driven model selection
Model + provider + max_tokens come from `models.yaml` entries via `build_llm_client_from_entry` ([llm/client.py:175-190](../llm/client.py#L175-L190)), and a single process can hold clients for multiple providers at once. No hardcoded model names in the loop. Doc 03's "model + base_url as config" is satisfied.

### ✅ Transport robustness lives in the right layer
Each client overrides `is_transient_error` with provider-specific knowledge — the OpenAI client classifies `408/429/500/502/503/504`, `RateLimitError`, `APITimeoutError`, `APIConnectionError` ([llm/providers/openai.py:155-177](../llm/providers/openai.py#L155-L177)); the base ABC handles `TimeoutError`/`ConnectionError` ([llm/client.py:77-86](../llm/client.py#L77-L86)). The loop owns the retry *policy* but asks the client whether a failure is transient — a clean split that matches Doc 03's "transport retries/timeouts in the client; don't scatter retry logic through business code." Finish-reason normalization maps provider vocab to the canonical `end_turn`/`max_tokens` ([llm/providers/openai.py:61-77](../llm/providers/openai.py#L61-L77)).

### ✅ Handles provider quirks at the edge
The OpenAI client strips a leading `<think>…</think>` reasoning block so Qwen3's chain-of-thought never pollutes the replayed `TextBlock` ([llm/providers/openai.py:80-84](../llm/providers/openai.py#L80-L84), [307](../llm/providers/openai.py#L307)). The Gemini client round-trips `provider_metadata` (`thought_signature`) — "never special-case one model deep in the core loop; messy compatibility lives at the edge" (Doc 03). Both honor it.

---

## Gaps & recommendations

### 🟡 (a) No capability detection — the weak-model story is incomplete
Doc 03's capability table is the heart of weak-model support: probe or configure `native-tools? / json-mode? / grammar? / context-size?` once and adapt. The harness instead *assumes* per provider:
- `thinking_level` is silently ignored by the OpenAI provider — logged at debug and dropped ([llm/providers/openai.py:142-144](../llm/providers/openai.py#L142-L144)). Correct per the contract, but it means the orchestrator's "intelligence on demand" knob is a no-op for the *default* model with no signal to the caller.
- There is **no prompted (ReAct-style) tool-calling fallback**. If a local model lacks native tool support, it "simply never emits tool_calls" ([llm/providers/openai.py:21-23](../llm/providers/openai.py#L21-L23)) — the harness can't fall back to describing tools in the prompt and parsing JSON, which Doc 03/04 call the key degradation path for weak models.

> **Recommendation (P1):** add a per-model **capability profile** to `models.yaml` entries (`supports_native_tools`, `json_mode`, `thinking`, `context_window`) and a **prompted-tool encoder** behind the same registry/dispatcher, selected by profile. The loop stays unaware (Doc 04: "same registry for native & prompted; pick the encoder by model profile").

### 🟡 (b) No token-count fallback
Doc 03: "many local servers under-report or omit `usage`. Keep a local tokenizer estimate as fallback so budgets still work. Never let missing `usage` disable your spend cap." The OpenAI client returns `Usage()` (all zeros) when the server omits usage ([llm/providers/openai.py:335-344](../llm/providers/openai.py#L335-L344)) and never populates thinking/cached. There is no estimator. This is benign today *only because there is no budget to break* — but it blocks the spend cap recommended in `06-reliability-safety.md`.

> **Recommendation (P1, pairs with budget):** add a local tokenizer estimate (e.g. `tiktoken` or a char/4 heuristic) used whenever provider `usage` is absent or zero.

### 🟡 (c) Structured-output repair ladder missing
Doc 03 prescribes a fallback ladder for typed output: native JSON schema → "respond with JSON" + parse → grammar-constrained → **retry-on-parse-failure with the error fed back**. The OpenAI client wires `response_format` (json_schema → json_object fallback) ([llm/providers/openai.py:270-287](../llm/providers/openai.py#L270-L287)), and the orchestrator unwraps markdown fences before parsing — but on failure the orchestrator *falls back to a default decision* rather than re-asking with the parse error. Also note: an unparseable tool-call argument string is silently swallowed to `{}` ([llm/providers/openai.py:315-319](../llm/providers/openai.py#L315-L319)) — fine for hosted models, a real risk for weak ones (a hallucinated-args failure that produces *no* error signal).

> **Recommendation (P2):** a bounded "repair retry" (cap ~2) that feeds the parse error back, for both structured output and malformed tool-call arguments.

### 🔴 (d) Documentation drift — understates the harness's own strength
`docs/architecture.md` and `docs/operations.md` state "One LLM provider implemented. Anthropic and OpenAI keys are recognized but `build_llm_client` / `build_llm_client_from_entry` raise `NotImplementedError`." This is **false**: `_build_openai` is registered ([llm/client.py:108-129](../llm/client.py#L108-L129)), `OpenAILLMClient` is a full 353-LOC implementation, and `models.yaml` runs it by default. The docs both *understate provider-agnosticism* and *misdescribe the default runtime* (a reader would think they're on Gemini when they're on local Qwen).

> **Recommendation (P0, doc-only):** correct both docs to "Gemini and OpenAI-compatible (incl. Ollama/local) implemented; Anthropic stubbed." Note the default model is a local OpenAI-compatible Qwen3.6.

---

## What works / what doesn't — scored

| Doc 03 criterion | Status | Evidence |
|---|---|---|
| Narrow `ModelClient`; providers behind adapters | ✅ | [llm/client.py:46-75](../llm/client.py#L46-L75) |
| OpenAI-compatible wire format targeted | ✅ | [llm/providers/openai.py:87-103](../llm/providers/openai.py#L87-L103) |
| Model + base_url as config | ✅ | `models.yaml` + [llm/client.py:175-190](../llm/client.py#L175-L190) |
| Capability detection + degradation | 🟡 | thinking ignored; no prompted-tool fallback ([openai.py:142-144](../llm/providers/openai.py#L142-L144)) |
| Local tokenizer estimate when usage missing | 🔴 | zeros, no estimator ([openai.py:335-344](../llm/providers/openai.py#L335-L344)) |
| Tolerant parsing + bounded repair loop | 🟡 | fences unwrapped; no repair retry; args→`{}` ([openai.py:315-319](../llm/providers/openai.py#L315-L319)) |
| Transport retries/timeouts/error-normalization in client | ✅ | [openai.py:155-177](../llm/providers/openai.py#L155-L177) |
| Model-specific knobs at the edge, not in core | ✅ | `<think>` strip, `thought_signature` |
