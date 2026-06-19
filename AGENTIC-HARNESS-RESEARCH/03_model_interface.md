# 03 — Model Interface (Provider- & Model-Agnostic)

This is where our key constraint lives: **the harness must run against many providers and many
models — hosted and local — without code changes.** We achieve this by targeting the
**OpenAI-compatible API** as the integration lingua franca and by assuming **some models will be
weak**.

## 1. Why OpenAI-compatible is the right integration surface

The `POST /v1/chat/completions` request/response shape has become a de-facto standard. The same
client talks to all of these by swapping only `base_url` (and `api_key`):

| Runtime | `base_url` example | Notes |
|---------|--------------------|-------|
| **Ollama** (local) | `http://localhost:11434/v1` | Local models; the headline portability target. |
| **vLLM** (self-host) | `http://host:8000/v1` | High-throughput self-hosting. |
| **LM Studio** (local) | `http://localhost:1234/v1` | Desktop local serving. |
| **llama.cpp server** | `http://localhost:8080/v1` | Minimal local serving. |
| **OpenRouter** | `https://openrouter.ai/api/v1` | One key, many hosted models. |
| **OpenAI / Azure / others** | provider URL | Hosted frontier models. |

**Design decision:** depend on the *wire format*, not a vendor SDK. Use the official OpenAI SDK (it
takes a `base_url`) or a thin HTTP client. This keeps the model client small and swappable, and it
means a developer can run the entire harness offline against Ollama.

> Caveat — "compatible" is a spectrum. Local servers vary in support for tool/function-calling,
> streaming deltas for tool calls, JSON mode, `logprobs`, and `usage` accounting. **Treat advanced
> features as capabilities to detect, not guarantees** (see §4).

## 2. The model-client interface (keep it narrow)

One small interface, behind which any provider hides:

```python
class ModelClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        stream: bool = False,
        response_format: dict | None = None,   # e.g. JSON schema, when supported
        temperature: float | None = None,
    ) -> ModelResponse: ...

# Normalize the response so the loop never sees provider quirks:
ModelResponse = {
    "text": str | None,
    "tool_calls": [ToolCall],     # normalized regardless of how the provider emitted them
    "usage": {"input_tokens": int, "output_tokens": int, "cost": float | None},
    "finish_reason": str,          # "stop" | "tool_calls" | "length" | "content_filter" | ...
}
```

The loop in [02](02_anatomy_and_the_loop.md) only ever touches this normalized shape. All
provider-specific glue — endpoint differences, header auth, error formats — stays inside the
adapter. **This single boundary is what makes the harness provider-agnostic.**

### Configuration, not code

Model choice is config, never a hardcoded constant:

```yaml
model:
  base_url: ${MODEL_BASE_URL:-http://localhost:11434/v1}   # default: local Ollama
  api_key:  ${MODEL_API_KEY:-ollama}
  name:     ${MODEL_NAME:-qwen2.5:14b}
  context_window: 32768          # used for budgeting (doc 05) — providers don't always report it
  supports_native_tools: auto    # auto-detect, or pin true/false
  max_output_tokens: 4096
```

## 3. Streaming, structured output, token accounting

- **Streaming** — stream for interactive UIs (latency feel) and to allow early cancel. Note local
  servers may not stream *tool-call* deltas well; if so, fall back to non-streaming for tool turns.
  Keep streaming behind the interface so the loop can ignore it.
- **Structured output** — when you need a typed object (not a tool call), prefer the server's JSON
  mode / JSON-schema `response_format` if supported. **Fallbacks (in order):** (1) native JSON
  schema → (2) "respond with JSON matching this schema" + parse → (3) grammar-constrained decoding
  (llama.cpp/vLLM support GBNF/JSON grammars) → (4) retry-on-parse-failure with the error fed back.
- **Token accounting** — many local servers under-report or omit `usage`. Keep a local tokenizer
  estimate as a fallback so budgets/limits ([02](02_anatomy_and_the_loop.md) §4) still work. Never
  let missing `usage` disable your spend cap.

## 4. Designing for weak / local models (graceful degradation)

This is the part most harnesses skip and then break on Ollama. **Assume the model may be small
(7B–14B), have a small context window (4k–32k), and call tools unreliably.** Degrade, don't fail.

### Capability detection, not assumption

Probe (or configure) capabilities once and adapt:

| Capability | If present | If absent (fallback) |
|------------|-----------|----------------------|
| Native tool/function calling | Use the `tools` param | **Prompted tool-calling**: describe tools in the system prompt, ask for a JSON action, parse it (see [04](04_tool_design.md) §native-vs-prompted) |
| JSON mode / schema | Use `response_format` | Prompt for JSON + tolerant parse + repair retry |
| Grammar-constrained decoding | Constrain tool/JSON output | Validate + re-ask on invalid output |
| Large context | Use it | Aggressive compaction & retrieval ([05](05_context_and_memory.md)) |
| Reliable instruction following | Fewer guardrails | More structure, fewer tools, smaller steps |

### Tactics that make weak models usable

- **Fewer tools, smaller steps.** Weak models degrade fast past ~10 tools and on multi-part
  instructions. Expose a minimal toolset and let the loop take more, smaller turns.
- **Tolerant parsing + repair loop.** Expect malformed tool calls/JSON. Try to extract the JSON
  block; if it fails, feed the parse error back and ask again (cap the repairs, e.g. 2).
- **Lower the altitude of the prompt.** Weak models need more explicit, concrete instructions than
  frontier models (the opposite of the "lean on the model" default — which is exactly why this
  extra scaffolding must be **optional/config-driven**, not baked into the core).
- **Constrain outputs at the decoder** when the server supports grammars — the most reliable way to
  get valid tool calls from a small model.
- **Validate every tool argument** against its schema before executing (catches hallucinated args —
  a top failure mode; see [09](09_antipatterns_and_checklist.md)).

### Don't over-fit to one model

Resist hardcoding prompt hacks for one specific local model. Put model-specific knobs (tool mode,
prompt verbosity, max tools) in a small **per-model profile** in config. The core loop stays clean;
the messy compatibility lives at the edge.

## 5. Transport-level robustness (belongs here, not in the loop)

The model client owns network reliability so the loop stays clean:

- **Timeouts** on every request (connect + read).
- **Retries with exponential backoff + jitter** on 429/5xx/transport errors; cap attempts.
- **Idempotency** — a retried `complete()` must not double-execute side effects (it shouldn't have
  any; side effects live in tools).
- **Graceful error normalization** — rate limits, context-length-exceeded, content filters, and
  connection errors each map to a typed error the loop/guardrails can handle distinctly. (Deeper
  treatment in [07](07_reliability_and_safety.md).)

## What works / What doesn't

| ✅ Works | ❌ Doesn't |
|---------|-----------|
| One narrow `ModelClient` interface; providers behind adapters | Threading provider SDK objects through the loop |
| Targeting the OpenAI-compatible wire format | Coupling to one vendor's proprietary SDK/features |
| Model + base_url as **config** | Hardcoded model names and endpoints |
| Detecting capabilities and degrading | Assuming native tools/JSON mode/large context everywhere |
| Tolerant parsing + a bounded repair loop | Crashing on the first malformed tool call |
| Local-model knobs in a per-model profile at the edge | Special-casing one model deep in the core loop |
| Transport retries/timeouts in the client | Scattering retry logic through business code |

**Next:** [04 — Tool Design](04_tool_design.md)
