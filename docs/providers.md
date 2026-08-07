# LLM providers

Hyphae isolates provider SDKs behind one provider-neutral generation contract.
The agent, application, orchestrator, and session layers exchange only the
types in `hyphae.llm.schemas` and the `LLMClient` interface in
`hyphae.llm.client`.

## Contract

`GenerationRequest` is a shallow-immutable value containing:

- a sequence of provider-neutral `Message` values;
- an optional sequence of tool schemas;
- optional system text, output-token cap, structured-output schema, and
  thinking-level hint.

`LLMClient.complete()` returns an `AssistantMessage` containing text and tool
blocks, normalized stop reason, model, usage, and optional sanitized reasoning.
`stream()` yields text/reasoning deltas followed by exactly one assembled
`StreamEnd`. A complete-only provider may inherit the coarse default stream.
Providers classify their own transient failures; retry and timeout policy stays
in the agent generation layer. Each client also owns and closes its SDK
resources through `aclose()`.

Provider adapters must preserve opaque round-trip metadata required by their
wire protocol, but provider objects and SDK response types must not escape the
adapter.

## Implemented providers

### Gemini

`hyphae/llm/providers/gemini/` contains:

- `client.py`, which owns the `google.genai.Client`, invocation, lifecycle, and
  Gemini-specific transient-error classification;
- `codec.py`, which translates requests and responses, including Gemini tool
  calls, structured output, usage, stop reasons, and thought signatures.

The provider registry name is `gemini`.

### OpenAI-compatible

`hyphae/llm/providers/openai_compatible/` contains:

- `client.py`, which owns `AsyncOpenAI`, buffered and streaming invocation,
  lifecycle, and OpenAI-specific transient-error classification;
- `codec.py`, which translates buffered request/response shapes;
- `stream.py`, which incrementally assembles text, reasoning, tool calls,
  usage, and the terminal message while owning stream cleanup.

The primary registry name is `openai_compatible`; `openai` is a compatibility
alias. An empty `OPENAI_COMPAT_BASE_URL` targets OpenAI. Setting it targets an
OpenAI-compatible endpoint such as Ollama; the configured `model` must be a
model identifier accepted by that endpoint.

## Model profiles

Each `models.yaml` row produces a `ModelProfile`:

- `supports_native_tools` selects native tool calls or the prompted-tool
  adapter;
- `thinking` declares `none`, a request `hint-param`, or emitted
  `think-tags`;
- `sampling` supplies optional `temperature`, `top_p`, and `top_k` values.

When native tools are disabled, `PromptedToolLLMClient` renders the selected
tools into a protocol prompt and converts one model-authored JSON action back
to a normal `ToolUseBlock`. The orchestrator control model must support native
structured output; prompted-only models remain valid downstream choices.

## Registry and multi-provider routing

`hyphae.llm.client._PROVIDERS` is the single provider registry. Builders import SDK
adapters lazily. `build_llm_client_from_entry()` resolves credentials by the
provider on that model row, so one process can own ready clients from multiple
providers. `LLMRegistry` preflights the configured clients and exposes only
ready model IDs for orchestration.

`ANTHROPIC_API_KEY` is reserved configuration only. No Anthropic provider is
registered or implemented.

## Adding a provider

Adding a provider is a deliberate contract change, not only a registry edit:

1. Add a provider package whose client implements `LLMClient` and whose codec
   keeps SDK translation local.
2. Normalize content blocks, stop reasons, usage, tool-call argument failures,
   reasoning, and any required opaque replay metadata.
3. Implement idempotent, cancellation-safe SDK cleanup and provider-specific
   transient-error classification. Add native streaming only when the SDK can
   satisfy the same terminal contract.
4. Add a lazy builder and registry entry in `hyphae.llm.client`.
5. Add typed credential/endpoint configuration only when the provider needs
   it, plus representative `models.yaml` documentation.
6. Test request/response translation, malformed provider output, usage and stop
   normalization, tool parity, retries, streaming cleanup if supported, and
   client shutdown. A provider intended for orchestration must also prove
   structured-output behavior.

See [Configuration](configuration.md) for provider settings and model rows,
and [Execution](execution.md) for where generation fits in an accepted turn.
