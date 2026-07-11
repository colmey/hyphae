# Code Review Recommendations

This document records follow-up recommendations from the post-implementation review.
It is advisory: the changes below are proposed work and are not yet implemented.

## 1. Enforce timeouts throughout streamed LLM reads

**Priority:** High

**Affected subsystem:** The streaming branch of the agent loop, particularly the
iteration over `LLMClient.stream()`.

**Finding:** The configured LLM timeout protects the wait for the first stream chunk,
but later reads are awaited without a timeout. The run deadline is checked before each
read, yet it cannot interrupt a read that has already blocked. A provider can therefore
emit one token and then stall indefinitely beyond both `LLM_TIMEOUT_SECONDS` and the
remaining run deadline.

**Recommended behavior:** Every blocked read from the provider stream should use an
effective timeout equal to the shorter positive duration of:

- the configured per-LLM-call timeout; and
- the remaining run deadline.

The timeout should be recalculated before every read. A per-call timeout should bound
each provider read, while the run deadline remains an absolute bound for the whole run.
If either expires, the loop should close the stream when possible and follow the
existing timeout/deadline event path. Retry remains permissible only when no visible
text has been emitted.

**Rationale:** Streaming must not weaken the bounded-run guarantees already provided
for ordinary completions. This is especially important for local or OpenAI-compatible
servers that may leave a connection open after partially responding.

**Acceptance criteria:**

- A provider that stalls before its first chunk is bounded by the effective timeout.
- A provider that emits a delta and then stalls is also bounded.
- An absolute run deadline interrupts a blocked read even when the per-call timeout is
  longer or disabled.
- No retry occurs after a visible delta has been emitted.
- Timeout and deadline termination produce the same event and done-reason semantics as
  the non-streaming path.
- Tests cover stalls before the first chunk, between chunks, and before `StreamEnd`.

## 2. Surface streaming failures accurately through the OpenAI API

**Priority:** High

**Affected subsystem:** The OpenAI-compatible SSE renderer and its mapping of core
events and done reasons to wire responses.

**Finding:** An unrecoverable LLM failure produces an `ErrorEvent` followed by
`DoneEvent(reason="llm_error")`. The SSE renderer currently ignores `ErrorEvent`, and
the fallback finish-reason mapping turns an unrecognized `llm_error` into `"stop"`.
Consequently, a client can receive partial output followed by what appears to be a
successful completion.

**Recommended behavior:** When the renderer receives an `ErrorEvent`, it should emit an
explicit OpenAI-style error object as an SSE data frame. It should remember that the
stream failed and must not subsequently emit a normal successful finish frame.
`llm_error` should also receive an explicit mapping or failure branch so it can never
silently degrade to `"stop"`. The stream should still end with the `[DONE]` sentinel if
that is required for compatibility with existing clients.

**Rationale:** Once response headers and partial content have been sent, the server
cannot replace the stream with an HTTP error response. The failure must therefore be
represented honestly in-band; presenting it as a successful stop can cause clients to
persist or act on an incomplete answer.

**Acceptance criteria:**

- `ErrorEvent` produces an OpenAI-style error SSE frame.
- A failed stream never emits a final frame whose `finish_reason` is `"stop"`.
- Partial text already emitted remains available to the client.
- `[DONE]` behavior is deliberate and covered by a compatibility test.
- Normal completion, length termination, budget termination, and deadline termination
  retain their existing finish-reason mappings.
- Tests cover failures before any text and after one or more content deltas.

## 3. Migrate the smoke suite to pytest

**Priority:** Medium

**Affected subsystem:** Test organization, test execution, and future CI integration.

**Finding:** The standalone smoke scripts are readable and have served the project
well, but the suite now repeats fake clients, stores, event collectors, assertions, and
setup logic. Test discovery is manual, and a test can accidentally depend on live local
configuration instead of remaining hermetic.

**Recommended behavior:** Incrementally migrate the scripts to pytest without changing
the behaviors they verify. Introduce shared fixtures for scripted LLM clients, MCP
fakes, sessions, settings, and event collection. Use parametrized tests for stream
chunk boundaries, reasoning-tag splits, timeout positions, provider responses, and
done-reason mappings. Mark live model, HTTP-server, and MCP tests explicitly and keep
them out of the default hermetic test command.

The migration should preserve readable scenario names and may proceed subsystem by
subsystem. During the transition, the default test command should run both migrated
tests and any remaining hermetic scripts so coverage is not temporarily lost.

**Rationale:** Pytest provides reliable discovery, reusable fixtures, focused failure
output, parametrization, and standard CI reporting. Explicit hermetic/live markers
also prevent developer-specific MCP or model configuration from leaking into routine
checks.

**Acceptance criteria:**

- One documented command discovers and runs every hermetic test.
- Hermetic tests do not connect to configured MCP servers or model endpoints.
- Live integration tests require an explicit marker or command-line selection.
- Shared fixtures replace duplicated fake infrastructure.
- Stream-boundary and error-mapping cases use parametrization.
- Existing smoke behaviors remain covered throughout the migration.

## 4. Preserve the current streaming architecture

**Priority:** Architectural constraint for all streaming work

**Affected subsystem:** Provider interfaces, the agent loop, session handling, tool
dispatch, usage accounting, policy, tracing, and API renderers.

**Assessment:** The current design makes provider streaming additive rather than
creating a second agent implementation. Providers emit SDK-independent stream chunks,
and each streamed turn converges into a complete `AssistantMessage`. The existing
assistant/session/tool/usage/policy/tracing path then handles that message.

**Recommended behavior:** Retain this structure. Provider implementations should own
SDK-specific stream assembly and normalization. The agent loop should own run policy
and translate text deltas into core events. HTTP adapters should render core events
without gaining provider-specific parsing or tool-dispatch responsibilities.

New providers that lack native streaming should continue to use the complete-call
fallback. Future streaming features should extend the shared chunk and event contracts
only when a provider-neutral behavior genuinely requires it.

**Rationale:** Early convergence prevents streaming and non-streaming behavior from
drifting in session persistence, usage accounting, tool policy, tracing, and terminal
semantics. It also keeps provider SDK types out of the core and API layers.

**Acceptance criteria:**

- Every provider stream terminates in one complete `AssistantMessage`.
- Session mutation, tool dispatch, policy checks, usage accounting, and tracing remain
  shared after that convergence point.
- Provider SDK objects do not cross into the agent or API packages.
- API renderers consume core events and contain no provider-specific stream parsing.
- Complete-only providers remain valid under streaming requests through the fallback.

