# Hyphae execution model

This document explains the state and lifecycle inside one request. For package
boundaries see [architecture](architecture.md); for the external wire contract
see the [HTTP API](api.md).

## Vocabulary

| Term | Lifetime | Owner |
|---|---|---|
| Application runtime | One server process | `hyphae.application.ApplicationRuntime`, assembled by `hyphae.main.lifespan` |
| Session | Conversation history across native turns | `hyphae.agent.Session` and `SessionStore` |
| HTTP request | One inbound native or OpenAI-compatible operation | `api` |
| Turn | One accepted user request, including routing, execution, publication, and cleanup | `hyphae.application.TurnRunner` |
| Run | One provider-neutral agent-loop execution inside a turn | `hyphae.agent.run_agent` and private `_AgentRun` |
| Iteration | One context assembly and downstream model response within a run | `_AgentRun` |
| Attempt | One provider call or retry inside an iteration | `hyphae.agent.generation` |

A typical accepted turn contains one agent run and that run can contain several
model iterations because tool results return to the model. Routing and optional
context compaction are auxiliary model calls, not additional user turns.

## Accepted-turn flow

Both HTTP adapters derive a `TurnRunner` from the process runtime. Native
routes resolve or create a persistent session; `/v1/chat/completions` builds a
fresh ephemeral session from the validated message history.

`TurnRunner.open()` then:

1. validates an explicitly requested model before session or MCP work;
2. claims the session ID, rejecting a concurrent claim;
3. starts one `RunContext` containing the run ID, absolute deadline, logger,
   and optional tracer;
4. reloads the latest persistent checkpoint and rejects a prompt that would
   exceed the retained-history bound;
5. opens the MCP turn runtime, which refreshes due/unhealthy catalogs before
   exposing one immutable inventory;
6. projects `ToolPolicy` over that inventory before routing or model exposure;
7. resolves routing, trusted system text, model limits, and selected tools;
8. returns `TurnExecution`, whose metadata and event iterator share that
   lifetime; and
9. keeps the session claim and MCP runtime until the iterator and its cleanup
   have finished.

Buffered callers use `TurnRunner.run()`, which consumes the same event iterator,
joins visible text, and requires exactly one final `DoneEvent`. Streaming
renderers consume it directly. Closing a stream early still closes the agent
generator, MCP runtime, and session claim.

## Routing modes and prompt authority

`OrchestratedRouting` contains a selection service, a ready-model registry, and
the trusted startup-loaded agent prompt. The router receives the prior
conversation's compact text tail, ready model descriptions, and the
policy-visible tool snapshot. Its structured proposal contains only:

- `selected_model_id`;
- `selected_tools`; and
- `thinking_level`.

Unknown model/tool IDs and duplicate tools are sanitized and recorded as stable
correction codes. Router control-call or parse failure selects the ready default
model with no tools. Router output never supplies a system prompt.

An explicit `/v1` model ID pins the downstream model while routing still
selects tools and thinking level. A `/v1` system message is the authorized
downstream override; otherwise orchestrated turns use `hyphae/config/agent_prompt.md`.

`UnorchestratedRouting` uses one fixed client and all tools remaining after
policy projection. It has no application system prompt; an optional `/v1`
system override still applies. When configured orchestration cannot use its
control model, startup may publish a fixed registry-default route that
advertises only that executable model.

## Agent run composition

`run_agent()` is the public provider-neutral event generator. It builds one
private `_AgentRun`, which owns resolved limits, cumulative usage, current
iteration, the pending generation result, one `ToolDispatcher`, and any active
tool batch.

Per iteration, `_AgentRun`:

1. checks the absolute deadline and cumulative token budget;
2. creates the effective request, including final-iteration tool withholding
   and wrap-up text when applicable;
3. assembles a context view that is known to fit the selected model's input
   budget;
4. consumes normalized generation chunks from `hyphae.agent.generation`;
5. records the complete canonical assistant response and usage;
6. publishes a complete non-tool checkpoint, or starts a tool batch;
7. dispatches requested tools sequentially and appends the complete matching
   result batch; and
8. continues until an explicit terminal reason is emitted.

### Context assembly

The input budget is:

```text
context window - maximum output tokens - safety margin
```

The estimate includes the effective system text and tool schemas. `naive`
passes through an in-budget history but rejects an over-budget request rather
than knowingly submitting it. `compaction` preserves the first user task and a
recent protocol-safe tail, summarizes the middle with the selected client, and
then verifies the result fits. If required protocol units cannot fit, the run
ends `budget_exceeded`. Session history itself is never compacted or rewritten.

Compaction usage is emitted and counted. Routing usage is carried on the
orchestration event and seeds the run's cumulative/terminal totals, so each real
model call is counted once.

### Generation

`hyphae.agent.generation` owns the shared attempt policy for buffered and streaming
calls:

- the smaller of the per-attempt timeout and remaining run deadline applies;
- provider-classified transient errors, ordinary timeouts, and canonical empty
  responses may retry within `LLM_MAX_RETRIES`;
- retry sleep is cancellable and consumes the same absolute deadline;
- visible answer text commits a streaming attempt and prevents replay;
- reasoning-only output remains provisional and receives an interruption marker
  before an eligible retry; and
- provider iterators are closed on completion, failure, timeout, cancellation,
  or early consumer close.

Providers return canonical terminal outcomes. The run maps `max_tokens` to
`truncated`, treats refusal/filter outcomes explicitly, and fails closed on an
unknown or incomplete provider termination.

### Tool execution

`hyphae.agent.tool_execution.ToolDispatcher` owns one run's dispatch state. Before an
MCP invocation it:

1. blocks an exact repeated name/argument call;
2. converts malformed provider argument JSON into a teaching error result;
3. validates arguments against the captured JSON Schema;
4. rechecks `ToolPolicy`; and
5. applies the smaller of the tool timeout and remaining run deadline.

Invalid, denied, repeated, timed-out, or failed calls become canonical error
results the model can observe. A transport failure invalidates the affected
catalog revision and is never replayed automatically. Calls remain sequential
because they may have side effects.

`ActiveToolBatch` preserves call order and matching IDs. If cancellation or
generator close interrupts a batch, the in-flight call receives an
outcome-unknown synthetic result and calls not yet started receive not-executed
results. The balanced batch is best-effort checkpointed for safe future replay,
while the original cancellation or close signal continues outward.

## Sessions and checkpoints

`Session` contains canonical messages plus identity and timestamps. Mutation
uses `append_user`, `append_assistant`, and `append_tool_results`; a system
prompt is call-time policy and is never stored as session history.

Persistent native turns stage a detached copy of the latest stored checkpoint.
The store validates tool-call/result ordering and detaches nested canonical
values at create/get/save boundaries. Only these states are publishable:

- a complete assistant response with no pending tool calls; or
- an assistant tool-call message followed by exactly one result for every call.

Prompt-only and unmatched-tool states are never published. A checkpoint that
would exceed `SESSION_HISTORY_MAX_CHARS` is rejected without truncating or
replacing the prior checkpoint.

`InMemorySessionStore` bounds idle age and count. Admission protects every
currently claimed session; if all possible eviction victims are active, a new
native session receives the stable capacity-unavailable outcome. Eviction is
lazy and runs only on admission.

`SessionGuard` rejects concurrent turns for the same ID rather than queueing
them. It is process-local, so multi-worker deployments need shared persistence
and a distributed claim if native sessions must cross workers.

OpenAI-compatible turns are ephemeral. The client supplies durable history on
each request, and those turns never create, save, evict, or consume capacity in
the native store.

## Events and terminal outcomes

The run emits typed dataclass events:

| Event | Meaning |
|---|---|
| `OrchestrationDecisionEvent` | Resolved model/tools plus safe routing corrections, fallback, control-call usage, thinking level, and latency |
| `ReasoningEvent` | Sanitized provider reasoning, excluded from session replay |
| `TextEvent` | Visible assistant text |
| `UsageEvent` | One downstream or compaction completion's canonical usage |
| `ToolCallEvent` | A tool is about to be considered/dispatched |
| `ToolResultEvent` | Real or synthetic result, error flag, and optional call latency |
| `ErrorEvent` | A safe failure message not sent back to the model |
| `DoneEvent` | The single explicit terminal state and cumulative usage |

Native buffered and OpenAI adapters intentionally map some terminal outcomes to
HTTP errors instead of presenting them as successful completion. The exact wire
mapping lives in [api.md](api.md).

## MCP lifetime

`MCPManager` owns one immutable catalog per enabled server. Startup and
request-driven refresh use short discovery connections that close after
`list_tools`. A turn captures the resulting routes; later refreshes cannot
mutate that accepted snapshot.

`TurnToolRuntime` creates no server worker merely because a tool is visible or
selected. The first actual call to a server creates one task-owned lease worker,
and later calls to that server in the same turn reuse it. The worker closes with
the turn. Concurrent turns have independent workers and connections.

There is no periodic refresh task and no progressive tool activation. Accepted
requests refresh stale catalogs and unhealthy catalogs whose backoff has
elapsed; `/health` only reads retained state.
