# Hyphae architecture

This document is the system atlas: package ownership, dependency direction,
process composition, and the invariants that keep the harness maintainable.
See [execution](execution.md) for turn/run mechanics and
[providers](providers.md) for LLM wire translation.

## System map

```text
HTTP client
    │
    ▼
api
  authentication, bounded body parsing, DTO validation, public errors,
  native/OpenAI response rendering
    │ obtains one process-owned ApplicationRuntime
    ▼
application
  TurnRunner: accepted-turn claim, deadline, tool snapshot, routing,
  trusted prompt precedence, persistence policy, buffered/event result
    ├───────────────┬───────────────────┐
    ▼               ▼                   ▼
orchestrator      agent               tooling
selection only   provider-neutral     immutable ToolSnapshot and
ready model +    run composition      neutral ToolRuntime contract
visible tools    │
                 ├─ generation ───────────────► llm
                 │   retries and streams        canonical request/response;
                 │                              provider codecs and SDK clients
                 └─ tool_execution ───────────► mcp_runtime
                     validation/policy          process catalogs and lazy,
                     sequential dispatch        turn-owned server leases
```

`main.lifespan` builds one immutable `application.ApplicationRuntime` and
publishes it as `app.state.runtime`. Routes read that value through the sole
dynamic framework adapter in `api.dependencies` and derive a `TurnRunner` from
it. No route assembles a partial runtime from independent state fields.

The orchestrator is a selection-only control call. It selects a structurally
ready model, a subset of the policy-visible tool snapshot, and a thinking
level. It cannot author downstream system instructions. Downstream authority
comes from the startup-loaded `config/agent_prompt.md`, or from the documented
`/v1` caller override.

## Package ownership

| Package | Owns | Does not own |
|---|---|---|
| `config` | Frozen settings, typed YAML schemas/loaders, path and validation policy | Provider SDKs or request execution |
| `tooling` | Provider-neutral tool values, immutable snapshots, dispatch protocol | MCP connections or agent policy |
| `mcp_runtime` | Catalog discovery/refresh, health, namespaced routes, task-owned turn leases | Model selection or LLM schemas |
| `llm` | Canonical messages, `GenerationRequest`, `LLMClient`, provider registry and adapters | Retries, sessions, MCP, or HTTP |
| `orchestrator` | One selection-only routing call and ready-model registry | Downstream prompt authority or tool execution |
| `agent` | Run limits/context, generation policy, tool dispatch policy, transcripts, checkpoints, typed events and tracing | HTTP rendering or process composition |
| `application` | Process runtime value and complete accepted-turn lifecycle | FastAPI, Starlette, or wire-format rendering |
| `api` | HTTP authentication, body/DTO boundaries, stable public errors, native/OpenAI renderers | Agent-loop mechanics or resource ownership |
| `main.py` | Startup/shutdown composition | Request-specific business logic |

## Repository layout

```text
hyphae/
├── main.py
├── pyproject.toml
├── uv.lock
├── setup.sh
├── runscript.sh
├── config/
│   ├── settings.py
│   ├── schemas.py
│   ├── loaders.py
│   ├── mcp_config.yaml
│   ├── models.yaml
│   ├── orchestrator_prompt.md
│   └── agent_prompt.md
├── tooling/
│   └── contracts.py
├── mcp_runtime/
│   ├── catalog.py
│   ├── client.py
│   ├── lease.py
│   └── manager.py
├── llm/
│   ├── client.py
│   ├── schemas.py
│   ├── tool_prompt_protocol.py
│   └── providers/
│       ├── gemini/
│       └── openai_compatible/
├── orchestrator/
│   ├── contracts.py
│   ├── schemas.py
│   ├── registry.py
│   └── orchestrator.py
├── agent/
│   ├── session.py
│   ├── runtime.py
│   ├── context.py
│   ├── generation.py
│   ├── tool_execution.py
│   ├── tool_policy.py
│   ├── events.py
│   ├── tracing.py
│   └── loop.py
├── application/
│   └── turn.py
├── api/
│   ├── dependencies.py
│   ├── request_body.py
│   ├── public_errors.py
│   ├── schemas.py
│   ├── routes.py
│   └── openai_compatible.py
├── scripts/
│   ├── render_settings_reference.py
│   └── check_markdown_links.py
├── tests/
└── docs/
```

## Process lifecycle

Startup proceeds in one direction:

1. Load `Settings` lazily. Importing application modules does not read `.env`
   or credentials.
2. Load the MCP configuration and discover each enabled server's catalog.
   Discovery connections close immediately; partial MCP failure degrades
   health without preventing unrelated servers from serving.
3. Build the dispatch policy, bounded in-memory session store, and session
   guard.
4. When orchestration is enabled and its files exist, load both prompt assets,
   construct the registry, and preflight every configured client. Only ready
   models enter the executable inventory.
5. Build either `OrchestratedRouting` or `UnorchestratedRouting` and start the
   optional tracer.
6. Publish one complete `ApplicationRuntime` and begin serving.

A missing optional orchestration file degrades to the independently configured
default client. Present but invalid configuration fails startup. An unavailable
default registry model also fails startup. An unavailable or unsuitable control
model uses a fixed ready registry default and does not advertise other models.

Shutdown isolates LLM, MCP, and tracer cleanup so one failure does not skip the
remaining owners. Constructed LLM identities are deduplicated before closing.
Active MCP turn workers are joined in their owning tasks, and healthy tracer
shutdown drains accepted records before closing.

## Dependency direction

The intended direction is:

```text
api ──► application ──► agent ──► llm
 │           ├───────────────► orchestrator
 │           ├───────────────► tooling
 │           ├───────────────► config (typed Settings value)
 │           ├───────────────► llm (client/usage types)
 │           └───────────────► mcp_runtime (health/catalog types)
 └───────────────────────────► mcp_runtime (health types only)

mcp_runtime ──► tooling
orchestrator ──► llm + tooling + config values
main ──► every composition participant
```

`application` must remain framework-neutral. Provider SDK shapes must remain
inside their provider packages. The agent and orchestrator consume only
provider-neutral LLM/tool contracts.

## Architectural invariants

1. **One process composition.** Every route derives dependencies from the same
   frozen `ApplicationRuntime`; tests replace that whole graph or one named
   value through typed helpers.
2. **One accepted-turn owner.** `TurnRunner` holds the session claim, deadline,
   MCP turn runtime, immutable inventory, event iterator, and cleanup envelope
   together.
3. **Selection is not authority.** Router output can select only a ready model,
   policy-visible tools, and thinking level. Trusted configuration or an
   explicit `/v1` system message owns downstream instructions.
4. **Visibility never expands authorization.** `ToolPolicy` is projected before
   routing/model exposure and checked again immediately before dispatch. A
   routing failure falls back to the ready default with no tools.
5. **Catalogs and connections have different lifetimes.** MCP catalogs are
   process-owned immutable snapshots; actual server connections are lazy,
   task-owned, and limited to the accepted turn that dispatches them.
6. **Providers normalize at the edge.** Canonical messages, stop reasons,
   usage, reasoning, and tool calls cross the LLM boundary; SDK objects and wire
   quirks do not.
7. **Generation and dispatch have focused owners.** `agent.generation` owns
   retry/timeout/stream cleanup. `agent.tool_execution` owns schema validation,
   repeat detection, policy enforcement, sequential invocation, and interrupted
   batch balancing. `_AgentRun` composes them.
8. **Published transcripts are protocol-safe.** Persistent turns stage copies
   and save only complete assistant responses or balanced tool-call/result
   batches. Store boundaries detach nested values and validate history.
9. **Failure is explicit and sanitized.** Agent events retain meaningful
   terminal reasons; API adapters map them to stable public errors without
   leaking provider, configuration, session, or filesystem details.
10. **Observability cannot own execution.** Logs and optional JSONL tracing
    correlate by run ID, remain bounded and nonfatal, and never replace the
    event/transcript owners.

## Tool and prompt boundaries

An accepted request receives one `ToolSnapshot`. `disabled_tools` has already
removed server-declared exclusions. The application then removes tools denied
by `ToolPolicy`; the router and downstream model see only that projection. The
router may narrow it further, but the dispatcher still authorizes every real
call. Tools execute sequentially because they may have side effects, and no
transport failure is automatically replayed.

Prompt-producing code remains beside its distinct owner:

- `config/orchestrator_prompt.md` governs routing only;
- `config/agent_prompt.md` supplies trusted downstream behavior;
- `/v1` system messages are an authorized per-request downstream override;
- `agent.context` owns compaction instructions;
- `agent.loop` owns terminal wrap-up text; and
- `llm.tool_prompt_protocol` owns the prose-tool action and repair protocol.

No temporal context, generic prompting framework, or progressive tool
activation is implemented.
