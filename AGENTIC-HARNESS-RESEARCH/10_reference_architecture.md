# 10 — Reference Architecture (Opinionated, Minimal, Modular)

This is the payoff: a concrete design that satisfies the thesis — **a thin deterministic shell
around a capable model, light and modular, provider-agnostic, degrading gracefully to local
models.** It pulls together docs 01–09.

## 1. Module map

Seven small modules behind narrow interfaces. **Core** is the irreducible harness; **optional**
bolts on when a task justifies it.

```
                       ┌──────────────────────────────────────────┐
                       │                  AGENT LOOP               │   CORE
                       │   gather → act → verify → repeat (doc 02) │
                       └───┬───────────┬───────────┬───────────┬───┘
                           │           │           │           │
                ┌──────────▼──┐  ┌─────▼──────┐ ┌──▼────────┐ ┌▼───────────┐
                │   MODEL     │  │   TOOL     │ │  CONTEXT  │ │  POLICY /  │  CORE
                │   CLIENT    │  │  REGISTRY  │ │  MANAGER  │ │ GUARDRAILS │
                │  (doc 03)   │  │ + DISPATCH │ │ (doc 05)  │ │  (doc 07)  │
                │             │  │  (doc 04)  │ │           │ │            │
                └──────┬──────┘  └─────┬──────┘ └─────┬─────┘ └─────┬──────┘
                       │               │              │             │
   OpenAI-compatible   │       ┌───────┼──────┐       │             │
   adapters: OpenAI /  │       │       │      │       │             │
   Ollama / vLLM / …   │   builtin  MCP    subagent   │             │
                       │   tools  adapter  -as-tool    │             │
                       │          (opt)    (opt,doc06) │             │
                ┌──────▼───────────────────────────────▼─────────────▼──────┐
                │              OBSERVABILITY  (trace = event log, doc 08)     │  CORE (thin)
                └─────────────────────────────────────────────────────────┘
                ┌─────────────────────────────────────────────────────────┐
                │         MEMORY  (files first; RAG/vector optional, doc 05)│  OPTIONAL
                └─────────────────────────────────────────────────────────┘
```

| Module | Responsibility | Core? | Doc |
|--------|----------------|-------|-----|
| **Agent loop** | Drive gather→act→verify; own control flow & state | ✅ Core | [02](02_anatomy_and_the_loop.md) |
| **Model client** | Provider-agnostic completion via OpenAI-compatible API; capability detection; transport retries | ✅ Core | [03](03_model_interface.md) |
| **Tool registry + dispatch** | Hold schemas; validate args; authorize; execute; normalize results | ✅ Core | [04](04_tool_design.md) |
| **Context manager** | Assemble messages; enforce token budget; compaction | ✅ Core | [05](05_context_and_memory.md) |
| **Policy / guardrails** | Caps (iter/spend/time); no-progress detection; permissions; approval gates | ✅ Core | [07](07_reliability_and_safety.md) |
| **Observability** | Serialize the event log to traces; metrics | ✅ Core (thin) | [08](08_observability_and_evals.md) |
| **Memory** | Working notes (files) → long-term/RAG | ⬜ Optional | [05](05_context_and_memory.md) |
| **MCP adapter** | Register external MCP tools into the registry | ⬜ Optional | [04](04_tool_design.md) |
| **Subagents** | Read-only fan-out exposed as a tool | ⬜ Optional | [06](06_orchestration_single_vs_multi.md) |

**Dependency rule:** the loop depends only on the four core module *interfaces*. Optional modules
register *into* core seams (a tool, a context strategy, a trace exporter) — they never become
something the core imports. Delete any optional module and the harness still runs.

## 2. The seams (where modularity lives)

Five narrow interfaces are the entire extensibility story — no plugin framework needed:

| Seam | Interface (shape) | Swap to get… |
|------|-------------------|--------------|
| Model | `ModelClient.complete(messages, tools) -> ModelResponse` | OpenAI ↔ Ollama ↔ vLLM ↔ OpenRouter |
| Tool | `Tool{name, schema, run(args) -> result}` registered in a `ToolRegistry` | builtin ↔ MCP ↔ subagent |
| Context | `assemble_context(state) -> messages` | naive ↔ compaction ↔ retrieval-augmented |
| Policy | `check(state, action) -> allow / deny / ask` | dev (permissive) ↔ prod (sandboxed) |
| Trace | `emit(event)` over the event log | JSONL ↔ OpenTelemetry exporter |

If adding a capability requires touching more than one seam, it's a signal you're over-reaching —
re-read [09](09_antipatterns_and_checklist.md).

## 3. Configuration (everything swappable is config)

```yaml
model:
  base_url: ${MODEL_BASE_URL:-http://localhost:11434/v1}   # default to local Ollama
  api_key:  ${MODEL_API_KEY:-ollama}
  name:     ${MODEL_NAME:-qwen2.5:14b}
  profile:  auto            # capability profile: native-tools? json-mode? grammar? (doc 03 §4)
  context_window: 32768
  max_output_tokens: 4096

limits:
  max_iterations: 30
  max_spend_usd:  1.00       # falls back to token cap when usage unreported
  wall_clock_s:   600

tools:
  builtin: [read_file, write_file, list_dir, search_files, http_get, finish]
  mcp_servers: []            # optional; empty = none
  subagents: false           # optional

context:
  strategy: compaction       # naive | compaction | retrieval
  reserve_tokens: 6000

policy:
  default: read_only
  allow_write_under: ["./workspace"]
  approve_before: [shell_exec, delete_file]   # human-in-the-loop gate (doc 07)

observability:
  trace: jsonl               # jsonl | otel
  trace_path: ./traces
```

The same binary runs fully **offline against Ollama** (the defaults above) or against a hosted
frontier model by changing three env vars — no code change. That portability is the whole point.

## 4. Build-vs-adopt

| Component | Build | Adopt |
|-----------|-------|-------|
| The loop, state, control flow | **Build** — it's small and you must own it (12-Factor #8) | — |
| Model client | Thin wrapper you build | over the **official OpenAI SDK** (takes `base_url`) |
| Tool schemas / validation | **Build** the registry | use a JSON-Schema validator lib |
| Tracing | **Build** the serializer (it's your event log) | adopt **OpenTelemetry** for export |
| Evals | **Build** a small runner | optionally adopt an eval/trace platform later |
| External tools | — | **Adopt MCP** servers (optional) |
| Sandboxing | — | **Adopt** containers/jails for dangerous tools |

Heuristic: **build the ~200-line core** (loop + the four interfaces); **adopt** for
commoditized, well-standardized concerns (HTTP/SDK, schema validation, OTel, sandboxes). Do **not**
adopt a framework that wants to own the loop.

## 5. Phased build path

Each phase is independently shippable and adds value alone. Don't skip ahead.

| Phase | Add | You can now… | Docs |
|-------|-----|--------------|------|
| **v0 — Walking skeleton** | Loop + model client + 2–3 tools + iteration cap + JSONL trace | Run a real agentic task against Ollama or hosted | 02, 03, 04 |
| **v1 — Make it safe** | Spend/time caps, no-progress detection, arg validation, errors-as-feedback, least-privilege + write allow-list | Trust it to run unattended on bounded tasks | 04, 07 |
| **v2 — Make it durable** | Context budget + compaction; structured note-taking (files-as-memory) | Handle long-horizon tasks without context rot | 05 |
| **v3 — Make it measurable** | Small eval suite in CI; OTel export | Change prompts/tools/models and *know* if it helped; swap models safely | 08 |
| **v4 — Extend (only if needed)** | MCP adapter, read-only subagents, approval gates / sandbox, RAG | Add capability without bloating the core | 04, 06, 07, 05 |

A team can stop at v1 for many internal tools, v3 for anything serious. v4 items are à la carte —
add the one the task demands, skip the rest.

## 6. How this honors the thesis

- **Thin deterministic shell** — the loop is ~one page; the model does the thinking. (01, 02)
- **Lean, earns complexity** — core is 4 modules; everything else is opt-in and deletable. (09)
- **Light & modular** — five narrow seams; no plugin framework, no DSL. (§2)
- **Provider/model-agnostic** — OpenAI-compatible client + config; runs on Ollama by default. (03)
- **Degrades gracefully** — capability profiles + tolerant parsing for weak models. (03, 04)
- **Measured** — tracing is free from the state log; evals gate changes and model swaps. (08)

## What works / What doesn't

| ✅ Works | ❌ Doesn't |
|---------|-----------|
| 4-module core + opt-in modules behind 5 seams | A monolith, or a sprawling plugin platform |
| Build the loop; adopt commodity pieces (SDK, OTel, sandbox) | Adopting a framework that owns the loop |
| Config-driven model/limits/policy | Hardcoded model, caps, and paths |
| Phased path where each phase ships value | Building v4 features before v0 works |
| Default config runs offline on Ollama | A design that only works against one hosted vendor |

---

**You've reached the end of the guide.** Back to the [README](README.md) · the lean core is
defined in [02](02_anatomy_and_the_loop.md) · the don't-do list is [09](09_antipatterns_and_checklist.md).
