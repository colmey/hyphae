# 04 — Tool Design (The Agent–Computer Interface)

Tools are how the model affects the world. **Tool quality determines agent quality more than almost
anything except the model itself.** Tools for a non-deterministic agent need fundamentally
different design than APIs for human programmers — Anthropic calls this the
[agent–computer interface](https://www.anthropic.com/engineering/writing-tools-for-agents) (ACI).

## 1. The mindset: tool descriptions are prompts

Every word in a tool's name, description, and parameter docs shapes how the model uses it. A tool
spec is not documentation that sits beside the code — **it is part of the prompt the model reasons
over.** Tools must be:

- **Clear** enough that the model can't easily misuse them.
- **Informative** enough to steer the model toward good strategies.
- **Efficient** enough to preserve the context budget ([05](05_context_and_memory.md)).

Anthropic's iterative process: **Prototype → Evaluate → Collaborate** (test tools with the model,
read the failures, refine the spec). Treat tool specs as code you iterate on with eval feedback.

## 2. Five design principles

| Principle | Do | Avoid |
|-----------|----|----|
| **High leverage** | Tools that meaningfully expand capability; consolidate common multi-step jobs into one tool | Thin 1:1 wrappers around every REST endpoint |
| **Clear, distinct names** | `search_orders`, `cancel_order` | `get`, `do`, `handler`, overlapping names |
| **Human-readable I/O** | Names, titles, enums the model understands | Opaque UUIDs/codes the model must memorize |
| **Token-efficient output** | Pagination, truncation, filtering, `concise` vs `detailed` modes | Dumping raw multi-KB JSON into context |
| **Defensive & self-describing** | Validate args; return errors that teach the fix | Crashing, or returning stack traces |

**The litmus test (from Anthropic):** *if a human engineer can't definitively say which tool to use
for a task, the model won't either.* Overlap is a design smell.

## 3. Keep the toolset small (this is a context decision)

Tool count is a primary driver of reliability:

- Selection quality degrades noticeably **above ~20 tools**, and weak/local models degrade much
  sooner (~10). ([09](09_antipatterns_and_checklist.md) "tool bloat".)
- Every tool schema is **tokens in every model call** — bloat taxes the context budget on *every*
  turn, not just when used.

Tactics: ship the minimum viable toolset; **consolidate** (`edit_file` over
`open`/`seek`/`write`/`close`); **gate tools by phase/context** (only expose what's relevant now);
prefer one flexible tool with an enum mode over five near-duplicates.

## 4. Tool schema & namespacing

Use **JSON Schema** for parameters — it's the OpenAI function-calling format and is portable across
providers ([03](03_model_interface.md)):

```json
{
  "name": "search_files",
  "description": "Search file CONTENTS by regex within a directory. Returns matching lines with paths and line numbers. Use this to locate code; use read_file to view a full file.",
  "parameters": {
    "type": "object",
    "properties": {
      "pattern": {"type": "string", "description": "Regular expression to search for."},
      "path":    {"type": "string", "description": "Directory to search (default: repo root)."},
      "max_results": {"type": "integer", "default": 50, "description": "Cap results to protect context."}
    },
    "required": ["pattern"]
  }
}
```

- **Descriptions do the heavy lifting.** State what it does, when to use it (and when *not* to),
  what it returns, and how it relates to sibling tools.
- **Namespace** tools when you have many sources: `github_search`, `fs_read`, `db_query`. Prevents
  collisions and helps selection.
- **Constrain with enums/types** rather than free-form strings wherever possible — fewer ways for
  the model to be wrong, and it enables grammar-constrained decoding for weak models.

## 5. Native function-calling vs prompted tool-calling (the local-model split)

Per [03](03_model_interface.md), tool support varies by model. Support both paths behind the same
tool registry:

| | Native function-calling | Prompted (ReAct-style) fallback |
|---|------------------------|--------------------------------|
| **How** | Pass `tools` to the API; model returns structured `tool_calls` | Render tool specs into the system prompt; ask the model to emit a JSON action; parse it |
| **When** | Frontier/hosted models, capable local models | Weak/local models with no/poor tool support |
| **Reliability** | High | Lower — needs tolerant parsing + repair retry ([03](03_model_interface.md) §4) |
| **Best with** | — | Grammar-constrained decoding to force valid JSON |

**Key:** the registry and dispatcher are identical for both — only the *encoding/decoding* of the
call differs. Pick the encoder by model profile; the rest of the harness is unaware.

## 6. Errors are feedback, not failures (12-Factor #9)

A failed tool call is a normal event in an agent loop. The result you feed back is a **teaching
signal**:

```
❌ Bad:  "Error: ENOENT"
❌ Bad:  <full Python traceback dumped into context>
✅ Good: "File 'sec/config.yaml' not found. Did you mean 'src/config.yaml'?
          Use list_dir('src') to see available files."
```

Good error results: state what went wrong, in plain language, and **point at the likely next
action**. Compact the error — don't flood context with stack traces (an anti-pattern). This single
practice converts dead-ends into self-correction and is most of what makes an agent feel "robust."

## 7. Validate before you execute

Between the model's tool call and execution, the dispatcher should:

1. **Validate arguments** against the JSON Schema (catches hallucinated/missing args — a top
   failure mode). Reject with a helpful message rather than executing garbage.
2. **Authorize** against policy (permissions/sandbox — see [07](07_reliability_and_safety.md)).
3. **Execute** with a timeout.
4. **Normalize the result** (truncate/paginate to protect context; mark success/failure clearly).

This is also the natural seam for an **approval gate** on dangerous tools (write/delete/shell/spend).

## 8. MCP — modular tools as plug-ins (optional)

[Model Context Protocol](https://modelcontextprotocol.io) is an open, provider-agnostic standard
where each external capability is an **MCP server** the harness connects to as a client. Tools are
discovered at runtime; you add/remove/update integrations **without touching the core**.

- **Why it fits our ethos:** it's the cleanest expression of "light and modular" — third parties (or
  separate internal teams) ship tools as independent processes; the harness just speaks the
  protocol. It's vendor-neutral (not Claude-only), so it preserves provider-agnosticism.
- **Keep it optional.** Don't make MCP load-bearing for the core. Expose an MCP *adapter* that
  registers discovered tools into the same tool registry. The harness must run perfectly with zero
  MCP servers.
- **Watch the cost:** every connected MCP server's tools count toward your tool budget and context
  (§3). Connecting "all the servers" is just tool bloat with extra steps. Curate.

## What works / What doesn't

| ✅ Works | ❌ Doesn't |
|---------|-----------|
| Few, high-leverage, clearly-named tools | 30–50 thin wrappers; overlapping names |
| Descriptions written *for the model* (when/why/relations) | Terse descriptions or none |
| Human-readable I/O; paginated/truncated output | Raw UUIDs and unbounded JSON dumps |
| Errors that teach the next action | Stack traces / opaque codes fed back |
| Arg validation + authorization before execution | Executing hallucinated arguments blindly |
| Same registry for native & prompted tool-calling | Two separate tool stacks for hosted vs local |
| MCP as an optional, curated adapter | MCP as a mandatory core dependency; connecting everything |

**Next:** [05 — Context & Memory](05_context_and_memory.md)
