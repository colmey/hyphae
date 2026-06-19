# 03 — Tool Design (The Agent–Computer Interface)

**Research basis:** Doc 04 (Tool Design), Doc 10 (Tool seam).
**Verdict:** 🟡 **Partial** — the tool *plumbing* is excellent (MCP adapter, namespacing, errors-as-feedback, source truncation, curated subsets). The gap is the **validate / authorize seam between the model's tool call and execution** that Doc 04 puts at the center of reliability and safety.

---

## What the research wants (Doc 04)

Tools are how the model affects the world; tool quality drives agent quality more than almost anything but the model. Keep the toolset **small** (selection degrades past ~20 tools, ~10 for weak models). Between the model's call and execution, the dispatcher must: **validate arguments against schema** (the top failure mode is hallucinated args), **authorize against policy**, **execute with a timeout**, and **normalize the result** (truncate/paginate to protect context). **Errors are feedback** — return plain-language messages that point at the next action, not stack traces. MCP is the ideal *optional, curated* tool adapter, but every server's tools count toward the context budget.

---

## What the code does

### ✅ MCP as an optional, namespaced adapter
`MCPManager` aggregates per-server `MCPClient`s and presents a unified registry, namespacing every tool as `{server}__{tool}` to prevent collisions ([mcp_layer/manager.py:30](../mcp_layer/manager.py#L30), [72-74](../mcp_layer/manager.py#L72-L74)); `__` is reserved and rejected in server names. Tools cross to the LLM layer in the generic `{name, description, input_schema}` shape ([mcp_layer/manager.py:108-121](../mcp_layer/manager.py#L108-L121)) — provider-agnostic, exactly Doc 04's portable JSON-Schema approach. This is the Tool seam from Doc 10.

### ✅ Optional and degradable — not load-bearing
Startup connects to enabled servers in parallel and **individual server failures are logged, not fatal** ([mcp_layer/manager.py:44-79](../mcp_layer/manager.py#L44-L79)) — "the harness can still operate with a degraded tool set." Unknown-tool and disconnected-server calls return `is_error` results rather than raising ([mcp_layer/manager.py:125-139](../mcp_layer/manager.py#L125-L139)). Doc 04's "harness must run perfectly with zero MCP servers" holds.

### ✅ Errors are feedback, not failures
This is the harness's strongest tool-layer property. In the loop, a tool exception, a timeout, and a stall all become `ToolResultBlock(is_error=True)` fed straight back to the model ([agent/loop.py:418-460](../agent/loop.py#L418-L460)); the MCP client itself catches exceptions and returns `ToolCallResult(is_error=True)` rather than throwing. Doc 04: "this single practice converts dead-ends into self-correction and is most of what makes an agent feel robust." Present.

### ✅ Token-efficient output — truncate at the source
`_clip_tool_content` bounds a flattened tool result to `tool_result_max_chars` before it enters history, with a marker recording how much was dropped ([agent/loop.py:136-146](../agent/loop.py#L136-L146)), clipped once so streamed and stored content stay identical ([agent/loop.py:436-439](../agent/loop.py#L436-L439)). This is Doc 04/05's "truncate/paginate at the source" — protecting the context budget on every later turn.

### ✅ Small, curated toolset
The orchestrator selects the *smallest tool subset* per request and hands the loop only that subset ([agent/loop.py:289-293](../agent/loop.py#L289-L293) consumes the pre-filtered `tools`). This actively fights Doc 04/09's "tool bloat" anti-pattern — a phase-gating analog most harnesses never build. With only two MCP servers configured, the toolset is well within the ~20-tool reliability ceiling.

### ✅ Sequential execution (deliberate)
Tools run one at a time ([agent/loop.py:396-417](../agent/loop.py#L396-L417)) because some MCP tools have side effects. This is a defensible safety choice; the docstring notes parallelism can be added later behind `asyncio.gather`.

---

## Gaps & recommendations

### 🔴 (a) No argument validation before execution
This is the central Doc 04 seam, and it's missing. `mcp.call_tool(tu.name, tu.input)` passes the model's arguments straight through ([agent/loop.py:412-417](../agent/loop.py#L412-L417)); `MCPManager.call_tool` routes by name only ([mcp_layer/manager.py:125-139](../mcp_layer/manager.py#L125-L139)) and forwards to the server with no check against the tool's `input_schema`. Hallucinated or missing arguments — Doc 04/09's *top* tool failure mode — are caught only by the remote MCP server, if at all, and surface as whatever opaque error that server returns. (The OpenAI provider compounds this by silently coercing an unparseable arguments string to `{}` — see `02-model-interface.md`.)

> **Recommendation (P0):** validate `tu.input` against the tool's `input_schema` (already available on the `Tool`) in a small dispatch seam *before* calling the server. On failure, return an `is_error` result that names the offending field and shows the expected shape — a teaching message, per Doc 04. A JSON-Schema validator (e.g. `jsonschema`) is the adopt-not-build choice (Doc 10).

### 🔴 (b) No authorization / policy seam at dispatch
Doc 04 §7 and Doc 07 want the dispatch point to be the natural seam for allow/deny/ask and for an approval gate on dangerous tools. Today every tool that the orchestrator exposed executes unconditionally. (Full treatment in `06-reliability-safety.md`; flagged here because the *seam* belongs in the tool dispatcher.)

> **Recommendation (P1):** insert a `policy.check(state, tool_call) -> allow | deny | ask` call at the same seam as (a). Read-only-by-default with a write allow-list is a few dozen lines and covers most risk.

### 🟡 (c) Error messages are passthrough, not "point at the next action"
The harness reliably *delivers* errors to the model, but their content is whatever the MCP server produced (or `tool {name} timed out…` / `tool execution raised: {e}`). Doc 04's ideal ("File 'sec/config.yaml' not found. Did you mean 'src/config.yaml'? Use list_dir('src')…") requires the dispatcher to enrich. Lower priority — getting validation (a) in place is the bigger win and naturally produces teaching messages.

### 🟡 (d) MCP content flattening drops structure
`MCPClient` flattens MCP content blocks (text/image/etc.) to a single string for v1. This is a reasonable simplification but means image/structured tool results are lossy. Acknowledged in code comments; defer.

### ⬜ (e) Curate as servers grow (Doc 04 cost warning)
With two servers this is a non-issue, but the research is explicit: "every MCP server's tools count toward your tool budget and context. Connecting 'all servers' is tool bloat with extra steps." The orchestrator's subset selection mitigates this at *call* time, but each connected server's schemas still inflate the orchestrator's own decision prompt. Worth a note in docs as the server list grows.

---

## What works / what doesn't — scored

| Doc 04 criterion | Status | Evidence |
|---|---|---|
| Few, high-leverage, clearly-named tools | ✅ | orchestrator subset; 2 servers |
| JSON-Schema params, namespaced | ✅ | [mcp_layer/manager.py:108-121](../mcp_layer/manager.py#L108-L121) |
| Validate args before execution | 🔴 | passthrough ([agent/loop.py:412-417](../agent/loop.py#L412-L417)) |
| Authorize before execution (policy seam) | 🔴 | none |
| Errors that teach the next action | 🟡 | delivered as feedback, content is passthrough |
| Output truncated/paginated at source | ✅ | [agent/loop.py:136-146](../agent/loop.py#L136-L146) |
| Native & prompted tool-calling, one registry | 🟡 | native only (see `02`) |
| MCP optional & curated, not load-bearing | ✅ | [mcp_layer/manager.py:44-79](../mcp_layer/manager.py#L44-L79) |
