# Future: Progressive Tool Discovery

Status: deferred pending evidence from a representative large MCP catalog.

Hyphae currently captures one immutable, policy-visible tool snapshot per
accepted turn. The orchestrator sees each tool's namespaced name and a bounded
one-line description, selects a subset, and exposes full schemas only for that
subset to the downstream agent. `ToolPolicy` remains the dispatch authority.

This is intentionally simpler than progressive discovery and remains the
supported behavior. A large MCP deployment—such as Home Assistant—may justify
revisiting it if catalog size materially increases routing cost or causes
repeatable selection failures.

## Evidence required to reopen

Measure the real deployment before designing production changes:

1. Record policy-visible tool count, compact orchestrator-catalog characters
   and estimated tokens, routing latency, and the fraction of the control
   model's usable input budget consumed by the catalog.
2. Run representative workflows and identify repeatable cases where the
   required tool is omitted because of catalog selection pressure—not MCP
   connectivity, policy denial, weak descriptions, or general model quality.
3. Compare current routing with a test-only bounded candidate method. It must
   recover required tools without reducing task success or materially adding
   iterations.
4. Evaluate simpler controls first: `disabled_tools`, `ToolPolicy`, clearer
   descriptions, or a smaller statically configured tool surface.

If those checks establish material pressure and behavioral failure, write a
new bounded implementation plan and obtain approval before changing production
behavior.

## Constraints on a future design

A future plan should preserve these current guarantees:

- The full policy-visible catalog has one owner and remains immutable for the
  accepted turn.
- Discovery cannot expose policy-hidden tools or create a second authorization
  path.
- Search or indexing opens no MCP connection; actual dispatch alone owns the
  turn-local lease.
- Tool execution remains sequential, and transport uncertainty never triggers
  automatic replay.
- Candidate and active-tool limits, schema/token accounting, additional
  iterations, and failure behavior are explicit and tested.
- No second catalog, background connection, persisted activation state, vector
  database, plugin framework, or generic retrieval subsystem is introduced
  without separately demonstrated need.

The evidence phase is read-only or test-only. Passing it produces a fresh
implementation proposal; it does not authorize implementation by itself.
