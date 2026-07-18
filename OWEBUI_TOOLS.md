# Tool calls as a single collapsible "Thinking" block in Open WebUI

## Context

The OpenAI-compatible adapter (`api/openai_compatible.py`) currently renders each
completed tool call as its own collapsible `<details><summary>🔧 …</summary>` block
injected into `delta.content`. In Open WebUI this produces a **long list of separate
collapsibles** interleaved with the answer, which is noisy. The goal is to consolidate
**all** tool activity for a turn into a **single** collapsible block while the final
answer **still streams live**.

### Why the current shape can't do this, and what does

Inline markup (`<details>` or `<think>` tags) lives inside the single `content` text
stream — one channel that can't be "reopened," so consolidating tools into one block
would force buffering the answer. Open WebUI provides a **second, independent delta
channel** for exactly this: `delta.reasoning_content` (also accepts `reasoning` /
`thinking`). Open WebUI aggregates **all** `reasoning_content` deltas into **one**
collapsible "Thinking" block regardless of interleaving, and streams `content` deltas
as the live answer. This is the DeepSeek-R1 / o-series pattern.

- Route **tool activity → `reasoning_content`** → collapses into one "Thinking" block.
- Route **answer text → `content`** → streams token-by-token, unchanged.

Verified against Open WebUI docs (recognized structured fields: `reasoning_content`,
`reasoning`, `thinking`) and behavior. Known risk: Open WebUI issue #24697 — on some
versions, `content` arriving after `reasoning_content` can be dropped by frontend
stream reconciliation. Version-dependent; must be verified on the target instance.
Fallback if it bites: buffered single-`<think>`-in-content block (loses live answer).

Scope: **streaming only** (per decision). Non-streaming `_completion_body` continues to
return answer text only (it already drops tool activity today).

## Changes — all in `api/openai_compatible.py`

### 1. Route tool blocks to the reasoning channel (`_stream`, ~line 295)
In the `ToolResultEvent` branch, emit the rendered block under `reasoning_content`
instead of `content`:
```python
elif isinstance(event, ToolResultEvent):
    block = _tool_summary(event, pending_args.pop(event.id, None), tool_block_max_chars)
    yield {"data": json.dumps(
        _chunk(cid, created, model, {"reasoning_content": block}, None))}
```
`TextEvent` handling is unchanged (still `{"content": event.text}`) so the answer
streams live. `ToolCallEvent` still just buffers `pending_args`. Because every tool
block goes to `reasoning_content`, Open WebUI concatenates them into one collapsible.

### 2. Simplify the block rendering (`_tool_details` → `_tool_summary`, ~line 121)
Open WebUI supplies the outer collapsible, so drop the `<details><summary>` wrapper.
Render clean markdown per tool that concatenates well within one reasoning block, e.g.
a `🔧 {name} {icon} · {ms}` heading, a fenced `json` args block, and a fenced result
block (keep the existing truncation at `max_chars`; the `</details>`→defang guard is
no longer needed once the wrapper is gone). Future "embed more stuff" lives here —
additional lines appended into the same reasoning stream.

### 3. Fix inbound history stripping (`_TOOL_BLOCK_RE` / `_strip_tool_blocks`, ~line 115)
Tool activity now leaves via `reasoning_content`, and Open WebUI serializes prior-turn
reasoning back into the assistant message `content` as a `<details type="reasoning">…</details>`
block on replay. Update the strip regex to remove **that** shape so tool spam is not
re-fed to the model:
```python
_TOOL_BLOCK_RE = re.compile(
    r"\n*<details\b[^>]*type=[\"']?reasoning[\"']?[^>]*>.*?</details>\n*",
    re.DOTALL,
)
```
Keep matching for the legacy `🔧 ` `<details>` form only if backward compat with
already-stored histories is needed; otherwise the reasoning-details rule suffices.
`_strip_tool_blocks` / `_prepare` call sites are unchanged.

## Files
- `api/openai_compatible.py` — the only file to modify (`_stream`, `_tool_details`,
  `_TOOL_BLOCK_RE`). No changes to the core loop, events, or providers.

## Verification (end-to-end)
1. Start the API server; point an Open WebUI instance at `/v1` (or replay a saved SSE).
2. Send a prompt that triggers ≥2 tool calls followed by a text answer.
3. Confirm in Open WebUI: **one** collapsible "Thinking" block containing all tool
   calls, and the final answer **streams live** as normal content (watch tokens appear).
4. Multi-round check: a prompt causing tool → text → tool → answer should still yield a
   single Thinking block (reasoning_content aggregates) with the answer streamed.
5. **Bug #24697 check:** confirm no answer content is dropped after the thinking block on
   this Open WebUI version. If dropped, fall back to buffered `<think>`-in-content.
6. History check: continue the conversation a second turn; confirm the replayed assistant
   history has the `<details type="reasoning">` tool block stripped (inspect the request
   the model receives / trace) so tool output isn't re-ingested.
7. Inspect raw SSE (curl `/v1/chat/completions` with `"stream": true`): tool frames carry
   `delta.reasoning_content`, text frames carry `delta.content`.

## References
- Open WebUI reasoning docs: https://docs.openwebui.com/features/chat-conversations/chat-features/reasoning-models/
- Interleaving/content-drop bug #24697: https://github.com/open-webui/open-webui/issues/24697
- `<think>` in OpenAI-compat SSE (Path 1 background) #23923: https://github.com/open-webui/open-webui/issues/23923
