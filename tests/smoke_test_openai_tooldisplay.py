# tests/smoke_test_openai_tooldisplay.py

"""
Tool-call presentation for the OpenAI `/v1` adapter: outbound render <-> inbound strip.

Hermetic and pure -- exercises the `_tool_details` / `_strip_tool_blocks` /
`_prepare` helpers directly, no app, LLM, or MCP backend needed. Verifies:
  - a completed tool call renders to a well-formed collapsible <details> block;
  - that block round-trips to nothing through the inbound strip;
  - the strip is precise (a model-authored <details> is left untouched);
  - a tool result containing a literal "</details>" can't end the block early
    (defang), so the whole block still strips cleanly;
  - `_prepare` strips blocks from assistant history but leaves the user prompt.

Run: ./runscript.sh tests/smoke_test_openai_tooldisplay.py
"""

from __future__ import annotations

from bootstrap import load_secrets

load_secrets()

from agent import ToolResultEvent
from api.openai_compatible import (
    _ChatMessage,
    _prepare,
    _strip_tool_blocks,
    _tool_details,
)


def _result(content: str, *, name: str = "web__search", is_error: bool = False,
            latency_ms: float | None = 12.0) -> ToolResultEvent:
    return ToolResultEvent(id="call_1", name=name, content=content,
                           is_error=is_error, latency_ms=latency_ms)


def test_render_then_strip_roundtrips() -> None:
    block = _tool_details(_result("the answer is 42"), {"q": "meaning"})
    assert "<details>" in block and "</details>" in block
    assert "🔧 web__search ✅" in block        # name + ok icon
    assert '"q": "meaning"' in block            # args fence
    assert "the answer is 42" in block          # result fence
    # The rendered block, embedded in surrounding model text, strips to just the text.
    surrounded = f"Here is what I found.{block}So, 42."
    assert _strip_tool_blocks(surrounded) == "Here is what I found.So, 42."
    print("ok: render -> strip round-trips, text preserved")


def test_error_icon() -> None:
    block = _tool_details(_result("boom", is_error=True), {"q": "x"})
    assert "🔧 web__search ❌" in block
    print("ok: error result renders the ❌ icon")


def test_strip_is_precise() -> None:
    # A model-authored <details> (no 🔧 marker) must survive untouched.
    authored = "intro\n<details>\n<summary>Notes</summary>\nkeep me\n</details>\nend"
    assert _strip_tool_blocks(authored) == authored
    print("ok: model-authored <details> is left untouched")


def test_defang_blocks_early_close() -> None:
    # A tool result that literally contains "</details>" must not end the block
    # early -- the whole block (and only it) strips, leaving no dangling fragment.
    block = _tool_details(_result("snippet with </details> inside it"), {"q": "html"})
    assert block.count("</details>") == 1          # only the block's own real closer
    assert "<\u200b/details>" in block             # the embedded one is ZWSP-defanged
    out = _strip_tool_blocks(f"before{block}after")
    assert out == "beforeafter", repr(out)
    print("ok: embedded </details> is defanged; block strips cleanly")


def test_prepare_strips_history_not_prompt() -> None:
    block = _tool_details(_result("tool output"), {"q": "y"})
    messages = [
        _ChatMessage(role="system", content="be terse"),
        _ChatMessage(role="user", content="first question"),
        _ChatMessage(role="assistant", content=f"I checked.{block}Done."),
        _ChatMessage(role="user", content="follow-up question"),
    ]
    system_override, history, prompt = _prepare(messages)
    assert system_override == "be terse"
    assert prompt == "follow-up question"           # final user, untouched
    # The assistant turn in history has the block removed.
    assistant_texts = [t for role, t in history if role == "assistant"]
    assert assistant_texts == ["I checked.Done."], repr(assistant_texts)
    assert all("<details>" not in t for _r, t in history)
    print("ok: _prepare strips assistant history blocks, prompt preserved")


def main() -> None:
    test_render_then_strip_roundtrips()
    test_error_icon()
    test_strip_is_precise()
    test_defang_blocks_early_close()
    test_prepare_strips_history_not_prompt()
    print("\ntool-display smoke test passed.")


if __name__ == "__main__":
    main()