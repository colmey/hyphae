"""Live pytest coverage for the configured orchestration layer.

Verifies:
  1. models.yaml loads and validates.
  2. orchestrator_prompt.txt loads.
  3. LLMRegistry constructs a client for the default model lazily.
  4. Orchestrator.decide() returns a non-fallback OrchestrationDecision
     for a real user message, against the live MCP tool inventory, with a
     valid thinking_level.

Plus two offline checks (no LLM call):
  - OrchestrationResult.thinking_level parsing: default, passthrough,
    casing, and lenient coercion of an unknown value.
  - _build_prompt folds a passed history into a CONVERSATION SO FAR block
    and omits it on a first turn.

Important: this test asserts that the orchestrator's LLM call SUCCEEDED.
A "passing" run where every decision.fallback_used is True is a FAIL --
it means the orchestrator never made a real decision, only the safe
default. The earlier version of this test missed that case because its
structural assertions (model_id in registry, tools in MCP inventory)
were satisfied by the fallback values too.

Run explicitly with ``./runscript.sh -m pytest -m "live and model and mcp"``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from config import get_settings, load_mcp_config_from_settings, reset_settings
from llm.schemas import Message, TextBlock
from mcp_layer import MCPManager, ToolSnapshot
from config import load_models_config, load_orchestrator_prompt
from orchestrator import LLMRegistry, Orchestrator, ToolPreferences
from orchestrator.schemas import OrchestrationResult


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
pytestmark = [pytest.mark.live, pytest.mark.model, pytest.mark.mcp, pytest.mark.anyio]


def _check_thinking_level_parsing() -> None:
    """Offline: thinking_level parses, defaults, and coerces leniently.

    No LLM call -- just exercises the OrchestrationResult schema so a bad
    thinking_level value can never sink an otherwise-valid decision.
    """
    print("=" * 72)
    print("Offline: OrchestrationResult.thinking_level parsing")
    print("=" * 72)
    base = {"selected_model_id": "x", "generated_system_prompt": "y"}

    assert OrchestrationResult.model_validate(base).thinking_level == "medium", (
        "omitted thinking_level should default to 'medium'"
    )
    assert (
        OrchestrationResult.model_validate(
            {**base, "thinking_level": "high"}
        ).thinking_level
        == "high"
    ), "valid thinking_level should pass through"
    assert (
        OrchestrationResult.model_validate(
            {**base, "thinking_level": "HIGH"}
        ).thinking_level
        == "high"
    ), "casing should be normalized"
    assert (
        OrchestrationResult.model_validate(
            {**base, "thinking_level": "extreme"}
        ).thinking_level
        == "medium"
    ), "unknown thinking_level should coerce to 'medium'"
    # Extra hallucinated fields are ignored (lenient schema), not rejected.
    assert (
        OrchestrationResult.model_validate(
            {**base, "thinking_level": "low", "made_up": 1}
        ).thinking_level
        == "low"
    ), "extra fields should be ignored"
    print("  thinking_level parsing OK (default/passthrough/casing/coerce/extra)\n")


def _check_history_block(orch: Orchestrator) -> None:
    """Offline: a passed history produces a CONVERSATION SO FAR block."""
    print("=" * 72)
    print("Offline: _build_prompt history block")
    print("=" * 72)
    history = [
        Message.user("List the tables in the customer database."),
        Message.assistant(
            [TextBlock(text="The tables are: customers, orders, invoices.")]
        ),
    ]
    tools = ToolSnapshot()
    with_hist = orch._build_prompt(
        "now do the same for last month", tools, history=history
    )
    assert "CONVERSATION SO FAR" in with_hist, "history block missing from prompt"
    assert "customer database" in with_hist, "history content missing from prompt"

    without_hist = orch._build_prompt("hello", tools, history=None)
    assert "CONVERSATION SO FAR" not in without_hist, (
        "first-turn prompt should have no history block"
    )
    print("  history block OK (present with history, absent without)\n")


async def test_configured_orchestration_decisions() -> None:
    reset_settings()
    settings = get_settings()

    _check_thinking_level_parsing()

    # ----- step 1: config -----
    print("=" * 72)
    print("Loading models.yaml and orchestrator_prompt.txt")
    print("=" * 72)
    models_config = load_models_config(settings.models_config_path)
    orch_prompt = load_orchestrator_prompt(settings.orchestrator_prompt_path)
    print(f"  models loaded: {list(models_config.models.keys())}")
    print(f"  default model: {models_config.default_id()}")
    print(f"  prompt length: {len(orch_prompt)} chars")
    print()

    # ----- step 2: registry -----
    print("=" * 72)
    print("Building LLMRegistry; verifying default client constructs")
    print("=" * 72)
    registry = LLMRegistry(models_config, settings)
    default_id = registry.default_id()
    client = registry.get(default_id)
    print(f"  registry default_id = {default_id}")
    print(f"  client class        = {type(client).__name__}")
    print()

    # ----- step 3: MCP (needed for tool inventory) -----
    mcp_config = load_mcp_config_from_settings(settings)
    mcp = MCPManager(mcp_config)
    await mcp.startup()
    all_tool_names = {name for name, _ in mcp.list_tools()}
    total_tools = len(all_tool_names)

    failures: list[str] = []

    try:
        # ----- step 4: orchestrator.decide() -----
        print("=" * 72)
        print("Running Orchestrator.decide() on three sample messages")
        print("=" * 72)
        orch = Orchestrator(
            registry=registry,
            system_prompt=orch_prompt,
            model_id=settings.orchestrator_model_id or None,
        )

        _check_history_block(orch)

        messages = [
            "What is 2 + 2?",
            "List the tables in the customer database.",
            "Compare the structure of the customer database with what you find via web search "
            "about typical ERP database schemas, then summarize the differences.",
        ]

        for i, msg in enumerate(messages, 1):
            print(f"\n--- decision {i} ---")
            print(f"  user: {msg!r}")
            snapshot = ToolSnapshot.from_llm_tools(mcp.get_tools_for_llm())
            decision = await orch.decide(msg, snapshot)
            result = decision.result

            print(f"  fallback_used:           {decision.fallback_used}")
            if decision.fallback_reason:
                print(f"  fallback_reason:         {decision.fallback_reason}")
            print(f"  selected_model_id:       {result.selected_model_id}")
            print(f"  thinking_level:          {result.thinking_level}")
            print(f"  selected_tools ({len(result.selected_tools)}/{total_tools}):")
            for t in result.selected_tools:
                print(f"      - {t}")
            sys_preview = result.generated_system_prompt.replace("\n", " ")
            if len(sys_preview) > 200:
                sys_preview = sys_preview[:197] + "..."
            print(f"  generated_system_prompt: {sys_preview}")

            # ----- structural assertions (cheap; always must hold) -----
            assert result.selected_model_id in registry.model_ids, (
                f"selected_model_id {result.selected_model_id!r} not in registry"
            )
            for t in result.selected_tools:
                assert t in all_tool_names, (
                    f"orchestrator returned unknown tool {t!r} (sanitize should have dropped it)"
                )
            assert result.thinking_level in ("low", "medium", "high"), (
                f"thinking_level {result.thinking_level!r} not a valid tier"
            )

            # ----- the critical assertion: the LLM call actually worked -----
            # If decide() fell back, we got the SAFE DEFAULT, not a real
            # orchestration decision. The first version of this test missed
            # that case because the safe default IS structurally valid.
            # Collect rather than raise so we see all three decisions before
            # the test bails out.
            if decision.fallback_used:
                failures.append(
                    f"decision {i}: orchestrator fell back to safe defaults. "
                    f"Reason: {decision.fallback_reason}"
                )

        # ----- preference: a named tool is guaranteed into the selection -----
        if all_tool_names:
            pref_tool = sorted(all_tool_names)[0]
            server, _, tool = pref_tool.partition("__")
            prefs = ToolPreferences.from_request(
                [SimpleNamespace(name=server, tools={tool: ["arg1"]})]
            )
            print(f"\n--- preference decision (prioritizing {pref_tool!r}) ---")
            snapshot = ToolSnapshot.from_llm_tools(mcp.get_tools_for_llm())
            decision = await orch.decide(
                "Do a trivial task.", snapshot, preferences=prefs
            )
            print(f"  selected_tools: {decision.result.selected_tools}")
            assert pref_tool in decision.result.selected_tools, (
                f"preferred tool {pref_tool!r} was not guaranteed into the selection"
            )

        print()
        if failures:
            print("=" * 72)
            print(
                f"FAIL: {len(failures)} of {len(messages)} decisions used the fallback path."
            )
            print("=" * 72)
            for f in failures:
                print(f"  - {f}")
            print()
            print("The orchestrator's LLM call is failing. Inspect the warning log")
            print("above for the underlying error (typical causes: API key missing,")
            print("model name wrong, response_schema not supported by provider).")
            raise SystemExit(1)

        print("orchestrator smoke test passed (all decisions made real LLM calls).")
    finally:
        await mcp.shutdown()
