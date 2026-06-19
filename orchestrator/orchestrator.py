# orchestrator/orchestrator.py

"""
The Orchestrator: one LLM call -> OrchestrationDecision.

Per request flow:
  1. Build the orchestrator prompt: system prompt (from file) +
     models inventory + tools inventory + user message.
  2. Call the orchestrator's LLM client with response_schema=OrchestrationResult.
     (Gemini honors the schema; other providers fall back to raw text + parse.)
  3. Parse the JSON response into OrchestrationResult.
  4. Sanitize: drop tool names that don't exist; fall back to default model
     if the chosen one isn't in the registry.
  5. Wrap in OrchestrationDecision(result=..., fallback_used=False).

Failure handling: ANY failure (LLM error, parse error, validation error)
is caught and replaced with a safe fallback decision -- BUT the decision
is returned with `fallback_used=True` and a `fallback_reason` so the
caller (route, smoke test) can distinguish a real orchestration from a
degraded one. The orchestrator must never break a request; it's an
optimization layer, not a gate. The fallback flag is what makes the
degradation observable.

The Orchestrator does NOT touch sessions or the agent loop.
The route is responsible for invoking it and feeding its output into
run_agent().
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from llm.client import LLMClient
from llm.schemas import Message, Role, TextBlock
from mcp_layer import MCPManager

from .registry import LLMRegistry
from .schemas import OrchestrationDecision, OrchestrationResult, ToolPreferences

logger = logging.getLogger(__name__)

# How much prior conversation to show the orchestrator when routing a follow-up
# turn: the last few messages, each clipped. Enough context to resolve pronouns
# and keep thread-critical tools, without flooding the orchestrator prompt.
_HISTORY_MAX_MESSAGES = 6
_HISTORY_MAX_CHARS_PER_MSG = 500


class Orchestrator:
    """Picks a model, a tool subset, and a system prompt for one request."""

    def __init__(
        self,
        *,
        registry: LLMRegistry,
        mcp: MCPManager,
        system_prompt: str,
        model_id: str | None = None,
        fallback_system_prompt: str | None = None,
    ) -> None:
        """
        Args:
          registry: the LLMRegistry (used both to pick the orchestrator's own
                    LLM and to validate orchestrator output).
          mcp: the MCPManager (used to enumerate available tools for the
               orchestrator prompt).
          system_prompt: the orchestrator's system instruction (loaded from
                         orchestrator_prompt.txt).
          model_id: optional override for which model the orchestrator itself
                    uses to make its decision. Defaults to registry.default_id().
          fallback_system_prompt: system prompt used when orchestration fails.
                                  Defaults to a minimal generic instruction.
        """
        self._registry = registry
        self._mcp = mcp
        self._system_prompt = system_prompt
        self._orch_model_id = model_id or registry.default_id()
        self._fallback_system = (
            fallback_system_prompt
            or "You are a helpful assistant. Use the available tools when relevant."
        )

    # ----- public API -----

    async def decide(
        self,
        user_message: str,
        preferences: ToolPreferences | None = None,
        history: list[Message] | None = None,
    ) -> OrchestrationDecision:
        """Run one orchestration call. Always returns a valid decision.

        Never raises -- failures degrade to the fallback decision with
        `fallback_used=True` and are logged. Callers should check
        `decision.fallback_used` to detect degradation.

        `preferences` (optional) carries caller hints about which tools to
        prioritize. They are rendered into the prompt and the valid ones are
        guaranteed to be exposed (see _sanitize) -- a soft priority, not a
        filter: the orchestrator may still pick other tools as fallback.

        `history` (optional) is the session's prior messages, used to route a
        follow-up turn with the conversation in view. A compact tail is folded
        into the prompt as a CONVERSATION SO FAR block. None / empty behaves
        exactly as a first turn.
        """
        prompt = self._build_prompt(user_message, preferences, history)
        orch_llm = self._registry.get(self._orch_model_id)

        try:
            raw = await self._call_orchestrator_llm(orch_llm, prompt)
        except Exception as e:
            reason = f"orchestrator LLM call failed: {e}"
            logger.warning("%s; using fallback", reason)
            return self._fallback_decision(reason)

        try:
            result = self._parse_result(raw)
        except (json.JSONDecodeError, ValidationError) as e:
            reason = f"orchestrator output unparseable: {e}"
            logger.warning("%s; using fallback. raw=%r",
                           reason, raw[:500] if raw else raw)
            return self._fallback_decision(reason)

        sanitized = self._sanitize(result, preferences)
        return OrchestrationDecision(result=sanitized, fallback_used=False)

    # ----- helpers -----

    def _build_prompt(
        self,
        user_message: str,
        preferences: ToolPreferences | None = None,
        history: list[Message] | None = None,
    ) -> str:
        """Compose the full user-turn prompt: models + tools + history + message.

        When `preferences` is present, a PREFERRED TOOLS block is inserted so
        the orchestrator favors the caller's named tools (and folds their
        intended arguments into the generated system prompt) while keeping
        the rest of the inventory available.

        When `history` is present, a CONVERSATION SO FAR block is inserted just
        before the user message so a follow-up turn is routed with context.
        """
        models_block = self._registry.describe_for_prompt()
        tools_block = self._describe_tools_for_prompt()
        preferred_block = self._describe_preferences_for_prompt(preferences)
        history_block = self._describe_history_for_prompt(history)

        return (
            "AVAILABLE MODELS:\n"
            f"{models_block}\n\n"
            "AVAILABLE TOOLS:\n"
            f"{tools_block}\n\n"
            f"{preferred_block}"
            f"{history_block}"
            "USER MESSAGE:\n"
            f"{user_message}\n\n"
            "Return the JSON decision now."
        )

    def _describe_history_for_prompt(
        self, history: list[Message] | None
    ) -> str:
        """Render a compact tail of prior conversation, or "" when there's none.

        Last few messages, text blocks only, each clipped. Gives the
        orchestrator enough to interpret a pronoun-heavy follow-up ("now do the
        same for last month") and keep the tools the thread depends on, without
        dragging the whole transcript into its prompt. Tool-call / tool-result
        turns carry no plain text and are skipped.
        """
        if not history:
            return ""

        lines: list[str] = []
        for msg in history[-_HISTORY_MAX_MESSAGES:]:
            text = " ".join(
                b.text for b in msg.content
                if isinstance(b, TextBlock) and b.text
            ).strip()
            if not text:
                continue
            if len(text) > _HISTORY_MAX_CHARS_PER_MSG:
                text = text[:_HISTORY_MAX_CHARS_PER_MSG - 1] + "…"
            label = "User" if msg.role == Role.USER else "Assistant"
            lines.append(f"{label}: {text}")

        if not lines:
            return ""

        return (
            "CONVERSATION SO FAR (most recent last; use it to interpret the "
            "user message and keep the tools the thread depends on):\n"
            f"{chr(10).join(lines)}\n\n"
        )

    def _describe_preferences_for_prompt(
        self, preferences: ToolPreferences | None
    ) -> str:
        """Render caller tool preferences, or "" when there are none."""
        if not preferences:
            return ""

        lines: list[str] = []
        for name in preferences.preferred_tools:
            args = preferences.tool_arg_hints.get(name) or []
            if args:
                lines.append(f"- {name} (intended arguments: {', '.join(args)})")
            else:
                lines.append(f"- {name}")
        return (
            "PREFERRED TOOLS (favor these; other tools remain available):\n"
            f"{chr(10).join(lines)}\n\n"
        )

    def _describe_tools_for_prompt(self) -> str:
        """Format the MCP tool inventory for the orchestrator prompt."""
        tools = self._mcp.get_tools_for_llm()
        if not tools:
            return "(no tools available)"

        lines: list[str] = []
        for t in tools:
            name = t["name"]
            desc = (t.get("description") or "").strip().splitlines()
            first_line = desc[0] if desc else ""
            if len(first_line) > 200:
                first_line = first_line[:197] + "..."
            lines.append(f"- {name}\n    {first_line}")
        return "\n".join(lines)

    async def _call_orchestrator_llm(self, llm: LLMClient, prompt: str) -> str:
        """Run the LLM call and return raw text. Tools are NOT exposed to
        the orchestrator -- it must decide, not act.
        """
        response = await llm.complete(
            messages=[Message.user(prompt)],
            tools=None,
            system=self._system_prompt,
            response_schema=OrchestrationResult,
        )

        # Concatenate any text blocks; structured-output mode normally
        # produces a single block, but be defensive.
        parts: list[str] = []
        for block in response.text_blocks():
            if block.text:
                parts.append(block.text)
        return "".join(parts).strip()

    def _parse_result(self, raw: str) -> OrchestrationResult:
        """Parse raw text into OrchestrationResult.

        Tries strict JSON first. If the model wrapped the JSON in a
        markdown fence (despite being told not to), unwrap it and retry.
        """
        if not raw:
            raise json.JSONDecodeError("empty response", raw, 0)

        # Strip optional ```json ... ``` fences. Cheap, common defense.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # Drop a leading "json\n" if present.
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip("\n")
            # And any trailing fence remnant.
            cleaned = cleaned.rstrip("`").strip()

        data = json.loads(cleaned)
        return OrchestrationResult.model_validate(data)

    def _sanitize(
        self,
        result: OrchestrationResult,
        preferences: ToolPreferences | None = None,
    ) -> OrchestrationResult:
        """Coerce the orchestrator's decision to known-valid values.

        - Unknown model_id -> default_id().
        - Tool names not in MCPManager -> dropped.
        - Valid preferred tools -> unioned in, so a caller's priority hint is
          honored even if the orchestrator LLM omitted it. Preserves order:
          the LLM's picks first, then any preferred tools it left out.

        Returns a new OrchestrationResult; the input is not mutated. This
        sanitization is NOT counted as a "fallback" -- the orchestrator
        made a real decision, we just trimmed and topped it up. Only outright
        failure (LLM error, parse error) triggers the fallback flag.
        """
        known_models = set(self._registry.model_ids)
        if result.selected_model_id not in known_models:
            logger.warning(
                "orchestrator picked unknown model_id %r; correcting to %r",
                result.selected_model_id, self._registry.default_id(),
            )
            model_id = self._registry.default_id()
        else:
            model_id = result.selected_model_id

        known_tools = {name for name, _ in self._mcp.list_tools()}
        valid_tools: list[str] = []
        for t in result.selected_tools:
            if t in known_tools:
                valid_tools.append(t)
            else:
                logger.warning("orchestrator picked unknown tool %r; dropping", t)

        # Guarantee the caller's valid preferred tools are exposed. Keeps the
        # priority hint meaningful without locking out the orchestrator's own
        # picks, which stay first in the list.
        if preferences:
            for t in preferences.preferred_tools:
                if t not in known_tools:
                    logger.warning(
                        "preferred tool %r not in MCP inventory; ignoring", t
                    )
                elif t not in valid_tools:
                    valid_tools.append(t)

        return OrchestrationResult(
            selected_model_id=model_id,
            selected_tools=valid_tools,
            generated_system_prompt=result.generated_system_prompt,
            thinking_level=result.thinking_level,
        )

    def _fallback_decision(self, reason: str) -> OrchestrationDecision:
        """Safe decision used when the orchestrator call fails outright.

        Uses the default model and all available tools -- i.e. preserves
        the harness's pre-orchestrator behavior so requests still succeed.
        Returns OrchestrationDecision with fallback_used=True so the
        caller can detect the degradation.
        """
        all_tools = [name for name, _ in self._mcp.list_tools()]
        result = OrchestrationResult(
            selected_model_id=self._registry.default_id(),
            selected_tools=all_tools,
            generated_system_prompt=self._fallback_system,
        )
        return OrchestrationDecision(
            result=result,
            fallback_used=True,
            fallback_reason=reason,
        )