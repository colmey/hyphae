# orchestrator/orchestrator.py

"""One orchestration LLM call -> sanitized OrchestrationDecision."""

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

# Compact history tail for routing follow-up turns.
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
        self._registry = registry
        self._mcp = mcp
        self._system_prompt = system_prompt
        self._orch_model_id = model_id or registry.default_id()
        self._fallback_system = (
            fallback_system_prompt
            or "You are a helpful assistant. Use the available tools when relevant."
        )

    async def decide(
        self,
        user_message: str,
        preferences: ToolPreferences | None = None,
        history: list[Message] | None = None,
    ) -> OrchestrationDecision:
        """Run one orchestration call. Always returns a valid decision.

        Failures degrade to a fallback decision with `fallback_used=True`.
        Preferences are soft hints; valid preferred tools are guaranteed exposed.
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

    def _build_prompt(
        self,
        user_message: str,
        preferences: ToolPreferences | None = None,
        history: list[Message] | None = None,
    ) -> str:
        """Compose the full user-turn prompt."""
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
        """Render a compact text-only history tail, or "" when there's none."""
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
        """Run the LLM call and return raw text; no tools are exposed."""
        response = await llm.complete(
            messages=[Message.user(prompt)],
            tools=None,
            system=self._system_prompt,
            response_schema=OrchestrationResult,
        )

        parts: list[str] = []
        for block in response.text_blocks():
            if block.text:
                parts.append(block.text)
        return "".join(parts).strip()

    def _parse_result(self, raw: str) -> OrchestrationResult:
        """Parse raw text into OrchestrationResult, accepting fenced JSON."""
        if not raw:
            raise json.JSONDecodeError("empty response", raw, 0)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip("\n")
            cleaned = cleaned.rstrip("`").strip()

        data = json.loads(cleaned)
        return OrchestrationResult.model_validate(data)

    def _sanitize(
        self,
        result: OrchestrationResult,
        preferences: ToolPreferences | None = None,
    ) -> OrchestrationResult:
        """Coerce the orchestrator's decision to known-valid values.

        Unknown models fall back to default; unknown tools are dropped. Valid
        preferred tools are unioned in without marking the decision as fallback.
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

        # Keep the caller's valid preferred tools visible.
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
        """Safe default-model/all-tools decision for orchestration failures."""
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
