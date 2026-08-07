# hyphae/orchestrator/orchestrator.py

"""One orchestration LLM call -> sanitized OrchestrationDecision."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from time import perf_counter

from pydantic import ValidationError

from hyphae.llm.client import GenerationRequest, LLMClient
from hyphae.llm.schemas import CompletionUsage, Message, Role, TextBlock
from hyphae.agent.context import estimate_usage_tokens
from hyphae.tooling import ToolSnapshot

from .contracts import ModelRegistry
from .schemas import OrchestrationDecision, OrchestrationProposal, ToolPreferences

logger = logging.getLogger(__name__)

# Compact history tail for routing follow-up turns.
_HISTORY_MAX_MESSAGES = 6
_HISTORY_MAX_CHARS_PER_MSG = 500


class Orchestrator:
    """Picks a model and tool subset for one request."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        system_prompt: str,
        model_id: str | None = None,
    ) -> None:
        self._registry = registry
        self._system_prompt = system_prompt
        self._orch_model_id = model_id or registry.default_id()

    async def decide(
        self,
        user_message: str,
        tools: ToolSnapshot,
        preferences: ToolPreferences | None = None,
        history: list[Message] | None = None,
        timeout: float | None = None,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    ) -> OrchestrationDecision:
        """Run one orchestration call. Always returns a valid decision.

        Failures degrade to a fallback decision with `fallback_used=True`.
        Preferences are soft hints; valid preferred tools are guaranteed exposed.
        """
        decision_log = log or logger
        prompt = self._build_prompt(user_message, tools, preferences, history)
        started = perf_counter()

        try:
            if timeout is not None and timeout <= 0:
                raise TimeoutError("turn deadline exhausted before orchestration")
            orch_llm = self._registry.get(self._orch_model_id)
            if timeout is not None:
                async with asyncio.timeout(timeout):
                    raw, usage, latency_ms = await self._call_orchestrator_llm(
                        orch_llm, prompt
                    )
            else:
                raw, usage, latency_ms = await self._call_orchestrator_llm(
                    orch_llm, prompt
                )
        except Exception as exc:
            reason = "control_call_failed"
            decision_log.warning(
                "orchestrator control call failed (%s); using fallback",
                type(exc).__name__,
            )
            return self._fallback_decision(
                reason,
                latency_ms=(perf_counter() - started) * 1000,
            )

        try:
            proposal = self._parse_proposal(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            reason = "invalid_control_output"
            decision_log.warning(
                "orchestrator control output invalid (%s, chars=%d, sha256=%s); "
                "using fallback",
                type(exc).__name__,
                len(raw),
                hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            )
            return self._fallback_decision(reason, usage=usage, latency_ms=latency_ms)

        sanitized_proposal, corrections = self._sanitize_proposal(
            proposal, tools, preferences, log=decision_log
        )
        return OrchestrationDecision(
            result=sanitized_proposal,
            fallback_used=False,
            corrections=corrections,
            usage=usage,
            latency_ms=latency_ms,
            control_model_id=self._orch_model_id,
        )

    def _build_prompt(
        self,
        user_message: str,
        tools: ToolSnapshot,
        preferences: ToolPreferences | None = None,
        history: list[Message] | None = None,
    ) -> str:
        """Compose the full user-turn prompt."""
        models_block = self._registry.describe_for_prompt()
        tools_block = self._describe_tools_for_prompt(tools)
        preferred_block = self._describe_preferences_for_prompt(preferences, tools)
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

    def _describe_history_for_prompt(self, history: list[Message] | None) -> str:
        """Render a compact text-only history tail, or "" when there's none."""
        if not history:
            return ""

        lines: list[str] = []
        for msg in history[-_HISTORY_MAX_MESSAGES:]:
            text = " ".join(
                b.text for b in msg.content if isinstance(b, TextBlock) and b.text
            ).strip()
            if not text:
                continue
            if len(text) > _HISTORY_MAX_CHARS_PER_MSG:
                text = text[: _HISTORY_MAX_CHARS_PER_MSG - 1] + "…"
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
        self, preferences: ToolPreferences | None, tools: ToolSnapshot
    ) -> str:
        """Render caller tool preferences, or "" when there are none."""
        if not preferences:
            return ""

        lines: list[str] = []
        for name in preferences.preferred_tools:
            if name not in tools.names:
                continue
            args: tuple[str, ...] = preferences.tool_arg_hints.get(name, ())
            if args:
                lines.append(f"- {name} (intended arguments: {', '.join(args)})")
            else:
                lines.append(f"- {name}")
        if not lines:
            return ""
        return (
            "PREFERRED TOOLS (favor these; other tools remain available):\n"
            f"{chr(10).join(lines)}\n\n"
        )

    def _describe_tools_for_prompt(self, tools: ToolSnapshot) -> str:
        """Format the turn's tool snapshot for the orchestrator prompt."""
        if not tools:
            return "(no tools available)"

        lines: list[str] = []
        for tool in tools.tools:
            name = tool.name
            desc = tool.description.strip().splitlines()
            first_line = desc[0] if desc else ""
            if len(first_line) > 200:
                first_line = first_line[:197] + "..."
            lines.append(f"- {name}\n    {first_line}")
        return "\n".join(lines)

    async def _call_orchestrator_llm(
        self, llm: LLMClient, prompt: str
    ) -> tuple[str, CompletionUsage, float]:
        """Return raw text, canonical usage, and elapsed milliseconds; no tools."""
        request = GenerationRequest(
            messages=[Message.user(prompt)],
            system=self._system_prompt,
            response_schema=OrchestrationProposal,
        )
        started = perf_counter()
        response = await llm.complete(request)
        latency_ms = (perf_counter() - started) * 1000

        parts: list[str] = []
        for block in response.text_blocks():
            if block.text:
                parts.append(block.text)
        raw = "".join(parts).strip()
        return raw, estimate_usage_tokens(
            response.usage,
            messages=list(request.messages),
            system=request.system,
            response=response,
        ), latency_ms

    def _parse_proposal(self, raw: str) -> OrchestrationProposal:
        """Parse raw text into OrchestrationProposal, accepting fenced JSON."""
        if not raw:
            raise json.JSONDecodeError("empty response", raw, 0)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip("\n")
            cleaned = cleaned.rstrip("`").strip()

        data = json.loads(cleaned)
        return OrchestrationProposal.model_validate(data)

    def _sanitize_proposal(
        self,
        proposal: OrchestrationProposal,
        tools: ToolSnapshot,
        preferences: ToolPreferences | None = None,
        *,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger] = logger,
    ) -> tuple[OrchestrationProposal, tuple[str, ...]]:
        """Coerce the orchestrator's decision to known-valid values.

        Unknown models fall back to default; unknown tools are dropped. Valid
        preferred tools are unioned in without marking the decision as fallback.
        """
        unknown_model = False
        unknown_tool = False
        duplicate_tool = False
        known_models = set(self._registry.model_ids)
        if proposal.selected_model_id not in known_models:
            model_id = self._registry.default_id()
            unknown_model = True
        else:
            model_id = proposal.selected_model_id

        known_tools = tools.names
        valid_tools: list[str] = []
        seen_tools: set[str] = set()
        for t in proposal.selected_tools:
            if t not in known_tools:
                unknown_tool = True
            elif t in seen_tools:
                duplicate_tool = True
            else:
                valid_tools.append(t)
                seen_tools.add(t)

        # Keep the caller's valid preferred tools visible.
        if preferences:
            for t in preferences.preferred_tools:
                if t not in known_tools:
                    log.warning("preferred tool is not in MCP inventory; ignoring it")
                elif t not in valid_tools:
                    valid_tools.append(t)

        corrections = tuple(
            code
            for code, present in (
                ("unknown_model_id", unknown_model),
                ("unknown_tool_id", unknown_tool),
                ("duplicate_tool_id", duplicate_tool),
            )
            if present
        )
        if corrections:
            log.warning(
                "orchestrator selection corrected (codes=%s count=%d)",
                ",".join(corrections),
                len(corrections),
            )

        return OrchestrationProposal(
            selected_model_id=model_id,
            selected_tools=valid_tools,
            thinking_level=proposal.thinking_level,
        ), corrections

    def _fallback_decision(
        self,
        reason: str,
        *,
        usage: CompletionUsage | None = None,
        latency_ms: float = 0.0,
    ) -> OrchestrationDecision:
        """Safe default-model/no-tools decision for orchestration failures."""
        proposal = OrchestrationProposal(
            selected_model_id=self._registry.default_id(),
            selected_tools=[],
        )
        return OrchestrationDecision(
            result=proposal,
            fallback_used=True,
            fallback_reason=reason,
            usage=usage or CompletionUsage(),
            latency_ms=latency_ms,
            control_model_id=self._orch_model_id,
        )
