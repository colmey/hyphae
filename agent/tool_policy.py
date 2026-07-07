# agent/tool_policy.py

"""Loop-local policy for whether a requested tool call may execute."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from typing import Any, Sequence


class Verdict(Enum):
    ALLOW = "allow"
    DENY = "deny"
    # ASK reserved for a future human-approval flow; not implemented.


@dataclass(frozen=True)
class PolicyDecision:
    """Policy outcome; DENY carries a model-facing reason."""

    verdict: Verdict
    reason: str | None = None


class ToolPolicy:
    """Decide whether a requested tool call may execute.

    `allow_all` is the default; `allow_list` permits only fnmatch patterns from
    config. Denials become teaching tool-result errors in the loop.
    """

    def __init__(self, mode: str = "allow_all", allow: Sequence[str] = ()) -> None:
        self._mode = mode
        self._allow = tuple(allow)

    def check(self, tool_name: str, args: Any = None) -> PolicyDecision:
        if self._mode != "allow_list":
            return PolicyDecision(Verdict.ALLOW)
        if any(fnmatchcase(tool_name, pattern) for pattern in self._allow):
            return PolicyDecision(Verdict.ALLOW)
        allowed = ", ".join(self._allow) if self._allow else "(none)"
        return PolicyDecision(
            Verdict.DENY,
            reason=(
                f"tool call to {tool_name!r} was blocked by policy: this tool is "
                f"policy-restricted and cannot be run. Allowed tools match: {allowed}. "
                f"Use one of those or answer without this tool."
            ),
        )


_DEFAULT_POLICY = ToolPolicy()


def build_tool_policy(config: Any) -> ToolPolicy:
    """Build from a ToolPolicyConfig-like object without coupling config to agent."""
    return ToolPolicy(mode=config.mode, allow=config.allow)
