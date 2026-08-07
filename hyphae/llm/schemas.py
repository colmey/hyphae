# hyphae/llm/schemas.py
"""
Provider-agnostic message and tool types used internally by the harness.

These are the canonical shapes that the agent loop, session store, and
LLMClient implementations all speak. Each provider-specific client is
responsible for translating between these and its own SDK's types.

Design notes:
  - The "content" of a message is always a list of blocks, even when there's
    only one. This matches modern LLM APIs (Anthropic, Gemini, OpenAI) and
    lets a single assistant turn mix text and tool calls cleanly.
  - Tool use blocks carry an `id` that the matching tool_result must echo.
    For providers that don't have native call IDs (Gemini identifies calls
    by function name + position), the client mints synthetic IDs.
  - We keep things as plain dataclasses for now. Switch to Pydantic later
    only if we need validation or serialization beyond what session
    persistence requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeAlias, Union


class Role(str, Enum):
    """Roles in a conversation. `tool` is used for tool result messages."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


CanonicalStopReason: TypeAlias = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "empty",
    "content_filter",
    "refusal",
    "provider_error",
    "incomplete_stream",
]


# ----- content blocks -----


@dataclass
class TextBlock:
    text: str = ""
    # Opaque per-provider state that must round-trip back to the model in
    # later turns. Gemini 3+ attaches `thought_signature` here. Other
    # providers ignore it.
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    type: Literal["text"] = "text"


@dataclass
class ToolUseBlock:
    """A request from the model to invoke a tool.

    `name` is the namespaced tool name as exposed by MCPManager
    (e.g. "my-toolbox__list-tables").

    `provider_metadata` carries opaque per-provider state that must round-trip
    back to the model on subsequent turns. Gemini 3+ requires the original
    `thought_signature` to be echoed back on function_call parts in
    conversation history. Other providers ignore this field.
    """

    id: str
    name: str
    input: dict[str, Any]
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    # Set by a provider when the model's tool-call arguments were
    # unparseable; the loop turns this into a teaching is_error result.
    parse_error: str | None = None
    type: Literal["tool_use"] = "tool_use"


@dataclass
class ToolResultBlock:
    """The result of a tool call, sent back to the model in the next turn.

    `name` is the function name from the matching ToolUseBlock. Anthropic
    doesn't strictly need it (the id is enough), but Gemini's
    function_response requires the name. The agent loop populates both
    when constructing this from a tool call.
    """

    tool_use_id: str
    name: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock]


@dataclass
class CompletionUsage:
    """Token usage for a single LLM completion. Provider-agnostic.

    Each LLMClient maps its SDK's figures onto these fields; a provider that
    doesn't report a given field leaves it 0. `thinking_tokens` and
    `cached_tokens` are surfaced separately for visibility — they are already
    counted inside `total_tokens` where the provider includes them, so don't
    re-add them. `total_tokens` is the authoritative billed figure.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: "CompletionUsage") -> "CompletionUsage":
        return CompletionUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


def coerce_usage_count(value: Any) -> int:
    """Coerce provider token metadata without letting malformed values escape."""
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError, OverflowError):
        return 0


@dataclass(frozen=True)
class ModelProfile:
    """Declared model-interface capabilities, resolved from a models.yaml row."""

    supports_native_tools: bool = True
    thinking: str = "none"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    @classmethod
    def default(cls) -> "ModelProfile":
        return cls()


# ----- messages -----


@dataclass
class Message:
    """One turn in a conversation.

    User messages typically have a single TextBlock.
    Assistant messages can mix TextBlock and ToolUseBlock.
    Tool messages carry one or more ToolResultBlocks.
    """

    role: Role
    content: list[ContentBlock] = field(default_factory=list)

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role=Role.USER, content=[TextBlock(text=text)])

    @classmethod
    def assistant(cls, blocks: list[ContentBlock]) -> "Message":
        return cls(role=Role.ASSISTANT, content=list(blocks))

    @classmethod
    def tool_results(cls, results: list[ToolResultBlock]) -> "Message":
        # We use Role.TOOL internally; provider clients translate to whatever
        # role the API expects (Gemini uses "user" for function responses,
        # Anthropic uses "user" with tool_result blocks, etc.).
        return cls(role=Role.TOOL, content=list(results))


@dataclass
class AssistantMessage:
    """The structured response from an LLM completion.

    `stop_reason` is the provider-neutral terminal classification used by the
    loop. `raw_stop_reason` retains an optional provider-native value for
    diagnostics and later boundary mapping. Tool blocks remain authoritative
    when deciding whether dispatch is required.
    """

    content: list[ContentBlock]
    stop_reason: CanonicalStopReason | None = None
    model: str | None = None
    usage: CompletionUsage | None = None
    # Provider-extracted reasoning; never replayed to the model. Streaming
    # adapters may expose it through an explicitly configured reasoning channel.
    reasoning: str | None = None
    raw_stop_reason: str | None = None

    def text_blocks(self) -> list[TextBlock]:
        return [b for b in self.content if isinstance(b, TextBlock)]

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    def to_message(self) -> Message:
        return Message.assistant(self.content)


@dataclass
class TextDelta:
    """A fragment of assistant text produced mid-generation."""

    text: str
    type: Literal["text_delta"] = "text_delta"


@dataclass
class ReasoningDelta:
    """A sanitized reasoning fragment produced mid-generation."""

    text: str
    type: Literal["reasoning_delta"] = "reasoning_delta"


@dataclass
class StreamEnd:
    """Terminal stream chunk carrying the fully assembled assistant turn."""

    message: AssistantMessage
    type: Literal["stream_end"] = "stream_end"


StreamChunk = Union[TextDelta, ReasoningDelta, StreamEnd]
