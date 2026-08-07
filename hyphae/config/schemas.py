"""Strict immutable models for file-based harness configuration."""

from __future__ import annotations

from collections.abc import Mapping
from ipaddress import ip_address
from types import MappingProxyType
from typing import Annotated, Literal, Union
from urllib.parse import urlsplit

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from ._validation import (
    reject_bool_or_float_for_int,
    validate_nonblank_bounded,
)

_POLICY_PATTERN_MAX_CHARS = 512
_DESCRIPTION_MAX_CHARS = 20_000
_MODEL_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)
_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)


def _validate_http_url(value: str) -> str:
    if not value.strip():
        raise PydanticCustomError("nonblank_required", "MCP URL must be nonblank")
    if any(character.isspace() or not character.isprintable() for character in value):
        raise PydanticCustomError("url_whitespace", "MCP URL contains whitespace")
    try:
        parsed = _HTTP_URL_ADAPTER.validate_python(value)
        source_hostname = urlsplit(value).hostname
    except (ValueError, ValidationError) as exc:
        raise PydanticCustomError("url_malformed", "MCP URL is malformed") from exc

    if source_hostname is None:
        raise PydanticCustomError("url_host", "MCP URL host is malformed")
    hostname = parsed.host or ""
    ip_hostname = (
        hostname[1:-1]
        if hostname.startswith("[") and hostname.endswith("]")
        else hostname
    )
    try:
        ip_address(ip_hostname)
        return value
    except ValueError:
        pass

    labels = hostname.rstrip(".").split(".")
    if len(hostname) > 253 or any(
        not label or len(label) > 63 or label.startswith("-") or label.endswith("-")
        for label in labels
    ):
        raise PydanticCustomError("url_host", "MCP URL host is malformed")
    return value


class _ImmutableConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _MCPServerBase(_ImmutableConfigModel):
    disabled: bool = False
    disabled_tools: tuple[str, ...] = ()

    @field_validator("disabled_tools")
    @classmethod
    def _validate_disabled_tools(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            validate_nonblank_bounded(value, label="disabled tool identifier")
            for value in values
        )


class StreamableHTTPServer(_MCPServerBase):
    transport: Literal["streamable-http"]
    url: str

    _url_is_http = field_validator("url")(_validate_http_url)


class SSEServer(_MCPServerBase):
    transport: Literal["sse"]
    url: str

    _url_is_http = field_validator("url")(_validate_http_url)


class StdioServer(_MCPServerBase):
    transport: Literal["stdio"]
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = Field(default_factory=dict)

    @field_validator("command")
    @classmethod
    def _command_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise PydanticCustomError(
                "nonblank_required",
                "stdio command must be nonblank",
            )
        return value

    @field_validator("env")
    @classmethod
    def _freeze_env(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("env")
    def _serialize_env(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


MCPServerConfig = Annotated[
    Union[StreamableHTTPServer, SSEServer, StdioServer],
    Field(discriminator="transport"),
]


class ToolPolicyConfig(_ImmutableConfigModel):
    """Dispatch-time policy for what tools may execute."""

    mode: Literal["allow_all", "allow_list"] = "allow_all"
    allow: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_allow_list(self) -> "ToolPolicyConfig":
        if self.mode == "allow_list" and not self.allow:
            raise PydanticCustomError(
                "allow_list_required",
                "tool_policy.mode 'allow_list' requires a non-empty 'allow' list",
            )
        for pattern in self.allow:
            validate_nonblank_bounded(
                pattern,
                label="tool policy pattern",
                max_chars=_POLICY_PATTERN_MAX_CHARS,
            )
        return self


class MCPConfig(_ImmutableConfigModel):
    mcp_servers: Mapping[str, MCPServerConfig] = Field(alias="mcpServers")
    tool_policy: ToolPolicyConfig = Field(default_factory=ToolPolicyConfig)

    @field_validator("mcp_servers")
    @classmethod
    def _freeze_servers(
        cls,
        value: Mapping[str, MCPServerConfig],
    ) -> Mapping[str, MCPServerConfig]:
        return MappingProxyType(dict(value))

    @field_serializer("mcp_servers")
    def _serialize_servers(
        self,
        value: Mapping[str, MCPServerConfig],
    ) -> dict[str, MCPServerConfig]:
        return dict(value)

    @model_validator(mode="after")
    def _validate_names(self) -> "MCPConfig":
        for name in self.mcp_servers:
            validate_nonblank_bounded(name, label="server identifier")
            if "__" in name:
                raise PydanticCustomError(
                    "identifier_separator",
                    "server identifier cannot contain '__'",
                )
            if not name.replace("-", "").replace("_", "").isalnum():
                raise PydanticCustomError(
                    "identifier_characters",
                    "server identifier must be alphanumeric with optional dashes/underscores",
                )
        return self

    def enabled_servers(self) -> Mapping[str, MCPServerConfig]:
        """Return an immutable view of servers not marked disabled."""
        return MappingProxyType(
            {
                name: server
                for name, server in self.mcp_servers.items()
                if not server.disabled
            }
        )


class SamplingParams(_ImmutableConfigModel):
    """Optional per-model sampling parameters passed through to providers."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)

    _strict_integers = field_validator("top_k", mode="before")(
        reject_bool_or_float_for_int
    )


class ModelEntry(_ImmutableConfigModel):
    """One model the orchestrator may route to."""

    provider: str
    model: str
    description: str
    max_tokens: int | None = Field(default=None, gt=0)
    context_window: int | None = Field(default=None, gt=0)
    default: bool = False
    supports_native_tools: bool = True
    thinking: Literal["none", "hint-param", "think-tags"] = "none"
    sampling: SamplingParams | None = None

    _strict_integers = field_validator(
        "max_tokens",
        "context_window",
        mode="before",
    )(reject_bool_or_float_for_int)

    @field_validator("provider")
    @classmethod
    def _provider_is_valid(cls, value: str) -> str:
        return validate_nonblank_bounded(value, label="provider identifier")

    @field_validator("model")
    @classmethod
    def _model_is_valid(cls, value: str) -> str:
        return validate_nonblank_bounded(value, label="model identifier")

    @field_validator("description")
    @classmethod
    def _description_is_valid(cls, value: str) -> str:
        return validate_nonblank_bounded(
            value,
            label="model description",
            max_chars=_DESCRIPTION_MAX_CHARS,
        )

    @model_validator(mode="after")
    def _validate_explicit_bounds(self) -> "ModelEntry":
        if (
            self.max_tokens is not None
            and self.context_window is not None
            and self.max_tokens >= self.context_window
        ):
            raise PydanticCustomError(
                "inconsistent_context_bounds",
                "model max_tokens must be smaller than context_window",
            )
        return self


class ModelsConfig(_ImmutableConfigModel):
    """Typed parse of models.yaml. Maps model_id to immutable entries."""

    models: Mapping[str, ModelEntry] = Field(default_factory=dict)

    @field_validator("models")
    @classmethod
    def _freeze_models(
        cls,
        value: Mapping[str, ModelEntry],
    ) -> Mapping[str, ModelEntry]:
        return MappingProxyType(dict(value))

    @field_serializer("models")
    def _serialize_models(
        self,
        value: Mapping[str, ModelEntry],
    ) -> dict[str, ModelEntry]:
        return dict(value)

    @model_validator(mode="after")
    def _validate(self) -> "ModelsConfig":
        if not self.models:
            raise PydanticCustomError(
                "missing_model",
                "models config must define at least one model",
            )

        defaults = sum(1 for model in self.models.values() if model.default)
        if len(self.models) > 1 and defaults == 0:
            raise PydanticCustomError(
                "missing_default",
                "multiple models require exactly one explicit default",
            )
        if len(self.models) > 1 and defaults > 1:
            raise PydanticCustomError(
                "multiple_defaults",
                "models config has multiple default models",
            )

        for model_id in self.models:
            validate_nonblank_bounded(model_id, label="model identifier")
            if set(model_id) - _MODEL_ID_CHARS:
                raise PydanticCustomError(
                    "identifier_characters",
                    "model identifier contains disallowed characters",
                )
        return self

    def default_id(self) -> str:
        """Return the explicit default or the sole implicit default."""
        for model_id, model in self.models.items():
            if model.default:
                return model_id
        return next(iter(self.models))
