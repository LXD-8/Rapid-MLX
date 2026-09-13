# SPDX-License-Identifier: Apache-2.0
"""Versioned state and event contracts for the Rapid Agent Runtime."""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_serializer,
    model_validator,
)

AgentEventType = Literal[
    "run.created",
    "model.requested",
    "tool.requested",
    "approval.required",
    "approval.resolved",
    "tool.completed",
    "synthesis.required",
    "run.completed",
    "run.failed",
    "run.cancelled",
]


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AgentRunStatus(str, Enum):
    READY = "ready"
    AWAITING_MODEL = "awaiting_model"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_TOOL_RESULT = "awaiting_tool_result"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolRisk(str, Enum):
    READ_ONLY = "read_only"
    LOCAL_CHANGE = "local_change"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"

    @property
    def requires_approval(self) -> bool:
        return self is ToolRisk.EXTERNAL_SIDE_EFFECT


class AgentProfile(_WireModel):
    """Immutable model-specific runtime limits persisted with every run."""

    model_config = ConfigDict(frozen=True)

    name: StrictStr = Field(min_length=1, max_length=128)
    max_visible_tools: StrictInt = Field(ge=0, le=64)
    max_tool_rounds: StrictInt = Field(ge=0, le=128)
    repeated_call_limit: StrictInt = Field(ge=0, le=16)
    attach_ledger_to_tool_results: StrictBool = True


class ToolSpec(_WireModel):
    """The small policy surface the runtime needs from any tool registry."""

    model_config = ConfigDict(frozen=True)

    name: StrictStr = Field(min_length=1, max_length=128)
    description: StrictStr = Field(default="", max_length=4096)
    parameters_json: StrictStr = Field(default="{}", repr=False)
    risk: ToolRisk = ToolRisk.READ_ONLY

    @field_validator("parameters_json")
    @classmethod
    def validate_parameters_json(cls, value: str) -> str:
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("tool parameters must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("tool parameters must be a JSON object")
        return json.dumps(
            decoded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def parameters(self) -> dict[str, JsonValue]:
        """Return a detached schema so the advertised snapshot stays immutable."""

        return cast(dict[str, JsonValue], json.loads(self.parameters_json))

    @model_validator(mode="before")
    @classmethod
    def accept_wire_parameters(cls, value):
        if isinstance(value, dict) and "parameters" in value:
            normalized = dict(value)
            if "parameters_json" in normalized:
                raise ValueError(
                    "tool spec must not contain both parameters and parameters_json"
                )
            parameters = normalized.pop("parameters")
            normalized["parameters_json"] = json.dumps(
                parameters,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            return normalized
        return value

    @model_serializer(mode="plain")
    def serialize_wire(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "risk": self.risk.value,
        }


class AgentToolCall(_WireModel):
    model_config = ConfigDict(frozen=True)

    id: StrictStr = Field(min_length=1, max_length=256)
    name: StrictStr = Field(min_length=1, max_length=128)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class AgentToolResult(_WireModel):
    call_id: StrictStr = Field(min_length=1, max_length=256)
    content: StrictStr = Field(max_length=262_144)
    is_error: StrictBool = False
    safe_summary: StrictStr | None = Field(default=None, max_length=1024)


class AgentModelTurn(_WireModel):
    content: StrictStr = Field(default="", max_length=262_144)
    tool_calls: list[AgentToolCall] = Field(default_factory=list)


class AgentEvent(_WireModel):
    """Append-only event envelope consumed by both HTTP and Desktop clients."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    sequence: StrictInt = Field(ge=1)
    type: AgentEventType
    created_at: float = Field(ge=0, allow_inf_nan=False)
    payload_json: StrictStr = Field(default="{}", repr=False)

    @field_validator("payload_json")
    @classmethod
    def validate_payload_json(cls, value: str) -> str:
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("event payload must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("event payload must be a JSON object")
        return json.dumps(
            decoded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def data(self) -> dict[str, JsonValue]:
        """Return a detached payload so consumers cannot rewrite history."""

        return cast(dict[str, JsonValue], json.loads(self.payload_json))

    @model_validator(mode="before")
    @classmethod
    def accept_wire_data(cls, value):
        if isinstance(value, dict) and "data" in value:
            normalized = dict(value)
            if "payload_json" in normalized:
                raise ValueError("event must not contain both data and payload_json")
            data = normalized.pop("data")
            normalized["payload_json"] = json.dumps(
                data,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            return normalized
        return value

    @model_serializer(mode="plain")
    def serialize_wire(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "type": self.type,
            "created_at": self.created_at,
            "data": self.data,
        }


class AgentRun(_WireModel):
    """Serializable reducer state; no model reasoning or credentials are stored."""

    model_config = ConfigDict(validate_assignment=True)

    schema_version: Literal[1] = Field(default=1, frozen=True)
    id: StrictStr = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=1,
        max_length=256,
        frozen=True,
    )
    model: StrictStr = Field(min_length=1, frozen=True)
    goal: StrictStr = Field(min_length=1, max_length=65_536, frozen=True)
    profile: AgentProfile = Field(frozen=True)
    status: AgentRunStatus = AgentRunStatus.READY
    model_turns: StrictInt = Field(default=0, ge=0)
    tool_rounds: StrictInt = Field(default=0, ge=0)
    final_synthesis: StrictBool = False
    visible_tools: tuple[ToolSpec, ...] = Field(default_factory=tuple, frozen=True)
    pending_call: AgentToolCall | None = Field(default=None, frozen=True)
    pending_risk: ToolRisk | None = Field(default=None, frozen=True)
    call_counts: tuple[tuple[StrictStr, StrictInt], ...] = Field(
        default_factory=tuple,
        frozen=True,
    )
    used_call_ids: tuple[StrictStr, ...] = Field(default_factory=tuple, frozen=True)
    final_content: StrictStr | None = Field(default=None, max_length=262_144)
    failure_code: StrictStr | None = Field(default=None, max_length=128)
    events: tuple[AgentEvent, ...] = Field(default_factory=tuple, frozen=True)

    @model_validator(mode="after")
    def validate_event_history(self) -> AgentRun:
        expected = list(range(1, len(self.events) + 1))
        actual = [event.sequence for event in self.events]
        if actual != expected:
            raise ValueError("event sequences must be contiguous and start at 1")
        if len(self.used_call_ids) != len(set(self.used_call_ids)):
            raise ValueError("used tool call IDs must be unique")
        call_count_names = [fingerprint for fingerprint, _ in self.call_counts]
        if len(call_count_names) != len(set(call_count_names)):
            raise ValueError("tool call fingerprints must be unique")
        return self

    def append_event(
        self,
        event_type: AgentEventType,
        data: dict[str, JsonValue] | None = None,
        *,
        now: float | None = None,
    ) -> AgentEvent:
        event = AgentEvent(
            sequence=len(self.events) + 1,
            type=event_type,
            created_at=time.time() if now is None else now,
            payload_json=json.dumps(
                data or {},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
        object.__setattr__(self, "events", (*self.events, event))
        return event
