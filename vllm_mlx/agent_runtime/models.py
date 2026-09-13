# SPDX-License-Identifier: Apache-2.0
"""Versioned state and event contracts for the Rapid Agent Runtime."""

from __future__ import annotations

import json
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
    risk: ToolRisk

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


class RedactedPendingCall(_WireModel):
    """Persistent call identity; raw argument values are adapter-owned."""

    model_config = ConfigDict(frozen=True)

    id: StrictStr = Field(min_length=1, max_length=256)
    name: StrictStr = Field(min_length=1, max_length=128)


class AgentToolResult(_WireModel):
    call_id: StrictStr = Field(min_length=1, max_length=256)
    content: StrictStr = Field(max_length=262_144)
    is_error: StrictBool = False
    executed: StrictBool = True
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
    pending_call: RedactedPendingCall | None = Field(default=None, frozen=True)
    pending_risk: ToolRisk | None = Field(default=None, frozen=True)
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
        if self.events:
            self._validate_restored_state()
        return self

    def _validate_restored_state(self) -> None:
        from .profiles import resolve_agent_profile

        exact_event_keys = {
            "run.created": {"model", "profile"},
            "model.requested": {
                "model_turn",
                "visible_tools",
                "tools",
                "final_synthesis",
            },
            "tool.requested": {"call", "risk"},
            "approval.required": {"call_id", "tool", "risk"},
            "approval.resolved": {"call_id", "approved"},
            "run.completed": {"content"},
            "run.failed": {"code"},
            "run.cancelled": set(),
        }
        for event in self.events:
            data = event.data
            expected = exact_event_keys.get(event.type)
            if expected is not None and set(data) != expected:
                raise ValueError(f"{event.type} contains unexpected payload fields")
            if event.type == "synthesis.required" and (
                "reason" not in data or not set(data).issubset({"reason", "tool"})
            ):
                raise ValueError("synthesis.required contains invalid payload fields")
            if event.type == "tool.completed":
                if not set(data).issubset({"result", "ledger"}) or "result" not in data:
                    raise ValueError("tool.completed contains invalid payload fields")
                result = data["result"]
                if not isinstance(result, dict):
                    raise ValueError("tool.completed contains invalid result fields")
                result_keys = set(result)
                base_result_keys = {
                    "call_id",
                    "is_error",
                    "executed",
                    "content_bytes",
                }
                if result_keys != base_result_keys and result_keys != (
                    base_result_keys | {"safe_summary"}
                ):
                    raise ValueError("tool.completed contains invalid result fields")

        created = self.events[0]
        if created.type != "run.created":
            raise ValueError("event history must start with run.created")
        created_data = created.data
        if created_data.get("model") != self.model:
            raise ValueError("run model does not match run.created")
        if created_data.get("profile") != self.profile.model_dump(mode="json"):
            raise ValueError("run profile does not match run.created")

        required = resolve_agent_profile(self.model)
        if (
            self.profile.max_visible_tools > required.max_visible_tools
            or self.profile.max_tool_rounds > required.max_tool_rounds
            or self.profile.repeated_call_limit > required.repeated_call_limit
            or (
                required.attach_ledger_to_tool_results
                and not self.profile.attach_ledger_to_tool_results
            )
        ):
            raise ValueError("stored profile weakens required model limits")

        model_requests = [
            event for event in self.events if event.type == "model.requested"
        ]
        tool_requests = [
            event for event in self.events if event.type == "tool.requested"
        ]
        requested_ids_list: list[str] = []
        request_risks: dict[str, ToolRisk] = {}
        approval_required: set[str] = set()
        approval_resolved: dict[str, bool] = {}
        completed_ids: set[str] = set()
        for event in self.events:
            data = event.data
            if event.type == "tool.requested":
                call_data = data.get("call")
                if not isinstance(call_data, dict):
                    raise ValueError("tool.requested must contain a call object")
                call_id = call_data.get("id")
                call_name = call_data.get("name")
                argument_names = call_data.get("argument_names")
                if (
                    set(call_data) != {"id", "name", "argument_names"}
                    or not isinstance(call_id, str)
                    or not isinstance(call_name, str)
                    or not isinstance(argument_names, list)
                    or not all(isinstance(name, str) for name in argument_names)
                ):
                    raise ValueError("tool.requested contains malformed call metadata")
                try:
                    risk = ToolRisk(data.get("risk"))
                except ValueError as exc:
                    raise ValueError("tool.requested contains an invalid risk") from exc
                if call_id in request_risks:
                    raise ValueError("tool call IDs must be unique")
                requested_ids_list.append(call_id)
                request_risks[call_id] = risk
            elif event.type == "approval.required":
                call_id = data.get("call_id")
                if (
                    not isinstance(call_id, str)
                    or request_risks.get(call_id) is not ToolRisk.EXTERNAL_SIDE_EFFECT
                    or call_id in approval_required
                ):
                    raise ValueError(
                        "approval.required does not match an external call"
                    )
                approval_required.add(call_id)
            elif event.type == "approval.resolved":
                call_id = data.get("call_id")
                approved = data.get("approved")
                if (
                    not isinstance(call_id, str)
                    or type(approved) is not bool
                    or call_id not in approval_required
                    or call_id in approval_resolved
                ):
                    raise ValueError(
                        "approval.resolved does not match a pending approval"
                    )
                approval_resolved[call_id] = approved
            elif event.type == "tool.completed":
                result_data = data.get("result")
                if not isinstance(result_data, dict):
                    raise ValueError("tool.completed must contain result metadata")
                call_id = result_data.get("call_id")
                executed = result_data.get("executed")
                if (
                    not isinstance(call_id, str)
                    or type(executed) is not bool
                    or call_id not in request_risks
                    or call_id in completed_ids
                ):
                    raise ValueError("tool.completed does not match one requested call")
                completed_ids.add(call_id)
                if (
                    executed
                    and request_risks[call_id] is ToolRisk.EXTERNAL_SIDE_EFFECT
                    and approval_resolved.get(call_id) is not True
                ):
                    raise ValueError(
                        "executed external call requires matching approval"
                    )
        requested_ids = tuple(requested_ids_list)
        if self.model_turns != len(model_requests):
            raise ValueError("model_turns does not match event history")
        if self.tool_rounds != len(tool_requests):
            raise ValueError("tool_rounds does not match event history")
        if self.tool_rounds > self.profile.max_tool_rounds:
            raise ValueError("tool_rounds exceeds the stored profile limit")
        if self.used_call_ids != requested_ids:
            raise ValueError("used_call_ids does not match event history")
        if self.final_synthesis != any(
            event.type == "synthesis.required" for event in self.events
        ):
            raise ValueError("final_synthesis does not match event history")

        active_policy = self.status in {
            AgentRunStatus.AWAITING_MODEL,
            AgentRunStatus.AWAITING_APPROVAL,
            AgentRunStatus.AWAITING_TOOL_RESULT,
        }
        if active_policy:
            if not model_requests:
                raise ValueError("active run requires a model request")
            expected_tools = model_requests[-1].data.get("tools")
            actual_tools = [tool.model_dump(mode="json") for tool in self.visible_tools]
            if expected_tools != actual_tools:
                raise ValueError("visible tool policy does not match event history")
            if len(self.visible_tools) > self.profile.max_visible_tools:
                raise ValueError("visible tool policy exceeds the profile limit")
        elif self.visible_tools:
            raise ValueError("inactive run cannot retain a visible tool policy")

        pending_status = self.status in {
            AgentRunStatus.AWAITING_APPROVAL,
            AgentRunStatus.AWAITING_TOOL_RESULT,
        }
        if pending_status:
            if (
                self.pending_call is None
                or self.pending_risk is None
                or not tool_requests
            ):
                raise ValueError("pending status requires a pending call and risk")
            latest = tool_requests[-1].data
            latest_call = latest.get("call")
            if not isinstance(latest_call, dict):
                raise ValueError("latest tool request has malformed call metadata")
            if latest_call.get("id") != self.pending_call.id:
                raise ValueError("pending call does not match latest tool request")
            if latest_call.get("name") != self.pending_call.name:
                raise ValueError("pending tool does not match latest tool request")
            if latest.get("risk") != self.pending_risk.value:
                raise ValueError("pending risk does not match latest tool request")
            if self.pending_call.id in completed_ids:
                raise ValueError("pending call is already completed")
            if (
                self.status is AgentRunStatus.AWAITING_TOOL_RESULT
                and self.pending_risk.requires_approval
            ):
                resolution = self.events[-1]
                resolution_data = resolution.data
                if (
                    resolution.type != "approval.resolved"
                    or resolution_data.get("call_id") != self.pending_call.id
                    or resolution_data.get("approved") is not True
                ):
                    raise ValueError(
                        "external side effect requires matching approved resolution"
                    )
        elif self.pending_call is not None or self.pending_risk is not None:
            raise ValueError("non-pending status cannot retain a pending call")

        expected_last = {
            AgentRunStatus.AWAITING_MODEL: "model.requested",
            AgentRunStatus.AWAITING_APPROVAL: "approval.required",
            AgentRunStatus.COMPLETED: "run.completed",
            AgentRunStatus.FAILED: "run.failed",
            AgentRunStatus.CANCELLED: "run.cancelled",
        }.get(self.status)
        if expected_last is not None and self.events[-1].type != expected_last:
            raise ValueError(f"{self.status.value} run has inconsistent final event")
        if self.status is AgentRunStatus.AWAITING_TOOL_RESULT and self.events[
            -1
        ].type not in {
            "tool.requested",
            "approval.resolved",
        }:
            raise ValueError("awaiting_tool_result run has inconsistent final event")
        if self.status is AgentRunStatus.READY and self.events[-1].type not in {
            "run.created",
            "tool.completed",
            "synthesis.required",
        }:
            raise ValueError("ready run has inconsistent final event")

        if self.status is AgentRunStatus.COMPLETED:
            if (
                self.final_content is None
                or self.events[-1].data.get("content") != self.final_content
            ):
                raise ValueError("completed run must retain matching final content")
        elif self.final_content is not None:
            raise ValueError("non-completed run cannot retain final content")
        if self.status is AgentRunStatus.FAILED:
            if (
                self.failure_code is None
                or self.events[-1].data.get("code") != self.failure_code
            ):
                raise ValueError("failed run must retain matching failure code")
        elif self.failure_code is not None:
            raise ValueError("non-failed run cannot retain a failure code")
