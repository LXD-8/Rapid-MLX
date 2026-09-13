# SPDX-License-Identifier: Apache-2.0
"""Deterministic, side-effect-free reducer for bounded agent runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from .models import (
    AgentModelTurn,
    AgentProfile,
    AgentRun,
    AgentRunStatus,
    AgentToolCall,
    AgentToolResult,
    RedactedPendingCall,
    ToolSpec,
)
from .profiles import resolve_agent_profile


class AgentRuntimeError(ValueError):
    """A caller attempted an invalid state transition."""


@dataclass(frozen=True)
class AgentRuntimeOutput:
    """Transient adapter work that must never be serialized with a run."""

    call: AgentToolCall | None = None
    observation: AgentToolResult | None = None


def _call_fingerprint(call: AgentToolCall) -> str:
    payload = json.dumps(
        {"name": call.name, "arguments": call.arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _persisted_call_data(call: AgentToolCall) -> dict[str, JsonValue]:
    """Return enough call metadata for audit without retaining payload values."""

    return {
        "id": call.id,
        "name": call.name,
        "argument_names": sorted(call.arguments),
    }


class AgentRuntime:
    """Own progress, budgets, approvals, and the shared event contract.

    The class deliberately does not call a model or execute a tool. A server
    adapter can drive it around Rapid's in-process engine and MCP executor;
    Desktop can execute its existing built-in tools and post the result back.
    Both surfaces therefore receive identical loop and safety behavior.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        import time

        self._clock = clock or time.time
        # Key by live object identity, not caller-provided run ID. The strong
        # reference prevents id() reuse until terminal cleanup.
        self._call_counts_by_run: dict[int, tuple[AgentRun, dict[str, int]]] = {}

    def create_run(
        self,
        *,
        model: str,
        goal: str,
        run_id: str | None = None,
        profile: AgentProfile | None = None,
    ) -> AgentRun:
        required = resolve_agent_profile(model)
        selected = profile or required
        if (
            selected.max_visible_tools > required.max_visible_tools
            or selected.max_tool_rounds > required.max_tool_rounds
            or selected.repeated_call_limit > required.repeated_call_limit
            or (
                required.attach_ledger_to_tool_results
                and not selected.attach_ledger_to_tool_results
            )
        ):
            raise AgentRuntimeError(
                f"profile {selected.name!r} weakens required limits for model {model!r}"
            )
        run = (
            AgentRun(model=model, goal=goal, profile=selected)
            if run_id is None
            else AgentRun(id=run_id, model=model, goal=goal, profile=selected)
        )
        run.append_event(
            "run.created",
            {"model": model, "profile": selected.name},
            now=self._clock(),
        )
        self._call_counts_by_run[id(run)] = (run, {})
        return run

    def request_model(
        self,
        run: AgentRun,
        tools: Sequence[ToolSpec],
    ) -> list[ToolSpec]:
        selected = run.profile
        self._require_status(run, AgentRunStatus.READY)
        self._require_no_pending_call(run)

        restored_without_repeat_history = (
            id(run) not in self._call_counts_by_run and run.tool_rounds > 0
        )
        if restored_without_repeat_history and not run.final_synthesis:
            object.__setattr__(run, "final_synthesis", True)
            run.append_event(
                "synthesis.required",
                {"reason": "repeat_history_unavailable_after_restore"},
                now=self._clock(),
            )
        if id(run) not in self._call_counts_by_run:
            self._call_counts_by_run[id(run)] = (run, {})

        already_final = run.final_synthesis
        final_synthesis = already_final or run.tool_rounds >= selected.max_tool_rounds
        # Snapshot the exact schema and risk used for this turn. The caller may
        # rebuild or mutate its registry while a model request is in flight;
        # approval must still use the policy the model actually saw.
        visible = (
            [] if final_synthesis else [tool.model_copy(deep=True) for tool in tools]
        )
        if len(visible) > selected.max_visible_tools:
            raise AgentRuntimeError(
                f"profile {selected.name!r} permits at most "
                f"{selected.max_visible_tools} visible tools; got {len(visible)}"
            )
        names = [tool.name for tool in visible]
        if len(names) != len(set(names)):
            raise AgentRuntimeError("visible tool names must be unique")

        object.__setattr__(run, "status", AgentRunStatus.AWAITING_MODEL)
        object.__setattr__(run, "model_turns", run.model_turns + 1)
        object.__setattr__(run, "final_synthesis", final_synthesis)
        object.__setattr__(run, "visible_tools", tuple(visible))
        if final_synthesis and not already_final:
            run.append_event(
                "synthesis.required",
                {"reason": "tool_round_budget_exhausted"},
                now=self._clock(),
            )
        run.append_event(
            "model.requested",
            {
                "model_turn": run.model_turns,
                "visible_tools": names,
                "tools": [tool.model_dump(mode="json") for tool in visible],
                "final_synthesis": final_synthesis,
            },
            now=self._clock(),
        )
        return [tool.model_copy(deep=True) for tool in visible]

    def accept_model_turn(
        self,
        run: AgentRun,
        turn: AgentModelTurn,
    ) -> AgentRuntimeOutput | None:
        """Reduce a model turn and return transient adapter work.

        Accepted calls are returned with their raw arguments for live execution;
        only redacted identity enters the run. Safety transitions such as a
        repeated-call block return an observation the adapter must append to
        model history before requesting synthesis.
        """
        selected = run.profile
        self._require_status(run, AgentRunStatus.AWAITING_MODEL)

        if not turn.tool_calls:
            content = turn.content.strip()
            if not content:
                self._fail(run, "empty_model_turn")
                return None
            object.__setattr__(run, "status", AgentRunStatus.COMPLETED)
            object.__setattr__(run, "final_content", content)
            object.__setattr__(run, "visible_tools", ())
            run.append_event("run.completed", {"content": content}, now=self._clock())
            self._call_counts_by_run.pop(id(run), None)
            return None

        if run.final_synthesis or not run.visible_tools:
            self._fail(run, "tool_call_during_final_synthesis")
            return None
        if len(turn.tool_calls) > 1:
            self._fail(run, "parallel_tool_call_limit_exceeded")
            return None

        call = turn.tool_calls[0].model_copy(deep=True)
        by_name = {tool.name: tool for tool in run.visible_tools}
        if call.name not in by_name:
            self._fail(run, "unadvertised_tool_call")
            return None
        if call.id in run.used_call_ids:
            self._fail(run, "reused_tool_call_id")
            return None
        object.__setattr__(run, "used_call_ids", (*run.used_call_ids, call.id))

        fingerprint = _call_fingerprint(call)
        entry = self._call_counts_by_run.get(id(run))
        if entry is None or entry[0] is not run:
            entry = (run, {})
            self._call_counts_by_run[id(run)] = entry
        counts = entry[1]
        count = counts.get(fingerprint, 0) + 1
        counts[fingerprint] = count
        if count > selected.repeated_call_limit:
            risk = by_name[call.name].risk
            object.__setattr__(run, "tool_rounds", run.tool_rounds + 1)
            run.append_event(
                "tool.requested",
                {"call": _persisted_call_data(call), "risk": risk.value},
                now=self._clock(),
            )
            blocked = AgentToolResult(
                call_id=call.id,
                content=(
                    "This identical tool call was blocked because it repeated "
                    "without making progress. Answer using the available results."
                ),
                is_error=True,
                executed=False,
                safe_summary="Repeated tool call blocked; final synthesis required.",
            )
            run.append_event(
                "tool.completed",
                self._tool_result_event_data(run, blocked),
                now=self._clock(),
            )
            object.__setattr__(run, "status", AgentRunStatus.READY)
            object.__setattr__(run, "final_synthesis", True)
            object.__setattr__(run, "visible_tools", ())
            run.append_event(
                "synthesis.required",
                {"reason": "repeated_tool_call", "tool": call.name},
                now=self._clock(),
            )
            return AgentRuntimeOutput(observation=blocked)

        risk = by_name[call.name].risk
        # Argument values remain in the adapter-owned model turn. Persist only
        # identity: approval/result correlation needs no payload values.
        object.__setattr__(
            run,
            "pending_call",
            RedactedPendingCall(id=call.id, name=call.name),
        )
        object.__setattr__(run, "pending_risk", risk)
        object.__setattr__(run, "tool_rounds", run.tool_rounds + 1)
        run.append_event(
            "tool.requested",
            {"call": _persisted_call_data(call), "risk": risk.value},
            now=self._clock(),
        )
        if risk.requires_approval:
            object.__setattr__(run, "status", AgentRunStatus.AWAITING_APPROVAL)
            run.append_event(
                "approval.required",
                {"call_id": call.id, "tool": call.name, "risk": risk.value},
                now=self._clock(),
            )
        else:
            object.__setattr__(run, "status", AgentRunStatus.AWAITING_TOOL_RESULT)
        return AgentRuntimeOutput(call=call)

    def resolve_approval(
        self,
        run: AgentRun,
        *,
        call_id: str,
        approved: bool,
    ) -> AgentRuntimeOutput | None:
        """Resolve exactly one pending approval and return a denial observation."""

        if type(approved) is not bool:
            raise AgentRuntimeError("approved must be a boolean")
        self._require_status(run, AgentRunStatus.AWAITING_APPROVAL)
        call = self._pending_call(run)
        if call_id != call.id:
            raise AgentRuntimeError(
                f"approval {call_id!r} does not match pending call {call.id!r}"
            )
        run.append_event(
            "approval.resolved",
            {"call_id": call.id, "approved": approved},
            now=self._clock(),
        )
        if approved:
            object.__setattr__(run, "status", AgentRunStatus.AWAITING_TOOL_RESULT)
            return None
        object.__setattr__(run, "status", AgentRunStatus.AWAITING_TOOL_RESULT)
        denied = AgentToolResult(
            call_id=call.id,
            content="The user denied this tool call.",
            is_error=True,
            executed=False,
            safe_summary="User denied the tool call.",
        )
        self._complete_tool_result(run, denied)
        return AgentRuntimeOutput(observation=denied)

    def accept_tool_result(self, run: AgentRun, result: AgentToolResult) -> None:
        self._require_status(run, AgentRunStatus.AWAITING_TOOL_RESULT)
        call = self._pending_call(run)
        if result.call_id != call.id:
            raise AgentRuntimeError(
                f"tool result {result.call_id!r} does not match pending call {call.id!r}"
            )
        self._complete_tool_result(run, result)

    def _complete_tool_result(
        self,
        run: AgentRun,
        result: AgentToolResult,
    ) -> None:
        run.append_event(
            "tool.completed",
            self._tool_result_event_data(run, result),
            now=self._clock(),
        )
        object.__setattr__(run, "pending_call", None)
        object.__setattr__(run, "pending_risk", None)
        object.__setattr__(run, "visible_tools", ())
        object.__setattr__(run, "status", AgentRunStatus.READY)

    def cancel(self, run: AgentRun) -> None:
        if run.status in {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }:
            return
        object.__setattr__(run, "status", AgentRunStatus.CANCELLED)
        object.__setattr__(run, "pending_call", None)
        object.__setattr__(run, "pending_risk", None)
        object.__setattr__(run, "visible_tools", ())
        run.append_event("run.cancelled", now=self._clock())
        self._call_counts_by_run.pop(id(run), None)

    @staticmethod
    def ledger_context(run: AgentRun) -> str:
        """Small host-authored state block safe to attach to a tool result."""

        return (
            f"Goal: {run.goal}\n"
            f"Progress: {run.tool_rounds} tool round(s), "
            f"{run.model_turns} model turn(s).\n"
            "Choose only the next necessary action, or answer the user if complete."
        )

    def _fail(self, run: AgentRun, code: str) -> None:
        object.__setattr__(run, "status", AgentRunStatus.FAILED)
        object.__setattr__(run, "failure_code", code)
        object.__setattr__(run, "pending_call", None)
        object.__setattr__(run, "pending_risk", None)
        object.__setattr__(run, "visible_tools", ())
        run.append_event("run.failed", {"code": code}, now=self._clock())
        self._call_counts_by_run.pop(id(run), None)

    @staticmethod
    def _tool_result_event_data(
        run: AgentRun, result: AgentToolResult
    ) -> dict[str, JsonValue]:
        encoded = result.content.encode()
        persisted_result: dict[str, JsonValue] = {
            "call_id": result.call_id,
            "is_error": result.is_error,
            "executed": result.executed,
            "content_bytes": len(encoded),
            "content_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        if result.safe_summary is not None:
            persisted_result["safe_summary"] = result.safe_summary
        data: dict[str, JsonValue] = {"result": persisted_result}
        if run.profile.attach_ledger_to_tool_results:
            data["ledger"] = AgentRuntime.ledger_context(run)
        return data

    @staticmethod
    def _pending_call(run: AgentRun) -> RedactedPendingCall:
        if run.pending_call is None:
            raise AgentRuntimeError("run has no pending tool call")
        return run.pending_call

    @staticmethod
    def _require_no_pending_call(run: AgentRun) -> None:
        if run.pending_call is not None:
            raise AgentRuntimeError("run still has a pending tool call")

    @staticmethod
    def _require_status(run: AgentRun, expected: AgentRunStatus) -> None:
        if not run.events:
            raise AgentRuntimeError("run was not created by AgentRuntime")
        try:
            run._validate_restored_state()
        except (KeyError, TypeError, ValueError) as exc:
            raise AgentRuntimeError(f"run state is inconsistent: {exc}") from exc
        if run.status is not expected:
            raise AgentRuntimeError(
                f"run status must be {expected.value!r}; got {run.status.value!r}"
            )
