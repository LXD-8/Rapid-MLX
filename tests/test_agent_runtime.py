# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from vllm_mlx.agent_runtime import (
    AgentModelTurn,
    AgentProfile,
    AgentRun,
    AgentRunStatus,
    AgentRuntime,
    AgentRuntimeError,
    AgentToolCall,
    AgentToolResult,
    ToolRisk,
    ToolSpec,
    resolve_agent_profile,
)


def _clock():
    value = 100.0
    while True:
        yield value
        value += 1


def _runtime() -> AgentRuntime:
    ticks = _clock()
    return AgentRuntime(clock=lambda: next(ticks))


READ = ToolSpec(name="read_file", risk=ToolRisk.READ_ONLY)
SEND = ToolSpec(name="send_message", risk=ToolRisk.EXTERNAL_SIDE_EFFECT)


def _call(call_id: str = "call-1", **arguments) -> AgentToolCall:
    return AgentToolCall(id=call_id, name="read_file", arguments=arguments)


def test_minicpm_profile_is_alias_and_repo_aware():
    for model in (
        "minicpm5-2b-4bit",
        "openbmb/MiniCPM5-2B-MLX",
        "mlx-community/MiniCPM5_2B_8bit",
    ):
        profile = resolve_agent_profile(model)
        assert profile.name == "minicpm5-2b"
        assert profile.max_visible_tools == 6
        assert profile.max_tool_rounds == 8

    assert resolve_agent_profile("qwen3.5-4b-4bit").name == "default"


def test_run_identity_and_profile_are_immutable_after_creation():
    run = _runtime().create_run(model="minicpm5-2b-4bit", goal="Do the task")

    with pytest.raises(ValidationError, match="Field is frozen"):
        run.profile = resolve_agent_profile("qwen3.5-4b-4bit")
    with pytest.raises(ValidationError, match="Field is frozen"):
        run.model = "different-model"


def test_tool_arguments_are_json_only_at_the_wire_boundary():
    with pytest.raises(ValidationError):
        AgentToolCall(id="call-1", name="read_file", arguments={"bad": object()})
    with pytest.raises(ValidationError):
        AgentToolCall(id="call-1", name="read_file", arguments={"bad": float("nan")})


def test_successful_tool_round_has_stable_events_and_roundtrips():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read the report")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[_call(path="report.md")]),
    )
    assert run.status is AgentRunStatus.AWAITING_TOOL_RESULT

    runtime.accept_tool_result(
        run,
        AgentToolResult(call_id="call-1", content="Revenue fell 12%."),
    )
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(content="Revenue fell 12%."),
    )

    assert run.status is AgentRunStatus.COMPLETED
    assert [event.sequence for event in run.events] == list(
        range(1, len(run.events) + 1)
    )
    assert [event.type for event in run.events] == [
        "run.created",
        "model.requested",
        "tool.requested",
        "tool.completed",
        "model.requested",
        "run.completed",
    ]
    restored = AgentRun.model_validate_json(run.model_dump_json())
    assert restored == run
    assert "Choose only the next necessary action" in run.events[3].data["ledger"]


def test_minicpm_rejects_an_oversized_tool_surface():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Do the task")
    tools = [ToolSpec(name=f"tool_{index}") for index in range(7)]

    with pytest.raises(AgentRuntimeError, match="at most 6 visible tools"):
        runtime.request_model(run, tools)

    assert run.status is AgentRunStatus.READY
    assert run.model_turns == 0


def test_profile_override_cannot_weaken_model_limits():
    runtime = _runtime()
    weakened = AgentProfile(
        name="unsafe",
        max_visible_tools=7,
        max_tool_rounds=8,
        repeated_call_limit=2,
    )

    with pytest.raises(AgentRuntimeError, match="weakens required limits"):
        runtime.create_run(
            model="minicpm5-2b-4bit",
            goal="Do the task",
            profile=weakened,
        )


def test_unadvertised_tool_call_fails_closed():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read the report")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(
            tool_calls=[AgentToolCall(id="call-1", name="exec", arguments={})]
        ),
    )

    assert run.status is AgentRunStatus.FAILED
    assert run.failure_code == "unadvertised_tool_call"
    assert run.events[-1].type == "run.failed"


def test_tool_policy_is_snapshotted_before_the_model_turn():
    runtime = _runtime()
    source = ToolSpec(
        name="send_message",
        parameters={"properties": {"text": {"type": "string"}}},
        risk=ToolRisk.EXTERNAL_SIDE_EFFECT,
    )
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Send the update")

    offered = runtime.request_model(run, [source])
    offered[0].parameters["properties"].clear()
    source.parameters["properties"]["text"]["type"] = "integer"
    runtime.accept_model_turn(
        run,
        AgentModelTurn(
            tool_calls=[
                AgentToolCall(
                    id="send-1",
                    name="send_message",
                    arguments={"text": "Done"},
                )
            ]
        ),
    )

    assert run.status is AgentRunStatus.AWAITING_APPROVAL
    assert run.pending_risk is ToolRisk.EXTERNAL_SIDE_EFFECT
    assert run.visible_tools[0].parameters == {
        "properties": {"text": {"type": "string"}}
    }

    detached = run.visible_tools[0].parameters
    detached["properties"].clear()
    assert run.visible_tools[0].parameters == {
        "properties": {"text": {"type": "string"}}
    }


def test_external_side_effect_pauses_for_approval_and_denial_is_evidence():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Send the update")
    runtime.request_model(run, [SEND])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(
            tool_calls=[
                AgentToolCall(
                    id="send-1",
                    name="send_message",
                    arguments={"to": "Mina", "text": "Done"},
                )
            ]
        ),
    )

    assert run.status is AgentRunStatus.AWAITING_APPROVAL
    denied = runtime.resolve_approval(run, call_id="send-1", approved=False)
    assert run.status is AgentRunStatus.READY
    assert run.pending_call is None
    assert denied is not None
    assert denied.observation is not None
    assert denied.observation.content == "The user denied this tool call."
    assert run.events[-2].type == "approval.resolved"
    assert run.events[-1].data["result"]["is_error"] is True
    assert run.events[-1].data["result"]["safe_summary"] == (
        "User denied the tool call."
    )


def test_stale_approval_does_not_authorize_the_pending_call():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Send the update")
    runtime.request_model(run, [SEND])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[AgentToolCall(id="current", name="send_message")]),
    )

    with pytest.raises(AgentRuntimeError, match="does not match pending call"):
        runtime.resolve_approval(run, call_id="stale", approved=True)

    assert run.status is AgentRunStatus.AWAITING_APPROVAL
    assert run.pending_call is not None
    assert run.pending_call.id == "current"


def test_mismatched_tool_result_does_not_advance_the_run():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read the report")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))

    with pytest.raises(AgentRuntimeError, match="does not match pending call"):
        runtime.accept_tool_result(
            run, AgentToolResult(call_id="different", content="wrong")
        )

    assert run.status is AgentRunStatus.AWAITING_TOOL_RESULT
    assert run.pending_call is not None


def test_tool_result_content_is_transient_not_persisted_in_the_event_log():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read the clipboard")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))

    secret = "clipboard-secret-123"
    runtime.accept_tool_result(
        run,
        AgentToolResult(
            call_id="call-1",
            content=secret,
            safe_summary="Clipboard read completed.",
        ),
    )

    serialized = run.model_dump_json()
    assert secret not in serialized
    assert "Clipboard read completed." in serialized
    assert run.events[-1].data["result"]["content_bytes"] == len(secret)


def test_tool_argument_values_are_transient_not_persisted():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Use a credential ref")
    runtime.request_model(run, [READ])
    secret = "credential-secret-456"
    turn = AgentModelTurn(tool_calls=[_call(path="report.md", token=secret)])

    output = runtime.accept_model_turn(run, turn)

    serialized = run.model_dump_json()
    assert secret not in serialized
    assert run.pending_call is not None
    assert run.pending_call.arguments == {}
    assert output is not None
    assert output.call is not None
    assert output.call.arguments["token"] == secret
    assert run.events[-1].data["call"]["argument_names"] == ["path", "token"]
    # The adapter still owns the transient input needed for immediate execution.
    assert turn.tool_calls[0].arguments["token"] == secret


def test_reused_tool_call_id_fails_closed():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read two files")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))
    runtime.accept_tool_result(run, AgentToolResult(call_id="call-1", content="one"))
    runtime.request_model(run, [READ])

    runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[_call(call_id="call-1", path="other.md")]),
    )

    assert run.status is AgentRunStatus.FAILED
    assert run.failure_code == "reused_tool_call_id"


def test_repeated_identical_call_forces_tools_off_instead_of_looping():
    profile = AgentProfile(
        name="test",
        max_visible_tools=2,
        max_tool_rounds=8,
        repeated_call_limit=1,
    )
    runtime = _runtime()
    run = runtime.create_run(model="test-model", goal="Read once", profile=profile)

    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))
    runtime.accept_tool_result(
        run, AgentToolResult(call_id="call-1", content="contents")
    )
    runtime.request_model(run, [READ])
    blocked = runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[_call(call_id="call-2")]),
    )

    assert run.status is AgentRunStatus.READY
    assert run.final_synthesis is True
    assert run.pending_call is None
    assert run.events[-1].data["reason"] == "repeated_tool_call"
    assert [event.type for event in run.events[-3:]] == [
        "tool.requested",
        "tool.completed",
        "synthesis.required",
    ]
    assert run.events[-2].data["result"]["call_id"] == "call-2"
    assert run.events[-2].data["result"]["is_error"] is True
    assert blocked is not None
    assert blocked.observation is not None
    assert blocked.observation.call_id == "call-2"
    assert "blocked" in blocked.observation.content

    visible = runtime.request_model(run, [READ])
    assert visible == []
    assert run.events[-1].type == "model.requested"
    runtime.accept_model_turn(run, AgentModelTurn(content="Here is the result."))
    assert run.status is AgentRunStatus.COMPLETED


def test_tool_budget_reserves_a_tools_disabled_final_synthesis():
    profile = AgentProfile(
        name="one-round",
        max_visible_tools=1,
        max_tool_rounds=1,
        repeated_call_limit=2,
    )
    runtime = _runtime()
    run = runtime.create_run(model="test", goal="Read", profile=profile)
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))
    runtime.accept_tool_result(run, AgentToolResult(call_id="call-1", content="ok"))

    assert runtime.request_model(run, [READ]) == []
    assert run.final_synthesis is True
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call(call_id="again")]))
    assert run.status is AgentRunStatus.FAILED
    assert run.failure_code == "tool_call_during_final_synthesis"


def test_parallel_calls_fail_instead_of_being_partially_executed():
    runtime = _runtime()
    run = runtime.create_run(model="qwen3.5-4b-4bit", goal="Read two files")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[_call(call_id="one"), _call(call_id="two")]),
    )

    assert run.status is AgentRunStatus.FAILED
    assert run.failure_code == "parallel_tool_call_limit_exceeded"
    assert not any(event.type == "tool.requested" for event in run.events)


def test_cancel_is_idempotent_and_clears_pending_state():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))

    runtime.cancel(run)
    runtime.cancel(run)

    assert run.status is AgentRunStatus.CANCELLED
    assert run.pending_call is None
    assert [event.type for event in run.events].count("run.cancelled") == 1


def test_restored_run_rejects_noncontiguous_event_sequences():
    run = _runtime().create_run(model="minicpm5-2b-4bit", goal="Read")
    payload = run.model_dump(mode="json")
    payload["events"][0]["sequence"] = 5

    with pytest.raises(ValidationError, match="event sequences must be contiguous"):
        AgentRun.model_validate(payload)


def test_event_history_and_payload_are_immutable_to_consumers():
    run = _runtime().create_run(model="minicpm5-2b-4bit", goal="Read")
    event = run.events[0]

    with pytest.raises(ValidationError, match="Field is frozen"):
        run.events = ()
    with pytest.raises(ValidationError, match="Instance is frozen"):
        event.sequence = 9

    detached = event.data
    detached["model"] = "rewritten"
    assert event.data["model"] == "minicpm5-2b-4bit"


def test_reducer_safety_collections_are_immutable_to_consumers():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read")
    runtime.request_model(run, [READ])

    with pytest.raises(ValidationError, match="Field is frozen"):
        run.visible_tools = (*run.visible_tools, SEND)
    with pytest.raises(ValidationError, match="Field is frozen"):
        run.used_call_ids = ()
    with pytest.raises(ValidationError, match="Field is frozen"):
        run.call_counts = ()

    output = runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))
    assert run.pending_risk is ToolRisk.READ_ONLY
    assert output is not None
    assert output.call is not None
    assert output.call.arguments == {}

    with pytest.raises(ValidationError, match="Field is frozen"):
        run.pending_call = AgentToolCall(id="rewritten", name="send_message")
    with pytest.raises(ValidationError, match="Instance is frozen"):
        run.pending_call.id = "rewritten"
