# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from vllm_mlx.agent_runtime import (
    AgentModelTurn,
    AgentRunStatus,
    AgentToolCall,
    AgentToolResult,
    ToolRisk,
    ToolSpec,
    resolve_agent_profile,
)
from vllm_mlx.agent_runtime.server import (
    AgentApprovalRequest,
    AgentRunCapacityError,
    AgentRunConflictError,
    AgentRunCreateRequest,
    AgentRunNotFoundError,
    AgentServerService,
    AgentToolResultRequest,
    AgentToolSelectionError,
    MCPToolRegistry,
    classify_mcp_tool,
)

READ = ToolSpec(
    name="files__read_file",
    description="Read a file",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    risk=ToolRisk.READ_ONLY,
)
SEND = ToolSpec(
    name="mail__send_message",
    description="Send a message",
    parameters={
        "type": "object",
        "properties": {"body": {"type": "string"}},
        "required": ["body"],
        "additionalProperties": False,
    },
    risk=ToolRisk.EXTERNAL_SIDE_EFFECT,
)


class FakeRegistry:
    def __init__(self, tools: Sequence[ToolSpec] = (READ, SEND)) -> None:
        self.tools = list(tools)
        self.calls: list[AgentToolCall] = []

    def list_tools(self) -> Sequence[ToolSpec]:
        return list(self.tools)

    async def execute(self, call: AgentToolCall) -> AgentToolResult:
        self.calls.append(call)
        return AgentToolResult(
            call_id=call.id,
            content=f"result for {call.name}",
            safe_summary="Tool completed.",
        )


class ScriptedDriver:
    def __init__(self, *turns: AgentModelTurn) -> None:
        self.turns = list(turns)
        self.requests: list[tuple[str, list[dict], list[ToolSpec], object]] = []

    async def __call__(self, model, messages, tools, settings):
        self.requests.append((model, messages, list(tools), settings))
        if not self.turns:
            raise AssertionError("unexpected model turn")
        return self.turns.pop(0)


async def wait_for_status(
    service: AgentServerService,
    run_id: str,
    *statuses: AgentRunStatus,
):
    for _ in range(100):
        view = service.get(run_id)
        if view.status in statuses:
            return view
        await asyncio.sleep(0)
    raise AssertionError(f"run did not reach {statuses!r}")


@pytest.mark.asyncio
async def test_direct_answer_completes_without_tools_and_keeps_output_out_of_events():
    driver = ScriptedDriver(AgentModelTurn(content="Done."))
    service = AgentServerService(registry=FakeRegistry(()), chat_driver=driver)

    created = await service.create(
        AgentRunCreateRequest(goal="private goal"), model="minicpm5-2b-4bit"
    )
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert done.profile == "minicpm5-2b"
    assert done.output == "Done."
    assert done.pending_action is None
    wire = service.events(done.id).model_dump_json()
    assert "private goal" not in wire
    assert "Done." not in wire


@pytest.mark.asyncio
async def test_server_mode_executes_read_only_tool_and_attaches_transient_ledger():
    call = AgentToolCall(
        id="call-read", name=READ.name, arguments={"path": "private.txt"}
    )
    driver = ScriptedDriver(
        AgentModelTurn(tool_calls=[call]),
        AgentModelTurn(content="The file was inspected."),
    )
    registry = FakeRegistry((READ,))
    service = AgentServerService(registry=registry, chat_driver=driver)

    created = await service.create(
        AgentRunCreateRequest(goal="Inspect private.txt"), model="minicpm5-2b-4bit"
    )
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert registry.calls == [call]
    assert done.tool_rounds == 1
    second_history = driver.requests[1][1]
    assert second_history[-1]["role"] == "tool"
    assert "[Rapid task state]" in second_history[-1]["content"]
    assert "Inspect private.txt" in second_history[-1]["content"]
    events_wire = service.events(done.id).model_dump_json()
    assert "private.txt" not in events_wire


@pytest.mark.asyncio
async def test_side_effect_waits_for_exact_approval_before_server_execution():
    call = AgentToolCall(
        id="call-send", name=SEND.name, arguments={"body": "private message"}
    )
    driver = ScriptedDriver(
        AgentModelTurn(tool_calls=[call]), AgentModelTurn(content="Sent.")
    )
    registry = FakeRegistry((SEND,))
    service = AgentServerService(registry=registry, chat_driver=driver)

    created = await service.create(
        AgentRunCreateRequest(goal="Send it"), model="minicpm5-2b-4bit"
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )

    assert registry.calls == []
    assert waiting.pending_action is not None
    assert waiting.pending_action.arguments == {}
    assert waiting.pending_action.approval_required is True
    assert "private message" not in service.events(created.id).model_dump_json()

    with pytest.raises(AgentRunConflictError, match="does not match"):
        await service.approve(
            created.id, AgentApprovalRequest(call_id="wrong", approved=True)
        )
    approved = await service.approve(
        created.id, AgentApprovalRequest(call_id=call.id, approved=True)
    )
    assert approved.pending_action is not None
    assert approved.pending_action.arguments == {}
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert registry.calls == [call]
    assert done.output == "Sent."


@pytest.mark.asyncio
async def test_denial_becomes_tool_observation_and_does_not_execute():
    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "no"})
    driver = ScriptedDriver(
        AgentModelTurn(tool_calls=[call]), AgentModelTurn(content="Not sent.")
    )
    registry = FakeRegistry((SEND,))
    service = AgentServerService(registry=registry, chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="Maybe send"), model="minicpm5-2b-4bit"
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )
    assert waiting.pending_action is not None
    assert waiting.pending_action.arguments == {}

    await service.approve(
        created.id, AgentApprovalRequest(call_id=call.id, approved=False)
    )
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert registry.calls == []
    assert done.output == "Not sent."
    assert "user denied" in driver.requests[1][1][-1]["content"].casefold()


@pytest.mark.asyncio
async def test_client_mode_releases_call_then_accepts_one_matching_result():
    call = AgentToolCall(id="call-read", name=READ.name, arguments={"path": "x"})
    driver = ScriptedDriver(
        AgentModelTurn(tool_calls=[call]), AgentModelTurn(content="Client result used.")
    )
    registry = FakeRegistry((READ,))
    service = AgentServerService(registry=registry, chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="Read x", execution="client"),
        model="minicpm5-2b-4bit",
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_TOOL_RESULT
    )

    assert waiting.pending_action is not None
    assert waiting.pending_action.approval_required is False
    assert registry.calls == []
    with pytest.raises(AgentRunConflictError, match="does not match"):
        await service.submit_result(
            created.id,
            AgentToolResultRequest(call_id="wrong", content="result"),
        )

    await service.submit_result(
        created.id,
        AgentToolResultRequest(
            call_id=call.id,
            content="client secret result",
            executed=False,
        ),
    )
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert done.output == "Client result used."
    event_json = service.events(done.id).model_dump_json()
    assert "client secret result" not in event_json
    assert "Client tool was not executed." in event_json


@pytest.mark.asyncio
async def test_client_side_effect_requires_approval_before_result():
    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "x"})
    driver = ScriptedDriver(
        AgentModelTurn(tool_calls=[call]), AgentModelTurn(content="Client sent it.")
    )
    service = AgentServerService(registry=FakeRegistry((SEND,)), chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="Send", execution="client"),
        model="minicpm5-2b-4bit",
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )
    assert waiting.pending_action is not None
    assert waiting.pending_action.arguments == {}

    with pytest.raises(AgentRunConflictError, match="awaiting_tool_result"):
        await service.submit_result(
            created.id, AgentToolResultRequest(call_id=call.id, content="forged")
        )
    approved = await service.approve(
        created.id, AgentApprovalRequest(call_id=call.id, approved=True)
    )
    assert approved.status is AgentRunStatus.AWAITING_TOOL_RESULT
    assert approved.pending_action is not None
    assert approved.pending_action.arguments == {"body": "x"}
    await service.submit_result(
        created.id, AgentToolResultRequest(call_id=call.id, content="sent")
    )

    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)
    assert done.output == "Client sent it."


@pytest.mark.asyncio
async def test_server_mode_rejects_client_result_injection():
    blocker = asyncio.Event()

    async def blocked_driver(*_args):
        await blocker.wait()
        return AgentModelTurn(content="done")

    service = AgentServerService(registry=FakeRegistry(()), chat_driver=blocked_driver)
    created = await service.create(AgentRunCreateRequest(goal="Wait"), model="model")

    with pytest.raises(AgentRunConflictError, match="do not accept"):
        await service.submit_result(
            created.id, AgentToolResultRequest(call_id="invented", content="inject")
        )
    await service.cancel(created.id)


@pytest.mark.asyncio
async def test_capacity_never_evicts_an_active_run():
    blocker = asyncio.Event()

    async def blocked_driver(*_args):
        await blocker.wait()
        return AgentModelTurn(content="done")

    service = AgentServerService(
        registry=FakeRegistry(()), chat_driver=blocked_driver, max_runs=1
    )
    first = await service.create(AgentRunCreateRequest(goal="First"), model="model")

    with pytest.raises(AgentRunCapacityError, match="all agent run slots are active"):
        await service.create(AgentRunCreateRequest(goal="Second"), model="model")
    assert service.get(first.id).id == first.id
    await service.cancel(first.id)


@pytest.mark.asyncio
async def test_old_terminal_run_is_evicted_to_make_room():
    driver = ScriptedDriver(
        AgentModelTurn(content="one"), AgentModelTurn(content="two")
    )
    service = AgentServerService(
        registry=FakeRegistry(()), chat_driver=driver, max_runs=1
    )
    first = await service.create(AgentRunCreateRequest(goal="First"), model="model")
    await wait_for_status(service, first.id, AgentRunStatus.COMPLETED)

    second = await service.create(AgentRunCreateRequest(goal="Second"), model="model")

    with pytest.raises(AgentRunNotFoundError):
        service.get(first.id)
    assert service.get(second.id).id == second.id
    await service.close()


@pytest.mark.asyncio
async def test_terminal_ttl_expires_without_a_background_reaper():
    now = [10.0]
    service = AgentServerService(
        registry=FakeRegistry(()),
        chat_driver=ScriptedDriver(AgentModelTurn(content="done")),
        terminal_ttl_seconds=5,
        monotonic=lambda: now[0],
    )
    created = await service.create(AgentRunCreateRequest(goal="Task"), model="model")
    await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)
    now[0] = 15.0

    with pytest.raises(AgentRunNotFoundError, match="expired"):
        service.get(created.id)


@pytest.mark.asyncio
async def test_adapter_exception_fails_closed_without_leaking_message():
    async def broken_driver(*_args):
        raise RuntimeError("secret prompt and /private/path")

    service = AgentServerService(registry=FakeRegistry(()), chat_driver=broken_driver)
    created = await service.create(
        AgentRunCreateRequest(goal="secret goal"), model="model"
    )
    failed = await wait_for_status(service, created.id, AgentRunStatus.FAILED)

    assert failed.failure_code == "agent_adapter_failure"
    wire = service.events(created.id).model_dump_json()
    assert "secret" not in wire
    assert "/private/path" not in wire


@pytest.mark.asyncio
async def test_cancel_aborts_inflight_generation_and_close_is_idempotent():
    started = asyncio.Event()

    async def blocked_driver(*_args):
        started.set()
        await asyncio.Future()

    service = AgentServerService(registry=FakeRegistry(()), chat_driver=blocked_driver)
    created = await service.create(AgentRunCreateRequest(goal="Wait"), model="model")
    await started.wait()

    cancelled = await service.cancel(created.id)
    cancelled_again = await service.cancel(created.id)
    await service.close()
    await service.close()

    assert cancelled.status is AgentRunStatus.CANCELLED
    assert cancelled_again.status is AgentRunStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_returns_existing_terminal_outcome_without_500():
    service = AgentServerService(
        registry=FakeRegistry(()),
        chat_driver=ScriptedDriver(AgentModelTurn(content="done")),
    )
    created = await service.create(AgentRunCreateRequest(goal="finish"), model="model")
    completed = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    after_cancel = await service.cancel(created.id)

    assert after_cancel.status is AgentRunStatus.COMPLETED
    assert after_cancel.output == completed.output == "done"

    async def broken_driver(*_args):
        raise RuntimeError("failure")

    failed_service = AgentServerService(
        registry=FakeRegistry(()), chat_driver=broken_driver
    )
    failed_created = await failed_service.create(
        AgentRunCreateRequest(goal="fail"), model="model"
    )
    failed = await wait_for_status(
        failed_service, failed_created.id, AgentRunStatus.FAILED
    )
    failed_after_cancel = await failed_service.cancel(failed.id)
    assert failed_after_cancel.status is AgentRunStatus.FAILED


@pytest.mark.asyncio
async def test_cancel_racing_approval_never_schedules_side_effect():
    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "x"})
    driver = ScriptedDriver(AgentModelTurn(tool_calls=[call]))
    registry = FakeRegistry((SEND,))
    service = AgentServerService(registry=registry, chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="Send"), model="minicpm5-2b-4bit"
    )
    await wait_for_status(service, created.id, AgentRunStatus.AWAITING_APPROVAL)
    entry = service._entry(created.id)

    await entry.lock.acquire()
    approval = asyncio.create_task(
        service.approve(
            created.id, AgentApprovalRequest(call_id=call.id, approved=True)
        )
    )
    await asyncio.sleep(0)
    cancellation = asyncio.create_task(service.cancel(created.id))
    await asyncio.sleep(0)
    entry.lock.release()

    with pytest.raises(AgentRunConflictError, match="cancellation"):
        await approval
    cancelled = await cancellation
    await asyncio.sleep(0)

    assert cancelled.status is AgentRunStatus.CANCELLED
    assert registry.calls == []


@pytest.mark.asyncio
async def test_cancel_after_side_effect_dispatch_preserves_outcome_before_cancel():
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowRegistry(FakeRegistry):
        async def execute(self, call):
            self.calls.append(call)
            started.set()
            await release.wait()
            return AgentToolResult(
                call_id=call.id,
                content="remote action completed",
                safe_summary="Tool completed.",
            )

    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "x"})
    registry = SlowRegistry((SEND,))
    service = AgentServerService(
        registry=registry,
        chat_driver=ScriptedDriver(AgentModelTurn(tool_calls=[call])),
    )
    created = await service.create(
        AgentRunCreateRequest(goal="Send"), model="minicpm5-2b-4bit"
    )
    await wait_for_status(service, created.id, AgentRunStatus.AWAITING_APPROVAL)
    await service.approve(
        created.id, AgentApprovalRequest(call_id=call.id, approved=True)
    )
    await started.wait()

    cancellation = asyncio.create_task(service.cancel(created.id))
    await asyncio.sleep(0)
    assert not cancellation.done()
    release.set()
    cancelled = await cancellation

    event_types = [event.type for event in service.events(created.id).events]
    assert cancelled.status is AgentRunStatus.CANCELLED
    assert registry.calls == [call]
    assert event_types[-2:] == ["tool.completed", "run.cancelled"]


def test_tool_selection_is_exact_bounded_and_read_first_by_default():
    tools = [
        ToolSpec(name=f"mutate_{index}", risk=ToolRisk.EXTERNAL_SIDE_EFFECT)
        for index in range(6)
    ] + [READ]
    registry = FakeRegistry(tools)
    service = AgentServerService(registry=registry)
    profile = resolve_agent_profile("minicpm5-2b-4bit")

    selected = service._select_tools(None, profile, registry)
    assert len(selected) == 6
    assert READ in selected

    with pytest.raises(AgentToolSelectionError, match="unknown"):
        service._select_tools(["missing"], profile, registry)
    with pytest.raises(AgentToolSelectionError, match="at most 6"):
        service._select_tools([tool.name for tool in tools], profile, registry)


@pytest.mark.parametrize(
    ("name", "declared", "risk"),
    [
        ("files__read_file", ["files__read_file"], ToolRisk.READ_ONLY),
        ("web__search", [], ToolRisk.EXTERNAL_SIDE_EFFECT),
        ("get_and_delete", [], ToolRisk.EXTERNAL_SIDE_EFFECT),
        ("mail__send_message", ["other__send_message"], ToolRisk.EXTERNAL_SIDE_EFFECT),
    ],
)
def test_mcp_risk_classifier_requires_exact_operator_declaration(name, declared, risk):
    assert classify_mcp_tool(name, declared_read_only=declared) is risk


def test_request_contract_rejects_duplicate_tools_and_non_boolean_controls():
    assert AgentRunCreateRequest(goal="x", tool_names=None).tool_names is None
    assert AgentRunCreateRequest(goal="x", tool_names=["a"]).tool_names == ["a"]
    with pytest.raises(ValidationError, match="1-128"):
        AgentRunCreateRequest(goal="x", tool_names=[""])
    with pytest.raises(ValidationError, match="unique"):
        AgentRunCreateRequest(goal="x", tool_names=["a", "a"])
    with pytest.raises(ValidationError):
        AgentRunCreateRequest(goal="x", enable_thinking=1)
    with pytest.raises(ValidationError):
        AgentApprovalRequest(call_id="call", approved="yes")
    with pytest.raises(ValidationError, match="safe_summary"):
        AgentToolResultRequest(
            call_id="call",
            content="result",
            safe_summary="client-controlled event payload",
        )


def test_store_configuration_must_be_bounded():
    with pytest.raises(ValueError, match="max_runs"):
        AgentServerService(max_runs=0)
    with pytest.raises(ValueError, match="terminal_ttl"):
        AgentServerService(terminal_ttl_seconds=-1)


def test_event_cursor_returns_only_new_events():
    async def scenario():
        service = AgentServerService(
            registry=FakeRegistry(()),
            chat_driver=ScriptedDriver(AgentModelTurn(content="done")),
        )
        created = await service.create(AgentRunCreateRequest(goal="x"), model="model")
        done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)
        all_events = service.events(done.id)
        tail = service.events(done.id, after=all_events.events[-2].sequence)
        ahead = service.events(done.id, after=999)
        return all_events, tail, ahead

    all_events, tail, ahead = asyncio.run(scenario())
    assert len(tail.events) == 1
    assert tail.events[0].type == "run.completed"
    assert tail.next_after == all_events.next_after
    assert ahead.events == []
    assert ahead.next_after == 999


@pytest.mark.asyncio
async def test_assistant_history_uses_json_arguments_not_python_repr():
    call = AgentToolCall(id="c", name=READ.name, arguments={"path": "你好"})
    driver = ScriptedDriver(AgentModelTurn(tool_calls=[call]))
    service = AgentServerService(registry=FakeRegistry((READ,)), chat_driver=driver)
    run = await service.create(
        AgentRunCreateRequest(goal="x", execution="client"), model="model"
    )
    await wait_for_status(service, run.id, AgentRunStatus.AWAITING_TOOL_RESULT)
    message = service._entry(run.id).messages[-1]

    encoded = message["tool_calls"][0]["function"]["arguments"]
    assert json.loads(encoded) == {"path": "你好"}


def test_mcp_projection_uses_declared_risk_and_skips_unsupported_schemas():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config
    from vllm_mlx.mcp.types import MCPTool

    cfg = reset_config()
    cfg.mcp_manager = SimpleNamespace(
        config=SimpleNamespace(agent_read_only_tools=["files__read_file"]),
        get_all_tools=lambda: [
            MCPTool("files", "read_file", "read", {"type": "object"}),
            MCPTool("files", "write_file", "write", {"type": "object"}),
            MCPTool("refs", "search", "bad", {"$ref": "#/$defs/x"}),
            MCPTool("bad.server", "tool", "bad name", {"type": "object"}),
        ],
    )

    tools = list(MCPToolRegistry().list_tools())

    assert [(tool.name, tool.risk) for tool in tools] == [
        ("files__read_file", ToolRisk.READ_ONLY),
        ("files__write_file", ToolRisk.EXTERNAL_SIDE_EFFECT),
    ]
    reset_config()


@pytest.mark.asyncio
async def test_mcp_execution_preserves_sandbox_and_audit():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config
    from vllm_mlx.mcp.types import MCPToolResult

    audited = []

    class Sandbox:
        def validate_tool_execution(self, *args):
            assert args == ("read_file", "files", {"path": "x"})

        def record_execution(self, *args, **kwargs):
            audited.append((args, kwargs))

    class Manager:
        def resolve_tool_target(self, name):
            assert name == "files__read_file"
            return "files", "read_file"

        async def execute_tool(self, name, arguments):
            return MCPToolResult(name, {"text": "ok"})

    cfg = reset_config()
    cfg.mcp_manager = Manager()
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())
    call = AgentToolCall(id="call", name="files__read_file", arguments={"path": "x"})

    result = await MCPToolRegistry().execute(call)

    assert result.content == '{"text": "ok"}'
    assert result.executed is True
    assert audited[0][1]["success"] is True
    reset_config()


@pytest.mark.asyncio
async def test_mcp_snapshot_never_executes_against_reloaded_registry():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config
    from vllm_mlx.mcp.types import MCPTool, MCPToolResult

    calls = []

    class Sandbox:
        def validate_tool_execution(self, *_args):
            return None

        def record_execution(self, *_args, **_kwargs):
            return None

    class Manager:
        def __init__(self, generation):
            self.generation = generation
            self.config = SimpleNamespace(agent_read_only_tools=[])

        def get_all_tools(self):
            return [MCPTool("same", "tool", "tool", {"type": "object"})]

        def resolve_tool_target(self, _name):
            return "same", "tool"

        async def execute_tool(self, *_args):
            calls.append(self.generation)
            return MCPToolResult("same__tool", "ok")

    cfg = reset_config()
    cfg.mcp_manager = Manager("advertised")
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())
    snapshot = MCPToolRegistry().snapshot()
    assert [tool.name for tool in snapshot.list_tools()] == ["same__tool"]

    cfg.mcp_manager = Manager("reloaded")
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())
    await snapshot.execute(AgentToolCall(id="call", name="same__tool", arguments={}))

    assert calls == ["advertised"]
    reset_config()


@pytest.mark.asyncio
async def test_run_never_switches_to_replacement_model_generation():
    from vllm_mlx.config import reset_config
    from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from vllm_mlx.service.helpers import get_engine

    first_engine = object()
    replacement_engine = object()
    first = ModelEntry(
        engine=first_engine,
        model_name="canonical",
        model_path="/models/first",
        aliases={"served"},
    )
    registry = ModelRegistry()
    registry.add(first, is_default=True)
    cfg = reset_config()
    cfg.model_registry = registry
    seen_engines = []

    async def driver(model, _messages, _tools, _settings):
        seen_engines.append(get_engine(model))
        if len(seen_engines) == 1:
            return AgentModelTurn(
                tool_calls=[
                    AgentToolCall(id="call", name=READ.name, arguments={"path": "x"})
                ]
            )
        return AgentModelTurn(content="done")

    service = AgentServerService(registry=FakeRegistry((READ,)), chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="read", execution="client"),
        model="canonical",
        request_model="served",
        model_generation=first,
    )
    await wait_for_status(service, created.id, AgentRunStatus.AWAITING_TOOL_RESULT)

    registry.remove("canonical")
    registry.add(
        ModelEntry(
            engine=replacement_engine,
            model_name="canonical",
            model_path="/models/replacement",
            aliases={"served"},
        ),
        is_default=True,
    )
    await service.submit_result(
        created.id,
        AgentToolResultRequest(call_id="call", content="result"),
    )
    failed = await wait_for_status(service, created.id, AgentRunStatus.FAILED)

    assert failed.failure_code == "agent_adapter_failure"
    assert seen_engines == [first_engine]
    reset_config()


@pytest.mark.asyncio
async def test_mcp_sandbox_rejection_is_reported_as_unexecuted():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config
    from vllm_mlx.mcp.security import MCPSecurityError

    executed = []
    audited = []

    class Sandbox:
        def validate_tool_execution(self, *_args):
            raise MCPSecurityError("secret policy detail")

        def record_execution(self, *args, **kwargs):
            audited.append((args, kwargs))

    class Manager:
        def resolve_tool_target(self, _name):
            return "shell", "execute"

        async def execute_tool(self, *_args):
            executed.append(True)

    cfg = reset_config()
    cfg.mcp_manager = Manager()
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())

    result = await MCPToolRegistry().execute(
        AgentToolCall(id="call", name="shell__execute", arguments={"cmd": "x"})
    )

    assert result.executed is False
    assert result.is_error is True
    assert "secret policy detail" not in result.content
    assert executed == []
    assert audited[0][1]["success"] is False
    assert audited[0][1]["error_message"] == "blocked by server security policy"
    reset_config()


@pytest.mark.asyncio
async def test_mcp_unavailable_and_disappeared_calls_are_unexecuted():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config

    call = AgentToolCall(id="call", name="files__read_file", arguments={})
    cfg = reset_config()
    assert (await MCPToolRegistry().execute(call)).executed is False

    cfg.mcp_manager = SimpleNamespace(
        resolve_tool_target=lambda _name: (None, "read_file")
    )
    cfg.mcp_executor = SimpleNamespace(sandbox=object())
    assert (await MCPToolRegistry().execute(call)).executed is False

    audited = []

    class Sandbox:
        def record_execution(self, *args, **kwargs):
            audited.append((args, kwargs))

    cfg.mcp_manager = SimpleNamespace(
        resolve_tool_target=lambda _name: ("files", "read_file"),
        get_client=lambda _name: SimpleNamespace(is_connected=False),
    )
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())
    unavailable = await MCPToolRegistry().execute(call)
    assert unavailable.executed is False
    assert unavailable.is_error is True
    assert audited[0][1]["error_message"] == "MCP server unavailable"
    reset_config()


@pytest.mark.asyncio
async def test_mcp_result_shapes_and_execution_exception_are_audited():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config
    from vllm_mlx.mcp.types import MCPToolResult

    audited = []

    class Sandbox:
        def validate_tool_execution(self, *_args):
            return None

        def record_execution(self, *args, **kwargs):
            audited.append((args, kwargs))

    class Manager:
        result = MCPToolResult("tool", "plain")

        def resolve_tool_target(self, _name):
            return "server", "tool"

        async def execute_tool(self, *_args):
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    manager = Manager()
    cfg = reset_config()
    cfg.mcp_manager = manager
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())
    call = AgentToolCall(id="call", name="server__tool", arguments={})

    plain = await MCPToolRegistry().execute(call)
    assert plain.content == "plain"

    manager.result = MCPToolResult(
        "tool", None, is_error=True, error_message="expected failure"
    )
    failed = await MCPToolRegistry().execute(call)
    assert failed.content == "expected failure"
    assert failed.executed is True

    manager.result = MCPToolResult("tool", "x" * 250_000)
    truncated = await MCPToolRegistry().execute(call)
    assert truncated.content.endswith("[tool result truncated by Rapid]")

    manager.result = RuntimeError("private exception")
    with pytest.raises(RuntimeError, match="private exception"):
        await MCPToolRegistry().execute(call)
    assert audited[-1][1]["error_message"] == "RuntimeError"
    reset_config()


@pytest.mark.asyncio
async def test_registry_without_mcp_has_no_tools_and_internal_request_stays_live():
    from vllm_mlx.agent_runtime.server import _InternalRequest
    from vllm_mlx.config import reset_config

    reset_config()
    assert MCPToolRegistry().list_tools() == []
    assert await _InternalRequest().is_disconnected() is False


@pytest.mark.asyncio
async def test_close_cancels_active_work_and_rejects_new_runs():
    started = asyncio.Event()

    async def blocked_driver(*_args):
        started.set()
        await asyncio.Future()

    service = AgentServerService(registry=FakeRegistry(()), chat_driver=blocked_driver)
    created = await service.create(AgentRunCreateRequest(goal="wait"), model="model")
    await started.wait()

    await service.close()

    assert service.get(created.id).status is AgentRunStatus.CANCELLED
    with pytest.raises(AgentRunCapacityError, match="shutting down"):
        await service.create(AgentRunCreateRequest(goal="new"), model="model")


@pytest.mark.asyncio
async def test_schedule_rejects_parallel_driver_for_same_run():
    blocker = asyncio.Event()

    async def blocked_driver(*_args):
        await blocker.wait()
        return AgentModelTurn(content="done")

    service = AgentServerService(registry=FakeRegistry(()), chat_driver=blocked_driver)
    created = await service.create(AgentRunCreateRequest(goal="wait"), model="model")
    entry = service._entry(created.id)

    with pytest.raises(AgentRunConflictError, match="already has work"):
        service._schedule(entry)
    await service.cancel(created.id)


@pytest.mark.asyncio
async def test_runtime_rejection_is_marked_terminal_by_adapter():
    driver = ScriptedDriver(
        AgentModelTurn(
            tool_calls=[AgentToolCall(id="bad", name="not_advertised", arguments={})]
        )
    )
    service = AgentServerService(registry=FakeRegistry((READ,)), chat_driver=driver)
    created = await service.create(AgentRunCreateRequest(goal="x"), model="model")

    failed = await wait_for_status(service, created.id, AgentRunStatus.FAILED)

    assert failed.failure_code == "unadvertised_tool_call"


@pytest.mark.asyncio
async def test_repeated_call_guard_observation_reaches_final_synthesis():
    calls = [
        AgentToolCall(id=f"call-{index}", name=READ.name, arguments={"path": "x"})
        for index in range(3)
    ]
    driver = ScriptedDriver(
        *(AgentModelTurn(tool_calls=[call]) for call in calls),
        AgentModelTurn(content="Stopped repeating."),
    )
    registry = FakeRegistry((READ,))
    service = AgentServerService(registry=registry, chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="read x"), model="minicpm5-2b-4bit"
    )

    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert done.output == "Stopped repeating."
    assert len(registry.calls) == 2
    assert driver.requests[-1][2] == []
    assert "blocked because it repeated" in driver.requests[-1][1][-1]["content"]
