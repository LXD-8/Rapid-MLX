# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient

from vllm_mlx.agent_runtime import AgentRunStatus, ToolRisk, ToolSpec
from vllm_mlx.agent_runtime.server import (
    AgentEventsView,
    AgentRunCapacityError,
    AgentRunConflictError,
    AgentRunCreateRequest,
    AgentRunNotFoundError,
    AgentRunView,
    AgentServerError,
    AgentToolSelectionError,
    generate_chat_turn,
)
from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
)
from vllm_mlx.config import get_config, reset_config
from vllm_mlx.middleware.auth import check_rate_limit, verify_api_key
from vllm_mlx.routes import agents as agent_routes


@pytest.mark.asyncio
async def test_chat_driver_reuses_non_stream_route_and_decodes_tool_call(monkeypatch):
    captured = []
    response = ChatCompletionResponse(
        model="served",
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function=FunctionCall(
                                name="files__read_file",
                                arguments=json.dumps({"path": "notes.txt"}),
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    async def fake_chat(request, raw_request):
        captured.append((request, raw_request))
        return Response(
            content=response.model_dump_json(exclude_none=True),
            media_type="application/json",
        )

    from vllm_mlx.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "create_chat_completion", fake_chat)
    settings = AgentRunCreateRequest(goal="Read notes")
    tool = ToolSpec(
        name="files__read_file",
        risk=ToolRisk.READ_ONLY,
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    )

    turn = await generate_chat_turn(
        "served", [{"role": "user", "content": "Read notes"}], [tool], settings
    )

    request = captured[0][0]
    assert request.stream is False
    assert request.parallel_tool_calls is False
    assert request.max_tokens == 900
    assert request.timeout == 300.0
    assert request.temperature == 0.7
    assert request.top_p == 0.95
    assert request.enable_thinking is False
    assert turn.tool_calls[0].arguments == {"path": "notes.txt"}


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["not json", "[]"])
async def test_chat_driver_fails_closed_on_malformed_tool_arguments(
    monkeypatch, arguments
):
    response = ChatCompletionResponse(
        model="served",
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function=FunctionCall(name="tool", arguments=arguments),
                        )
                    ]
                )
            )
        ],
    )

    async def fake_chat(*_args):
        return Response(content=response.model_dump_json(exclude_none=True))

    from vllm_mlx.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "create_chat_completion", fake_chat)

    with pytest.raises(AgentServerError, match="tool arguments"):
        await generate_chat_turn(
            "served",
            [{"role": "user", "content": "x"}],
            [],
            AgentRunCreateRequest(goal="x"),
        )


@pytest.mark.asyncio
async def test_chat_driver_rejects_unsuccessful_or_ambiguous_response(monkeypatch):
    from vllm_mlx.routes import chat as chat_routes

    async def unavailable(*_args):
        return Response(status_code=503)

    monkeypatch.setattr(chat_routes, "create_chat_completion", unavailable)
    with pytest.raises(AgentServerError, match="successful response"):
        await generate_chat_turn(
            "served",
            [{"role": "user", "content": "x"}],
            [],
            AgentRunCreateRequest(goal="x"),
        )

    ambiguous = ChatCompletionResponse(
        model="served",
        choices=[
            ChatCompletionChoice(message=AssistantMessage(content="one")),
            ChatCompletionChoice(message=AssistantMessage(content="two")),
        ],
    )

    async def two_choices(*_args):
        return Response(content=ambiguous.model_dump_json(exclude_none=True))

    monkeypatch.setattr(chat_routes, "create_chat_completion", two_choices)
    with pytest.raises(AgentServerError, match="choice count"):
        await generate_chat_turn(
            "served",
            [{"role": "user", "content": "x"}],
            [],
            AgentRunCreateRequest(goal="x"),
        )


class _RouteService:
    def __init__(self):
        self.created = []

    async def create(
        self,
        request,
        *,
        model,
        request_model=None,
        profile_model_config=None,
        profile_tool_call_parser=None,
        model_generation=None,
    ):
        if request.goal == "capacity":
            raise AgentRunCapacityError("full")
        if request.goal == "bad tools":
            raise AgentToolSelectionError("bad selection")
        self.created.append(
            (
                request,
                model,
                request_model,
                profile_model_config,
                profile_tool_call_parser,
                model_generation,
            )
        )
        return self.view()

    @staticmethod
    def view():
        return AgentRunView(
            id="run-1",
            model="model",
            profile="minicpm5-2b",
            status=AgentRunStatus.READY,
            model_turns=0,
            tool_rounds=0,
            final_synthesis=False,
        )

    def get(self, run_id):
        if run_id == "missing":
            raise AgentRunNotFoundError("missing")
        return self.view()

    def events(self, run_id, *, after):
        if run_id == "missing":
            raise AgentRunNotFoundError("missing")
        return AgentEventsView(
            run_id=run_id,
            status=AgentRunStatus.READY,
            events=[],
            next_after=after,
        )

    async def approve(self, run_id, request):
        if run_id == "conflict":
            raise AgentRunConflictError("conflict")
        if run_id == "missing":
            raise AgentRunNotFoundError("missing")
        return self.view()

    async def submit_result(self, run_id, request):
        return await self.approve(run_id, request)

    async def cancel(self, run_id):
        if run_id == "missing":
            raise AgentRunNotFoundError("missing")
        return self.view()


def test_agent_routes_require_bearer_and_bind_profile_to_real_model(monkeypatch):
    cfg = reset_config()
    cfg.api_key = "secret"
    cfg.model_name = "pretty-served-name"
    cfg.model_alias = "minicpm5-2b-4bit"
    cfg.model_path = "openbmb/MiniCPM5-2B-MLX"
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        assert client.post("/v1/agent/runs", json={"goal": "x"}).status_code == 401
        response = client.post(
            "/v1/agent/runs",
            headers={"Authorization": "Bearer secret"},
            json={"goal": "x", "model": "pretty-served-name"},
        )

    assert response.status_code == 202
    assert service.created[0][1:3] == (
        "openbmb/MiniCPM5-2B-MLX",
        "pretty-served-name",
    )
    reset_config()


def test_agent_create_rejects_unknown_model_before_starting_background_work(
    monkeypatch,
):
    cfg = reset_config()
    cfg.model_name = "known"
    cfg.model_path = "openbmb/MiniCPM5-2B-MLX"
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        response = client.post("/v1/agent/runs", json={"goal": "x", "model": "unknown"})

    assert response.status_code == 404
    assert service.created == []
    reset_config()


def test_agent_create_requires_a_configured_model(monkeypatch):
    reset_config()
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        response = client.post("/v1/agent/runs", json={"goal": "x"})

    assert response.status_code == 503
    assert service.created == []
    reset_config()


def test_agent_create_resolves_registry_model_identity(monkeypatch):
    from types import SimpleNamespace

    class Registry:
        def __bool__(self):
            return True

        def __contains__(self, name):
            return name == "served"

        def get_entry(self, name):
            assert name == "served"
            return SimpleNamespace(
                model_path="openbmb/MiniCPM5-2B-MLX",
                model_name="canonical",
                tool_call_parser="minicpm",
            )

    cfg = reset_config()
    cfg.model_name = "served"
    cfg.model_registry = Registry()
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        response = client.post("/v1/agent/runs", json={"goal": "x"})

    assert response.status_code == 202
    assert service.created[0][1:3] == (
        "openbmb/MiniCPM5-2B-MLX",
        "served",
    )
    reset_config()


def test_agent_create_qualifies_custom_local_minicpm_from_metadata(monkeypatch):
    from types import SimpleNamespace

    minicpm_config = {
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 42,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "vocab_size": 130560,
    }

    class Registry:
        def __bool__(self):
            return True

        def __contains__(self, name):
            return name == "friendly-name"

        def get_entry(self, _name):
            return SimpleNamespace(
                model_path="/models/arbitrary-folder",
                model_name="friendly-name",
                tool_call_parser="minicpm",
            )

    cfg = reset_config()
    cfg.model_name = "friendly-name"
    cfg.model_registry = Registry()
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    monkeypatch.setattr(
        agent_routes,
        "read_model_metadata",
        lambda _path: SimpleNamespace(config=minicpm_config),
    )
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        response = client.post("/v1/agent/runs", json={"goal": "x"})

    assert response.status_code == 202
    assert service.created[0][1] == "/models/arbitrary-folder"
    assert service.created[0][3] == minicpm_config
    assert service.created[0][4] == "minicpm"
    reset_config()


def test_agent_http_surface_maps_success_and_stable_failures(monkeypatch):
    cfg = reset_config()
    cfg.model_name = "known"
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        assert client.get("/v1/agent/runs/run-1").status_code == 200
        assert client.get("/v1/agent/runs/missing").status_code == 404
        events = client.get("/v1/agent/runs/run-1/events?after=7")
        assert events.status_code == 200
        assert events.json()["next_after"] == 7
        assert client.get("/v1/agent/runs/missing/events").status_code == 404

        approval = {"call_id": "call", "approved": True}
        assert (
            client.post("/v1/agent/runs/run-1/approval", json=approval).status_code
            == 200
        )
        assert (
            client.post("/v1/agent/runs/conflict/approval", json=approval).status_code
            == 409
        )
        assert (
            client.post("/v1/agent/runs/missing/approval", json=approval).status_code
            == 404
        )

        result = {"call_id": "call", "content": "ok"}
        assert (
            client.post("/v1/agent/runs/run-1/tool-result", json=result).status_code
            == 200
        )
        assert (
            client.post("/v1/agent/runs/conflict/tool-result", json=result).status_code
            == 409
        )
        assert client.post("/v1/agent/runs/run-1/cancel").status_code == 200
        assert client.post("/v1/agent/runs/missing/cancel").status_code == 404

        assert (
            client.post("/v1/agent/runs", json={"goal": "capacity"}).status_code == 503
        )
        assert (
            client.post("/v1/agent/runs", json={"goal": "bad tools"}).status_code == 422
        )

    reset_config()


@pytest.mark.asyncio
async def test_agent_route_singleton_closes_and_resets(monkeypatch):
    closed = []

    class Service:
        async def close(self):
            closed.append(True)

    monkeypatch.setattr(agent_routes, "_service", Service())
    await agent_routes.close_agent_service()
    await agent_routes.close_agent_service()

    assert closed == [True]
    assert agent_routes._service is None


def test_agent_route_singleton_is_lazy(monkeypatch):
    instance = object()
    monkeypatch.setattr(agent_routes, "_service", None)
    monkeypatch.setattr(agent_routes, "AgentServerService", lambda: instance)

    assert agent_routes.get_agent_service() is instance
    assert agent_routes.get_agent_service() is instance


def test_all_agent_routes_share_auth_and_rate_limit_dependencies():
    route_dependencies = [
        route.dependencies
        for route in agent_routes.router.routes
        if hasattr(route, "dependencies")
    ]

    assert route_dependencies
    expected = {verify_api_key, check_rate_limit}
    assert all(
        {dependency.dependency for dependency in dependencies} == expected
        for dependencies in route_dependencies
    )
    assert get_config() is not None
