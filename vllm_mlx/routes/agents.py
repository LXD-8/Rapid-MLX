# SPDX-License-Identifier: Apache-2.0
"""Authenticated HTTP surface for bounded local agent runs."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..agent_runtime.server import (
    AgentApprovalRequest,
    AgentEventsView,
    AgentRunCapacityError,
    AgentRunConflictError,
    AgentRunCreateRequest,
    AgentRunNotFoundError,
    AgentRunView,
    AgentServerService,
    AgentToolResultRequest,
    AgentToolSelectionError,
)
from ..config import get_config
from ..middleware.auth import check_rate_limit, verify_api_key
from ..model_metadata import read_model_metadata
from ..service.helpers import _validate_model_name

router = APIRouter(
    prefix="/v1/agent",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)

_service: AgentServerService | None = None


def get_agent_service() -> AgentServerService:
    global _service
    if _service is None:
        _service = AgentServerService()
    return _service


async def close_agent_service() -> None:
    global _service
    service, _service = _service, None
    if service is not None:
        await service.close()


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, AgentRunNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, AgentRunCapacityError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, AgentToolSelectionError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


@router.post(
    "/runs",
    response_model=AgentRunView,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_agent_run(request: AgentRunCreateRequest) -> AgentRunView:
    cfg = get_config()
    request_model = request.model or cfg.model_alias or cfg.model_name
    if not request_model:
        raise HTTPException(status_code=503, detail="no text model is configured")
    _validate_model_name(request_model)
    profile_model = cfg.model_path or cfg.model_alias or request_model
    profile_model_config = None
    profile_tool_call_parser = cfg.tool_call_parser
    model_generation = cfg.engine
    if cfg.model_registry is not None:
        try:
            entry = cfg.model_registry.get_entry(request_model)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        profile_model = entry.model_path or entry.model_name
        metadata = await asyncio.to_thread(read_model_metadata, entry.model_path)
        profile_model_config = metadata.config if metadata is not None else None
        profile_tool_call_parser = entry.tool_call_parser
        model_generation = entry
    elif cfg.model_path:
        metadata = await asyncio.to_thread(read_model_metadata, cfg.model_path)
        profile_model_config = metadata.config if metadata is not None else None
    try:
        return await get_agent_service().create(
            request,
            model=profile_model,
            request_model=request_model,
            profile_model_config=profile_model_config,
            profile_tool_call_parser=profile_tool_call_parser,
            model_generation=model_generation,
        )
    except (AgentRunCapacityError, AgentToolSelectionError) as exc:
        raise _http_error(exc) from exc


@router.get("/runs/{run_id}", response_model=AgentRunView)
async def get_agent_run(run_id: str) -> AgentRunView:
    try:
        return get_agent_service().get(run_id)
    except AgentRunNotFoundError as exc:
        raise _http_error(exc) from exc


@router.get("/runs/{run_id}/events", response_model=AgentEventsView)
async def get_agent_events(
    run_id: str, after: int = Query(default=0, ge=0)
) -> AgentEventsView:
    try:
        return get_agent_service().events(run_id, after=after)
    except AgentRunNotFoundError as exc:
        raise _http_error(exc) from exc


@router.post("/runs/{run_id}/approval", response_model=AgentRunView)
async def approve_agent_action(
    run_id: str, request: AgentApprovalRequest
) -> AgentRunView:
    try:
        return await get_agent_service().approve(run_id, request)
    except (AgentRunNotFoundError, AgentRunConflictError) as exc:
        raise _http_error(exc) from exc


@router.post("/runs/{run_id}/tool-result", response_model=AgentRunView)
async def submit_agent_tool_result(
    run_id: str, request: AgentToolResultRequest
) -> AgentRunView:
    try:
        return await get_agent_service().submit_result(run_id, request)
    except (AgentRunNotFoundError, AgentRunConflictError) as exc:
        raise _http_error(exc) from exc


@router.post("/runs/{run_id}/cancel", response_model=AgentRunView)
async def cancel_agent_run(run_id: str) -> AgentRunView:
    try:
        return await get_agent_service().cancel(run_id)
    except AgentRunNotFoundError as exc:
        raise _http_error(exc) from exc
