import logging
import uuid
from uuid import UUID
from fastapi import APIRouter, Path, Query, Body, Header, BackgroundTasks
from fastapi.exceptions import HTTPException
from gigachat.exceptions import GigaChatException
from typing import Dict, Any

from datetime import datetime

from contract.schemas_dnie import (
    Response,
    CreateRunPayload,
)

from small_agent.main import AIAgent_DNIE


agent = AIAgent_DNIE()

small_router = APIRouter()

logger_api = logging.getLogger("api")
logger_agent = logging.getLogger("agent")

# TODO временное решение
results = {}


@small_router.post("/api/v1/invoke", response_model=Response)
async def response(
    input_data: CreateRunPayload,
    x_agent_id: str = Header(
        ...,
        alias="x-agent-id",
        description="K9 агента-оркестратора (TBD)",
    ),
    x_trace_id: UUID = Header(
        ...,
        alias="X-Trace-Id",
        description="Идентификатор запроса",
    ),
):
    resp = await agent.conversation(input_data)
    return resp
