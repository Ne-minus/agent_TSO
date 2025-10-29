import logging
from uuid import UUID
from fastapi import APIRouter, Path, Query, Body, Header

from datetime import datetime

# from uvicorn.config import logger

from contract.schemas import (
    UserContext,
    CreateNewDialogRs,
    MessageToAgentRs,
    MessageToAgentRq,
)

from agent.main import AIAgent


agent = AIAgent()

router = APIRouter()

logger = logging.getLogger("api")


@router.post("/dialogs/", response_model=CreateNewDialogRs)
async def dialogs(
    user_context: UserContext,
    x_request_id: UUID = Header(
        ..., description="Уникальный идемнтификатор запроса. Ключ идепотентности"
    ),
):
    time = datetime.now()
    logger.debug(f"На API получен запрос {x_request_id}.")
    response = agent.create_conversation(user_context)
    logger.debug(f"Ответ на запрос API {x_request_id} отправлен. Время выполнения {datetime.now() - time}.")
    return response


@router.post("/dialogs/{dialogId}", response_model=MessageToAgentRs)
async def dialog_by_id(
    message: MessageToAgentRq,
    dialogId: UUID = Path(
        ..., description="ID диалога, в контексте которого отправляется сообщение"
    ),
    x_request_id: UUID = Header(
        ..., description="Уникальный идентификатор запроса. Ключ идемпотентности"
    ),
):
    time = datetime.now()
    logger.debug(f"На API получен запрос {x_request_id}.")
    if not message.context:
        message.context = None
    response = agent.continue_conversation(
        str(dialogId), message.message, context=message.context
    )
    logger.debug(f"Ответ на запрос API {x_request_id} отправлен. Время выполнения {datetime.now() - time}.")
    return response
