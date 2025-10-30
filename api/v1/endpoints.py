import logging
from uuid import UUID
from fastapi import APIRouter, Path, Header
from fastapi.exceptions import HTTPException
from gigachat.exceptions import GigaChatException

from datetime import datetime

from contract.schemas import (
    UserContext,
    CreateNewDialogRs,
    MessageToAgentRs,
    MessageToAgentRq,
)

from agent.main import AIAgent


agent = AIAgent()

router = APIRouter()

logger_api = logging.getLogger("api")
logger_agent = logging.getLogger("agent")


@router.post("/dialogs/", response_model=CreateNewDialogRs)
async def dialogs(
    user_context: UserContext,
    x_request_id: UUID = Header(
        ..., description="Уникальный идентификатор запроса. Ключ идемпотентности"
    ),
):
    # todo сделать middleware для логгирования запросов
    try:
        time = datetime.now()
        logger_api.debug(f"x_request_id {x_request_id} - инициация диалога")
        response = await agent.create_conversation(user_context)
        logger_api.debug(f"x_request_id {x_request_id} - сформирован ответ")
        logger_api.debug(f"x_request_id {x_request_id} - {response.model_dump()}")
        logger_api.debug(f"x_request_id {x_request_id} - отправлен ответ на запрос - время выполнения {datetime.now() - time}")
        return response
    except GigaChatException as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"GigaChat_error: {e}"))
    except Exception as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"Agent_error: {e}"))

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
    # todo сделать middleware для логгирования запросов
    try:
        time = datetime.now()
        logger_api.debug(f"x_request_id {x_request_id} - продолжение диалога {dialogId}")
        logger_api.debug(f"x_request_id {x_request_id} - {message.model_dump()}")
        if not message.context:
            message.context = None
        response = await agent.continue_conversation(
            str(dialogId), message.message, context=message.context
        )
        logger_api.debug(f"x_request_id {x_request_id} - сформирован ответ")
        logger_api.debug(f"x_request_id {x_request_id} - {response.model_dump()}")
        logger_api.debug(f"x_request_id {x_request_id} - отправлен ответ на запрос - время выполнения {datetime.now() - time}")
        return response
    except GigaChatException as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"GigaChat_error: {e}"))
    except Exception as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"Agent_error: {e}"))
