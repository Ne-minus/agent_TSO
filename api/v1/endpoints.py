import logging
import uuid
from uuid import UUID
from fastapi import APIRouter, Path, Query, Body, Header, BackgroundTasks
from fastapi.exceptions import HTTPException
from gigachat.exceptions import GigaChatException
from typing import Dict, Any

from datetime import datetime

# from uvicorn.config import logger

from contract.schemas import (
    UserContext,
    CreateNewDialogRs,
    MessageToAgentRs,
    MessageToAgentRq,
    TaskResponse,
    CreateTaskRs,
)

from agent.main import AIAgent


agent = AIAgent()

router = APIRouter()

logger_api = logging.getLogger("api")
logger_agent = logging.getLogger("agent")

results = {}


@router.post("/dialogs/", response_model=CreateNewDialogRs)
async def dialogs(
    user_context: UserContext,
    x_request_id: UUID = Header(
        ..., description="Уникальный идемнтификатор запроса. Ключ идепотентности"
    ),
):
    try:
        time = datetime.now()
        logger_api.debug(f"На API получен запрос {x_request_id}.")
        response = await agent.create_conversation(user_context)
        logger_api.debug(
            f"Ответ на запрос API {x_request_id} отправлен. Время выполнения {datetime.now() - time}."
        )
        return response
    except GigaChatException as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"GigaChat_error: {e}"))
    except Exception as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"Agent_error: {e}"))


async def process_task(
    task_id: str,
    message: MessageToAgentRq,
    dialogId: UUID = Path(
        ..., description="ID диалога, в контексте которого отправляется сообщение"
    ),
    x_request_id: UUID = Header(
        ..., description="Уникальный идентификатор запроса. Ключ идемпотентности"
    ),
) -> None:
    logger_api.info(f"Processing task {task_id}")
    results[task_id] = {}

    try:
        time = datetime.now()
        logger_api.debug(f"На API получен запрос {x_request_id}.")
        if not message.context:
            message.context = None

        agent_message = await agent.continue_conversation(
            str(dialogId), message.message, context=message.context
        )
        logger_api.debug(
            f"Ответ на запрос API {x_request_id} отправлен. Время выполнения {datetime.now() - time}."
        )
        results[task_id] = agent_message

        logger_api.info(f"Task {task_id} result: {agent_message}")

    except GigaChatException as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"GigaChat_error: {e}"))
    except Exception as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"Agent_error: {e}"))


@router.post("/dialogs/{dialogId}")
async def create_task(
    message: MessageToAgentRq,
    background_tasks: BackgroundTasks,
    dialogId: UUID = Path(
        ..., description="ID диалога, в контексте которого отправляется сообщение"
    ),
    x_request_id: UUID = Header(
        ..., description="Уникальный идентификатор запроса. Ключ идемпотентности"
    ),
) -> CreateTaskRs:

    task_id = str(uuid.uuid4())

    background_tasks.add_task(process_task, task_id, message, dialogId, x_request_id)

    return CreateTaskRs(task_id=task_id)


@router.get("/dialogs/{dialogId}/{taskId}")
async def get_task_result(task_id: str) -> TaskResponse:
    result = results.get(task_id)

    if result is None:
        return TaskResponse(error=f"No task with id: {task_id}")
    if result == {}:
        return TaskResponse(status=f"in_progress")
    return TaskResponse(status="done", result=result)
