import logging
import uuid
from uuid import UUID
from fastapi import APIRouter, Path, Query, Body, Header, BackgroundTasks
from fastapi.exceptions import HTTPException
from gigachat.exceptions import GigaChatException
from typing import Dict, Any

from datetime import datetime

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

# TODO временное решение
results = {}


async def process_task(
    task_id: str,
    message: MessageToAgentRq,
    dialogId: UUID = Path(
        ..., description="ID диалога, в контексте которого отправляется сообщение"
    ),
    x_request_id: UUID = Header(
        ..., description="Уникальный идентификатор запроса. Ключ идемпотентности"
    ),
):

    try:
        time = datetime.now()
        results[task_id] = {}
        logger_api.debug(f"x_request_id {x_request_id} - создание задачи агента {dialogId}")
        logger_api.debug(f"x_request_id {x_request_id} - {message.model_dump()}")
        if not message.context:
            message.context = None

        response = await agent.continue_conversation(
            str(dialogId), message.message, context=message.context
        )

        results[task_id] = response
        logger_api.info(f"x_request_id {x_request_id} - task {task_id} result: {response.model_dump()}")
        logger_api.debug(f"x_request_id {x_request_id} - ответ агента сформирован - время выполнения {datetime.now() - time}")
    except GigaChatException as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"GigaChat_error: {e}"))
    except Exception as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"Agent_error: {e}"))


@router.post("/dialogs/", response_model=CreateNewDialogRs)
async def dialogs(
    user_context: UserContext,
    x_request_id: UUID = Header(
        ..., description="Уникальный идентификатор запроса. Ключ идемпотентности"
    ),
):
    try:
        time = datetime.now()
        logger_agent.debug(f"x_request_id {x_request_id} - инициация диалога")
        response = await agent.create_conversation(user_context)
        logger_agent.debug(f"x_request_id {x_request_id} - сформирован ответ")
        logger_agent.debug(f"x_request_id {x_request_id} - {response.model_dump()}")
        logger_agent.debug(f"x_request_id {x_request_id} - отправлен ответ на запрос - время выполнения {datetime.now() - time}")
        return response
    except GigaChatException as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"GigaChat_error: {e}"))
    except Exception as e:
        logger_agent.error(e)
        raise HTTPException(status_code=500, detail=str(f"Agent_error: {e}"))


@router.post("/dialogs/{dialogId}", response_model=CreateTaskRs)
async def send_message(
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


@router.get("/dialogs/task_status/{task_id}", response_model=TaskResponse)
async def get_response(
        task_id: str,
        x_request_id: UUID = Header(
        ..., description="Уникальный идентификатор запроса. Ключ идемпотентности"
    ),
) -> TaskResponse:
    response = results.get(task_id)

    if response is None:
        logger_api.error(f"No such task: {task_id}")
        raise HTTPException(status_code=404, detail=f"no_such_task - {task_id}")
    if response == {}:
        return TaskResponse(status=f"in_progress")
    return TaskResponse(status="done", result=response)
