import logging
import uuid
from uuid import UUID
from fastapi import APIRouter, Path, Query, Body, Header, BackgroundTasks
from fastapi.exceptions import HTTPException
from gigachat.exceptions import GigaChatException
from typing import Dict, Any
from itertools import count

from datetime import datetime

from contract.schemas import (
    UserContext,
    CreateNewDialogRs,
    MessageToAgentRs,
    MessageToAgentRq,
    TaskResponse,
    CreateTaskRs,
)
from fastapi.responses import JSONResponse


from agent.main import AIAgent


agent = AIAgent()
preview_agent = AIAgent(model="GigaChat-2-Max-preview")

router = APIRouter()

logger_api = logging.getLogger("api")
logger_agent = logging.getLogger("agent")

# TODO временное решение
results = {}
hops = {}
preview_ids = []
counter = count(1)


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
        hops[task_id] = 1
        logger_api.debug(
            f"x_request_id {x_request_id} - создание задачи агента {dialogId}"
        )
        if not message.context:
            message.context = None
        if dialogId in preview_ids:
            response = await preview_agent.continue_conversation(
                str(dialogId), message.message, context=message.context
            )
        else:
            response = await agent.continue_conversation(
                str(dialogId), message.message, context=message.context
            )

        results[task_id] = response
        logger_api.info(
            f"x_request_id {x_request_id} - task {task_id} result: {response.model_dump()}"
        )
        logger_api.debug(
            f"x_request_id {x_request_id} - время выполнения задачи {task_id} {datetime.now() - time}"
        )
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
        current = next(counter)
        if current % 20 == 0:
            logger_agent.info(
                f"x_request_id {x_request_id} - диалог отправлен на preview модель"
            )
            response = await preview_agent.create_conversation(user_context)
            preview_ids.append(response.dialogId)
        else:
            response = await agent.create_conversation(user_context)

        logger_agent.debug(f"x_request_id {x_request_id} - сформирован ответ")
        logger_agent.debug(f"x_request_id {x_request_id} - {response.model_dump()}")
        logger_agent.debug(
            f"x_request_id {x_request_id} - отправлен ответ на запрос - время выполнения {datetime.now() - time}"
        )
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
    logger_api.debug(
        f"x_request_id {x_request_id} - получен запрос: {message.model_dump()}"
    )
    task_id = str(uuid.uuid4())
    background_tasks.add_task(process_task, task_id, message, dialogId, x_request_id)
    response = CreateTaskRs(task_id=task_id)
    logger_api.debug(
        f"x_request_id {x_request_id} - направлен ответ: {response.model_dump()}"
    )
    return CreateTaskRs(task_id=task_id)


@router.get("/dialogs/task_status/{task_id}", response_model=TaskResponse)
async def get_response(
    task_id: str,
    x_request_id: UUID = Header(
        ..., description="Уникальный идентификатор запроса. Ключ идемпотентности"
    ),
) -> TaskResponse:
    logger_api.debug(f"x_request_id {x_request_id} - запрос статуса задачи {task_id}")

    if hops.get(task_id) is None:
        logger_api.error(f"No such task: {task_id}")
        raise HTTPException(status_code=404, detail=f"no_such_task - {task_id}")

    hops[task_id] += 1
    if hops[task_id] >= 50:
        response = TaskResponse(
            status="error",
            result={
                "message": f"Превышено количество запросов статуса задачи {task_id}"
            },
        )
        logger_api.error(
            f"x_request_id {x_request_id} - направлен ответ {response.model_dump()}"
        )
        return JSONResponse(
            status_code=429,
            content={
                "status": "error",
                "result": {
                    "message": "Превышено количество запросов",
                    "action": None,
                    "ticketData": None,
                },
            },
        )

    task_result = results.get(task_id)
    response = TaskResponse(status="in_progress")
    if task_result is None:
        logger_api.error(f"No such task: {task_id}")
        # return TaskResponse(status="no_such_task")
        raise HTTPException(status_code=404, detail=f"no_such_task - {task_id}")
    if task_result == {}:
        return response
    response.status = "done"
    response.result = task_result
    logger_api.debug(
        f"x_request_id {x_request_id} - направлен ответ {response.model_dump()}"
    )
    return response
