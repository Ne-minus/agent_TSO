from uuid import UUID
from fastapi import APIRouter, Path, Query, Body, Header

from contract.schemas import (
    UserContext,
    CreateNewDialogRs,
    MessageToAgentRs,
    MessageToAgentRq,
)

from agent.main import AIAgent


agent = AIAgent()

router = APIRouter()


@router.post("/dialogs/", response_model=CreateNewDialogRs)
async def dialogs(
    user_context: UserContext,
    x_request_id: UUID = Header(
        ..., description="Уникальный идемнтификатор запроса. Ключ идепотентности"
    ),
):
    response = agent.create_conversation(user_context)
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
    if not message.context:
        message.context = None
    response = agent.continue_conversation(
        str(dialogId), message.message, context=message.context
    )

    return response
