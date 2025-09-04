from uuid import UUID
from fastapi import APIRouter, Path, Query, Body, Header

from contract.schemas import (
    UserContext,
    CreateNewDialogRs,
    MessageToAgentRs,
    MessageToAgentRq,
    Action,
)

from contract.schemas import User

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
    user = User(
        first_name=user_context.name.firstname,
        last_name=user_context.name.lastname,
        middle_name=user_context.name.middlename,
        tabel=user_context.empid,
    )
    response = agent.create_conversation(user)
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
    action = None
    if response.action:
        action = Action(response.action)
    response = MessageToAgentRs(
        message=response.message,
        action=action,
        ticketData=response.ticketData,
    )
    return response
