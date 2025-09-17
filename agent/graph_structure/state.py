from typing import Annotated, Sequence, TypedDict, Optional, Dict, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from contract.schemas import UserContext, UserContextNoLocation


class AgentState(TypedDict):
    # История диалога (редьюсер)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    user_info: UserContext | UserContextNoLocation
    ticket_active: bool = False
    ticket_data: bool | str
    awaiting_param: Optional[str]

    user_validated: bool | Literal["in progress"]
    action: Optional[
        Literal["CREATE_TICKET", "SELECT_ASUN_BUILDING", "SELECT_INNER_CLIENT"]
    ]
    change_to_name: bool
    missing_params: list | None = None
