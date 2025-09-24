from typing import Annotated, Sequence, TypedDict, Optional, Dict, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from contract.schemas import UserContext, UserContextNoLocation, AsunEntry


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    user_info: UserContext
    real_requester: UserContext
    current_building: AsunEntry
    ticket_active: bool = False
    ticket_data: bool | str
    awaiting_param: Optional[str]

    user_validated: bool | str
    parameters_to_val: list[str]
    action: Optional[
        Literal["CREATE_TICKET", "SELECT_ASUN_BUILDING", "SELECT_INNER_CLIENT"]
    ]
    missing_params: list | None = None
    we_need_to_start_params: bool
    chosen_scenario: str
    stk_insr: Dict
