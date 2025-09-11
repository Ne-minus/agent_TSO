from typing import Annotated, Sequence, TypedDict, Optional, Dict, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from contract.schemas import UserContext, UserContextNoLocation


class AgentState(TypedDict):
    # История диалога (редьюсер)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    user_info: UserContext | UserContextNoLocation
    ticket_active: bool
    ticket_data: Optional[Dict]
    awaiting_param: Optional[str]
