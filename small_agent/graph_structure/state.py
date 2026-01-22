from typing import TypedDict
from typing import Annotated, Sequence, Literal, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from contract.schemas_dnie import Address, Person


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    status: Literal["IN_PROGRESS", "DONE"]
    ticket_dict: dict

    user_info: Optional[Person]
    user_position: Optional[str]
    user_department: Optional[str]
    address: Optional[Address]
