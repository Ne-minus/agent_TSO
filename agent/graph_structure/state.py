from typing import Annotated, Sequence, TypedDict, Optional, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # История диалога (редьюсер)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Лёгкий флаг режима «идёт оформление заявки».
    # Основная «память» заявки ведётся внутри ticket_* тулов (по memory_key="default"),
    # но этот флаг нужен для маршрутизации из общей ветки обратно в Ticket node.
    ticket_active: bool
    active_scenario_id: Optional[str]
    filled: Dict[str, str]
