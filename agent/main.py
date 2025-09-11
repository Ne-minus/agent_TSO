from typing import Any, Dict, Optional, Tuple, AsyncGenerator, Literal
from uuid import uuid4

from langchain_core.messages import HumanMessage, BaseMessage
from agent.graph_structure.state import AgentState
from agent.graph_structure.graph import build_graph
from contract.schemas import UserContext, AsunEntry, FinalAnswer

# если в build_graph уже вшит checkpointer — просто вызываем его.
GRAPH = build_graph()


def _thread_config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


class AIAgent:
    """
    Тонкая обёртка над LangGraph с двумя публичными методами:
      - create_conversation: первый ход, создаёт новый thread_id
      - continue_conversation: продолжение по существующему thread_id
    Возвращаем объект совместимый со схемой FinalAnswer/MessageToAgentRs.
    """

    def __init__(self):
        self._graph = GRAPH

    def _finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Берём последнее ассистентское сообщение как финальный ответ
        msg: BaseMessage = state["messages"][-1]
        text = getattr(msg, "content", "") if msg else ""
        # Готовим Action и ticketData из стейта
        # (если твои узлы пишут action в другое место — подстрой тут)
        action = state.get("action")
        ticket = state.get("ticket_data")
        return {
            "message": text,
            "action": action,
            "ticketData": ticket,
        }

    def create_conversation(
        self,
        user_text: str,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        dialog_id = str(uuid4())
        config = _thread_config(dialog_id)

        init_state: AgentState = {
            "messages": [HumanMessage(content=user_text)],
            "user_info": context,  # UserContext | UserContextNoLocation
            "action": Optional[
                Literal["GiveAsunObject", "GiveContact", "CreateTicket"]
            ],
            "ticket_active": False,
            "ticket_data": None,
            "awaiting_param": None,
        }
        if metadata:
            init_state["metadata"] = metadata

        result_state = self._graph.invoke(init_state, config=config)
        final = self._finalize(result_state)
        return dialog_id, final

    def continue_conversation(
        self,
        dialog_id: str,
        user_text: str,
        context: UserContext | AsunEntry | None = None,
    ) -> FinalAnswer:
        config = _thread_config(dialog_id)
        # достаточно передать только новое HumanMessage — редьюсер add_messages сам смёрджит историю
        inputs: Dict[str, Any] = {"messages": [HumanMessage(content=user_text)]}

        if isinstance(context, UserContext):
            inputs["user_info"] = context
        elif isinstance(context, AsunEntry):
            inputs["user_info"].workPlaceLocation = context

        result_state = self._graph.invoke(inputs, config=config)
        return self._finalize(result_state)
