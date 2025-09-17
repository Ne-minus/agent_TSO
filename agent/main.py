from typing import Any, Dict, Optional, Tuple, AsyncGenerator, Literal
from uuid import uuid4
from dotenv import load_dotenv
import json

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from agent.graph_structure.state import AgentState
from agent.graph_structure.tools import (
    user_interaction_tool,
    kb_search_tool,
    scenario_search_tool,
    get_params_tool,
    fill_params_tool,
    validate_user_tool,
)
from agent.graph_structure.graph import get_graph
from agent.model.model_init import get_llm
from contract.schemas import (
    UserContext,
    AsunEntry,
    TicketData,
    MessageToAgentRs,
    Action,
    CreateNewDialogRs,
    Name,
)

# если в build_graph уже вшит checkpointer — просто вызываем его.


def _thread_config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _combine_info_for_ticket(
    user_info: Dict, current_building: Dict, ticket_data: Dict
) -> TicketData: ...


class AIAgent:
    """
    Тонкая обёртка над LangGraph с двумя публичными методами:
      - create_conversation: первый ход, создаёт новый thread_id
      - continue_conversation: продолжение по существующему thread_id
    Возвращаем объект совместимый со схемой FinalAnswer/MessageToAgentRs.
    """

    def __init__(self):
        tools_list = [
            user_interaction_tool,
            kb_search_tool,
            scenario_search_tool,
            get_params_tool,
            fill_params_tool,
            # validate_user_tool,
        ]
        self.llm = get_llm().bind_tools(tools_list)

        self._graph = get_graph(self.llm)

        png = self._graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
        with open("graph.png", "wb") as f:
            f.write(png)
        print("Сохранено в graph.png")

    def _maybe_parse_json(self, text: str):
        t = text.strip()
        if not t:
            return None
        if not (t.startswith("{") or t.startswith("[")):
            return None
        try:
            output = json.loads(t)
            text = output.get("question") or output.get("answer") or ""
            action = output.get("action") or None
            return text, action
        except Exception:
            return None

    def _normalize_result(self, msg: BaseMessage) -> str:
        print("RUINING MESSAGE: ", msg)
        print(msg.content)
        return self._maybe_parse_json(msg.content)

    def _finalize(self, state: AgentState) -> MessageToAgentRs:
        msg: BaseMessage = state["messages"][-1]
        text, action = self._normalize_result(msg) if msg else ""
        # Готовим Action и ticketData из стейта
        # (если твои узлы пишут action в другое место — подстрой тут)
        state_action = state.get("action")
        print("ACTION: ", action)
        ticket = None
        if state_action == "CREATE_TICKET":
            action = state_action
            ticket = _combine_info_for_ticket(
                state.get("user_info"),
                state.get("current_building"),
                state.get("ticket_data"),
            )
        if action:
            action = Action(action)

        return MessageToAgentRs(message=text, action=action, ticketData=ticket)

    def create_conversation(
        self,
        user: UserContext,
    ) -> CreateNewDialogRs:
        dialog_id = str(uuid4())
        config = _thread_config(dialog_id)

        init_state: AgentState = {
            "user_info": user.model_dump(),
            "current_building": None,
            "action": None,
            "ticket_active": False,
            "ticket_data": None,
            "awaiting_param": None,
            "missing_params": None,
            "user_validated": None,
            "parameters_to_val": ["SELECT_INNER_CLIENT", "SELECT_ASUN_BUILDING"],
        }

        self._graph.update_state(config, init_state)

        init_message = f"Здравствуйте! Чем могу помочь?"
        self._graph.update_state(
            config, {"messages": [AIMessage(content=init_message)]}
        )
        return CreateNewDialogRs(dialogId=dialog_id, message=init_message)

    def continue_conversation(
        self,
        dialog_id: str,
        user_text: str,
        context: UserContext | AsunEntry | None = None,
    ) -> MessageToAgentRs:
        config = _thread_config(dialog_id)
        inputs: Dict[str, Any] = {"messages": [HumanMessage(content=user_text)]}

        if context is not None:
            ctx = context.model_dump()

            if isinstance(context, UserContext):
                self._graph.update_state(config, {"user_info": ctx})

            elif isinstance(context, AsunEntry):
                self._graph.update_state(config, {"current_building": ctx})

        result_state = self._graph.invoke(inputs, config=config)
        print("RESULT_STATE: ", result_state)
        return self._finalize(result_state)


if __name__ == "__main__":

    # Load enviroment
    load_dotenv()

    # Initialization
    agent = AIAgent()

    user = UserContext(
        name=Name(lastname="Денисова", firstname="Анна", middlename="Александровна"),
        empid="22334455",
        departamentCode="10323702",
        departamentName="Группа разработки",
    )

    building = AsunEntry(
        asunId="77", addr="г. Москва, пр-кт Кутузовский, 32 к3 стрБ, Б.05.05, Б.05.05.1"
    )

    # Handshake - get a conversation id
    answer = agent.create_conversation(user)
    dialog_id = answer.dialogId
    print(answer.message)

    # Conversation loop
    while True:
        query = input("Ваше сообщение: ")
        context = None

        if answer.action == "SELECT_INNER_CLIENT":
            if query == "да":
                context = user
            else:
                context = UserContext(
                    name=Name(
                        lastname="Неминова",
                        firstname="Екатерина",
                        middlename="Сергеевна",
                    ),
                    empid="22434455",
                    departamentCode="10393702",
                    departamentName="Группа разработки",
                )

        answer = agent.continue_conversation(str(dialog_id), query, context=context)

        print("Агент: ", answer.message)
        print()
