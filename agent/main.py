import logging
import re
import asyncio
from typing import Any, Dict, Optional, Tuple, AsyncGenerator, Literal
from uuid import uuid4
from dotenv import load_dotenv
from markdown_it import MarkdownIt
import json

from datetime import datetime

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from agent.graph_structure.state import AgentState
from agent.graph_structure.tools import (
    user_interaction_tool,
    kb_search_tool,
    scenario_search_tool,
    get_params_tool,
    fill_params_tool,
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
from agent.utils.create_ticket import Formalize
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("agent")

logging.basicConfig(level=logging.DEBUG)
md = MarkdownIt()


class RawHTTPCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.time = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.time = datetime.now()
        logger.debug(f"Отправка запроса в GigaChat")

    def on_llm_end(self, response, **kwargs):
        self.time = datetime.now() - self.time
        logger.debug(
            f"GigaChat ответил - x-request-id: {response.llm_output['x_headers']['x-request-id']} - time: {self.time} - prompt_tokens: {response.llm_output['token_usage']['prompt_tokens']} - total_tokens: {response.llm_output['token_usage']['total_tokens']}"
        )


def _thread_config(thread_id: str) -> Dict[str, Any]:
    # return {"configurable": {"thread_id": thread_id}}
    return {"configurable": {"thread_id": thread_id}, "callbacks": [RawHTTPCallback()]}


def _combine_info_for_ticket(state: AgentState) -> TicketData:
    user = f"<b>Заявитель:</b> {state.get('real_requester')._compile_name()}<br />"
    building = f"<b>Адрес объекта:</b> {state.get('current_building').addr}<br />"
    malfunction = f"<b>Неисправность:</b> {state.get('chosen_scenario').description}"
    ticket = f"<b>Выбранная заявка:</b> {state.get('chosen_scenario').scenario}<br />"

    params_str = ""
    params = state.get("ticket_data")
    for param in params:
        # logging.info("+" * 20)
        # logging.info(param)
        # logging.info(param.get("value"))
        # logging.info("+" * 20)
        params_str += (
            f"<b>{params[param]['description']}</b>: {params[param]['value']}<br />"
        )
    final_ticket = f"{user}<br />{building}<br />{malfunction}<br />{ticket}<br />{params_str}<br />"

    return final_ticket


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
        try:
            output = json.loads(t)
            text = output.get("question") or output.get("answer") or ""
            action = output.get("action") or None
            return text, action
        except Exception:
            return text, None

    def _normalize_result(self, msg: BaseMessage) -> str:
        # print("RUINING MESSAGE: ", msg)
        return self._maybe_parse_json(msg)

    def _finalize(self, state: AgentState) -> MessageToAgentRs:
        msg: BaseMessage = state["messages"][-1]
        # print("MESSAGE: ", type(msg))
        text, action = self._normalize_result(msg.content) if msg else ""
        text = re.sub("\\n", "<br />", md.render(text))
        text = re.sub("\\\\n", "<br />", text)
        # Готовим Action и ticketData из стейта
        # (если твои узлы пишут action в другое место — подстрой тут)
        state_action = state.get("action")
        # print("ACTION: ", action)
        # print("STATE ACTION: ", state_action)
        ticket_data = None
        if state_action == "CREATE_TICKET":
            action = state_action
            form = Formalize()
            ticket_data = form.create_ticket(state)
            text += f"<br />{_combine_info_for_ticket(state)}"

        if action:
            action = Action(action)

        return MessageToAgentRs(message=text, action=action, ticketData=ticket_data)

    async def create_conversation(
        self,
        user: UserContext,
    ) -> CreateNewDialogRs:
        dialog_id = str(uuid4())
        config = _thread_config(dialog_id)

        init_state: AgentState = {
            "user_info": user,
            "real_requester": user,
            "current_building": None,
            "action": None,
            "ticket_active": False,
            "ticket_data": None,
            "awaiting_param": None,
            "missing_params": None,
            "user_validated": None,
            "parameters_to_val": ["SELECT_INNER_CLIENT", "SELECT_ASUN_BUILDING"],
            "we_need_to_start_params": False,
            "chosen_scenario": "",
            "stk_insr": "",
            "ticket_name_chosen": None,
            "if_comment": None,
            "curr_question": None,
            "last_asked_question": None,
            "ticket_not_started": True,
            "choice_in_progress": False,
            "awaiting_fallback_confirmation": False,
            "messages": [AIMessage(content="Здравствуйте! Чем могу помочь?")],
            "last_user_message": "",
            "node_name": "",
        }

        self._graph.update_state(config, init_state)

        # init_message = f"Здравствуйте! Чем могу помочь?"
        # self._graph.update_state(
        #     config, {"messages": [AIMessage(content=init_message)]}
        # )
        return CreateNewDialogRs(
            dialogId=dialog_id, message="Здравствуйте! Чем могу помочь?"
        )

    async def continue_conversation(
        self,
        dialog_id: str,
        user_text: str,
        context: UserContext | AsunEntry | None = None,
    ) -> MessageToAgentRs:
        config = _thread_config(dialog_id)
        inputs: Dict[str, Any] = {"messages": [HumanMessage(content=user_text)]}

        if context is not None:
            # ctx = context.model_dump()

            if isinstance(context, UserContext):
                user_initial = self._graph.get_state(config)
                if user_initial == context:
                    self._graph.update_state(config, {"user_info": context})
                else:
                    self._graph.update_state(config, {"real_requester": context})

            elif isinstance(context, AsunEntry):
                self._graph.update_state(config, {"current_building": context})

        result_state = await self._graph.ainvoke(inputs, config=config)
        return self._finalize(result_state)


if __name__ == "__main__":
    load_dotenv()

    agent = AIAgent()

    user = UserContext(
        name=Name(lastname="Денисова", firstname="Анна", middlename="Александровна"),
        empid="22334455",
        departamentCode="10323702",
        departamentName="Группа разработки",
        gosbCode="ГОСБ",
        terbankCode="Волговятский банк",
    )

    building = AsunEntry(
        asunId="77", addr="г. Москва, пр-кт Кутузовский, 32 к3 стрБ, Б.05.05, Б.05.05.1"
    )

    async def main():
        # Инициализация диалога
        answer = await agent.create_conversation(user)
        dialog_id = answer.dialogId
        print(answer.message)

        # Основной цикл общения
        while True:
            query = input("Ваше сообщение: ")
            context = None

            # === обработка выбора клиента ===
            if answer.action == "SELECT_INNER_CLIENT":
                if query.strip().lower() in ["да", "ага", "верно"]:
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

            # === обработка выбора здания ===
            elif answer.action == "SELECT_ASUN_BUILDING":
                context = AsunEntry(
                    asunId="77",
                    addr="г. Москва, пр-кт Кутузовский, 32 к3 стрБ, Б.05.05, Б.05.05.1",
                )

            # ⚠️ Асинхронный вызов агента
            answer = await agent.continue_conversation(
                str(dialog_id), query, context=context
            )

            print("Агент:", answer.message)
            print()

    asyncio.run(main())
