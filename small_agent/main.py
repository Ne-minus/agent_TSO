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
from small_agent.graph_structure.state import State
from small_agent.graph_structure.tools import (
    user_interaction_tool,
    format_output_tool,
)
from small_agent.graph_structure.graph import build_graph
from agent.model.model_init import get_llm
from contract.schemas_dnie import Response, CreateRunPayload

# from agent.utils.create_ticket import Formalize
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


class AIAgent_DNIE:
    """
    Тонкая обёртка над LangGraph с двумя публичными методами:
      - create_conversation: первый ход, создаёт новый thread_id
      - continue_conversation: продолжение по существующему thread_id
    Возвращаем объект совместимый со схемой FinalAnswer/MessageToAgentRs.
    """

    def __init__(self):
        tools_list = [user_interaction_tool, format_output_tool]
        self.llm = get_llm().bind_tools(tools_list)

        self._graph = build_graph(self.llm)

    def _maybe_parse_json(self, text: str):
        t = text.strip()
        if not t:
            return None
        try:
            output = json.loads(t)
            text = output.get("text")
        except:
            text = t
        return text

    def _normalize_result(self, msg: BaseMessage) -> str:
        return self._maybe_parse_json(msg)

    def _finalize(self, state: State, thread_id) -> Response:
        msg: BaseMessage = state["messages"][-1]
        print("MESSAGE: ", msg)
        text = self._normalize_result(msg.content) if msg else ""
        text = re.sub("\\n", "<br />", md.render(text))
        text = re.sub("\\\\n", "<br />", text)

        status = state.get("status")
        print(status)
        return Response(thread_id=thread_id, content=text, status=status)

    def _init_state(self, input_data: CreateRunPayload) -> State:

        if input_data.additional_data:
            init_state: State = {
                "user_info": input_data.additional_data.author,
                "user_position": input_data.additional_data.position,
                "user_department": input_data.additional_data.department,
                "address": input_data.additional_data.address,
                "status": "IN_PROGRESS",
                "messages": [],
                "ticket_dict": {},
            }
        else:
            init_state: State = {
                "user_info": None,
                "user_position": None,
                "user_department": None,
                "address": None,
                "status": "IN_PROGRESS",
                "messages": [],
                "ticket_dict": {},
            }

        return init_state

    async def conversation(self, input_data: CreateRunPayload):
        config = _thread_config(input_data.thread_id)
        user_text = input_data.input
        inputs: Dict[str, Any] = {"messages": [HumanMessage(content=user_text)]}

        if not self._graph.get_state(config).values:
            self._graph.update_state(config, self._init_state(input_data))

        result_state = await self._graph.ainvoke(inputs, config=config)
        return self._finalize(result_state, input_data.thread_id)


if __name__ == "__main__":
    load_dotenv()

    agent = AIAgent_DNIE()

    async def main():
        # Инициализация диалога
        answer = await agent.create_conversation(user)
        dialog_id = answer.dialogId
        print(answer.message)

        # Основной цикл общения
        while True:
            query = input("Ваше сообщение: ")
            context = None

            # ⚠️ Асинхронный вызов агента
            answer = await agent.continue_conversation(
                str(dialog_id), query, context=context
            )

            print("Агент:", answer.message)
            print()

    asyncio.run(main())
