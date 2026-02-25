import aiohttp
import os

from langchain_core.tools import tool, InjectedToolCallId
from typing import Annotated, Dict, Any, Literal
from langgraph.prebuilt import InjectedState

from langgraph.types import Command
from langchain_core.messages import ToolMessage

from small_agent.utils.create_ticket import Formalize
from small_agent.utils.encript_logs import encript_ticketdata

form = Formalize()


@tool
async def user_interaction_tool(text: Annotated[str, "текст для пользователя"]) -> dict:
    """
    Универсальный инструмент для взаимодействия с пользователем.
    """
    return {"type": "response", "text": text}


@tool
async def format_output_tool(
    name: Annotated[str, "Имя проблемы"],
    problem_type: Annotated[str, "Категория проблемы"],
    subproblem_type: Annotated[str, "Подкатегория проблемы"],
    problem_description: Annotated[str, "Описание проблемы от пользователя"],
    short_problem: Annotated[
        Literal["ТСВ", "СКУД", "СОТС"], "Аббревиатура категории проблемы"
    ],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[dict, InjectedState] = None,
) -> dict:
    """
    Инструмент для форматирования заявки.
    Форматирует данные в структуру: Заявка: Название проблемы из предложенных, Категория и подкатегория, Комментарий: Описание проблемы пользователем.
    """

    url = os.environ["ARSENAL_URL"] + "/fault-manager/api/v1/ai-hub/create-tso-task"
    form._create_scenario_request(
        state, problem_type, subproblem_type, name, problem_description, short_problem
    )
    payload = form.create_ticket(state)

    payload_log = payload.model_dump()
    payload_log = encript_ticketdata(payload_log)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload.model_dump(exclude_none=True)
        ) as resp:
            response = await resp.json()
            status_code = resp.status

    if status_code == 200:
        task_id = response["message"]
        formatted_text = f"Спасибо за обращение! Ваша заявка переданая специалисту.\n\nЗаявка: {task_id}\nКомментарий: {problem_description}"
    else:
        formatted_text = (
            f"Кажется, что-то пошло не так... Пожалуйста, повторите попытку позже."
        )

    ticket_dict = {
        "Заявка": name,
        "Категория проблемы": problem_type,
        "Подкатегория проблемы": subproblem_type,
        "Описание проблемы": problem_description,
    }

    formatted_text = f"Спасибо за обращение! Ваша заявка переданая специалисту.\n\nЗаявка: {task_id}\nКомментарий: {problem_description}"
    return Command(
        update={
            "messages": [
                ToolMessage(
                    formatted_text,
                    tool_call_id=tool_call_id,
                )
            ],
            "ticket_dict": ticket_dict,
            "status": "DONE",
        },
    )
