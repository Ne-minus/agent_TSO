import json
import uuid

from typing import Set, List, Sequence, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command

from agent.graph_structure.state import AgentState
from agent.prompts.prompts import (
    create_system_prompt,
    get_ticket_prompt,
    get_react_instructions,
    get_scenario_validation_prompt,
    get_formatting_prompt,
    get_scenario_prompt,
)

from agent.graph_structure.tools import (
    user_interaction_tool,
    ask_user_with_action_tool,
    kb_search_tool,
    scenario_search_tool,
    get_params_tool,
    fill_params_tool,
    GENERAL_TOOLS,
    TICKET_TOOLS,
)

from agent.utils.structured_output import FinalResponse


TICKET_TOOL_NAMES: Set[str] = {t.name for t in TICKET_TOOLS if hasattr(t, "name")}
GENERAL_TOOL_NAMES: Set[str] = {t.name for t in GENERAL_TOOLS if hasattr(t, "name")}


TOOLS_BY_NAME = {t.name: t for t in (GENERAL_TOOLS + TICKET_TOOLS)}


def _last_tool_message(state: AgentState):
    for m in reversed(list(state.get("messages", []))):
        if isinstance(m, ToolMessage):
            return m
    return None


def _extract_question(state: AgentState) -> str | None:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, ToolMessage):
            try:
                data = (
                    json.loads(m.content) if isinstance(m.content, str) else m.content
                )
            except Exception:
                continue
            if isinstance(data, dict) and data.get("type") == "ask_user":
                return data.get("question")
    return None


def _was_question_asked(state: AgentState) -> bool:
    tm = _last_tool_message(state)
    if not tm:
        return False
    data = tm.content
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return False
    return (
        isinstance(data, dict) and data.get("type") == "ask_user" and "question" in data
    )


class Solution(BaseModel):
    """Joke to tell user."""

    intent: Literal["create_ticket", "other"] = Field(
        description="То, что пользоватеь хочет сделать (завести заявку или просто получить ответ на вопрос)"
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Имя необходимого инструмента, если пользователь хочет получить ответ на вопрос.",
    )


## NODES


def _compose_prompt(if_ticket: bool = False, extra: str = "") -> str:
    """System prompt with extra instuctions of needed."""
    if if_ticket:
        system_prompt = get_ticket_prompt()
    else:
        system_prompt = create_system_prompt()
    return system_prompt + (("\n" + extra) if extra else "")


def reflect_node(state: AgentState, config: RunnableConfig, model):
    messages = list(state["messages"])
    system = SystemMessage(_compose_prompt())

    if state.get("choice_in_progress"):
        return Command(goto="scenario_node")

    resp = model.bind_tools(GENERAL_TOOLS).invoke([system] + messages, config)

    if resp.content == "scenario_node":
        return {"messages": state["messages"] + [AIMessage(content="scenario_node")]}

    return {"messages": [resp]}


def await_user_node(state: AgentState, *_):
    """
    Останавливает граф, спрашивает пользователя и ждёт resume с {"answer": "..."}.
    """
    # print("WE ARE WAITING FOR USER")

    question = _extract_question(state) or "Пожалуйста, ответьте на вопрос."
    payload = interrupt({"question": question})

    answer = payload.get("answer")
    # print("THIS IS PAYLOAD: ", payload)
    if not answer:
        return {}

    # Превращаем ответ в HumanMessage и продолжаем граф.
    return {"messages": [HumanMessage(answer)]}


def scenario_node(state: AgentState, config: RunnableConfig, model):
    print("We've got here")
    messages = list(state["messages"])
    print(messages[-1])
    # print(messages[-3:])
    if state["ticket_not_started"]:
        print("WE ARE HERE")
        messages += [
            f"\nСейчас нужно задать пользователю дополнительные вопросы. Для этого вызови search_scenario_tool(entrypoint='<НЕОБХОДИМЫЙ ВОПРОС>'). Заполнение параметра entrypoint зависит от запроса пользователя."
        ]
        system = get_scenario_prompt()
        resp = model.bind_tools([scenario_search_tool]).invoke(
            [system] + messages, config
        )
        print("FIRST ITER RESPNSE", resp)
    else:
        messages += [
            f"\nЕсли ты ранее получил вопрос из search_scenario_tool, но не задал его пользователю, то нужно спросить у пользователя ответ на этот помощью user_interaction_tool. Если пользователь тебе ответил, далее вызови search_scenario_tool(), не передавая никаких аргументов, чтобы продолжить задавать вопросы. Не нужно ответ пользователя передавать в инструмент."
        ]
        system = get_scenario_prompt()
        resp = model.bind_tools([scenario_search_tool, user_interaction_tool]).invoke(
            [system] + messages, config
        )

        if resp.additional_kwargs["function_call"]["name"] == "scenario_search_tool":
            resp.additional_kwargs["function_call"]["arguments"] = {}

        print("HERE'S THE RESPONSE: ", resp)

    # возвращаем всё сразу
    return {"messages": [resp], "choice_in_progress": True}


def validate_scenario_node(state: AgentState, config: RunnableConfig, model):
    messages = list(state["messages"])
    print("WE VALIDATE TICKET")
    system = SystemMessage(get_scenario_validation_prompt())

    resp = model.bind_tools([user_interaction_tool, get_params_tool]).invoke(
        [system] + messages, config
    )
    print(resp)
    if "no" in resp.content:
        return {"messages": [resp], "ticket_name_chosen": "Иные неисправности СКУД"}

    return {"messages": [resp]}


def validate_user_node(state: AgentState, config: RunnableConfig, model):
    messages = list(state["messages"])
    system = SystemMessage(_compose_prompt())
    if state.get("parameters_to_val") != [] and state.get("user_validated") == False:

        mixture = {
            "SELECT_ASUN_BUILDING": "По какому адресу вы создаете заявку?",
            "SELECT_INNER_CLIENT": "Вы заводите заявку от своего имени?",
        }

        params_to_val = state.get("parameters_to_val")
        user_validated = state.get("user_validated")
        for param in params_to_val:
            messages += [
                f"\nСейчас нужно провалидировать данные пользователя. Для этого вызови ask_user_with_action_tool(text='{mixture[param]}', action='{param}'). Нельзя продролжать заполнение заявки."
            ]

            # print("ASK FOR ACTION: ", messages[-1])

            resp = model.bind_tools([ask_user_with_action_tool]).invoke(
                [system] + messages + messages, config
            )
            params_to_val.pop(0)
            if params_to_val == []:
                user_validated = True

            return {
                "messages": [resp],
                "parameters_to_val": params_to_val,
                # "user_validated": user_validated,
            }


def fill_ticket_node(state: AgentState, config: RunnableConfig, model):
    messages = list(state["messages"])
    system = SystemMessage(_compose_prompt())
    if state.get("we_need_to_start_params"):
        messages += [
            f"\nСейчас нужно начать запрашивать параметры по заявке. Обязательно вызови fill_param_tools()\n"
        ]

        resp = model.bind_tools([get_params_tool, fill_params_tool]).invoke(
            [system] + messages, config
        )

        return {
            "messages": [resp],
            "we_need_to_start_params": False,
        }

    elif state.get("awaiting_param") is not None:
        # print("WE FILL PARAMS")

        # print("Now we check missing")
        if state.get("missing_params"):
            # print("Now we check missing")

            messages += [
                f"\nСейчас нужно заполнить параметр {state.get('awaiting_param')} через user_interaction_tool. Далее вызови fill_param_tools(), так как  остались незаполненными другие необходимые параметры. Процесс заполнения заявки завершать НЕЛЬЗЯ!\n"
            ]

        resp = model.bind_tools(
            [fill_params_tool, user_interaction_tool] + GENERAL_TOOLS
        ).invoke([SystemMessage(get_formatting_prompt())] + messages, config)

        parameters_to_fill = state.get("ticket_data")

        new_missing = state.get("missing_params")
        if new_missing:
            new_missing.pop(0)
        return {
            "messages": [resp],
            "ticket_active": True,
            "ticket_data": parameters_to_fill,
            "missing_params": new_missing,
        }


## EDGES


def should_route_after_reflect(state: AgentState):
    """
    После reflect:
      - если есть tool_calls → use_general_tool
      - если модель хочет перейти в сценарий → scenario_node
      - иначе → end
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []

    if calls:
        return "use_general_tool"

    if "scenario_node" in getattr(last, "content", ""):
        return "scenario_node"

    return "end"


def after_general_tool(state: AgentState):
    """
    После выполнения общих инструментов:
      - если спросили пользователя → await_user
      - если активен сценарий → scenario_node
      - если нужно проверить заявку → validate_scenario
      - если нужно заполнить тикет → fill_ticket
      - если снова вызвали инструмент → use_general_tool
      - иначе → reflect
    """
    last = state["messages"][-1]
    content = getattr(last, "content", "")
    calls = getattr(last, "tool_calls", None) or []

    # Пользователю задан вопрос
    if _was_question_asked(state):
        return "await_user"

    # Активен сценарий (идём обратно)
    if state.get("choice_in_progress"):
        return "scenario_node"

    # Просьба проверить заявку
    if state.get("ticket_name_chosen"):
        if state.get("if_comment"):
            return "reflect"
        else:
            return "validate_scenario"

    if state.get("ticket_name_final"):
        ...

    # Начало заполнения тикета
    if "fill_ticket" in content.lower():
        return "fill_ticket"

    # Ещё один tool_call
    if calls:
        return "use_general_tool"

    # Возврат к размышлению
    return "reflect"


def after_scenario_node(state: AgentState):
    """
    После сценарного этапа:
      - если вызвали инструмент → use_general_tool
      - иначе → validate_scenario
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []

    if calls:
        return "use_general_tool"

    return "validate_scenario"


def after_validate_scenario(state: AgentState):
    """
    После validate_scenario:
      - если вызван инструмент → use_general_tool
      - если пользователь подтвердил заявку → validate_user
      - иначе → end
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []

    if calls:
        return "use_general_tool"

    # Проверка structured output или контента
    try:
        if isinstance(last, dict):
            resp = last.get("final_output", {}).get("response")
            if resp:
                return "validate_user"
        elif hasattr(last, "content") and "подходит" in last.content.lower():
            return "validate_user"
    except Exception:
        pass

    return "end"


def after_validate_user(state: AgentState):
    """
    После validate_user:
      - если вызван инструмент → use_general_tool
      - если есть ещё параметры для проверки → остаёмся
      - иначе → fill_ticket
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    params_left = state.get("parameters_to_val") or []

    if calls:
        return "use_general_tool"

    if params_left:
        return "validate_user"

    return "fill_ticket"


def after_fill_ticket(state: AgentState):
    """
    После fill_ticket:
      - если был вызван инструмент → use_general_tool
      - если задали вопрос пользователю → await_user
      - иначе → end
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []

    if calls:
        return "use_general_tool"

    if _was_question_asked(state):
        return "await_user"

    return "end"


def after_await_user(state: AgentState):
    """
    После ответа пользователя:
      - если вызвали инструмент → use_general_tool
      - иначе → reflect
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []

    if calls:
        return "use_general_tool"

    return "reflect"
