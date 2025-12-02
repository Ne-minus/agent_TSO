import json
import re
import logging
import uuid
import asyncio

from typing import Set, List, Sequence
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command

from agent.utils.extract_params import ParamExtractor
from agent.utils.structured_outputs import Decision, NodeRouting
from agent.graph_structure.state import AgentState
from agent.prompts.prompts import (
    create_system_prompt,
    get_ticket_prompt,
    get_react_instructions,
    get_formatting_prompt,
    get_scenario_prompt,
    get_tsv_prompt,
    get_ts_prompt,
    get_validation_prompt,
)

from agent.graph_structure.tools import (
    user_interaction_tool,
    ask_user_with_action_tool,
    check_archive_tool,
    scenario_search_tool,
    scenario_ts_search_tool,
    get_params_tool,
    fill_params_tool,
    format_output_tool,
    GENERAL_TOOLS,
    TICKET_TOOLS,
    # SCENARIO_TOOLS,
)


logger = logging.getLogger("nodes")


TICKET_TOOL_NAMES: Set[str] = {t.name for t in TICKET_TOOLS if hasattr(t, "name")}
GENERAL_TOOL_NAMES: Set[str] = {t.name for t in GENERAL_TOOLS if hasattr(t, "name")}


TOOLS_BY_NAME = {t.name: t for t in (GENERAL_TOOLS + TICKET_TOOLS)}


def _last_tool_message(state: AgentState):
    for m in reversed(list(state.get("messages", []))):
        if isinstance(m, ToolMessage):
            return m
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


def _compose_prompt(if_ticket: bool = False, extra: str = "") -> str:
    """System prompt with extra instuctions of needed."""
    if if_ticket:
        system_prompt = get_ticket_prompt()
    else:
        system_prompt = create_system_prompt()
    return system_prompt + get_react_instructions() + (("\n" + extra) if extra else "")


def _choose_other_ticket(node_name: str) -> str:
    if "tsv" in node_name:
        return "Иные неисправности СКУД"
    elif "skud" in node_name:
        return "Иные неисправности ТСВ"


async def ticket_reflect_node(state: AgentState, config: RunnableConfig, model):
    """
    Reflection during the process of ticket filling.
    Both general and ticket tools are allowed.
    After using general tools, we go back here till filling process is active.
    """

    messages = list(state["messages"])
    system = SystemMessage(_compose_prompt(if_ticket=True))

    if state.get("choice_in_progress"):
        return Command(goto=state.get("node_name"))

    if state.get("awaiting_fallback_confirmation"):
        other_ticket = _choose_other_ticket(state.get("node_name"))

        # from agent.utils.extract_params import ParamExtractor
        extractor = ParamExtractor(state.get("llm"), state)
        user_response = await extractor.simple_extraction(
            f"Продолжим с заявкой {other_ticket}?"
        )

        if user_response == "положительно":
            # Пользователь подтвердил, переходим к заполнению параметров
            messages.append(
                HumanMessage(
                    content=f"Пользователь подтвердил создание заявки {other_ticket}. Вызови get_params_tools c данной заявкой"
                )
            )
            resp = await model.bind_tools([get_params_tool]).ainvoke(
                [system] + messages, config
            )
            return {"messages": [resp], "awaiting_fallback_confirmation": False}

        elif user_response == "отрицательно":
            # Пользователь отказался
            resp = AIMessage(
                content="Хорошо, если у вас возникнут вопросы — обращайтесь снова.",
                tool_calls=[
                    {
                        "name": "user_interaction_tool",
                        "args": {
                            "text": "Хорошо, если у вас возникнут вопросы — обращайтесь снова.",
                            "mode": "answer",
                        },
                        "id": f"call_{uuid.uuid4()}",
                        "type": "tool_call",
                    }
                ],
            )
            return {"messages": [resp], "awaiting_fallback_confirmation": False}
        else:
            # Ещё не ответил, продолжаем ждать
            return {}

    # AFTER WE GOT TO THE END OF THE TREE
    if state.get("ticket_name_chosen") and state.get("if_comment"):
        messages += f"Нам не нужно заводить заявку, даем пользователю подсказку с обращением в стороннюю систему. Вызови user_interaction_tool с mode='answer', где тебе нужно будет сказать, что ты понял запрос пользователя и  по данной проблеме нужно завести заявку в другой системе: {state.get('if_comment')}."
        resp = await model.bind_tools([user_interaction_tool]).ainvoke(
            [system] + messages, config
        )

        return {"messages": [resp], "ticket_name_chosen": None}

    if state.get("parameters_to_val") != [] and state.get("user_validated") == False:

        return Command(goto="validate_user_node")

    if state.get("we_need_to_start_params"):
        messages += [
            f"\nСейчас нужно начать запрашивать параметры по заявке. Обязательно вызови fill_param_tools()\n"
        ]

        resp = await model.bind_tools([fill_params_tool]).ainvoke(
            [system] + messages, config
        )

        return {
            "messages": [resp],
            "we_need_to_start_params": False,
        }

    elif state.get("awaiting_param") is not None:
        if state.get("missing_params"):

            messages += [
                f"\nСейчас нужно заполнить параметр {state.get('awaiting_param')} через user_interaction_tool. Далее вызови fill_param_tools(), так как  остались незаполненными другие необходимые параметры. Процесс заполнения заявки завершать НЕЛЬЗЯ!\n"
            ]

        resp = await model.bind_tools(
            [fill_params_tool, user_interaction_tool] + GENERAL_TOOLS
        ).ainvoke([SystemMessage(get_formatting_prompt())] + messages, config)

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
    resp = await model.bind_tools(
        [
            get_params_tool,
            fill_params_tool,
            ask_user_with_action_tool,
        ]
        + GENERAL_TOOLS
    ).ainvoke([system] + messages, config)

    if "route" in resp.content:
        last_user_msg = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            None,
        )
        node_name = re.search(r"<route>(.+?)<\/route>", resp.content).group(1)
        print(node_name)
        return Command(
            goto=node_name,
            update={
                "last_user_message": last_user_msg,
                "choice_in_progress": True,
                "node_name": node_name,
            },
        )
    # logger.debug(f"TICKET RESPONSE: {resp}")

    return {"messages": [resp]}


async def validate_user_node(state: AgentState, config: RunnableConfig, model):
    messages = list(state["messages"])
    system = SystemMessage(get_validation_prompt())

    resp = await model.bind_tools(
        [user_interaction_tool, ask_user_with_action_tool]
    ).ainvoke([system] + messages, config)
    print(resp)

    return {"messages": [resp]}


async def scenario_skud_node(state: AgentState, config: RunnableConfig, model):
    messages = list(state["messages"])
    if state["ticket_not_started"]:
        messages += [
            f"\nСейчас нужно задать пользователю дополнительные вопросы. Для этого вызови scenario_search_tool(entrypoint='<НЕОБХОДИМЫЙ ВОПРОС>'). Заполнение параметра entrypoint зависит от запроса пользователя."
        ]
        system = get_scenario_prompt()
        resp = await model.bind_tools([scenario_search_tool]).ainvoke(
            [system] + messages, config
        )
    else:
        messages += [
            f"\nЕсли ты ранее получил вопрос из scenario_search_tool, но не задал его пользователю, то нужно спросить у пользователя ответ на этот помощью user_interaction_tool. Если пользователь тебе ответил, далее вызови scenario_search_tool() без каких-либо аргументов, чтобы продолжить задавать вопросы."
        ]
        system = get_scenario_prompt()
        resp = await model.bind_tools(
            [scenario_search_tool, user_interaction_tool]
        ).ainvoke([system] + messages, config)

    # logger.debug(f"SCENARIO RESPONSE: {resp}")
    return {"messages": [resp], "choice_in_progress": True}


# async def scenario_ts_node(state: AgentState, config: RunnableConfig, model):
#     messages = list(state["messages"])
#     if state["ticket_not_started"]:
#         messages += [
#             f"\nСейчас нужно задать пользователю дополнительные вопросы. Для этого вызови scenario_ts_search_tool(entrypoint='<НЕОБХОДИМЫЙ ВОПРОС>'). Заполнение параметра entrypoint зависит от запроса пользователя."
#         ]
#         system = get_ts_prompt()
#         resp = await model.bind_tools([scenario_ts_search_tool]).ainvoke(
#             [system] + messages, config
#         )
#     else:
#         messages += [
#             f"\nЕсли ты ранее получил вопрос из scenario_ts_search_tool, но не задал его пользователю, то нужно спросить у пользователя ответ на этот помощью user_interaction_tool. Если пользователь тебе ответил, далее вызови scenario_ts_search_tool() без каких-либо аргументов, чтобы продолжить задавать вопросы."
#         ]
#         system = get_ts_prompt()
#         resp = await model.bind_tools(
#             [scenario_ts_search_tool, user_interaction_tool]
#         ).ainvoke([system] + messages, config)

#     # logger.debug(f"SCENARIO RESPONSE: {resp}")
#     return {"messages": [resp], "choice_in_progress": True}


async def scenario_tsv_node(state: AgentState, config: RunnableConfig, model):
    messages = list(state["messages"])
    system = SystemMessage(get_tsv_prompt())

    llm = model.bind_tools(
        [user_interaction_tool, check_archive_tool, format_output_tool]
    )

    response = await llm.ainvoke([system] + messages, config)

    return {"messages": [response], "choice_in_progress": True}


async def scenario_ts_node(state: AgentState, config: RunnableConfig, model):
    messages = list(state["messages"])
    system = SystemMessage(get_ts_prompt())

    llm = model.bind_tools([user_interaction_tool, format_output_tool])

    response = await llm.ainvoke([system] + messages, config)
    print(response)

    return {"messages": [response], "choice_in_progress": True}


async def await_user_node(state: AgentState, *_):
    """
    Останавливает граф, спрашивает пользователя и ждёт resume с {"answer": "..."}.
    """
    question = _extract_question(state) or "Пожалуйста, ответьте на вопрос."
    payload = interrupt({"question": question})
    answer = payload.get("answer")
    if not answer:
        return {}

    # Превращаем ответ в HumanMessage и продолжаем граф.
    return {"messages": [HumanMessage(answer)]}


### -----------------------
### EDGES
### -----------------------


def should_route_scenario_reflect(state: AgentState):
    """
    Куда идти после Reflection:
      - если LLM вызвала ticket-инструмент → сначала в Ticket node (а не сразу в ToolNode),
      - если вызвала общий инструмент → в общий ToolNode,
      - если не было tool_calls → END.
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return "end"
    for c in calls:
        if c["name"] in TICKET_TOOL_NAMES:
            return "use_ticket_tool"
    return "use_general_tool"


def should_route_after_ticket_reflect(state: AgentState):
    last = state["messages"][-1]
    content = getattr(last, "content", "")
    if isinstance(content, str) and "scenario_node" in content:
        return "scenario_node"

    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return "end"

    names = {c["name"] for c in calls}
    if names & TICKET_TOOL_NAMES:
        return "use_ticket_tool"
    if names & GENERAL_TOOL_NAMES:
        return "use_general_tool"

    return "end"


def should_continue_after_ticket_tool(state: AgentState):
    """
    После выполнения тикетных инструментов:
    - если задался вопрос пользователю → await_user
    - если финализация → end
    - иначе → вернуться в ticket
    """
    # финализация
    for m in reversed(list(state.get("messages", []))):
        if isinstance(m, ToolMessage) and getattr(m, "name", "") in {
            "ticket_finalize",
            "finalize",
        }:
            data = m.content
            if not isinstance(data, dict):
                try:
                    data = json.loads(data)
                except Exception:
                    data = {}
            if isinstance(data, dict) and data.get("ready") is True:
                return "end"
            break

    if _was_question_asked(state):
        return "await_user"

    return "ticket"


def after_general_tool(state: AgentState):
    """
    После общих инструментов:
    - если спросили пользователя → await_user
    - если активен тикет → ticket
    - иначе → reflect
    """
    # ADDED:
    last = state["messages"][-1]
    content = getattr(last, "content", "")
    if _was_question_asked(state):
        return "await_user"

    if "scenario_tsv_node" in content:
        return "scenario_tsv_node"
    elif "scenario_skud_node" in content:
        return "scenario_skud_node"
    elif "scenario_ts_node" in content:
        return "scenario_ts_node"

    return "ticket"
