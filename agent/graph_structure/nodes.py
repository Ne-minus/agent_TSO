import json
import uuid

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

from agent.graph_structure.state import AgentState
from agent.prompts.prompts import (
    create_system_prompt,
    get_ticket_prompt,
    get_react_instructions,
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


def run_tools_and_wrap(ai_msg: AIMessage) -> list[ToolMessage]:
    """Выполнить все tool_calls из AIMessage и вернуть список ToolMessage."""
    tool_msgs: list[ToolMessage] = []
    for tc in ai_msg.tool_calls or []:
        name = tc["name"]
        args = tc.get("args", {}) or {}
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            raise RuntimeError(f"LLM вызвал неизвестный инструмент: {name}")

        try:
            result = tool.invoke(args) if hasattr(tool, "invoke") else tool.func(**args)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        if not isinstance(result, str):
            try:
                content = json.dumps(result, ensure_ascii=False)
            except Exception:
                content = str(result)
        else:
            content = result

        tool_msgs.append(
            ToolMessage(
                content=content,
                tool_call_id=tc["id"],
                name=name,
            )
        )
    return tool_msgs


def _compose_prompt(if_ticket: bool = False, extra: str = "") -> str:
    """System prompt with extra instuctions of needed."""
    if if_ticket:
        system_prompt = get_ticket_prompt()
    else:
        system_prompt = create_system_prompt()
    return system_prompt + get_react_instructions() + (("\n" + extra) if extra else "")


def ticket_reflect_node(state: AgentState, config: RunnableConfig, model):
    """
    Reflection during the process of ticket filling.
    Both general and ticket tools are allowed.
    After using general tools, we go back here till filling process is active.
    """

    messages = list(state["messages"])
    system = SystemMessage(_compose_prompt(if_ticket=True))

    print("FLAG FOR SCENARIO:", state.get("choice_in_progress"))

    if state.get("choice_in_progress"):
        print("WE ARE CHOOSING SCENARIO")
        return Command(goto="scenario_node")

    # AFTER WE GOT TO THE END OF THE TREE
    # if state.get("ticket_name_chosen"):
    #     print("WE'VE CHOSEN SCENARIO")
    #     if state.get("if_comment"):
    #         resp = AIMessage(
    #             content="Нам не нужно заводить заявку, даем пользователю подсказку с обращением в стороннюю систему.",
    #             tool_calls=[
    #                 {
    #                     "name": "user_interaction_tool",
    #                     "args": {
    #                         "text": state.get("ticket_name_chosen"),
    #                         "mode": "answer",
    #                     },
    #                     "id": f"call_{uuid.uuid4()}",
    #                     "type": "tool_call",
    #                 }
    #             ],
    #         )
    #         return {**state, "messages": messages + [resp], "ticket_name_chosen": None}
    #     else:
    #         print("WE'VE CALLED PARAMS TOOL")
    #         resp = AIMessage(
    #             content="Нам нужно завести заявку, даем пользователю подсказку с обращением в стороннюю систему.",
    #             tool_calls=[
    #                 {
    #                     "name": "get_params_tool",
    #                     "args": {
    #                         "text": state.get("ticket_name_chosen"),
    #                         "mode": "answer",
    #                     },
    #                     "id": f"call_{uuid.uuid4()}",
    #                     "type": "tool_call",
    #                 }
    #             ],
    #         )
    #         return {**state, "messages": messages + [resp], "ticket_name_chosen": None}

    print(f"WE ABOUT TO FILL PARAMS: {state.get('awaiting_param') }")

    print(
        "IMPORTANT CONDITIONS: ",
        state.get("parameters_to_val"),
        state.get("user_validated"),
    )
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

            print("ASK FOR ACTION: ", messages[-1])

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

    print("CONDITION: ", state.get("we_need_to_start_params"))

    if state.get("we_need_to_start_params"):
        messages += [
            f"\nСейчас нужно начать запрашивать параметры по заявке. Обязательно вызови fill_param_tools()\n"
        ]

        resp = model.bind_tools([fill_params_tool]).invoke([system] + messages, config)

        return {
            "messages": [resp],
            "we_need_to_start_params": False,
        }

    elif state.get("awaiting_param") is not None:
        print("WE FILL PARAMS")

        print("Now we check missing")
        if state.get("missing_params"):
            print("Now we check missing")

            messages += [
                f"\nСейчас нужно заполнить параметр {state.get('awaiting_param')} через user_interaction_tool. Далее вызови fill_param_tools(), так как  остались незаполненными другие необходимые параметры. Процесс заполнения заявки завершать НЕЛЬЗЯ!\n"
            ]

        resp = model.bind_tools(
            [fill_params_tool, user_interaction_tool] + GENERAL_TOOLS
        ).invoke([SystemMessage(get_formatting_prompt())] + messages, config)

        parameters_to_fill = state.get("ticket_data")
        # parameters_to_fill[state.get("awaiting_param")]["value"] = resp
        print("OUR PARAMETERS FILLING: ", parameters_to_fill)
        new_missing = state.get("missing_params")
        if new_missing:
            new_missing.pop(0)
        return {
            "messages": [resp],
            "ticket_active": True,
            "ticket_data": parameters_to_fill,
            "missing_params": new_missing,
        }

    resp = model.bind_tools(
        [get_params_tool, fill_params_tool, ask_user_with_action_tool] + GENERAL_TOOLS
    ).invoke([system] + messages, config)

    return {"messages": [resp]}


def scenario_node(state: AgentState, config: RunnableConfig, model):
    messages = list(state["messages"])
    print(messages[-3:])
    if state["ticket_not_started"]:
        messages += [
            f"\nСейчас нужно задать пользователю дополнительные вопросы. Для этого вызови search_scenario_tool(entrypoint='<НЕОБХОДИМЫЙ ВОПРОС>'). Заполнение параметра entrypoint зависит от запроса пользователя."
        ]
        system = get_scenario_prompt()
        print("WE ARE GONNA GET RESPONSE FROM GIGACHAT")
        resp = model.bind_tools([scenario_search_tool]).invoke(
            [system] + messages, config
        )
    else:
        messages += [
            f"\nЕсли ты ранее получил вопрос из search_scenario_tool, но не задал его пользователю, то нужно спросить у пользователя ответ на этот помощью user_interaction_tool. Если пользователь тебе ответил, далее вызови search_scenario_tool() без каких-либо аргументов, чтобы продолжить задавать вопросы."
        ]
        system = get_scenario_prompt()
        print("WE ARE GONNA GET RESPONSE FROM GIGACHAT")
        resp = model.bind_tools([scenario_search_tool, user_interaction_tool]).invoke(
            [system] + messages, config
        )
        print(resp)

    # возвращаем всё сразу
    return {"messages": [resp], "choice_in_progress": True}


def await_user_node(state: AgentState, *_):
    """
    Останавливает граф, спрашивает пользователя и ждёт resume с {"answer": "..."}.
    """
    print("WE ARE WAITING FOR USER")

    question = _extract_question(state) or "Пожалуйста, ответьте на вопрос."
    payload = interrupt({"question": question})
    print("THIS IS PAYLOAD: ", payload)
    answer = payload.get("answer")
    print("THIS IS PAYLOAD: ", payload)
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
        print("WAS QUESTION ASKED")
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
    if _was_question_asked(state):
        return "await_user"
    return "ticket"
