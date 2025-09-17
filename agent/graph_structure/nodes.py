import json

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
)

from agent.graph_structure.tools import (
    user_interaction_tool,
    kb_search_tool,
    scenario_search_tool,
    get_params_tool,
    fill_params_tool,
    validate_user_tool,
    GENERAL_TOOLS,
    TICKET_TOOLS,
)


TICKET_TOOL_NAMES: Set[str] = {t.name for t in TICKET_TOOLS if hasattr(t, "name")}
GENERAL_TOOL_NAMES: Set[str] = {t.name for t in GENERAL_TOOLS if hasattr(t, "name")}


# Словарь доступных инструментов по имени
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
    tm = _last_tool_message(state)
    if not tm:
        return None
    data = tm.content
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return None
    return data.get("question")


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


def reflect_node(state: AgentState, config: RunnableConfig, model):
    """
    General reflection on whether we need QA or Ticket functional.
    """
    system = SystemMessage(_compose_prompt())
    resp = model.bind_tools(GENERAL_TOOLS + TICKET_TOOLS).invoke(
        [system] + list(state["messages"]), config
    )
    return {"messages": [resp]}


def ticket_reflect_node(state: AgentState, config: RunnableConfig, model):
    """
    Reflection during the process of ticket filling.
    Both general and ticket tools are allowed.
    After using general tools, we go back here till filling process is active.
    """

    messages = list(state["messages"])
    system = SystemMessage(_compose_prompt(if_ticket=True))
    print(f"WE ABOUT TO FILL PARAMS: {state.get("awaiting_param") }")

    if state.get("awaiting_param") is not None:
        print("WE FILL PARAMS")

        print("Now we check missing")
        if state.get("missing_params"):

            messages += [
                f"\nСейчас нужно заполнить параметр {state.get('awaiting_param')} через user_interaction_tool. Далее вызови fill_param_tools(), так как  остались незаполненными другие необходимые параметры. Процесс заполнения заявки завершать НЕЛЬЗЯ!\n"
            ]

        resp = model.bind_tools(TICKET_TOOLS + GENERAL_TOOLS).invoke(
            [SystemMessage(get_formatting_prompt())] + messages, config
        )

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

    if state.get("user_validated") == "in progress":
        messages += [
            f"\nЕще не все параметры пользователя проверены. Вызови далее validate_user_tool.\n"
        ]
    elif state.get("user_validated"):
        messages += [
            f"\nВсе параметры пользователя проверены, можно продолжить заведение заявки. "
        ]

    resp = model.bind_tools(TICKET_TOOLS + GENERAL_TOOLS).invoke(
        [system] + messages, config
    )

    return {"messages": [resp]}


def await_user_node(state: AgentState, *_):
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


def should_route_after_reflect(state: AgentState):
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
            return "use_ticket_tool"  # ← СРАЗУ исполняем тикет-тул
    return "use_general_tool"


def should_route_after_ticket_reflect(state: AgentState):
    last = state["messages"][-1]
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

    return "ticket_loop"


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
    return "ticket" if state.get("ticket_active") else "reflect"
