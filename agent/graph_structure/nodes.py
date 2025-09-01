import json

from typing import Set
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.graph_structure.state import AgentState
from agent.prompts import create_system_prompt, get_react_instructions

from agent.graph_structure.tools import (
    response_tool,
    kb_search_tool,
    question_user_tool,
    scenario_search_tool,
    scenario_get_tool,
    ticket_select_scenario,
    ticket_sync_from_history,
    ticket_process_input,
    ticket_finalize,
)

GENERAL_TOOLS = [
    response_tool,
    question_user_tool,
    kb_search_tool,
]

TICKET_TOOLS = [
    scenario_search_tool,
    scenario_get_tool,
    ticket_select_scenario,
    ticket_sync_from_history,
    ticket_process_input,
    ticket_finalize,
    response_tool,
]

TICKET_TOOL_NAMES: Set[str] = {t.name for t in TICKET_TOOLS if hasattr(t, "name")}
GENERAL_TOOL_NAMES: Set[str] = {t.name for t in GENERAL_TOOLS if hasattr(t, "name")}


# Словарь доступных инструментов по имени
TOOLS_BY_NAME = {t.name: t for t in (GENERAL_TOOLS + TICKET_TOOLS)}


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


def _compose_prompt(extra: str = "") -> str:
    """System prompt with extra instuctions of needed."""
    return (
        create_system_prompt()
        + get_react_instructions()
        + (("\n" + extra) if extra else "")
    )


def reflect_node(state: AgentState, config: RunnableConfig, model):
    """
    General reflection on whether we need QA or Ticket functional.
    """
    system = SystemMessage(_compose_prompt())
    resp = model.bind_tools(GENERAL_TOOLS).invoke(
        [system] + list(state["messages"]), config
    )
    return {"messages": [resp]}


def ticket_reflect_node(state: AgentState, config: RunnableConfig, model):
    """
    Reflection during the process of ticket filling.
    Both general and ticket tools are allowed.
    After using general tools, we go back here till filling process is active.
    """
    ticket_rules = """
[Ticket mode]
- Ты в процессе оформления заявки. Предпочитай ticket_* инструменты, чтобы продвигаться по сценарию (scenario_search → select → sync_from_history → process_input → finalize).
- Если в процессе задают вопрос — можешь вызвать GENERAL инструменты (например, kb_search_tool), кратко ответь и вернись к сбору параметров.
- Всегда отправляй сообщения пользователю через response_tool.
- Для всех ticket_* используй memory_key="default".
"""
    messages = list(state["messages"])
    last = messages[-1] if messages else None

    # 1) Если предыдущий шаг уже вернул tool_calls — СНАЧАЛА закрываем их ToolMessage-ами
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        tool_msgs = run_tools_and_wrap(last)
        # Возвращаем только ToolMessage — следующий шаг графа уже пойдёт дальше без 422
        return {"messages": tool_msgs, "ticket_active": True}

    system = SystemMessage(_compose_prompt(ticket_rules))
    resp = model.bind_tools(GENERAL_TOOLS).invoke([system] + messages, config)
    return {"messages": [resp], "ticket_active": True}


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
            return "ticket"  # СНАЧАЛА в Ticket node
    return "use_general_tool"


def should_route_after_ticket_reflect(state: AgentState):
    """
    После Ticket node:
      - если выбран ticket-инструмент → в Ticket ToolNode,
      - если общий инструмент → в общий ToolNode (ответить на вопрос и вернуться),
      - если нет tool_calls → END.
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return "end"
    for c in calls:
        if c["name"] in TICKET_TOOL_NAMES:
            return "use_ticket_tool"
    return "use_general_tool"


def should_continue_after_ticket_tool(state: AgentState):
    """
    После выполнения ticket-инструмента:
      - если только что был ticket_finalize и {'ready': True} → END,
      - иначе продолжаем тикет-цикл (возвращаемся в Ticket node).
    """
    msgs = list(state["messages"])
    # ищем последний ToolMessage с именем ticket_finalize
    for m in reversed(msgs):
        name = getattr(m, "name", None)
        if name == "ticket_finalize":
            data = None
            if isinstance(m.content, dict):
                data = m.content
            else:
                try:
                    import json

                    data = json.loads(m.content)
                except Exception:
                    data = None
            if isinstance(data, dict) and data.get("ready") is True:
                return "end"
            break
    return "ticket_loop"


def after_general_tool(state: AgentState):
    """
    После общего ToolNode:
      - если мы в процессе оформления (ticket_active=True) → вернуться в Ticket node,
      - иначе → в Reflection node.
    """
    return "ticket" if state.get("ticket_active") else "reflect"
