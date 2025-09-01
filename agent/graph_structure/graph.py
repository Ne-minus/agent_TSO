from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from agent.graph_structure.state import AgentState
from agent.graph_structure.nodes import (
    reflect_node,
    ticket_reflect_node,
    should_route_after_reflect,
    should_route_after_ticket_reflect,
    should_continue_after_ticket_tool,
    after_general_tool,
    GENERAL_TOOLS,
    TICKET_TOOLS,
)


def get_graph(model):
    checkpointer = InMemorySaver()
    g = StateGraph(AgentState)

    # Синие узлы (рефлексия)
    g.add_node("reflect", partial(reflect_node, model=model))
    g.add_node("ticket", partial(ticket_reflect_node, model=model))

    # Жёлтые узлы (исполнение инструментов)
    g.add_node("use_general_tool", ToolNode(GENERAL_TOOLS))
    g.add_node("use_ticket_tool", ToolNode(TICKET_TOOLS))

    # Точка входа
    g.set_entry_point("reflect")

    # Reflect → сразу в нужный ToolNode, если есть tool_calls
    g.add_conditional_edges(
        "reflect",
        should_route_after_reflect,
        {
            "use_general_tool": "use_general_tool",  # общий инструмент → сразу выполняем
            "use_ticket_tool": "use_ticket_tool",  # ТИКЕТ-инструмент → сразу выполняем (исправление!)
            "ticket": "ticket",  # перейти в режим тикета (без tool_calls)
            "end": END,
        },
    )

    # Ticket → если модель хочет инструмент, идём сразу в соответствующий ToolNode
    g.add_conditional_edges(
        "ticket",
        should_route_after_ticket_reflect,
        {
            "use_ticket_tool": "use_ticket_tool",
            "use_general_tool": "use_general_tool",
            "end": END,
        },
    )

    # После выполнения тикет-инструмента: либо крутим тикет дальше, либо END
    g.add_conditional_edges(
        "use_ticket_tool",
        should_continue_after_ticket_tool,
        {
            "ticket_loop": "ticket",
            "end": END,
        },
    )

    # После общего инструмента: вернуться в ticket (если активен) или в reflect
    g.add_conditional_edges(
        "use_general_tool",
        after_general_tool,
        {
            "ticket": "ticket",
            "reflect": "reflect",
        },
    )

    return g.compile(checkpointer=checkpointer)
