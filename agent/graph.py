from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.nodes import (
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

    # Reflection → условно: общий ToolNode / Ticket node / END
    g.add_conditional_edges(
        "reflect",
        should_route_after_reflect,
        {
            "use_general_tool": "use_general_tool",
            "ticket": "ticket",  # если выбран ticket-инструмент — сначала в Ticket node
            "end": END,
        },
    )

    # Ticket node → условно: ticket ToolNode / общий ToolNode / END
    g.add_conditional_edges(
        "ticket",
        should_route_after_ticket_reflect,
        {
            "use_ticket_tool": "use_ticket_tool",
            "use_general_tool": "use_general_tool",  # временно «уходим» ответить на вопрос
            "end": END,
        },
    )

    # Execute ticket tools → условно: продолжить тикет / END (после finalize.ready=True)
    g.add_conditional_edges(
        "use_ticket_tool",
        should_continue_after_ticket_tool,
        {
            "ticket_loop": "ticket",
            "end": END,
        },
    )

    # Execute general tools → обратно: в Ticket node, если активен тикет, иначе в Reflection
    g.add_conditional_edges(
        "use_general_tool",
        after_general_tool,
        {
            "ticket": "ticket",
            "reflect": "reflect",
        },
    )

    return g.compile(checkpointer=checkpointer)
