from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from agent.graph_structure.state import AgentState
from agent.graph_structure.tools import GENERAL_TOOLS, TICKET_TOOLS
from agent.graph_structure.nodes import (
    ticket_reflect_node,
    should_route_scenario_reflect,
    should_route_after_ticket_reflect,
    should_continue_after_ticket_tool,
    after_general_tool,
    await_user_node,
    scenario_node,
)


def get_graph(model):
    checkpointer = InMemorySaver()
    g = StateGraph(AgentState)

    # Синие узлы (рефлексия)
    g.add_node("ticket", partial(ticket_reflect_node, model=model))
    g.add_node("await_user", await_user_node)
    g.add_node("scenario_node", partial(scenario_node, model=model))

    # Жёлтые узлы (исполнение инструментов)
    g.add_node("use_general_tool", ToolNode(GENERAL_TOOLS))
    g.add_node("use_ticket_tool", ToolNode(TICKET_TOOLS))

    g.set_entry_point("ticket")

    g.add_conditional_edges(
        "ticket",
        should_route_after_ticket_reflect,
        {
            "use_ticket_tool": "use_ticket_tool",
            "use_general_tool": "use_general_tool",
            "scenario_node": "scenario_node",
            "end": END,
        },
    )

    g.add_conditional_edges(
        "use_ticket_tool",
        should_continue_after_ticket_tool,
        {
            "ticket": "ticket",
            "await_user": "await_user",  # ADDED
            "end": END,
        },
    )

    g.add_conditional_edges(
        "scenario_node",
        should_route_scenario_reflect,
        {
            "use_ticket_tool": "use_ticket_tool",
            "use_general_tool": "use_general_tool",
            "end": END,
        },
    )

    g.add_conditional_edges(
        "use_general_tool",
        after_general_tool,
        {
            "await_user": "await_user",  # ADDED (раньше было END — ломало историю)
            "ticket": "ticket",
        },
    )

    return g.compile(checkpointer=checkpointer)
