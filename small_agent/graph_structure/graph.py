from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from small_agent.graph_structure.state import State
from small_agent.graph_structure.tools import (
    user_interaction_tool,
    format_output_tool,
)
from small_agent.graph_structure.nodes import (
    analyze_query,
    should_route_scenario_reflect,
)


def build_graph(model):
    checkpointer = InMemorySaver()
    workflow = StateGraph(State)

    workflow.add_node("reflect", partial(analyze_query, model=model))
    workflow.add_node(
        "tools",
        ToolNode([user_interaction_tool, format_output_tool]),
    )

    workflow.set_entry_point("reflect")
    workflow.add_conditional_edges(
        "reflect",
        should_route_scenario_reflect,
        {"tools": "tools", "end": END},
    )
    return workflow.compile(checkpointer=checkpointer)
