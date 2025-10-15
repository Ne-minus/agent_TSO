from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

# Импорты из твоего файла
from agent.graph_structure.state import AgentState
from agent.graph_structure.tools import GENERAL_TOOLS, TICKET_TOOLS
from agent.graph_structure.nodes2 import (
    reflect_node,
    scenario_node,
    validate_scenario_node,
    validate_user_node,
    fill_ticket_node,
    await_user_node,
    should_route_after_reflect,
    after_general_tool,
    after_scenario_node,
    after_validate_scenario,
    after_validate_user,
    after_fill_ticket,
    after_await_user,
)


def get_graph(model):
    """Создаём граф с узлами и переходами для основного агента."""
    checkpointer = InMemorySaver()
    g = StateGraph(AgentState)

    # === Узлы ===
    g.add_node("reflect", partial(reflect_node, model=model))
    g.add_node("scenario_node", partial(scenario_node, model=model))
    g.add_node("validate_scenario", partial(validate_scenario_node, model=model))
    g.add_node("validate_user", partial(validate_user_node, model=model))
    g.add_node("fill_ticket", partial(fill_ticket_node, model=model))
    g.add_node("await_user", await_user_node)

    # Узел для выполнения general-инструментов
    g.add_node("use_general_tool", ToolNode(GENERAL_TOOLS))

    # === Точка входа ===
    g.set_entry_point("reflect")

    # === Переходы ===

    g.add_conditional_edges(
        "reflect",
        should_route_after_reflect,
        {
            "scenario_node": "scenario_node",
            "use_general_tool": "use_general_tool",
            "end": END,
        },
    )

    g.add_conditional_edges(
        "use_general_tool",
        after_general_tool,
        {
            "await_user": "await_user",
            "scenario_node": "scenario_node",
            "validate_scenario": "validate_scenario",
            "fill_ticket": "fill_ticket",
            "reflect": "reflect",
        },
    )

    g.add_conditional_edges(
        "scenario_node",
        after_scenario_node,
        {
            "use_general_tool": "use_general_tool",
            "validate_scenario": "validate_scenario",
        },
    )

    g.add_conditional_edges(
        "validate_scenario",
        after_validate_scenario,
        {
            "use_general_tool": "use_general_tool",
            "validate_user": "validate_user",
            "end": END,
        },
    )

    g.add_conditional_edges(
        "validate_user",
        after_validate_user,
        {
            "use_general_tool": "use_general_tool",
            "fill_ticket": "fill_ticket",
        },
    )

    g.add_conditional_edges(
        "fill_ticket",
        after_fill_ticket,
        {
            "use_general_tool": "use_general_tool",
            "end": END,
        },
    )

    # g.add_conditional_edges(
    #     "await_user",
    #     after_await_user,
    #     {
    #         "use_general_tool": "use_general_tool",
    #     },
    # )

    # === Компиляция графа ===
    return g.compile(checkpointer=checkpointer)
