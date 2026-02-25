from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from small_agent.graph_structure.state import State
from small_agent.graph_structure.tools import (
    user_interaction_tool,
    format_output_tool,
)
from small_agent.prompts.prompt import create_prompt


async def analyze_query(state: State, config: RunnableConfig, model) -> State:
    messages = list(state["messages"])
    system = SystemMessage(create_prompt())
    address = state["address"].full_address
    status = state["status"]

    if not address:
        messages.append("Важно: У пользователя не задан адрес в профиле Сберчат.")
        status = "DONE"

    llm = model.bind_tools([user_interaction_tool, format_output_tool])

    response = await llm.ainvoke([system] + messages, config)

    return {"messages": [response], "agent_message": response.content, "status": status}


def should_route_scenario_reflect(state: State):

    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []

    if calls:
        return "tools"
    else:
        return "end"
