from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

import json
from colorama import init, Fore, Style
import os
from langchain_gigachat import GigaChat

from agent.graph_structure.tools import (
    user_interaction_tool,
    kb_search_tool,
    scenario_search_tool,
    get_params_tool,
    fill_params_tool,
    get_params_tool,
)
from agent.graph_structure.graph import get_graph
from agent.model.model_init import get_llm

model = get_llm()

tools_list = [
    user_interaction_tool,
    kb_search_tool,
    scenario_search_tool,
    get_params_tool,
    fill_params_tool,
]

# print("Tool names handed to graph:", [t.name for t in tools_list])

model = model.bind_tools(tools_list)

graph = get_graph(model)

# png = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
# with open("graph.png", "wb") as f:
#     f.write(png)
# print("Сохранено в graph.png")


THREAD_ID = "cli-session-001"


prompt = None
config = {
    "configurable": {
        "thread_id": THREAD_ID,
        "prompt": prompt,
        # optional: separate memory per ticket/flow
        # "checkpoint_ns": f"ticket:{ticket_id}",
    }
}

conversation = {"messages": []}

print("Чем могу помочь?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    conversation["messages"].append(HumanMessage(content=user_input))

    stream = graph.stream(
        conversation,
        stream_mode="values",
        config=config,
    )
    print(stream)

    for step in stream:
        if "messages" in step and step["messages"]:
            msg = step["messages"][-1]
            # msg может быть HumanMessage, AIMessage или SystemMessage
            print(f"{msg.__class__.__name__}:", msg.content)
        elif "events" in step:
            print("Event:", step["events"])
        else:
            print("Step:", step)
        try:
            if msg in conversation["messages"]:
                continue

            if isinstance(msg, AIMessage):
                print(f"{Fore.YELLOW}{msg.content}{Style.RESET_ALL}")
            elif getattr(msg, "name", "") == "response_tool":
                data = json.loads(msg.content)
                print(f"{Fore.GREEN}{data.get('answer', '')}{Style.RESET_ALL}")
            else:
                msg.pretty_print()
            conversation["messages"].append(msg)
        except AttributeError:
            print(msg)
