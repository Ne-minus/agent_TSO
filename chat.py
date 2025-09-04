import streamlit as st

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

import json
from colorama import init, Fore, Style
import os
from langchain_gigachat import GigaChat

# забрал всё из main.py

from agent.graph_structure.tools import (
    response_tool,
    question_user_tool,
    kb_search_tool,
    scenario_search_tool,
)
from agent.graph_structure.graph import get_graph
from agent.model.model_init import get_llm


# Инициализируем модель
model = get_llm()

tools_list = [
    response_tool,  # Для коммуникации с пользователем
    question_user_tool,  # Искать в интернете
    kb_search_tool,
    scenario_search_tool,
]

print("Tool names handed to graph:", [t.name for t in tools_list])

model = model.bind_tools(tools_list)

graph = get_graph(model)

# Pick a stable ID for this conversation/session/user
THREAD_ID = "cli-session-001"  # e.g., f"user:{user_id}:conv:{conv_id}"
# or: THREAD_ID = str(uuid4())         # stable only for this process run

prompt = None
config = {
    "configurable": {
        "thread_id": THREAD_ID,
        "prompt": prompt,
        # optional: separate memory per ticket/flow
        # "checkpoint_ns": f"ticket:{ticket_id}",
    }
}

# вот здесь начинается streamlit
st.title('Chat')
st.caption('🚀Chat')

# Стартовое сообщение сохраняем в session_state
if "message" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Чем могу помочь?"}]

# Выводим стартовое сообщение в чат
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

conversation = {"messages": []} # это по аналогии с main.py, чтобы в терминале дублировалось все

if prompt := st.chat_input(): # пользователь пишет
    st.session_state.messages.append({"role": "user", "content": HumanMessage(content=prompt)}) # добавляем сообщение пользователя в session_state
    st.chat_message("user").write(prompt) # выводим сообщение пользователя в чат

    conversation["messages"].append(HumanMessage(content=prompt))

    stream = graph.stream(
        conversation,
        stream_mode="values",
        config=config,
    )

    for step in stream:
        msg = step["messages"][-1]
        try:
            if msg in conversation["messages"]:
                continue

            if isinstance(msg, AIMessage):
                print(f"{Fore.YELLOW}{msg.content}{Style.RESET_ALL}")
                st.session_state.messages.append({"role": "assistant", "content": msg})
                conversation["messages"].append(msg)
            elif getattr(msg, "name", "") == "response_tool":
                data = json.loads(msg.content)
                print(f"{Fore.GREEN}{data.get('answer', '')}{Style.RESET_ALL}")
            else:
                msg.pretty_print()
            conversation["messages"].append(msg)
            st.session_state.messages.append(msg)
            st.chat_message("assistant").write(msg.content)
        except AttributeError:
            print(msg)
