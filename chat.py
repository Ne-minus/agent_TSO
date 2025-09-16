import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import json
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph_structure.tools import (
    user_interaction_tool,
    kb_search_tool,
    scenario_search_tool,
    get_params_tool,
    fill_params_tool,
)
from agent.graph_structure.graph import get_graph
from agent.model.model_init import get_llm


# ---------- Инициализация одноразовая (не вносите в session_state то, что не нужно сериализовать) ----------
@st.cache_resource(show_spinner=False)
def _init_graph():
    model = get_llm()
    tools_list = [
        user_interaction_tool,
        kb_search_tool,
        scenario_search_tool,
        get_params_tool,
        fill_params_tool,
    ]
    model = model.bind_tools(tools_list)
    graph = get_graph(model)
    return graph


graph = _init_graph()

THREAD_ID = "cli-session-001"
config = {"configurable": {"thread_id": THREAD_ID, "prompt": None}}

# ---------- session_state ----------
if "ui_messages" not in st.session_state:
    # Лента для UI (строки): [{"role": "assistant"|"user", "content": "..."}, ...]
    st.session_state.ui_messages = [
        {"role": "assistant", "content": "Чем могу помочь?"}
    ]

if "lc_messages" not in st.session_state:
    # Лента для LangChain: [HumanMessage(...), AIMessage(...), ToolMessage(...)]
    st.session_state.lc_messages = []

if "pending_tool_question" not in st.session_state:
    # Если граф задал вопрос через question_user_tool — держим тут текст вопроса
    st.session_state.pending_tool_question = None

st.title("Chat")
st.caption("🚀 Chat")

# ---------- Рендер истории ----------
for m in st.session_state.ui_messages:
    st.chat_message(m["role"]).write(m["content"])

# Если есть незавершённый вопрос от графа — покажем подсказку
if st.session_state.pending_tool_question:
    st.info("Вопрос от ассистента: " + st.session_state.pending_tool_question)

# ---------- Ввод пользователя ----------
user_text = st.chat_input("Введите сообщение...")

if user_text:
    # 1) UI-лента
    st.session_state.ui_messages.append({"role": "user", "content": user_text})
    st.chat_message("user").write(user_text)

    # 2) LangChain-лента
    st.session_state.lc_messages.append(HumanMessage(content=user_text))

    # 3) Запуск графа на основе всей истории
    conversation = {"messages": st.session_state.lc_messages.copy()}

    # Подготовим placeholder под «поточную печать» ассистента
    assistant_box = st.chat_message("assistant")
    placeholder = assistant_box.empty()
    incremental_text = ""

    # Сбросим флаг незавершённого вопроса: этот ход может на него отвечать
    st.session_state.pending_tool_question = None

    # 4) Поток
    stream = graph.stream(conversation, stream_mode="values", config=config)

    # Флаг: надо остановиться и дождаться ответа пользователя (когда граф задал вопрос)
    must_wait_for_user = False

    for step in stream:
        msg = step["messages"][-1]

        # Пропускаем дубли, если такие есть
        if msg in st.session_state.lc_messages:
            continue

        tool_name = getattr(msg, "name", "")

        # --- ВАЖНО: Ничего не печатаем для AIMessage (рефлексии/план/мысли) ---
        if isinstance(msg, AIMessage):
            # Храним в lc-истории для корректной работы графа, но в UI не показываем
            st.session_state.lc_messages.append(msg)
            print(f"LOGS!!!\n\n{msg}")
            continue

        # --- Показываем только «финальный» ответ из response_tool ---
        if tool_name == "user_interaction_tool":
            data = json.loads(msg.content) if msg.content else {}
            answer = data.get("answer", "")
            question = data.get("question", "")
            if answer:
                incremental_text = (
                    incremental_text + ("\n" if incremental_text else "") + answer
                ).strip()
                placeholder.write(incremental_text)
                st.session_state.ui_messages.append(
                    {"role": "assistant", "content": answer}
                )

            elif question:
                st.session_state.pending_tool_question = question
                assistant_text = f"Нужны уточнения: {question}"
                incremental_text = (
                    incremental_text
                    + ("\n" if incremental_text else "")
                    + assistant_text
                ).strip()
                placeholder.write(incremental_text)
                st.session_state.ui_messages.append(
                    {"role": "assistant", "content": assistant_text}
                )

                must_wait_for_user = True
                break

            st.session_state.lc_messages.append(msg)
            continue

        # # --- И только вопросы на уточнение из question_user_tool ---
        # if tool_name == "question_user_tool":
        #     data = json.loads(msg.content) if msg.content else {}
        #     q = data.get("question") or data.get("text") or "Можете уточнить детали?"
        #     st.session_state.pending_tool_question = q
        #     assistant_text = f"Нужны уточнения: {q}"
        #     incremental_text = (
        #         incremental_text + ("\n" if incremental_text else "") + assistant_text
        #     ).strip()
        #     placeholder.write(incremental_text)
        #     st.session_state.ui_messages.append(
        #         {"role": "assistant", "content": assistant_text}
        #     )
        #     st.session_state.lc_messages.append(msg)
        #     must_wait_for_user = True
        #     break

        # Прочие системные/инструментальные сообщения — не выводим в чат
        st.session_state.lc_messages.append(msg)
