from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import List, Annotated, Dict
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langgraph.types import Command
from langgraph.prebuilt import InjectedState, InjectedStore

from agent.rag_module import FaissSearch
from agent.config import Settings
from agent.model.model_init import get_embeddings

KNOWLEDGE_BASE = FaissSearch(
    get_embeddings(), Settings.docs.knowledge.full_vector_store_path
)
SCENARIO = FaissSearch(
    get_embeddings(), Settings.docs.tso_scenario.full_vector_store_path
)


@tool
def response_tool(
    answer: Annotated[str, "текст, который нужно отправить пользователю"],
) -> dict:
    """Единый способ отдать ответ пользователю. Всегда вызывай этот инструмент для финального текста ответа на шаге."""
    return {"answer": answer}


@tool
def question_user_tool(
    question: Annotated[
        str, "уточняющий вопрос, который необходимо задать пользователю."
    ],
) -> dict:
    """Инструмент для того, чтобы уточнить у пользователя, информацию, которой тебе не хватает.
    Использовать только для уточняющих вопросов."""
    # Print the question to the terminal
    print(f"\n[Follow-up question]: {question}")
    # Wait for the user's response
    answer = input("> ")
    return {"answer": answer}


@tool
def kb_search_tool(query: Annotated[str, "вопрос пользователя по базе знаний"]) -> dict:
    """RAG для ответа на общие вопросы, касающиеся технических средств охраны в банке."""
    ctx_docs: list[Document] = KNOWLEDGE_BASE.similarity_search(query, k=2)
    if not ctx_docs:
        return {"found": False, "answer": None, "context": []}

    def doc_to_str(d: Document) -> str:
        title = (
            (d.metadata or {}).get("title") or (d.metadata or {}).get("source") or ""
        )
        body = (d.page_content or "").strip()
        return f"{title}: {body}" if title else body

    joined = "\n".join(doc_to_str(d) for d in ctx_docs)

    return {"found": True, "answer": joined, "context": [d.dict() for d in ctx_docs]}


# _TICKET_MEM: Dict[str, Dict] = {}  # memory_key -> {scenario_id, filled}


@tool
def scenario_search_tool(
    query: Annotated[str, "фраза пользователя, по которой подбираем сценарии"],
    top_k: Annotated[int, "сколько кандидатов вернуть"] = 10,
) -> dict:
    """Используется, если необходимо завести заявку о поломке или несиправности. Ищет сценарии через FAISS. Возвращает до top_k кандидатов с вопросами для развилки."""
    try:
        cands = SCENARIO.scenario_search(query, k=top_k)
    except Exception as e:
        return {"found": False, "error": f"search_failed: {e}"}
    return {"found": bool(cands), "candidates": cands}


@tool
def get_params_tool(
    query: Annotated[str, "описание заявки в базе данных"],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Находит необходимый сценарий, а затем извлекает список параметров, неоьходимых для заполнения заявки по данному сценарию.
    """
    cands = SCENARIO.scenario_search(query, k=1)

    parameters = {}
    for param in cands[0].parameters:
        parameters[param.name] = {
            "info": f"Описание: {param.description}Примеры: \n{param.examples}Подсказка: \n{param.hint}Запрос в стороннюю систему: \n{param.out_of_system}",
            "value": None,
        }

    msg_text = (
        f"Success: extracted {len(parameters)} parameters "
        f"for scenario '{getattr(cands[0], 'name', 'unknown')}'."
    )

    return Command(
        update={
            "messages": [ToolMessage(msg_text, tool_call_id=tool_call_id)],
            "ticket_data": parameters,  # keep your structured state too
        },
    )


@tool
def fill_params_tool(
    memory: InjectedStore,
    state: InjectedState,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command: ...
