from langchain_core.tools import tool
from typing import List, Annotated, Dict
from bs4 import BeautifulSoup

from agent.rag_module import FaissSearch
from agent.config import Settings
from agent.model.model_init import get_embeddings

KNOWLEDGE_BASE = FaissSearch(
    get_embeddings(), Settings.docs.knowledge.full_vector_store_path
)


@tool
def response_tool(
    answer: Annotated[str, "текст, который нужно отправить пользователю"],
) -> dict:
    """Единый способ отдать ответ пользователю. Всегда вызывай этот инструмент для финального текста ответа на шаге."""
    return {"answer": answer}


@tool
def question_user_tool(question: str) -> dict:
    """Задать пользователю вопрос"""
    # Print the question to the terminal
    print(f"\n[Follow-up question]: {question}")
    # Wait for the user's response
    answer = input("> ")
    return {"answer": answer}


@tool
def kb_search_tool(query: Annotated[str, "вопрос пользователя по базе знаний"]) -> dict:
    """RAG for general QA"""
    ctx = KNOWLEDGE_BASE.similarity_search(query, k=2)
    if not ctx:
        return {"found": False, "answer": None, "context": []}
    joined = "\n".join([f"{t}: {x}" for t, x in ctx])
    return {"found": True, "answer": joined, "context": ctx}


_TICKET_MEM: Dict[str, Dict] = {}  # memory_key -> {scenario_id, filled}


@tool
def scenario_search_tool(
    query: Annotated[str, "фраза пользователя, по которой подбираем сценарии"],
    top_k: Annotated[int, "сколько кандидатов вернуть"] = 3,
) -> dict:
    """Ищет сценарии через FAISS. Возвращает до top_k кандидатов с вопросами для развилки."""
    try:
        cands = _search_scenarios(query, k=max(1, min(10, top_k)))
    except Exception as e:
        return {"found": False, "error": f"search_failed: {e}"}
    return {"found": bool(cands), "candidates": cands}


@tool
def scenario_get_tool(scenario_id: Annotated[str, "точный id сценария"]) -> dict:
    """Возвращает ПОЛНЫЙ JSON сценария по id (из индекса)."""
    sc = _get_scenario_by_id(scenario_id)
    if not sc:
        return {"found": False, "error": "scenario_not_found_or_index_outdated"}
    return {"found": True, "scenario": sc}


@tool
def ticket_select_scenario(
    scenario_id: Annotated[str, "выбранный id сценария"],
    memory_key: Annotated[
        str, "ключ сессии (используй один и тот же, напр. 'default')"
    ],
) -> dict:
    """Фиксирует выбранный сценарий и инициализирует память заявки."""
    sc = _get_scenario_by_id(scenario_id)
    if not sc:
        return {"ok": False, "error": "unknown_scenario"}
    _TICKET_MEM[memory_key] = {"scenario_id": scenario_id, "filled": {}}
    required = [p["name"] for p in sc.get("params", []) if p.get("required")]
    return {"ok": True, "title": sc.get("title"), "required_params": required}


@tool
def ticket_sync_from_history(
    messages_text: Annotated[
        str, "вся история диалога одной строкой (role: текст ... )"
    ],
    memory_key: Annotated[str, "ключ сессии ('default')"],
) -> dict:
    """Парсит всю историю и заполняет известные параметры без переспроса."""
    cur = _TICKET_MEM.get(memory_key)
    if not cur:
        return {"error": "no_active_ticket"}
    sc = _get_scenario_by_id(cur["scenario_id"])
    if not sc:
        return {"error": "scenario_lost"}
    filled = cur["filled"].copy()
    extracted = _extract_entities(messages_text, sc)
    filled.update(extracted)
    cur["filled"] = filled
    _TICKET_MEM[memory_key] = cur
    missing = _next_missing(sc, filled)
    return {
        "filled": filled,
        "missing": (missing["name"] if missing else None),
        "ask": (missing.get("ask") if missing else None),
        "hint": (missing.get("hint") if missing else None),
    }


@tool
def ticket_process_input(
    user_text: Annotated[str, "последний ответ пользователя"],
    memory_key: Annotated[str, "ключ сессии ('default')"],
) -> dict:
    """Обновляет параметры по последнему сообщению; проверяет FAQ; отдаёт следующий незаполненный параметр."""
    cur = _TICKET_MEM.get(memory_key)
    if not cur:
        return {"error": "no_active_ticket"}
    sc = _get_scenario_by_id(cur["scenario_id"])
    if not sc:
        return {"error": "scenario_lost"}

    # FAQ евристика: все слова вопроса встречаются в сообщении
    faq_answer = None
    tl = user_text.lower()
    for qa in sc.get("faq", []):
        words = re.findall(r"[а-яА-Яa-zA-Z0-9]+", qa.get("q", "").lower())
        if words and all(w in tl for w in words):
            faq_answer = qa.get("a")
            break

    newly = _extract_entities(user_text, sc)
    if newly:
        cur["filled"].update(newly)
        _TICKET_MEM[memory_key] = cur

    missing = _next_missing(sc, cur["filled"])
    return {
        "faq_answer": faq_answer,
        "filled": cur["filled"],
        "missing": (missing["name"] if missing else None),
        "ask": (missing.get("ask") if missing else None),
        "hint": (missing.get("hint") if missing else None),
    }


@tool
def arsenal(param_list: List): ...


@tool
def ticket_finalize(memory_key: Annotated[str, "ключ сессии ('default')"]) -> dict:
    """Если все обязательные поля собраны — отдаёт сводку JSON и флаг готовности."""
    cur = _TICKET_MEM.get(memory_key)
    if not cur:
        return {"ready": False, "error": "no_active_ticket"}
    sc = _get_scenario_by_id(cur["scenario_id"])
    if not sc:
        return {"ready": False, "error": "scenario_lost"}
    missing = _next_missing(sc, cur["filled"])
    if missing:
        return {
            "ready": False,
            "missing": missing["name"],
            "ask": missing.get("ask"),
            "hint": missing.get("hint"),
        }
    return {"ready": True, "summary": _format_summary(sc, cur["filled"])}
