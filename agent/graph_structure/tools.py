from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import List, Annotated, Dict, Literal, Optional, Any
from langchain_core.documents import Document
from langgraph.types import Command
from langgraph.prebuilt import InjectedState, InjectedStore
from langchain_core.prompts import ChatPromptTemplate
import json


from agent.rag_module import FaissSearch
from agent.config import Settings
from agent.model.model_init import get_embeddings
from agent.utils.extract_params import ParamExtractor
from contract.schemas import UserValidation

KNOWLEDGE_BASE = FaissSearch(
    get_embeddings(), Settings.docs.knowledge.full_vector_store_path
)
SCENARIO = FaissSearch(
    get_embeddings(), Settings.docs.tso_scenario.full_vector_store_path
)


@tool
def user_interaction_tool(
    text: Annotated[str, "текст для пользователя (либо ответ, либо вопрос)"],
    mode: Annotated[
        Literal["answer", "ask"],
        "режим: 'answer' для ответа, 'ask' для уточняющего вопроса",
    ],
) -> dict:
    """
    Универсальный инструмент для взаимодействия с пользователем.
    - Если mode='answer' → возвращается финальный ответ.
    - Если mode='ask' → возвращается уточняющий вопрос.
    """
    if mode == "answer":
        return {"answer": text}
    elif mode == "ask":
        return {"type": "ask_user", "question": text}
    else:
        raise ValueError("Неверный mode. Используй 'answer' или 'ask'.")


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


@tool
def scenario_search_tool(
    query: Annotated[str, "фраза пользователя, по которой подбираем сценарии"],
    top_k: Annotated[int, "сколько кандидатов вернуть"] = 10,
) -> dict:
    """Используется, если необходимо завести заявку о поломке или несиправности. Ищет сценарии через FAISS. Возвращает до top_k кандидатов."""
    try:
        cands = SCENARIO.scenario_search(query, k=top_k)
    except Exception as e:
        return {"found": False, "error": f"search_failed: {e}"}
    return {"found": bool(cands), "candidates": cands}


@tool
def get_params_tool(
    id: Annotated[str, "ID заявки в базе знаний"],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Находит необходимый сценарий, а затем извлекает список параметров, неоьходимых для заполнения заявки по данному сценарию.
    """
    scenario_raw = SCENARIO.find_by_ids(id)
    print("CANDIDATES: ", scenario_raw)

    # TODO: fix database management and creation

    scenario_processed = scenario_raw[0].metadata["node"]

    parameters = {}
    for param in scenario_processed.parameters:
        parameters[param.name] = {
            "info": f"{param._pretty_print()}",
            "class_mode": param,
            "value": None,
        }

    msg_text = (
        f"Успешно: извлечено {len(parameters)} параметр(а/ов) "
        f"для сценария '{getattr(scenario_processed._pretty_print(), 'name', 'unknown')}'."
        f"Далее используй validate_user_tool, чтобы уточнить, кто и из какого здания заводит заявку."
    )

    return Command(
        update={
            "messages": [ToolMessage(msg_text, tool_call_id=tool_call_id)],
            "ticket_data": parameters,  # keep your structured state too
        },
    )


def _create_update(result: UserValidation) -> Command:
    return Command(
        update={
            "messages": [
                ToolMessage(
                    msg_text=result.message,
                    tool_call_id=result.tool_call_id,
                )
            ],
            "user_validated": result.user_validated,
            "action": result.action,
        }
    )


@tool
def validate_user_tool(
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Проверяет от чьего имени и на каком объекте охраны создается заявка.
    """
    user = state.get("user_info")
    validation_status = state.get("user_validated")
    # change_to_name = state.get("change_to_name")

    result = UserValidation(user=user, tool_call_id=tool_call_id)

    match validation_status:
        case False:
            # if change_to_name:
            result.action = "SELECT_INNER_CLIENT"
            result.user_validated = "in progress"
            result.message = f"""У тебя есть ФИО пользователя: {result.user.name}, его табельный номер: {result.user.empid}. 
                            Обязательно уточни, от своего имени он ее заводит или нет. Задай пользователю вопрос: 'Вы заводите заявку от своего имени?'"""

            return _create_update(result)
            # else:
            #     result.user_validated = "in progress"
            #     result.message = f"""У тебя есть ФИО пользователя: {result.user.name}, его табельный номер: {result.user.empid}.
            #                     Он заводит заявку от своего имени, далее провалидируй адрес."""

        case "in progress":
            result.action = "SELECT_ASUN_BUILDING"
            result.user_validated = True
            if hasattr(user, "workPlaceLocation"):
                result.message = f"""У тебя есть адрес, где находится пользователь: {result.user.workPlaceLocation}. 
                            Обязательно уточни, на этом ли объекте у него случилась поломка. Задай пользователю вопрос: 'Вы находитесь по адресу {str(result.user.workPlaceLocation)}?'"""
            else:
                result.message = f"""У тебя нет адреса, гле находится пользователь. 
                            Обязательно уточни, на каком объекте у него случилась поломка. Задай пользователю вопрос: 'По какому адресу вы находитесь?'"""

        case True:
            ...


@tool
def fill_params_tool(
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Используется сразу после выбора сценария.
    Сканирует историю диалога и находит ранее указанные пользователем значения параметров (если они есть).
    Проверяет, какие еще параметры нужно запросить у пользователя, чтобы точно заполнить заявку полность.
    """
    params_to_fill = state.get("ticket_data")
    exctractor = ParamExtractor(state.get("llm"), state)
    data_from_history = exctractor.run_exctraction()
    print("FROM HISTORY: ", data_from_history)

    for param, value in data_from_history.items():
        if not params_to_fill[param]["value"]:
            params_to_fill[param]["value"] = value

    if not state.get("missing_params"):
        missing = [
            param for param in params_to_fill if params_to_fill[param]["value"] is None
        ]
    else:
        missing = state.get("missing_params")

    print(f"MISSING: {missing}")

    if missing == []:
        # TODO: doublecheck logic
        msg_text = f"Success: filled {len(params_to_fill)} parameters "
        return Command(
            update={
                "messages": [
                    ToolMessage(msg_text, tool_call_id=tool_call_id, name="finalize")
                ],
                "ticket_active": "complete",
                "awaiting_param": None,
            },
        )

    for param in missing:
        print(params_to_fill)
        if params_to_fill[param]["value"] is None:

            msg_text = (
                f"Нужна информация от пользователя: необходимо значение параметра '{param}'."
                f"Описание и примеры заполнения: {params_to_fill[param]['info']}"
            )
            return Command(
                goto="ticket",
                update={
                    "messages": [ToolMessage(msg_text, tool_call_id=tool_call_id)],
                    "awaiting_param": param,
                    "missing_params": missing,
                },
            )


GENERAL_TOOLS = [user_interaction_tool, kb_search_tool]
TICKET_TOOLS = [scenario_search_tool, get_params_tool, fill_params_tool]
