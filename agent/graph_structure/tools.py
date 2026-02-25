import logging
import traceback
import json
import random


from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import List, Annotated, Dict, Literal, Optional, Any
from langchain_core.documents import Document
from langgraph.types import Command
from langgraph.prebuilt import InjectedState, InjectedStore
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime


from agent.rag_module import FaissSearch
from agent.config import Settings
from agent.model.model_init import get_embeddings
from agent.utils.extract_params import ParamExtractor
from agent.utils.tree_strcture import TREE, TS_TREE
from contract.schemas import UserValidation


logger = logging.getLogger("agent")

KNOWLEDGE_BASE = FaissSearch(
    get_embeddings(), Settings.docs.knowledge.full_vector_store_path
)
SCENARIO = FaissSearch(
    get_embeddings(), Settings.docs.tso_scenario.full_vector_store_path
)


def _choose_other_ticket(node_name: str) -> str:
    if "tsv" in node_name:
        return "Иные неисправности СКУД"
    elif "skud" in node_name:
        return "Иные неисправности ТСВ"


@tool
async def ask_user_with_action_tool(
    text: Annotated[str, "Вопрос пользователю"],
    action: Annotated[str, "Название параметра для стейта"],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> dict:
    """
    Инструмент для уточнения у пользователя параметров.
    """
    if action == "SELECT_ASUN_BUILDING":
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(
                            {"type": "ask_user", "question": text, "action": action},
                            ensure_ascii=False,
                        ),
                        tool_call_id=tool_call_id,
                        name="ask_user_with_action_tool",
                    )
                ],
                "user_validated": True,
            },
        )
    return {"type": "ask_user", "question": text, "action": action}


@tool
async def user_interaction_tool(
    text: Annotated[str, "текст для пользователя (либо ответ, либо вопрос)"],
    mode: Annotated[
        Literal["answer", "ask"],
        "режим: 'answer' для ответа, 'ask' для уточняющего вопроса, 'action' для уточнения параметров пользователя.",
    ],
) -> dict:
    """
    Универсальный инструмент для взаимодействия с пользователем.
    - Если mode='answer' → возвращается финальный ответ в формате словаря.
    - Если mode='ask' → возвращается уточняющий вопрос в формате словаря.
    """
    if mode == "answer":
        return {"type": "answer", "answer": text}
    elif mode == "ask":
        return {"type": "ask_user", "question": text}

    else:
        raise ValueError("Неверный mode. Используй 'answer', 'ask'.")


@tool
async def check_archive_tool(
    archive_type: Annotated[str, "тип архива/местоположения камеры"],
    data: Annotated[str, "Запрашиваемая дата в формате ДД.ММ.ГГГГ"],
):
    """Инструмент для подсчета, сохранилась ли запрашиваемая запись в архиве."""
    lengths = {
        "Банкоматная зона": 60,
        "ХЦК": 60,
        "УС в сторонней организации": 4,
        "Иное": 30,
    }

    d = datetime.strptime(data, "%d.%m.%Y").date()
    today = datetime.today().date()
    diff = (today - d).days

    logger.debug(
        f"Тип выбранного архива: {archive_type}. Дата: {data}. Разница между датой и текущим днем: {diff}"
    )

    if diff <= lengths[archive_type]:
        return True
    else:
        return False


@tool
async def format_output_tool(
    chosen_ticket: Annotated[str, "название выбранной заявки"],
    if_comment: Annotated[str, "Комментарий к зявке, если есть"],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
):
    """Инструмент для форматирования и сохранения названия выбранной заявки"""
    if_comment = None if if_comment in ("None", "", "null") else if_comment

    logger.debug(f"Выбранная заявка: {chosen_ticket}. Комментарий: {if_comment}.")
    if not if_comment:
        description, _ = await _get_candidates(chosen_ticket)
        msg = f"Пользоватеть выбрал заявку {chosen_ticket} с описанием {description}. Далее уточни у пользователя, подходит ли ему данная заявка"
    else:
        msg = f"В нашей системе мы не можем завести это заявку. {if_comment}"

    return Command(
        goto="ticket",
        update={
            "messages": [
                ToolMessage(
                    msg,
                    tool_call_id=tool_call_id,
                    name="format_output_tool",
                )
            ],
            "if_comment": if_comment,
            "ticket_name_chosen": chosen_ticket,
            "choice_in_progress": False,
        },
    )


@tool
async def kb_search_tool(
    query: Annotated[str, "вопрос пользователя по базе знаний"],
) -> dict:
    """RAG для ответа на общие вопросы, касающиеся технических средств охраны в банке."""
    ctx_docs: list[Document] = await KNOWLEDGE_BASE.similarity_search(query, k=2)
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


async def _get_candidates(ticket_chosen):
    variants = await SCENARIO.scenario_search(ticket_chosen, k=8)

    other_variants = []

    for i in variants:
        if i["scenario_obj"].ticket_name == ticket_chosen:
            scenario_processed = i["scenario"]
        else:
            other_variants.append(i["scenario"])

    other_variants = "\n\n".join(other_variants[:2])

    return scenario_processed, other_variants


async def _search_next_one(
    curr_question: str,
    tree: Dict[str, Dict],
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    preambule: str = None,
    scenario_type: str = "scenario_search_tool",  # Добавляем параметр для определения типа сценария
):
    if tree[curr_question]["is_last"]:
        # Проверяем, является ли это успешным завершением (проблема решена)
        if tree[curr_question].get("is_resolved", False):
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"Проблема решена. Сообщи пользователю: {tree[curr_question]['if_comment']}",
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "choice_in_progress": False,
                    "ticket_name_chosen": None,
                    "if_comment": True,
                },
                goto="ticket",
            )
        elif tree[curr_question]["if_comment"]:

            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"Не нужно заводить заявку, только даем комментарий: {tree[curr_question]['if_comment']}",
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "choice_in_progress": False,
                    "ticket_name_chosen": tree[curr_question]["if_comment"],
                    "if_comment": True,
                },
                goto="ticket",
            )
        else:
            scenario_processed, _ = await _get_candidates(curr_question)
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"Необходимо завести заявку со следующим названием {curr_question} и описанием: {scenario_processed}",
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "choice_in_progress": False,
                    "ticket_name_chosen": curr_question,
                },
                goto="ticket",
            )
    else:
        extractor = ParamExtractor(state.get("llm"), state)
        answer = await extractor.simple_extraction(curr_question)

        if answer:
            return await _search_next_one(
                tree[curr_question][answer],
                tree,
                state,
                tool_call_id,
                preambule=None,
                scenario_type=scenario_type,
            )
        else:
            if preambule:
                to_ask = f"{preambule} {curr_question}"
            else:
                to_ask = curr_question
            return Command(
                goto="ticket",
                update={
                    "messages": [
                        ToolMessage(
                            # content=json.dumps(
                            #     {"type": "ask_user", "question": curr_question},
                            #     ensure_ascii=False,
                            # ),
                            f"Нужно уточнить у пользователя ответ на следующий вопрос, не изменяя формулировку: {to_ask}",
                            tool_call_id=tool_call_id,
                            name=scenario_type,
                        )
                    ],
                    "choice_in_progress": True,
                    "curr_question": curr_question,
                    "ticket_not_started": False,
                },
            )


@tool
async def scenario_search_tool(
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    entrypoint: Optional[str] = None,
):
    """
    Инструмент для поиска сценария, исходя из ответов пользователя на дополнительные вопросы.
    """

    phrases = [
        "Понимаю, что проблема связана с неисправностью. Чтобы разобраться точнее и понять, как это исправить, мне нужно задать вам несколько уточняющих вопросов.",
        "Я вижу, что речь идёт о поломке. Чтобы определить причину и подобрать решение, позвольте задать несколько вопросов.",
        "Похоже, возникла неисправность. Чтобы понять, в чём именно дело, мне нужно уточнить некоторые детали.",
        "Понимаю, что у вас случилась поломка. Чтобы разобраться, что именно вышло из строя, мне потребуется задать пару уточняющих вопросов.",
        "Похоже, что проблема связана с поломкой. Чтобы точно определить источник неисправности и помочь вам, я задам несколько уточняющих вопросов.",
    ]
    if entrypoint:
        curr_question = entrypoint
        preambule = random.choice(phrases)
    else:
        curr_question = state["curr_question"]
        preambule = None

    try:
        result = await _search_next_one(
            curr_question, TREE, state, tool_call_id, preambule
        )
        return result
    except Exception as e:
        logger.error(e)
        traceback.print_exc()


@tool
async def scenario_ts_search_tool(
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    entrypoint: Optional[str] = None,
):
    """
    Инструмент для поиска сценария, исходя из ответов пользователя на дополнительные вопросы.
    """

    phrases = [
        "Понимаю, что проблема связана с неисправностью. Чтобы разобраться точнее и понять, как это исправить, мне нужно задать вам несколько уточняющих вопросов.",
        "Я вижу, что речь идёт о поломке. Чтобы определить причину и подобрать решение, позвольте задать несколько вопросов.",
        "Похоже, возникла неисправность. Чтобы понять, в чём именно дело, мне нужно уточнить некоторые детали.",
        "Понимаю, что у вас случилась поломка. Чтобы разобраться, что именно вышло из строя, мне потребуется задать пару уточняющих вопросов.",
        "Похоже, что проблема связана с поломкой. Чтобы точно определить источник неисправности и помочь вам, я задам несколько уточняющих вопросов.",
    ]
    if entrypoint:
        curr_question = entrypoint
        preambule = random.choice(phrases)
    else:
        curr_question = state["curr_question"]
        preambule = None

    try:
        result = await _search_next_one(
            curr_question,
            TS_TREE,
            state,
            tool_call_id,
            preambule,
            scenario_type="scenario_ts_search_tool",
        )
        return result
    except Exception as e:
        logger.error(e)
        traceback.print_exc()


@tool
async def get_params_tool(
    ticket_name: Annotated[str, "Название заявки"],
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Находит необходимый сценарий, а затем извлекает список параметров, неоьходимых для заполнения заявки по данному сценарию.
    """

    # Проверяем, был ли отказ от предыдущей заявки и переход на "Иные неисправности СКУД"
    messages = state.get("messages", [])
    node = _choose_other_ticket(state.get("node_name"))

    # Ищем вопрос о подходящей заявке в последних сообщениях
    for m in reversed(messages[-5:]):  # Проверяем последние 5 сообщений
        if (
            isinstance(m, ToolMessage)
            and getattr(m, "name", "") == "user_interaction_tool"
        ):
            content = str(getattr(m, "content", ""))
            try:
                parsed = json.loads(content)
                question = parsed.get("question", "")
                # Если нашли вопрос о подходящей заявке
                if (
                    "Вам подходит заявка" in question
                    and "Иные неисправности" in question
                ):
                    # Проверяем ответ пользователя
                    # from agent.utils.extract_params import ParamExtractor
                    extractor = ParamExtractor(state.get("llm"), state)
                    user_response = await extractor.simple_extraction(
                        f"Вам подходит заявка"
                    )

                    if user_response == "отрицательно":
                        # Пользователь отказался, нужно сначала оповестить о переходе на "Иные неисправности"
                        return Command(
                            update={
                                "messages": [
                                    ToolMessage(
                                        content=json.dumps(
                                            {
                                                "type": "ask_user",
                                                "question": f"Так как данная заявка Вам не подходит, предлагаю завести обобщенную заявку {node}, где я подробно зафиксирую вашу неисправность. Продолжим?",
                                            },
                                            ensure_ascii=False,
                                        ),
                                        tool_call_id=tool_call_id,
                                        name="user_interaction_tool",
                                    )
                                ],
                                "awaiting_fallback_confirmation": True,
                            },
                            goto="ticket",
                        )
                    break
            except json.JSONDecodeError:
                pass

    scenario_raw = await SCENARIO.scenario_search(ticket_name, k=10)

    # TODO: fix database management and creation

    try:

        for i in scenario_raw:
            if i["scenario_obj"].ticket_name == ticket_name:
                scenario_processed = i["scenario_obj"]
                break

        parameters = {}
        for param in scenario_processed.parameters:
            parameters[param.name] = {
                "info": f"{param._pretty_print()}",
                "class_mode": param,
                "description": param.description,
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
                "user_validated": False,
                "ticket_active": True,
                "we_need_to_start_params": True,
                "chosen_scenario": scenario_processed,
                "stk_insr": scenario_processed.answers,
            }
        )

    except Exception as e:

        traceback.print_exc()


@tool
async def fill_params_tool(
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
    data_from_history = await exctractor.run_exctraction()

    for param, value in data_from_history.items():
        if not params_to_fill[param]["value"]:
            params_to_fill[param]["value"] = value

    if not state.get("missing_params"):
        missing = [
            param for param in params_to_fill if params_to_fill[param]["value"] is None
        ]
    else:
        missing = state.get("missing_params")

    logger.debug(f"Не заполненные поля заявки: {missing}")

    if missing == []:
        # TODO: doublecheck logic
        logger.debug("Все поля заполнены, сохранения action-а в state.")

        msg_text = f"Success: filled {len(params_to_fill)} parameters "
        return Command(
            update={
                "messages": [
                    ToolMessage(msg_text, tool_call_id=tool_call_id, name="finalize")
                ],
                "awaiting_param": None,
                "ticket_data": params_to_fill,
                "action": "CREATE_TICKET",
            },
        )
    for param in missing:
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
                    "ticket_active": False,
                    "ticket_data": params_to_fill,
                },
            )


GENERAL_TOOLS = [user_interaction_tool, kb_search_tool]
TICKET_TOOLS = [
    scenario_search_tool,
    scenario_ts_search_tool,
    get_params_tool,
    fill_params_tool,
    ask_user_with_action_tool,
    check_archive_tool,
    format_output_tool,
    # validate_user_tool,
]
