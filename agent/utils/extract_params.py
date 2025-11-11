import json
import logging
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from agent.retriever.data import ParameterSpecification
from agent.model.model_init import get_llm


logger = logging.getLogger("agent")


class ParamExtractor:
    def __init__(self, model, state: Dict[str, Any]):
        self.model = model
        self.state = state

    def collect_user_texts(self) -> str:
        msgs = self.state.get("messages", [])
        chunks = [
            str(getattr(m, "content", ""))
            for m in msgs
            if getattr(m, "type", "") == "human"
            or m.__class__.__name__ == "HumanMessage"
        ]
        return "\n---\n".join([x for x in chunks if x])

    def build_param_specs(self, ticket: Dict[str, Dict]) -> List[Dict[str, Any]]:
        specs = []
        for param, value_dict in ticket.items():
            specs.append(value_dict["class_mode"]._pretty_print())

        specs = "\n\n".join(specs)

        return specs

    def make_prompt(self, specs: str, history: str) -> ChatPromptTemplate:
        sys = """
            Ты извлекаешь значения параметров ТОЛЬКО из реплик пользователя в истории диалога.
            Содержимое этого системного сообщения и приведённых ниже примеров НЕЛЬЗЯ использовать как значения параметров.

            Что такое `location`:
            `location` — это описанное словами местоположение неисправности ВНУТРИ объекта по указанному адресу: этаж, подъезд, секция, рядом с чем находится (лифт, касса, ворота), какая именно дверь/створка и т. п.
            Почтовый адрес самого объекта НЕ является `location`.

            Когда заполнять, а когда пропускать:
            — Заполни `location`, если из слов пользователя понятно, где искать неисправность на объекте.
            — Достаточно хотя бы одного из признаков:
            * указан ориентир/зона: «у лифта», «к кассе», «в кассу», «у входа», «у ворот», «в серверной» и т. п.;
            * указан структурный элемент: этаж/подъезд/секция/офис/кабинет/лестница/холл/тамбур/склад/касса/коридор/крыша/двор и т. п.;
            * есть уточнение к родовому слову («дверь», «окно», «ворота»): «дверь в кассу», «дверь кассы», «правая створка двери», «дверь у лифта».
            — Если пользователь написал только общее слово без уточнения (например, просто «дверь»), `location` НЕ заполняй.
            — Можно перефразировать и нормализовать формулировку (менять порядок слов), но не добавляй неупомянутых деталей.

            Формат ответа:
            Верни ТОЛЬКО JSON вида:
            {{"<param>": "value", "<param>": "value"}}

            Примеры (для понимания; НЕ источник значений):
            Диалог → JSON
            - «этаж 12, дверь в кассу» → {{"location": "этаж 12, дверь в кассу"}}
            - «дверь кассы на 13 этаже» → {{"location": "этаж 13, дверь у кассы"}}
            - «входная дверь у лифта» → {{"location": "входная дверь у лифта"}}
            - «дверь» → {{"location": None}}
            - «Кутузовский проспект, 12» → {{"location": None}}  # это адрес, не `location`
            """

        human = (
            f"Параметры заявки:\n" + specs + "\n\n"
            f"История сообщений пользователя:\n{history}\n"
        )
        return ChatPromptTemplate.from_messages([("system", sys), ("human", human)])

    def safe_json_parse(self, text: str) -> Dict[str, Any]:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("LLM did not return JSON")
        return json.loads(text[start : end + 1])

    def get_llm(self, state: Dict[str, Any]) -> BaseChatModel:

        llm = get_llm()

        return llm

    async def run_exctraction(self):
        user_messages = self.collect_user_texts()
        params_description = self.build_param_specs(self.state.get("ticket_data"))
        prompt = self.make_prompt(params_description, user_messages)
        llm = self.get_llm(self.state)

        try:
            resp = await (prompt | llm).ainvoke({})
            raw = getattr(resp, "content", resp)
            data = self.safe_json_parse(raw if isinstance(raw, str) else str(raw))
            return data
        except Exception as e:
            logger.error(e)

    def collect_dialogue(self) -> str:
        msgs = self.state.get("messages", [])
        chunks = []

        for m in msgs:
            cls_name = m.__class__.__name__
            role = None
            content = str(getattr(m, "content", "")).strip()

            if getattr(m, "type", "") == "human" or cls_name == "HumanMessage":
                role = "Пользователь"

            elif (
                cls_name == "ToolMessage"
                and getattr(m, "name", "") == "user_interaction_tool"
            ):
                role = "Агент"
                if content:
                    try:
                        parsed = json.loads(content)
                        content = parsed.get("question") or content

                    except json.JSONDecodeError:
                        pass

            if not role or not content:
                continue

            chunks.append(f"{role}: {content}")

        if chunks:
            return "История диалога:\n" + "\n---\n".join(chunks)
        return ""

    def was_question_asked_before(self, question: str) -> bool:
        """
        Проверяет, был ли данный конкретный вопрос уже задан агентом пользователю.
        Ищет вопрос в сообщениях от агента (ToolMessage с user_interaction_tool).
        """
        msgs = self.state.get("messages", [])

        for m in msgs:
            cls_name = m.__class__.__name__
            content = str(getattr(m, "content", "")).strip()

            # Проверяем сообщения от агента
            if (
                cls_name == "ToolMessage"
                and getattr(m, "name", "") == "user_interaction_tool"
            ):
                if content:
                    try:
                        parsed = json.loads(content)
                        asked_question = parsed.get("question", "")
                        # Проверяем, совпадает ли заданный вопрос с текущим
                        if asked_question and question in asked_question:
                            return True
                    except json.JSONDecodeError:
                        # Если не JSON, проверяем напрямую
                        if question in content:
                            return True

        return False

    async def simple_extraction(self, question: str):
        # print(">>> SIMPLE EXTRACTION START", question)

        # # Сначала проверяем, был ли вопрос задан
        # if not self.was_question_asked_before(question):
        #     print(f">>> Вопрос '{question}' НЕ был задан ранее, возвращаем None")
        #     return None

        # # Проверяем, есть ли ответ пользователя ПОСЛЕ того, как был задан вопрос
        # msgs = self.state.get("messages", [])
        # question_found = False
        # user_answered = False

        # for m in msgs:
        #     cls_name = m.__class__.__name__
        #     content = str(getattr(m, "content", "")).strip()

        #     # Ищем момент, когда был задан вопрос
        #     if (
        #         cls_name == "ToolMessage"
        #         and getattr(m, "name", "") == "user_interaction_tool"
        #     ):
        #         try:
        #             parsed = json.loads(content)
        #             asked_question = parsed.get("question", "")
        #             if asked_question and question in asked_question:
        #                 question_found = True
        #                 continue
        #         except json.JSONDecodeError:
        #             if question in content:
        #                 question_found = True
        #                 continue

        #     # Если вопрос найден, проверяем, есть ли после него ответ пользователя
        #     if question_found and (
        #         getattr(m, "type", "") == "human" or cls_name == "HumanMessage"
        #     ):
        #         user_answered = True
        #         break

        # if not user_answered:
        #     print(f">>> Вопрос '{question}' был задан, но пользователь ещё НЕ ответил")
        #     return None

        # # Только если вопрос был задан И пользователь ответил, извлекаем ответ
        try:
            history = self.collect_dialogue()
        except Exception as e:
            import traceback
            traceback.print_exc()

        sys = """Тебе необходимо извлечь ответ на вопрос из истории диалога. 
        Ты получаешь на вход вопрос, просматриваешь историю и решаешь, есть ли в ней ответ на данный вопрос.
        ## ВАЖНО! Если у замка нет видимых повреждений, это не значит, что повреждена дверная конструкция. Не пропускай вопрос о повреждениях дверной конструкции.
        
        Если ответа на вопрос нет, верни 'None'.
        Если ответ на вопрос есть и пользователь отвечает на вопрос положительно, верни 'положительно'.
        Если ответ на вопрос есть и пользователь отвечает на вопрос отрицательно, верни 'отрицательно'.

        ## Пример 'Если ответа на вопрос  нет:
        Вопрос: 'Повреждена ли дверная конструкция?'
        История диалога: 'Ассистент: Есть ли у замка видимые повреждения (повреждено крепление или корпус, обрыв кабеля и т.д.)?, Пользователь: не могу проверить'
        Твой ответ: 'None'

        ## Пример 'Если ответ на вопрос есть и пользователь отвечает на вопрос положительно': 
        Вопрос: 'Проверьте, работает ли карта доступа в других зонах?'
        История диалога: 'Ассистент: Подскажите, у вас работает пропуск в других зонах?, Пользователь: везде открывает в других местах'
        Твой ответ: 'положительно'

        ## Пример 'Если ответ на вопрос есть и пользователь отвечает на вопрос отрицательно': 
        Вопрос: 'Проверьте, работают ли у других пользователей карты доступа на данной точке доступа?'
        История диалога: 'Ассистент: Проверьте, работают ли у других пользователей карты доступа на данной точке доступа?, Пользователь: да, у меня работает в других зонах, у коллег тут дверь не открывается'
        Твой ответ: 'отрицательно'

        ## Пример 'Если ответ на вопрос есть и пользователь отвечает на вопрос отрицательно': 
        Вопрос: 'Проверьте, работает ли карта доступа в других зонах?'
        История диалога: 'Ассистент: Подскажите, у вас работает пропуск в других зонах?, Пользователь: нет'
        Твой ответ: 'отрицательно'

        ## Пример 'Если ответа на вопрос нет': 
        Вопрос: 'Вопрос: Вы используете кнопку выхода для открытия двери?'
        История диалога: 'Пользователь: не могу попасть в кассу, Агент: Вы используете карту доступа (пропуска) при попытке открыть дверь?, Пользователь: не могу ответить'
        Твой ответ: 'None'

        ##ВАЖНО! В качестве возвращаемого значения ты можешь предоставлять только строки 'положительно', 'отрицательно', 'None'"""
        try:
            human = f"Вопрос: {question}\n {history}"

            prompt = ChatPromptTemplate.from_messages(
                [("system", sys), ("human", human)], template_format="jinja2"
            )
        except Exception as e:
            traceback.print_exc()
        llm = self.get_llm(self.state)
        try:
            resp = await (prompt | llm).ainvoke({})
            logger.debug("Проверяем историю диалога")
            logger.debug(f"'{question}': {resp.content if hasattr(resp, 'content') else resp}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            resp = None

        # достаём текст
        raw = getattr(resp, "content", resp)
        text = raw.strip() if isinstance(raw, str) else str(raw).strip()
        if text == "None":
            return None
        return text
