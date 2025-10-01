import json
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from agent.retriever.data import ParameterSpecification
from agent.model.model_init import get_llm


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
        sys = (
            "Ты извлекаешь значения параметров из истории диалога, исходя из их описания и примеров заполнения, которые тебе представитт пользователь."
            "Если пользователь не писал ничего похожего на значение какого-либо параметра, не записывай его в итоговый словарь."
            "Верни ТОЛЬКО JSON вида: "
            '{{"<param>": "value", "<param>": "value"}}'
        )

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

    def run_exctraction(self):
        user_messages = self.collect_user_texts()
        params_description = self.build_param_specs(self.state.get("ticket_data"))
        prompt = self.make_prompt(params_description, user_messages)
        llm = self.get_llm(self.state)

        try:
            resp = (prompt | llm).invoke({})
            raw = getattr(resp, "content", resp)
            data = self.safe_json_parse(raw if isinstance(raw, str) else str(raw))
        except Exception as e:
            data = {}

        return data

    def collect_dialogue(self) -> str:
        msgs = self.state.get("messages", [])
        chunks = []
        for m in msgs:
            role = None
            if (
                getattr(m, "type", "") in ("human",)
                or m.__class__.__name__ == "HumanMessage"
            ):
                role = "Пользователь"
            elif (
                getattr(m, "type", "") in ("ai",) or m.__class__.__name__ == "AIMessage"
            ):
                role = "Агент"

            if role:
                content = str(getattr(m, "content", "")).strip()
                if not content:
                    continue

                # фильтрация для агента
                if role == "Агент":
                    # исключаем размышления и планирование
                    if any(
                        content.startswith(prefix)
                        for prefix in ["Reflection", "Planning", "Thought", "Action"]
                    ):
                        continue
                    # берём только вопросительные реплики
                    if "?" not in content:
                        continue

                chunks.append(f"{role}: {content}")

        # один заголовок
        if chunks:
            return "История диалога:\n" + "\n---\n".join(chunks)
        return ""

    def simple_extraction(self, question: str):
        print(">>> SIMPLE EXTRACTION START", question)
        history = self.collect_dialogue()

        sys = """Тебе необходимо извлечь ответ на вопрос из истории диалога. 
        Ты получаешь на вход вопрос, просматриваешь историю и решаешь, есть ли в ней ответ на данный вопрос.
        Если пользователь выражает согласие, то верни 'yes'. Если не согласен, то верни 'no'. 
        Если пользователь не знает ответа на вопрос или ты считаешь, что ответа нет в истории, то верни 'None'.
        Например, 
        Вопрос: 'Проверьте, работает ли карта доступа в других зонах?'
        История диалога: 'Асистент: Подскажите, у вас работает пропуск в других зонах?, Пользователь: везде открывает в других местах'
        Твой ответ: 'yes'
        ##ВАЖНО! В качестве ответа ты можешь предоставлять только строки 'yes', 'no', 'None'"""
        human = f"Вопрос: {question}\nИстория диалога: {history}"

        prompt = ChatPromptTemplate.from_messages([("system", sys), ("human", human)])
        print("PROMPT: ", prompt)
        llm = self.get_llm(self.state)
        print(">>> SIMPLE EXTRACTION BEFORE LLM", question)
        try:
            resp = (prompt | llm).invoke({})
            print(">>> SIMPLE EXTRACTION RAW:", resp)
        except Exception as e:
            import traceback

            print(">>> ERROR in invoke:", e)
            traceback.print_exc()
            resp = None

        # достаём текст
        raw = getattr(resp, "content", resp)
        text = raw.strip() if isinstance(raw, str) else str(raw).strip()
        print(">>> SIMPLE EXTRACTION TEXT:", text)

        if text == "None":
            return None
        return text
