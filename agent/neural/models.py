import ast
import os

# Local ollama (LLM model) and huggingface (embeddings model)
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings

# Gigachat API connection
# import os
# from langchain_gigachat.chat_models import GigaChat
# from langchain_gigachat.embeddings import GigaChatEmbeddings

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from contract.config import Settings

from langchain_core.messages import BaseMessage
from typing import Literal

import logging


ERROR_MESSAGE = "Извините, что-то пошло не так. Пожалуйста, переформулируйте Ваш ответ или перезагрузите страницу."
DONOTKNOW_MESSAGE = "Простите, сейчас я не могу ответить на данный вопрос - у меня нет по нему никакой информации."


class MLModelShell:
    def __init__(
        self,
        embeddings_name: str,
        llm_model_type: str,
        inside_docker_container: bool | None = None,
        device: Literal["cpu", "cuda", "mps"] | None = None,
    ) -> None:
        if device is None:
            device = Settings.system.device

        if inside_docker_container is None:
            inside_docker_container = Settings.system.inside_docker_container

        self.embeddings = self._init_embeddings(embeddings_name, device)

        self.llm = self._init_llm_model(
            model_type=llm_model_type, inside_docker_container=inside_docker_container
        )

    def _init_llm_model(
        self, model_type: Literal["gemma3:4b"], inside_docker_container: bool = True
    ) -> GigaChat:  # GigaChat
        # Local Ollama LLM
        if inside_docker_container:
            engine_url = Settings.models.llm_base_url
        else:
            engine_url = None

        # llm = ChatOllama(model=model_type, base_url=engine_url)

        # Gigachat API
        llm = GigaChat(
            model=model_type,
            # Auth with credentials
            credentials=os.environ["GIGACHAIN_AUTH"],
            scope=os.environ["GIGACHAT_SCOPE"],
            # Auth with certs
            # base_url = Settings.models.base_url,
            # ca_bundle_file = Settings.certs.ca_chain,
            # cert_file = Settings.certs.cert,
            # key_file = Settings.certs.key,
            # key_file_password = Settings.certs.key_pass,
            verify_ssl_certs=False,
            profanity_check=False,
        )

        return llm

    def _init_embeddings(
        self, embeddings_name: str, device: Literal["cpu", "cuda", "mps"]
    ) -> GigaChatEmbeddings:  # GigaChatEmbeddings
        """Get an embeddings model"""
        # Local Huggingface embeddings models
        # embeddings = HuggingFaceEmbeddings(
        #     model_name=embeddings_name, model_kwargs={"device": device}
        # )

        # Gigachat API
        embeddings = GigaChatEmbeddings(
            model=embeddings_name,
            # Auth with credentials
            credentials=os.environ["GIGACHAIN_AUTH"],
            scope=os.environ["GIGACHAT_SCOPE"],
            # Auth with certs
            # base_url = Settings.models.base_url,
            # ca_bundle_file = Settings.certs.ca_chain,
            # cert_file = Settings.certs.cert,
            # key_file = Settings.certs.key,
            # key_file_password = Settings.certs.key_pass,
            verify_ssl_certs=False,
        )

        return embeddings

    def _parse_answer(
        self, answer: BaseMessage, llm_answer_parser: Literal["none", "json", "string"]
    ) -> BaseMessage | str | dict[str, str]:
        match llm_answer_parser:
            case "none":
                pass

            case "json":
                parser = JsonOutputParser()
                try:
                    answer = parser.invoke(answer)
                except Exception as error:
                    logging.error(error)
                    logging.info("Trying to parse with `ast` python module")
                    # Try parsing as Python literal if JSON fails
                    try:
                        answer = ast.literal_eval(answer.content)
                    except Exception as error:
                        logging.error(error)
                        answer = ERROR_MESSAGE

            case "string":
                parser = StrOutputParser()
                answer = parser.invoke(answer)

        return answer

    def get_emdeddings(self) -> GigaChatEmbeddings:
        return self.embeddings

    def llm_answer(
        self,
        messages: list[BaseMessage] | str | list[str],
        llm_answer_parser: Literal["none", "json", "string"] = "none",
    ) -> BaseMessage | str | dict[str, str]:  # depends on an output parser
        # Message type
        if type(messages) is str:
            messages = [HumanMessage(messages)]

        # Try to get the answer - if there are no errors appear
        try:
            # LLM call
            answer = self.llm.invoke(messages)
            logging.info(f"LLM's answer: {answer}")

            # Output parser
            answer = self._parse_answer(answer, llm_answer_parser)
            logging.info(f"LLM's answer after parser: {answer}")

        # If they are - show the error message
        except Exception as error:
            logging.error(error)
            answer = ERROR_MESSAGE

        return answer
