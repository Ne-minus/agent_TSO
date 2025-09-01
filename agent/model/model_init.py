import ast
import os

from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings

from agent.config import Settings


def get_embeddings() -> GigaChatEmbeddings:

    embeddings = GigaChatEmbeddings(
        model=Settings.models.llm_model_type,
        credentials=os.environ["GIGACHAIN_AUTH"],
        scope=os.environ["GIGACHAT_SCOPE"],
        verify_ssl_certs=False,
    )

    return embeddings


def get_llm() -> GigaChat:

    model = GigaChat(
        model=Settings.models.llm_model_type,
        credentials=os.environ["GIGACHAIN_AUTH"],
        scope=os.environ["GIGACHAT_SCOPE"],
        verify_ssl_certs=False,
        profanity_check=False,
    )

    return model
