import ast
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings


def get_embeddings() -> GigaChatEmbeddings:

    embeddings = GigaChatEmbeddings(
        model="Embeddings",
        credentials=os.environ["GIGACHAIN_AUTH"],
        scope=os.environ["GIGACHAT_SCOPE"],
        verify_ssl_certs=False,
    )

    return embeddings


def get_llm() -> GigaChat:

    model = GigaChat(
        # verbose=True,
        model=os.environ["LLM_MODEL_TYPE"],
        credentials=os.environ["GIGACHAIN_AUTH"],
        scope=os.environ["GIGACHAT_SCOPE"],
        verify_ssl_certs=False,
        profanity_check=False,
        timeout=60,
    )

    return model
