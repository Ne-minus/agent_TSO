import asyncio
import logging
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document


class FaissSearch:
    def __init__(
        self,
        embeddings: Embeddings,
        vectorstore_path: Optional[str] = None,
    ) -> None:
        """Инициализация FAISS-поиска."""
        self.path = vectorstore_path
        self.vector_store_faiss = FAISS.load_local(
            folder_path=self.path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logging.info("[INIT]: Faiss vector store loaded")

    # -------------------------
    # Основные async-функции
    # -------------------------

    async def similarity_search(
        self,
        query: str,
        k: int,
        score_threshold: Optional[float] = None,
        filter: Optional[dict[str, str]] = None,
    ) -> list[Document]:
        """Асинхронный поиск по FAISS (через поток)."""
        results = await asyncio.to_thread(
            self.vector_store_faiss.similarity_search_with_relevance_scores,
            query=query,
            k=k,
            score_threshold=score_threshold,
            filter=filter,
        )
        results = [doc for doc, _ in results]
        return results

    async def find_by_ids(self, ids: list[str]) -> list[Document]:
        """Асинхронное получение документов по ID."""
        results = await asyncio.to_thread(self.vector_store_faiss.get_by_ids, ids)
        return results

    async def scenario_search(
        self,
        query: str,
        k: int,
        score_threshold: Optional[float] = None,
        filter: Optional[dict[str, str]] = None,
    ) -> list[dict]:
        """Асинхронный поиск сценариев (через FAISS + обработка)."""
        top_k = await self.similarity_search(
            query=query, k=k, score_threshold=score_threshold, filter=filter
        )

        scenarios = []
        for doc in top_k:
            node = doc.metadata.get("node")
            if node and node not in [s["scenario_obj"] for s in scenarios]:
                scenarios.append(
                    {
                        "scenario_obj": node,
                        "scenario": node._pretty_print(),
                        "questions_to_ask": node._questions(),
                    }
                )

        return scenarios
