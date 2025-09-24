import faiss
import logging
from langchain_community.vectorstores import FAISS

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document


class FaissSearch:
    def __init__(
        self,
        embeddings: Embeddings,
        vectorstore_path: str | None = None,
    ) -> None:

        # Here we load already made index
        # (for setting up we'd have separate class)

        self.path = vectorstore_path
        self.vector_store_faiss = FAISS.load_local(
            folder_path=self.path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logging.info("[INIT]: Faiss vector store loaded")

    def similarity_search(
        self,
        query: str,
        k: int,
        score_threshold: float | None = None,
        filter: dict[str, str] | None = None,
    ) -> list[Document]:

        results = self.vector_store_faiss.similarity_search_with_relevance_scores(
            query=query, k=k, score_threshold=score_threshold, filter=filter
        )
        results = [doc for doc, score in results]
        logging.info(f"[INFO]: We get results: {results}")
        return results

    def find_by_ids(self, ids: list[str]) -> list[Document]:
        results = self.vector_store_faiss.get_by_ids([ids])
        return results

    def scenario_search(
        self,
        query: str,
        k: int,
        score_threshold: float | None = None,
        filter: dict[str, str] | None = None,
    ) -> list[Document]:
        """
        Search for scenarios and its parameters.
        """
        top_k = self.similarity_search(
            query=query, k=k, score_threshold=score_threshold, filter=filter
        )
        exit_scenarios = []
        for doc in top_k:
            node = doc.metadata["node"]
            if node not in exit_scenarios:

                exit_scenarios.append(
                    {
                        "id": doc.id,
                        "scenario": node._pretty_print(),
                        "questions_to_ask": node._questions(),
                    }
                )

        print(f"RESULTS: {top_k}")

        return exit_scenarios
