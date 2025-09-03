from agent.model.model_init import get_llm, get_embeddings
from .vectorbase import FaissStoreHandler
from .data import ExitScenarioPoint
from agent.config import Settings

from typing import Literal

import logging


class Retriever:
    def __init__(self, model):
        self.vector_stores = self._setup_vector_stores(get_llm())
        logging.info("[INIT]: Faiss vector stores were loaded")

    def _setup_vector_stores(self) -> dict[str, FaissStoreHandler]:
        scenario_vs = FaissStoreHandler(
            embeddings=get_embeddings(),
            vectorstore_path=Settings.docs.tso_scenario.full_vector_store_path,
        )

        knowledge_vs = FaissStoreHandler(
            embeddings=get_embeddings(),
            vectorstore_path=Settings.docs.knowledge.full_vector_store_path,
        )

        vector_stores = {"tso_scenario": scenario_vs, "all_knowledge": knowledge_vs}
        return vector_stores

    def retrieve(
        self,
        query: str,
        mode: Literal["tso_scenario", "tso_knowledge", "all_knowledge"],
        k: int,
        score_threshold: float | None = None,
    ) -> list[ExitScenarioPoint] | str:
        # Match right vector store
        if "knowledge" in mode:
            vector_store = self.vector_stores["all_knowledge"]
        else:
            vector_store = self.vector_stores[mode]

        # Check for filter
        filter = None
        # filter will be not-None if we have specific knowledge area
        if mode == "tso_knowledge":
            filter = None
            # filter = {"source": "<doc_name_without_extension>"}

        # Search for the closest blocks
        top_k = vector_store.similarity_search(
            query=query, k=k, score_threshold=score_threshold, filter=filter
        )

        # Prepare and return retrieved blocks
        ## Scenarios
        if mode == "tso_scenario":
            # Filter unique Scenario Exit Point
            exit_scenarios = []
            for doc in top_k:
                node = doc.metadata["node"]
                if node not in exit_scenarios:
                    exit_scenarios.append(node)

            return exit_scenarios

        ## Knowledge
        if mode in ["tso_knowledge", "all_knowledge"]:
            # Unite all retriever documetns in one string
            knowledge = "### Информация из базы знаний\n\n"
            for item in top_k:
                knowledge += item.page_content
                knowledge += "\n\n"

            return knowledge
