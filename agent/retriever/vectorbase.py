import faiss
from langchain_community.vectorstores import FAISS

from langchain_core.embeddings import Embeddings

from uuid import uuid4
from langchain_community.docstore.in_memory import InMemoryDocstore

from .data import (
    load_data, 
    make_exit_nodes, 
    make_documents, 
    load_and_split_file
)

from typing import Literal
from langchain_core.documents import Document

import logging


class FaissStoreHandler():
    """Additional class for faiss vector store manipulation"""
    def __init__(
        self, 
        embeddings: Embeddings,
        vectorstore_path: str | None = None,
        create_new: bool = False
    ) -> None:
        if create_new:
            self.vector_store_faiss = FAISS(
                embedding_function = embeddings,
                index = faiss.IndexFlatL2(len(embeddings.embed_query("Генрих Герц"))),
                docstore = InMemoryDocstore(),
                index_to_docstore_id = {},
            )
            logging.info("[INIT]: Blank faiss vector store created")            
        
        else:
            self.vector_store_faiss = FAISS.load_local(
                folder_path = vectorstore_path,
                embeddings = embeddings,
                allow_dangerous_deserialization = True
            )
            logging.info("[INIT]: Faiss vector store loaded")
        
        self.path = vectorstore_path
    
    def add_documents(self, split_documents: list[Document]) -> None:
        """Add list of Langchain Documents to faiss vector store"""
        uuids = [str(uuid4()) for _ in range(len(split_documents))]
        self.vector_store_faiss.add_documents(documents = split_documents, ids = uuids)
        logging.info("[INIT]: Document added to faiss vector store")
    
    def similarity_search(
            self, 
            query: str, 
            k: int, 
            score_threshold: float | None = None,
            filter: dict[str, str] | None = None
        ) -> list[Document]:
        """Search for the most relevant chunks to the user's query

        k parameter - how many documents need to retrieve
        """
        results = self.vector_store_faiss.similarity_search_with_relevance_scores(
            query = query,
            k = k,
            score_threshold = score_threshold,
            filter = filter
        )
        results = [doc for doc, score in results]
        return results
    
    def find_by_ids(self, ids: list[str]) -> list[Document]:
        results = self.vector_store_faiss.get_by_ids(ids)
        return results
    
    def save(self, path: str) -> None:
        """Save this Faiss vector store to the given path"""
        self.vector_store_faiss.save_local(path)
        logging.info(f"[INIT]: Faiss vector store saved in {path}")
    
    def add_and_save_raw_files(
        self,
        path_to_file: str,
        mode: Literal["scenario", "knowledge"],
        path_to_new_file: Literal["same"] | str | None = None,
        source: str | None = None
    ) -> None:
        """Add documents to this vector store from the raw .xlsx file
        
        Also could be saved in the vectorstore directory
        """
        match mode:
            case "scenario":
                data = load_data(path_to_file)
                exit_nodes = make_exit_nodes(data)
                docs = make_documents(exit_nodes)
            
            case "knowledge":
                if source is None:
                    source = path_to_file
                docs = load_and_split_file(
                    path_to_file = path_to_file,
                    source_name = source
                )

        self.add_documents(docs)

        if path_to_new_file is not None:
            if path_to_new_file == "same":
                self.save(self.path)
            
            else:
                self.save(path_to_new_file)