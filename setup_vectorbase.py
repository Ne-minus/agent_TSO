import os

from agent.neural.models import MLModelShell
from agent.retriever.vectorbase import FaissStoreHandler
from agent.retriever.data import FileEntity, load_data, data_to_txt
from contract.config import Settings

from dotenv import load_dotenv

from typing import Literal

import logging

logging.basicConfig(level=Settings.system.logging_level)


def vector_store_creation(
    mode: Literal["scenario", "knowledge"], model_shell: MLModelShell
):
    """Setup the Faiss vector store from scratch from raw excel (or txt) file

    Excel file MUST contain this fields:
        - branch
        - scenario
        - name
        - description
        - examples

    Txt file will be split by next line separator by default

    Every unique branch-scenario MUST have ONLY ONE type parameter - this
    will be used for the whole branch-scenario description
    """
    logging.info(
        f"[INIT]: Start to prepare vector store of {mode} and new documents..."
    )

    match mode:
        case "scenario":
            path_to_vector_store = Settings.docs.tso_scenario.full_vector_store_path
            path_to_docs = Settings.docs.tso_scenario.full_docs_path

        case "knowledge":
            path_to_vector_store = Settings.docs.knowledge.full_vector_store_path
            path_to_docs = Settings.docs.knowledge.full_docs_path

    match mode:
        case "scenario":
            prepared_docs = [doc.name for doc in os.scandir(path_to_docs)]

        case "knowledge":
            # Scan directory for files
            docs = [
                FileEntity(path=doc.path, name=doc.name)
                for doc in os.scandir(path_to_docs)
            ]

            # Remove everything except .xlsx and .txt files
            docs = [doc for doc in docs if doc.extension in ["xlsx", "txt"]]

            # Transform docs to main_extension file format
            prepared_docs = []
            for doc in docs:
                # Prepare some doc names and paths
                doc_path = path_to_docs + doc.name
                new_doc_name = (
                    f"{doc.name.split('.')[0]}.{Settings.docs.main_extension}"
                )
                new_doc_path = path_to_docs + new_doc_name

                match doc.extension:
                    case "xlsx":
                        data = load_data(doc_path)
                        data_to_txt(data, save_path=new_doc_path)

                    case "txt":
                        with open(doc_path, mode="r", encoding="utf-8") as file:
                            data = file.readlines()

                        # For now - we are splitting original text by \n. So, we have a readlines()
                        # function before, which already splitted by this separator. We just
                        # add a standard separator that we use here - \n\n - in the end of each line.
                        data = [line + "\n" for line in data if line != ""]

                        with open(new_doc_path, mode="w+", encoding="utf-8") as file:
                            file.writelines(data)

                prepared_docs.append(new_doc_name)

    logging.info(prepared_docs)

    logging.info("[INIT]: Creating new blank vector store")
    vector_store = FaissStoreHandler(
        embeddings=model_shell.get_emdeddings(),
        vectorstore_path=path_to_vector_store,
        create_new=True,
    )

    logging.info("[INIT]: Adding documents...")
    for doc in prepared_docs:
        logging.info(doc)
        doc_path = path_to_docs + doc
        vector_store.add_and_save_raw_files(
            path_to_file=doc_path,
            mode=mode,
            source=doc.split(".")[0],  # doc name without extension
        )

    logging.info("[INIT]: Saving...")
    vector_store.save(path_to_vector_store)

    logging.info(f"[INIT]: Path to actual vector store: {path_to_vector_store}")


if __name__ == "__main__":
    load_dotenv()

    logging.info("[INIT]: Models prepearing...")
    model_shell = MLModelShell(
        embeddings_name=Settings.models.embeddings,
        llm_model_type=Settings.models.llm_model_type,
    )

    vector_store_creation(mode="scenario", model_shell=model_shell)

    vector_store_creation(mode="knowledge", model_shell=model_shell)
