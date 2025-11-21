import pandas as pd
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from uuid import uuid4

from agent.config import Settings

from typing import Literal, List


# ----------------
# Setup Vectorbase
# ----------------
class FileEntity(BaseModel):
    path: str
    name: str

    @property
    def extension(self):
        return self.name.split(".")[-1]


# --------
# Scenario
# --------
class ParameterSpecification(BaseModel):
    name: str
    description: str
    examples: str
    hint: str | None = None
    out_of_system: str | None = None

    @property
    def description_examples(self) -> str:
        return f"{self.description}\n\n{self.examples}"

    def _pretty_print(self):
        result = f"Названине параметра для заполнения: {self.name}"
        result += f"Описание: {self.description}\nПримеры: {self.examples}\nПодсказка: {self.hint}\nЗапрос в стороннюю систему: {self.out_of_system}"

        return result


class ExitScenarioPoint(BaseModel):
    ticket_name: str
    branch: str
    scenario: str
    description: str
    questions: str
    answers: str
    examples: List[str]
    parameters: List[ParameterSpecification]

    def _pretty_print(self):
        msg = "ЗАЯВКА\n"
        msg += f"{self.branch} - {self.scenario}\n"
        msg += f"Описание: {self.description}"
        return msg

    def _questions(self):
        return (
            f"Вопросы, чтобы задать пользователю для уточнения заявки: {self.questions}"
        )


def load_data(path: str) -> pd.DataFrame:
    """Load data from the given path"""
    data = pd.read_excel(path)
    # If there are any empty cells, replace them to None value
    data = data.replace({float("nan"): None})
    return data


def get_unique_keys(data: pd.DataFrame) -> list[tuple[str, str]]:
    """Get the unique combination of primal keys

    By default it is this keys:
        - branch
        - scenario

    That keys MUST be in the given dataset
    """
    unique_branch_scenario = []

    for _, point in data.iterrows():
        br = point["branch"]
        sc = point["scenario"]
        if (br, sc) not in unique_branch_scenario:
            unique_branch_scenario.append((br, sc))

    return unique_branch_scenario


def filter_data_on_keys(data: pd.DataFrame, branch: str, scenario: str):
    """Filter data by the primal keys: the unqiue combination of branch and scenario"""
    return data[(data["branch"] == branch) & (data["scenario"] == scenario)]


def make_specification(filter_data: pd.DataFrame) -> dict:
    """Get the node specification based on filtered data

    Specification contains this params:
        - branch
        - scenario
        - description
        - parameters
            - type: description
    """
    specification = {}

    specification["branch"] = filter_data.iloc[0]["branch"]
    specification["scenario"] = filter_data.iloc[0]["scenario"]
    # print(filter_data[filter_data["name"] == "type"]["description"])
    specification["description"] = filter_data[filter_data["name"] == "type"][
        "description"
    ].item()
    specification["examples"] = (
        filter_data[filter_data["name"] == "type"]["examples"].item().split("\n")
    )

    filter_data = filter_data[filter_data["name"] != "type"]

    params: list[ParameterSpecification] = []
    for _, line in filter_data.iterrows():
        param = ParameterSpecification(
            name=line["name"],
            description=line["description"],
            examples=line["examples"],
            hint=line["hint"],
            out_of_system=line["out_of_system"],
        )
        params.append(param)

    specification["parameters"] = params

    return specification


def make_exit_nodes(data: pd.DataFrame) -> list[ExitScenarioPoint]:
    """Make the exit nodes based on `ExitScenarioPoint` schema with given dataset"""
    unique_branch_scenario = get_unique_keys(data)
    exit_nodes = []

    for branch, scenario in unique_branch_scenario:
        filter_data = filter_data_on_keys(data, branch, scenario)
        specification = make_specification(filter_data)
        # print(specification)
        node = ExitScenarioPoint(**specification)
        exit_nodes.append(node)

    return exit_nodes


def parse_excel(path: str) -> List[ExitScenarioPoint]:
    df = pd.read_excel(path).fillna("")

    exit_nodes = []
    # группируем по ticket_name
    # print(df.columns)
    for ticket, group in df.groupby("ticket_name"):
        # обязательные поля
        print("GROUP: ", group)
        print(group[group["field_name"] == "branch"])
        branch = group[group["field_name"] == "branch"]["field_value"].iloc[0]
        scenario = group[group["field_name"] == "scenario"]["field_value"].iloc[0]
        description = group[group["field_name"] == "ticket_description"][
            "field_value"
        ].iloc[0]
        questions = group[group["field_name"] == "questions"]["field_value"].iloc[0]
        answers = group[group["field_name"] == "answers"]["field_value"].iloc[0]

        examples = (
            group[group["field_name"] == "ticket_description"]["examples"]
            .iloc[0]
            .split("\n")
            if "examples" in group.columns
            and not group[group["field_name"] == "ticket_description"]["examples"]
            .isna()
            .all()
            else []
        )

        # остальные параметры
        params = []
        for _, row in group.iterrows():
            if row["field_name"] not in [
                "branch",
                "scenario",
                "ticket_description",
                "questions",
                "answers",
            ]:
                param = ParameterSpecification(
                    name=row["field_name"],
                    description=row["field_value"],
                    examples=row["examples"] if row["examples"] else "",
                    hint=None,
                    out_of_system=None,
                )
                params.append(param)

        node = ExitScenarioPoint(
            ticket_name=ticket,
            branch=branch,
            scenario=scenario,
            description=description,
            questions=questions,
            answers=answers,
            examples=examples,
            parameters=params,
        )
        exit_nodes.append(node)

    return exit_nodes


def make_documents(exit_nodes: list[ExitScenarioPoint]) -> list[Document]:
    """Make Langchain Documents based on ExitScenarioPoint

    The document will contain:
        - page_content: node.description (for embeddings)
        - metadata.node: node (for link)
    """
    documents = []

    for node in exit_nodes:
        page_content = f"{node.ticket_name}\n{node.description}"
        doc = Document(page_content=page_content, metadata={"node": node})
        documents.append(doc)

    return documents


# ---------
# Knowledge
# ---------
def data_to_txt(data: pd.DataFrame, save_path: str) -> None:
    n = len(data)
    questions = data["Вопрос"].to_list()
    answers = data["Ответ"].to_list()

    final = []

    for i in range(n):
        try:
            txt = questions[i] + "\n" + answers[i] + "\n\n"
            final.append(txt)
        except Exception as e:
            pass  # If there is no answer text, we don't create a Chunk

    with open(save_path, mode="w+t", encoding="utf-8") as file:
        file.writelines(final)


def get_standard_splitter() -> CharacterTextSplitter:
    """Возвращает стандартный CharacterTextSplitter (для подготовленных по структуре заранее документов) со следующими характеристиками:\n
    - separator = 'двойной перенос строки',
    - chunk_size = 100,
    - chunk_overlap = 0
    """
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter


def add_metadata(doc: Document) -> Document:
    doc.metadata["id"] = str(uuid4())

    # Вопрос пользователя
    phrase = doc.page_content.split("\n")[0]
    doc.metadata["question"] = phrase

    return doc


def load_and_split_file(path_to_file: str, source_name: str) -> list[Document]:
    splitter = get_standard_splitter()

    with open(path_to_file, encoding="utf8") as f:
        document_original = f.read()

    document_split = splitter.create_documents(
        texts=[document_original], metadatas=[{"source": source_name}]
    )

    for doc in document_split:
        doc = add_metadata(doc)

    return document_split


if __name__ == "__main__":
    data = load_data(Settings.docs.tso_scenario.full_docs_path)
    exit_nodes = make_exit_nodes(Settings.docs.tso_scenario.full_docs_path)
    docs = make_documents(exit_nodes)

    print(f"Всего выходных узлов: {len(docs)}")

    for doc in docs:
        print(doc)
        print()
        print()
