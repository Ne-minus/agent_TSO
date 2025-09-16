"""
The specification and load for the yaml config file. It is using OmegaConf for yaml hierarchical structure.

Most of the time you'll need the `Settings` variable with configuration specification.

You can run this .py file to print the config file and check if it is loaded correctly.
"""

from pathlib import Path
from pydantic import BaseModel
from omegaconf import OmegaConf, DictConfig

from typing import Literal

import logging

logger = logging.getLogger(__name__)

# Get the path to the yaml config file
pwd = Path(__file__).parent.parent
config_file = pwd / "config.yaml"


# ---------------------------------
# The whole configuration structure
# ---------------------------------
class DocsBase(BaseModel):
    main_dir: str
    docs_path: str
    vector_store_path: str
    top_k: int
    score_threshold: float | None

    @property
    def full_docs_path(self) -> str:
        return self.main_dir + self.docs_path

    @property
    def full_vector_store_path(self) -> str:
        return self.main_dir + self.vector_store_path


class Docs(BaseModel):
    main_extension: str
    knowledge: DocsBase
    tso_scenario: DocsBase


class System(BaseModel):
    silent: bool
    logging_file: str
    # LEVELS: 10 - Debug, 20 - Info, 30 - Warning, 40 - Error, 50 - Critical Error
    logging_level: Literal[10, 20, 30, 40, 50]
    ###
    device: Literal["cpu", "cuda", "mps"]
    inside_docker_container: bool
    ###
    ask_for_requester: bool
    knowledge_valuation: bool
    answer_without_knowledge: bool


class Models(BaseModel):
    embeddings: str
    llm_model_type: Literal[
        "gemma3:4b",
        "gemma3:27b",
        "GigaChat-2-Max",
        "GigaChat-2-Pro",
        "qwen2.5:3b",
        "qwen3:8b",
    ]
    llm_base_url: str
    language: Literal["ru", "en"]


class SomeApiSettings(BaseModel):
    port: int
    host: str
    next_line_placeholder: str


class Config(BaseModel):
    system: System
    docs: Docs
    models: Models
    api: SomeApiSettings


# Load the yaml config file
def _load_yaml_config(path: Path) -> DictConfig:
    """Load the yaml config file from the specific path.

    This function using OmegaConf for yaml hierarchical structure opportunity.

    Arguments
    ---------
    path: Path
        Path to the .yaml config file

    Returns
    -------
    config: DictConfig
        The config dictionary

    Raises
    ------
    FileNotFoundError
        If there is no .yaml config file in provided path
    """
    try:
        return OmegaConf.load(path)

    except FileNotFoundError as error:
        message = f"Error! There is no yaml file in {path}"
        logger.exception(message)
        raise FileNotFoundError(error, message) from error


Settings = Config(**_load_yaml_config(config_file))


if __name__ == "__main__":
    # We can run this .py file to check if the Settings was loaded correctly
    print(Settings.model_dump_json(indent=2))
