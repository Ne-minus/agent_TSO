from pathlib import Path
from typing import List

import yaml

# TODO come up with something better than this
# prompts_path = Path("./resources") / "prompts.yaml"
prompts_path = "agent/prompts/prompts_main.yaml"
tsv_path = "agent/prompts/prompts_tsv.yaml"
ts_path = "agent/prompts/prompts_ts.yaml"

with open(prompts_path, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

with open(tsv_path, "r", encoding="utf-8") as f:
    tsv_prompts = yaml.safe_load(f)

with open(ts_path, "r", encoding="utf-8") as f:
    ts_prompts = yaml.safe_load(f)

def create_system_prompt() -> str:
    return prompts["system_prompt"]


def get_react_instructions() -> str:
    return prompts["react_instructions"]


def get_ticket_prompt() -> str:
    return prompts["ticket_system_prompt"]


def get_formatting_prompt() -> str:
    return prompts["formatting_prompt"]


def get_scenario_prompt() -> str:
    return prompts["scenario_system_prompt"] + "\n" + prompts["scenario_instructions"]


def get_tsv_prompt() -> str:
    return tsv_prompts["tsv_system_prompt"] + "\n" + tsv_prompts["tsv_instructions"]

def get_ts_prompt() -> str:
    return ts_prompts["scenario_ts_system_prompt"] + "\n" + ts_prompts["scenario_ts_instructions"]