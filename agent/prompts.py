from pathlib import Path
from typing import List

import yaml

# TODO come up with something better than this
# prompts_path = Path("./resources") / "prompts.yaml"
prompts_path = "prompts.yaml"
with open(prompts_path, 'r', encoding='utf-8') as f:
    prompts = yaml.safe_load(f)


def create_system_prompt() -> str:
    return prompts['system_prompt']


def get_react_instructions() -> str:
    return prompts['react_instructions']
