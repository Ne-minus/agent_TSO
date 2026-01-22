import yaml

prompts_path = "small_agent/prompts/prompt_main.yaml"

with open(prompts_path, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)


def create_prompt() -> str:
    return prompts["system_prompt"]
