import os
import sys
from colorama import init, Fore, Style, Back

from langchain_gigachat import GigaChat

# Get API key from environment
api_key = os.getenv("GIGACHAT_API_KEY")
if not api_key:
    print(f"{Fore.RED}Error: GIGACHAT_API_KEY not found in environment variables{Style.RESET_ALL}")

def get_model(tools_list):
    model = GigaChat(
                credentials=api_key,
                scope="GIGACHAT_API_CORP",
                model="GigaChat-2-Max",
                base_url="https://gigachat-preview.devices.sberbank.ru/api/v1",
                verify_ssl_certs=False,
                profanity_check=False
            )

    if tools_list:
        model = model.bind_tools(tools_list)
        
    return model