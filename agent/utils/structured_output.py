from typing import Optional, Union, Literal
from typing_extensions import Annotated, TypedDict


class UseTool(TypedDict):
    tool_name: Annotated[str, ..., "Имя инструмента, который нужно вызвать"]


class UserVerdict(TypedDict):
    response: Annotated[
        str,
        ...,
        "Ответ подьзователя, где он подверждает или опровергает заведение подобранной заявки.",
    ]


class FinalResponse(TypedDict):
    final_output: Union[UseTool, UserVerdict]
