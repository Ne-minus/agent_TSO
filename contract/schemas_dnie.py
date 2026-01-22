from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    model_validator,
)
from typing_extensions import Annotated


class ResponseStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"


class HistoryItemType(str, Enum):
    human = "human"
    ai = "ai"


class HistoryItem(BaseModel):
    type: Optional[HistoryItemType] = Field(default=None)
    content: Optional[str] = Field(
        default=None,
        example="Не хватает информации для определения проблемы пользователя. Пожалуйста, уточните запрос.",
    )


class Response(BaseModel):
    thread_id: UUID
    content: str = Field(
        example="Не хватает информации для определения проблемы пользователя. Пожалуйста, уточните запрос."
    )
    history: Optional[List[HistoryItem]] = Field(default=None, min_length=1)
    status: ResponseStatus


class ValidationError(BaseModel):
    loc: List[Any] = Field(max_length=10, title="Location")
    msg: Annotated[
        str, StringConstraints(max_length=255, pattern=r"^[\w\W]{0,255}$")
    ] = Field(title="Message", description="description")
    type: Annotated[
        str, StringConstraints(max_length=255, pattern=r"^[\w\W]{0,255}$")
    ] = Field(title="Error Type", description="description")


class HTTPValidationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: Optional[List[ValidationError]] = Field(
        default=None, max_length=10, title="Detail"
    )
    Error: Optional[str] = Field(default=None, example="Name field is missing")


class Address(BaseModel):
    full_address: Optional[str] = Field(
        default=None,
        title="Полный адрес",
        example="ул Поклонная, 3к1, 11.B.01, 11.B.01.42",
    )
    city: Optional[str] = Field(default=None, title="Город", example="Москва")
    street: Optional[str] = Field(default=None, title="Улица", example="Поклонная")
    house: Optional[str] = Field(default=None, title="Дом", example="3к1")
    room: Optional[str] = Field(default=None, title="Помещение", example="11.B.01")


PersonalNumber = Annotated[
    str,
    StringConstraints(pattern=r"^(?!0)\d{1,10}$"),
]


class Person(BaseModel):
    """
    Контакт (заявитель или заказчик)

    В спецификации стоит:
      - required: personalNumber
      - oneOf: (fullname) OR (lastName+firstName)

    Ниже это реализовано валидатором (oneOf-правило).
    """

    fullname: Optional[str] = Field(default=None, example="Фамилия Имя Отчество")
    lastName: Optional[str] = Field(default=None, example="Фамилия")
    firstName: Optional[str] = Field(default=None, example="Имя")
    middleName: Optional[str] = Field(default=None, example="Отчество")

    personalNumber: str = Field(pattern=r"^\d{1,10}$", example="1898931")

    mobilePhone: Optional[Annotated[str, StringConstraints(max_length=17)]] = Field(
        default=None, example="+7(999)-888-77-66"
    )
    officePhone: Optional[Annotated[str, StringConstraints(max_length=10)]] = Field(
        default=None, example="8-78911625"
    )

    @model_validator(mode="after")
    def _check_one_of_name_variants(self) -> "Person":
        has_fullname = bool(self.fullname)
        has_split = bool(self.lastName) and bool(self.firstName)
        if not (has_fullname or has_split):
            raise ValueError(
                "Person: требуется либо 'fullname', либо пара 'lastName' + 'firstName'."
            )
        return self


class Incident(BaseModel):
    author: Person
    position: str = Field(title="Должность Заявителя", example="Сервисный менеджер")
    department: str = Field(title="Подразделение Заявителя", example="Сервисный центр")

    address: Optional[Address] = None
    timezone: Optional[str] = Field(
        default=None, title="Часовой пояс Пользователя", example="03"
    )
    attachments: Optional[List[HttpUrl]] = Field(
        default=None, title="Вложения", max_length=10
    )


class CreateRunPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: Optional[UUID] = Field(
        default=None,
        title="Thread Id",
        description="id чата. При отсутствии thread_id агент создает новый чат",
    )
    input: str = Field(
        title="Вопрос к агенту", example="У меня сломался стул на 13 этаже Поклонная 3"
    )
    additional_data: Optional[Incident] = None
