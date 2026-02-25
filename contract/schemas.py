from uuid import UUID

from enum import Enum

from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


from contract.config import Settings


class Action(str, Enum):
    SELECT_ASUN_BUILDING = "SELECT_ASUN_BUILDING"
    SELECT_INNER_CLIENT = "SELECT_INNER_CLIENT"
    CREATE_TICKET = "CREATE_TICKET"


class ExtSystem(str, Enum):
    FRIEND = "FRIEND"


class TicketEntry(BaseModel):
    position: Optional[int] = Field(
        None,
        ge=0,
        examples=[10],
        description="Порядковый номер элемента описания заявки",
    )
    label: Optional[str] = Field(
        None, examples=["Сценарий"], description="Именование элемента описания заявки"
    )
    value: Optional[str] = Field(
        None,
        min_length=1,
        max_length=4000,
        examples=["Система контроля и управления доступом (СКУД)"],
        description="Значение элемента описания заявки",
    )

    def to_dict(self) -> dict:
        result = {"position": self.position, "label": self.label, "value": self.value}
        return result


class Option(BaseModel):
    """Опции элемента"""

    text: Optional[str] = Field(max_length=4096, description="Значение поля")
    label: Optional[str] = Field(max_length=2048, description="Наименование поля")
    id: Optional[str] = Field(max_length=1024)


class DynaElement(BaseModel):
    """Динамический элемент"""

    logicalType: Optional[str] = Field(None)
    objectType: Optional[str] = Field(None)
    id: Optional[str] = Field(None)
    label: Optional[str] = Field(None)
    mandatory: Optional[bool] = Field(False)
    sbcommand: Optional[str] = Field(None)
    sbmask: Optional[str] = Field(None)
    sbmodify: Optional[bool] = Field(False)
    sbtask: Optional[str] = Field(None)
    sbtype: Optional[str] = Field(None)
    sbstyle: Optional[str] = Field(None)
    style: Optional[str] = Field(None)
    visible: Optional[bool] = Field(True)
    width: Optional[str] = Field(None)
    sbobject: Optional[str] = Field(None)
    sbdbfield: Optional[str] = Field(None)
    sbaction: Optional[str] = Field(None)
    sbmode: Optional[str] = Field(None)
    sbonload: Optional[str] = Field(None)
    sbcopyinfo: Optional[bool] = Field(False)
    childs: Optional[List[Dict]] = Field(None)
    groupid: Optional[str] = Field(None, description="Код группы элементов")
    text: Optional[str] = Field(None, description="Текстовое содержание компонента")
    options: Optional[List[Option]] = Field(None)
    type: Optional[str] = Field(None)
    multiline: Optional[bool] = Field(None)
    button: Optional[str] = Field(None)
    matchTable: Optional[str] = Field(None)
    matchField: Optional[str] = Field(None)
    query: Optional[str] = Field(None)
    hpcGroupByFields: Optional[str] = Field(None)
    subjectClass: Optional[str] = Field(None)
    sbfield: Optional[str] = Field(None)
    image: Optional[str] = Field(None)
    size: Optional[str] = Field(None)
    backgroundstyle: Optional[str] = Field(None)
    dcstyle: Optional[str] = Field(None)


class TicketData(BaseModel):
    callerId: Optional[str] = Field(
        None,
        examples=["1823818"],
        description="Табельный номер лица, инициировавшего заявку",
    )
    initiatorId: Optional[str] = Field(
        None,
        examples=["1823818"],
        description="Табельный номер лица, указанного пользователем в качестве контактного. "
        "Совпадает с табельным номером лица, инициировавшего заявку, если инициатор "
        "не указывал контактное лицо.",
    )
    extSystem: Optional[ExtSystem] = Field(
        "FRIEND",
        examples=["FRIEND"],
        description="Система исполнения заявки. Константа",
    )
    templateId: Optional[str] = Field(
        None,
        examples=["Ремонт_технических_средств_охраны_mip"],
        description="Идентификатор используемого шаблона",
    )
    templateName: Optional[str] = Field(
        None,
        examples=["Ремонт технических средств охраны"],
        description="Именование используемого шаблона",
    )
    description: Optional[str] = Field(
        None,
        max_length=128,
        examples=[
            "Ремонт технических средств охраны. Система контроля и управления доступом (СКУД)"
        ],
        description="Описание подготовленной заявки. Состоит из templateName + название сценария через '. '",
    )
    information: Optional[List[TicketEntry]] = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Массив элементов, описывающих непосредственно заявку",
    )
    scoringElements: Optional[List[DynaElement]] = Field(
        None,
        description="Массив структур типа DynaElement, используемымх для описания скоринга "
        "инцидента (на скольких сотрудников влияет, влияет ли на клиента, тип ВСП и прочее. Всегда пуст.",
    )
    optionElements: Optional[List[DynaElement]] = Field(
        None,
        min_length=1,
        max_length=128,
        description="Перечень элементов заявки, оформленных в элементы интерфейса для формирования тела "
        "созданной заявки в СберДруге",
    )
    treeId: Optional[str] = Field(None, description="Одно из обязательных полей заявки")


class CreateNewDialogRs(BaseModel):
    """Структура, описывающая ответ ИИ-агента на запрос нового диалога. С приветственным сообщением."""

    message: str = Field(
        ..., max_length=4000, description="Сообщение ИИ-агента для пользователя"
    )
    action: Action = Field(None)
    ticketData: Optional[TicketData] = Field(None)
    dialogId: UUID = Field(
        ...,
        examples=["e8a7a5fb-67f4-4306-93b3-379fa1cbaad7"],
        description="Индетификатор диалога. Генерируется ИИ-агентом при инциализации диалога",
    )


class MessageToAgentRs(BaseModel):
    """Ответ ИИ-агента на сообщение пользователя"""

    message: str = Field(
        ...,
        max_length=4000,
        examples=["Укажите адрес размещения ТСО"],
        description="Сообщение ИИ-агента для пользователя",
    )
    action: Optional[Action] = Field(
        None,
        description="""
            Ожидаемое действие. Отражает инструкцию, в соответствии с которой должен действовать PLю
            - SELECT_ASUN_BUILDING - Ожидание запроса пользователя с контекстом о здании.
            - SELECT_INNER_CLIENT - Ожидание запроса пользователя с контекстом о контактном лице.
            - CREATE_TICKET - Ожидание формирования обращения по результатам обработки ответа.
        """,
    )
    ticketData: Optional[TicketData] = Field(None)


class Name(BaseModel):
    lastname: Optional[str] = Field(
        None,
        min_length=1,
        max_length=512,
        examples=["Иванов"],
        description="Фамилия. SOURCE: Сервис UserService.",
    )
    firstname: Optional[str] = Field(
        None,
        min_length=1,
        max_length=512,
        examples=["Иван"],
        description="Имя. SOURCE: Сервис UserService.",
    )
    middlename: Optional[str] = Field(
        None,
        min_length=1,
        max_length=512,
        examples=["Иванович"],
        description="Отчество. SOURCE: Сервис UserService.",
    )


class AsunEntry(BaseModel):
    """
    Структура, описывающая расположение рабочего места Сотрудника SOURCE:
    - |AC| sberfriend/UsersService (кадровая информация) / fetchKadrById - для workPlaceLocation
    - Компонент AcAddressAsun - для context
    """

    asunId: str = Field(
        ...,
        examples=["RU/77/931"],
        description="Идентификатор здания размещения рабочего места в АС УН",
    )
    addr: str = Field(
        ...,
        examples=["г. Москва, пр-кт Кутузовский, 32 к3 стрБ, Б.05.05, Б.05.05.1"],
        description="Адрес размещения рабочего места. Структура: Адрес (по АС УН), название (номер) помещения, номер раб. места",
    )


class UserContext(BaseModel):
    """Структура, описывающая Пользователя. Используетсяя для идентификации собеседника в диалоге с ИИ-агентом, а также идентификации лица, указанного собеседником в заявке."""

    empid: str = Field(
        ...,
        examples=["1823818"],
        description="Идентификатор (табельный номер) собеседника - Внутреннего Клиента. SOURCE: Сервис User",
    )
    name: Name = Field(...)
    mobilePhoneNum: Optional[str] = Field(
        None, description="Мобильный телефонный номер SOURCE: Сервис UserService"
    )
    workPlaceLocation: Optional[AsunEntry] = Field(None)
    departamentCode: str = Field(
        ...,
        max_length=128,
        examples=["10323702"],
        description="Код подразделения, в котором работает сотрудник SOURCE:",
    )
    departamentName: str = Field(
        ...,
        max_length=256,
        examples=["Группа разработки"],
        description="Название подразделения, в котором работает сотрудник",
    )
    gosbCode: str = Field(
        ...,
        max_length=256,
        examples=["ГОСБ"],
        description="Название ГОСБа",
    )
    terbankCode: str = Field(
        ...,
        max_length=256,
        examples=["Сибирский банк"],
        description="Наименование территориального банка",
    )

    def _compile_name(self):
        return f"{self.name.lastname} {self.name.firstname} {self.name.middlename}"


class UserValidation(BaseModel):
    message: str
    user: Dict
    user_validated: Optional[bool | Literal["in progress"]]
    action: Optional[
        Literal["CREATE_TICKET", "SELECT_ASUN_BUILDING", "SELECT_INNER_CLIENT"]
    ]
    tool_call_id: str


class UserContextNoLocation(BaseModel):
    """Структура, описывающая Пользователя. Используетсяя для идентификации собеседника в диалоге с ИИ-агентом, а также идентификации лица, указанного собеседником в заявке."""

    empid: str = Field(
        ...,
        examples=["1823818"],
        description="Идентификатор (табельный номер) собеседника - Внутреннего Клиента. SOURCE: Сервис User",
    )
    name: Name = Field(...)
    mobilePhoneNum: Optional[str] = Field(
        None, description="Мобильный телефонный номер SOURCE: Сервис UserService"
    )
    departamentCode: str = Field(
        ...,
        max_length=128,
        examples=["10323702"],
        description="Код подразделения, в котором работает сотрудник SOURCE:",
    )
    departamentName: str = Field(
        ...,
        max_length=256,
        examples=["Группа разработки"],
        description="Название подразделения, в котором работает сотрудник",
    )

    def _compile_name(self):
        return f"{self.name.lastname} {self.name.firstname} {self.name.middlename}"


class MessageToAgentRq(BaseModel):
    """Запрос пользователя в адрес ИИ-агента"""

    # todo нужен или все-таки не нужен min_length
    message: str = Field(
        ...,
        max_length=4000,
        description="Текстовое сообщение пользователя в адрес ИИ-агента",
    )
    context: Optional[Union[UserContext, AsunEntry]] = Field(None)


class ErrorResponse(BaseModel):
    """Базовая структура для ответа с ошибкой"""

    reason: Optional[str] = Field(
        None, examples=["DUPLICATE"], description="Причина возникновения ошибки"
    )
    code: str = Field(
        "AuroraTSO-0000",
        examples=["AuroraTSO-0000"],
        description="Код операции от сервера",
    )
    errorUserMessage: Optional[str] = Field(
        None, description="Сообщение об ошибке для отображения пользователю"
    )
    stackTrace: Optional[str] = Field(None, description="StackTrace ошибки от сервера")
    totalCount: Optional[int] = Field(
        None, description="Количество элементов, используется в списковых запросах"
    )
    success: Optional[bool] = Field(
        True, description="true - запрос успешно обработан сервером"
    )


class TaskResponse(BaseModel):
    status: Optional[str] = Field(
        "in_progress",
        examples=["in_progress", "done", "error"],
        description="Статус выполнения задачи",
    )
    result: Optional[MessageToAgentRs] = Field(
        None, description="Ответ агента пользователю"
    )


class CreateTaskRs(BaseModel):
    task_id: str = Field(..., description="ID созданной задачи")
