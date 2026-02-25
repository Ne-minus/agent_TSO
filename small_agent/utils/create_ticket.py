import re

from pydantic import BaseModel
from typing import Optional
from small_agent.graph_structure.state import State

from contract.schemas import TicketData, ExtSystem, TicketEntry

from contract.schemas_dnie import Address, Person


INFORMATION_LABELS = [
    "Сценарий",
    "",
    "Гарантийная работа",
    "Неисправность",
    "Требуется срочное решение",
    "Объект КВО?",
    "Территориальный банк",
    "ГОСБ",
    "Номер ВСП",
    "Адрес",
    "Этаж",
    "Помещение",
    "id объекта",
    "Телефон для связи",
    "Комментарий",
    "ФИО ВК",
    "Подразделение",
    "Орг. единица",
    "Расчет КС",
    "",
]


class ScenarioRequest(BaseModel):
    user: Person
    branch: str
    scenario: str
    department: str
    description: str
    building: Address
    initial_problem: Optional[str]
    abb: str

    def __str__(self):
        result = "Заявка\n"
        result += f"Пользователь: {self.user.name.last_name} {self.user.name.first_name} {self.user.name.middle_name}\n"
        result += f"Тип заявки: {self.branch}. {self.scenario}\n"
        result += "Дополнительные параметры:\n"
        for param in self.parameters.keys():
            result += f"{param}: {self.parameters[param]}\n"
        return result


class Formalize:
    def _create_blank_information_for_request(self) -> list[TicketEntry]:
        information = []
        for i, label in enumerate(INFORMATION_LABELS):
            item = TicketEntry(position=i, label=label)
            information.append(item)

        return information

    def _create_scenario_request(
        self,
        state: State,
        branch: str,
        scenario: str,
        description: str,
        initial_problem: str,
        abb: str,
    ) -> None:
        # print("user_info")
        self.request = ScenarioRequest(
            user=state.get("user_info"),
            department=state.get("user_department"),
            branch=branch,
            scenario=scenario,
            description=description,
            building=state.get("address"),
            initial_problem=initial_problem,
            abb=abb,
        )

    def _fill_request(self) -> list[TicketEntry]:

        information = self._create_blank_information_for_request()
        for field in information:
            match field.label:
                case "Телефон для связи":
                    field.value = self.request.user.mobilePhone
                case "Адрес":
                    field.value = self.request.building.full_address
                case "Сценарий":
                    field.value = self.request.branch

                case "":
                    if field.position == 1:
                        field.value = self.request.abb

                case "Подразделение":
                    field.value = self.request.department

                case "Неисправность":
                    field.value = self.request.description

                case "Комментарий":
                    field.value = (
                        f"{self.request.scenario}\n{self.request.initial_problem}"
                    )

                case "ФИО ВК":
                    field.value = self.request.user.fullname

                case "Помещение":
                    field.value = self.request.building.room

                case _:
                    pass

        return [field.to_dict() for field in information]

    def _create_ticket_data_for_api(self) -> TicketData:
        information = self._fill_request()

        result = TicketData(
            callerId=self.request.user.personalNumber,
            initiatorId=self.request.user.personalNumber,
            extSystem=ExtSystem.FRIEND,
            templateId=f"Ремонт_технических_средств_охраны_mip",
            templateName="Ремонт_технических_средств_охраны",
            description=f"Ремонт_технических_средств_охраны_mip. {self.request.branch}",
            information=information,
        )
        return result

    def create_ticket(self, state: State) -> TicketData:
        ticket = self._create_ticket_data_for_api()
        return ticket
