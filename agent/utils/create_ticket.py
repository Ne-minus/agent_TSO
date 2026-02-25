from pydantic import BaseModel
from agent.graph_structure.state import AgentState

from contract.schemas import TicketData, ExtSystem, TicketEntry

from contract.schemas import UserContext, AsunEntry


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
    user: UserContext
    requester: UserContext
    branch: str
    scenario: str
    description: str
    parameters: dict
    building: AsunEntry
    initial_problem: str

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

    def _create_scenario_request(self, state: AgentState) -> ScenarioRequest:
        # print("user_info")
        request = ScenarioRequest(
            user=state.get("user_info"),
            requester=state.get("real_requester"),
            branch=state.get("chosen_scenario").branch,
            scenario=state.get("chosen_scenario").scenario,
            description=state.get("chosen_scenario").description,
            parameters=state.get("ticket_data"),
            building=state.get("current_building"),
            initial_problem=state.get("last_user_message"),
        )
        return request

    def _fill_request(
        self, request: ScenarioRequest, information: list[TicketEntry]
    ) -> list[TicketEntry]:
        for field in information:
            match field.label:
                case "Телефон для связи":
                    field.value = request.requester.mobilePhoneNum
                case "id объекта":
                    field.value = request.building.asunId
                case "Адрес":
                    field.value = request.building.addr
                case "Сценарий":
                    field.value = request.scenario

                case "":
                    if field.position == 1:
                        field.value = request.scenario

                case "Подразделение":
                    field.value = request.requester.departamentName

                case "Орг. единица":
                    field.value = request.requester.departamentCode

                case "Неисправность":
                    field.value = request.description

                case "Комментарий":

                    params = f"Основная проблема: {request.initial_problem}\n\n"
                    for param in request.parameters:
                        params += f"{request.parameters[param]['description']}: {request.parameters[param]['value']}\n"

                    field.value = params

                case "ФИО ВК":
                    field.value = request.requester._compile_name()

                case "ГОСБ":
                    field.value = request.requester.gosbCode

                case "Территориальный банк":
                    field.value = request.requester.terbankCode

                case _:
                    pass

        return information

    def _fill_information_field_in_request(
        self, request: ScenarioRequest
    ) -> list[dict[str, str | int]]:
        blank_information = self._create_blank_information_for_request()

        information = self._fill_request(request, blank_information)
        information = [field.to_dict() for field in information]

        return information

    def _create_ticket_data_for_api(self, request: ScenarioRequest) -> TicketData:
        information = self._fill_information_field_in_request(request)

        result = TicketData(
            callerId=request.user.empid,
            initiatorId=request.requester.empid,
            extSystem=ExtSystem.FRIEND,
            templateId=f"{request.branch}_mip",
            templateName=request.branch,
            description=f"{request.branch}. {request.scenario}",
            information=information,
        )
        return result

    def create_ticket(self, state: AgentState) -> TicketData:
        request = self._create_scenario_request(state)
        ticket = self._create_ticket_data_for_api(request)
        return ticket
