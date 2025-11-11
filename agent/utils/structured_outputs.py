from typing import Optional, Literal
from pydantic import BaseModel, Field


class Decision(BaseModel):
    """Ответ модели с уточнениями по выбору заявки"""

    ticket_chosen: Optional[str] = Field(
        default=None, description="Название выбранной в финале заявки"
    )
    if_comment: Optional[str] = Field(
        default=None, description="Значение поля if_comment для выбранной заявки"
    )
    response: str = Field(..., description="Объяснение действий, chain of thoughts")


class NodeRouting(BaseModel):
    """Ответ модели с уточнениями по переходам"""

    node: Optional[str] = Field(
        default=None,
        description="Нода, в которую необходимо перейти, если нужно создать заявку. Если пользователь просто задает вопрос, возвращаем None",
    )
    response: str = Field(..., description="Ответ агента, его рассуждения")
