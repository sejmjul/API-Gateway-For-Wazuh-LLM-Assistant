"""This file contains the session model for the application."""

from typing import (
    TYPE_CHECKING,
    List,
)

from sqlmodel import (
    Field,
    Relationship,
)

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.user import User


class Session(BaseModel, table=True):
    """Session model for storing chat sessions.

    Attributes:
        id: The primary key
        user_id: Foreign key to the user
        name: Name of the session (defaults to empty string)
        created_at: When the session was created
        messages: Relationship to session messages
        user: Relationship to the session owner
    """

    id: str = Field(primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    name: str = Field(default="")
    user: "User" = Relationship(back_populates="sessions")

    async def get_response(self, agent, chat_request):
        result = await agent.get_response(
            chat_request.messages, session_id="wazuh", user_id="wazuh"
        )
        return result
