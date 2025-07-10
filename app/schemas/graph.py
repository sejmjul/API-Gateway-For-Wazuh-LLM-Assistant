"""This file contains the graph schema for the application."""

import re
import uuid
from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class GraphState(BaseModel):
    """State definition for the LangGraph Agent/Workflow."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="The messages in the conversation"
    )
    session_id: str = Field(..., description="The unique identifier for the conversation session")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate that the session ID is a valid UUID or follows safe pattern.

        Args:
            v: The thread ID to validate

        Returns:
            str: The validated session ID

        Raises:
            ValueError: If the session ID is not valid
        """
        # Try to validate as UUID
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            # If not a UUID, check for safe characters only
            if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
                raise ValueError("Session ID must contain only alphanumeric characters, underscores, and hyphens")
            return v


class LangGraphAgent:
    def __init__(self):
        provider = settings.MODEL_PROVIDER
        if provider == "ollama":
            self.llm = ChatOllama(
                model=settings.llm_model,
                base_url=settings.LLM_BASE_URL,
                temperature=settings.DEFAULT_LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )
        elif provider == "openai":
            self.llm = ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                temperature=settings.DEFAULT_LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )
        else:
            raise ValueError(f"Unsupported MODEL_PROVIDER: {provider}")
        self.model_name = settings.llm_model
