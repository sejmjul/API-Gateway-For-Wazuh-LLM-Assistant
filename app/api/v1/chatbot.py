"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Header,
)
from fastapi.responses import StreamingResponse
from app.core.metrics import llm_stream_duration_seconds
from app.core.config import settings
from app.core.langgraph.graph import LangGraphAgent
from app.core.limiter import limiter
from app.core.logging import logger
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Message,
    StreamResponse,
)

router = APIRouter()
agent = LangGraphAgent()

async def require_api_key(x_api_key: str = Header(..., alias="X-API-KEY")):
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    authorized: bool = Depends(require_api_key),
):
    """Process a chat request using LangGraph.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        authorized: Whether the request is authorized via API key.

    Returns:
        ChatResponse: The processed chat response.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        # Use static session ID and user ID for API key auth
        session_id = "wazuh-api"
        user_id = "wazuh-api"
        
        logger.info(
            "chat_request_received",
            session_id=session_id,
            message_count=len(chat_request.messages),
        )

        result = await agent.get_response(
            chat_request.messages, session_id, user_id=user_id
        )

        logger.info("chat_request_processed", session_id=session_id)

        return ChatResponse(messages=result)
    except Exception as e:
        logger.error("chat_request_failed", session_id="wazuh-api", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    authorized: bool = Depends(require_api_key),
):
    """Process a chat request using LangGraph with streaming response.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        authorized: Whether the request is authorized via API key.

    Returns:
        StreamingResponse: A streaming response of the chat completion.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        # Use static session ID and user ID for API key auth
        session_id = "wazuh-api"
        user_id = "wazuh-api"
        
        logger.info(
            "stream_chat_request_received",
            session_id=session_id,
            message_count=len(chat_request.messages),
        )

        async def event_generator():
            """Generate streaming events."""
            try:
                full_response = ""
                with llm_stream_duration_seconds.labels(model=agent.model_name).time():
                    async for chunk in agent.get_stream_response(
                        chat_request.messages, session_id, user_id=user_id
                     ):
                        full_response += chunk
                        response = StreamResponse(content=chunk, done=False)
                        yield f"data: {json.dumps(response.model_dump())}\n\n"

                # Send final message indicating completion
                final_response = StreamResponse(content="", done=True)
                yield f"data: {json.dumps(final_response.model_dump())}\n\n"

            except Exception as e:
                logger.error(
                    "stream_chat_request_failed",
                    session_id=session_id,
                    error=str(e),
                    exc_info=True,
                )
                error_response = StreamResponse(content=str(e), done=True)
                yield f"data: {json.dumps(error_response.model_dump())}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(
            "stream_chat_request_failed",
            session_id="wazuh-api",
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


# Keep the session-based endpoints if needed, or remove if not
@router.get("/messages", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def get_session_messages(
    request: Request,
    x_api_key: str = Header(..., alias="X-API-KEY"),
):
    """Get all messages for a session."""
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        session_id = "wazuh-api"
        messages = await agent.get_chat_history(session_id)
        return ChatResponse(messages=messages)
    except Exception as e:
        logger.error("get_messages_failed", session_id="wazuh-api", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/messages")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def clear_chat_history(
    request: Request,
    authorized: bool = Depends(require_api_key),
):
    """Clear all messages for a session."""
    try:
        session_id = "wazuh-api"
        await agent.clear_chat_history(session_id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error("clear_chat_history_failed", session_id="wazuh-api", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))