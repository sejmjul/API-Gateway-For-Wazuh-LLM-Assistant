"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
import re
from typing import List, AsyncGenerator

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


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def chat(
    request: Request,
    chat_request: ChatRequest,
):
    """Process a stateless chat request.
    
    The client sends the full conversation history with each request.
    No database persistence - everything is processed fresh each time.
    """
    try:
        # Extract user message for logging
        user_message = chat_request.messages[-1].content if chat_request.messages else ""

        logger.info("chat_request_received", message_count=len(chat_request.messages), user_message=user_message[:200])
        print(f"\n{'-'*50}\n💬 User: {user_message}\n{'-'*50}")

        # Process the request (tools will be called if needed)
        result = await agent.get_response(
            messages=chat_request.messages,
            session_id=None,  # Optional: could be passed from client
            user_id="wazuh"
        )

        # Extract final answer
        clean_response = extract_final_answer(result)
        
        logger.info("chat_response_generated", response_length=len(clean_response))
        print(f"\n🤖 Assistant: {clean_response[:100]}...\n{'-'*50}")

        return ChatResponse(messages=[Message(role="assistant", content=clean_response)])
        
    except Exception as e:
        logger.error("chat_request_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
):
    """Process a stateless streaming chat request."""
    try:
        user_message = chat_request.messages[-1].content if chat_request.messages else ""
        logger.info("stream_request_received", message_count=len(chat_request.messages))
        print(f"\n{'-'*50}\n💬 User (Stream): {user_message}\n{'-'*50}")

        async def event_generator():
            try:
                full_response = ""
                async for chunk in agent.get_stream_response(
                    messages=chat_request.messages,
                    user_id="wazuh"
                ):
                    full_response += chunk
                    yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"

                print(f"\n🤖 Assistant (Stream): {full_response[:100]}...\n{'-'*50}")
                yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"

            except Exception as e:
                logger.error("stream_failed", error=str(e), exc_info=True)
                yield f"data: {json.dumps({'content': str(e), 'done': True})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error("stream_request_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def extract_final_answer(result: list[Message]) -> str:
    """Extract clean final answer from LLM response.
    
    Args:
        result: List of messages from the LLM
        
    Returns:
        str: Clean final answer text
    """
    if not result:
        return "I apologize, but I couldn't generate a response."
    
    # Get the last assistant message
    content = result[-1].content if hasattr(result[-1], 'content') else str(result[-1])
    
    # Try to extract from JSON format
    json_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if matches:
        try:
            json_obj = json.loads(matches[-1])
            if "final_answer" in json_obj:
                return json_obj["final_answer"]
        except:
            pass
    
    # Fallback: return as-is
    return content


# Remove these endpoints - not needed for stateless operation:
# - /cache (no database to clear)
# - /messages (no history to retrieve)