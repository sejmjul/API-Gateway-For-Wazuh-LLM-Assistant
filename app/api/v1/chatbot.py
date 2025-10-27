"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
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
    
    Uses stateless processing - each request is handled independently
    without retrieving or storing conversation history in the database.
    This is optimal for Wazuh integration where each request contains
    the full conversation context.
    """
    try:
        # Extract the latest user message for logging
        user_message = ""
        if chat_request.messages and len(chat_request.messages) > 0:
            last_message = chat_request.messages[-1]
            if last_message.role == "user":
                user_message = last_message.content

        # Log request with user message
        logger.info(
            "chat_request_received",
            message_count=len(chat_request.messages),
            user_message=user_message,
        )

        # Print user message to console for easier development/debugging
        print(f"\n{'-'*50}\n💬 User: {user_message}\n{'-'*50}")

        # Use stateless processing - no database dependency
        result = await agent.get_stateless_response(chat_request.messages)

        # Log response content
        response_content = ""
        if result and len(result) > 0:
            # Access as attribute instead of using .get()
            content = result[0].content if hasattr(result[0], 'content') else ""
            response_content = content[:100] + "..." if len(content) > 100 else content
        
        logger.info(
            "chat_request_processed",
            response_preview=response_content
        )
        
        # Print assistant response to console
        print(f"\n🤖 Assistant: {response_content}\n{'-'*50}")

        return ChatResponse(messages=result)
    except Exception as e:
        logger.error("chat_request_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    authorized: bool = Depends(require_api_key),
):
    """Process a streaming chat request using LangGraph.
    
    Uses stateless processing - each request is handled independently
    without retrieving or storing conversation history in the database.
    """
    try:
        # Extract the latest user message for logging
        user_message = ""
        if chat_request.messages and len(chat_request.messages) > 0:
            last_message = chat_request.messages[-1]
            if last_message.role == "user":
                user_message = last_message.content
        
        # Log request with user message
        logger.info(
            "stream_chat_request_received",
            message_count=len(chat_request.messages),
            user_message=user_message,
        )

        # Print user message to console for easier development/debugging
        print(f"\n{'-'*50}\n💬 User (Stream): {user_message}\n{'-'*50}")

        async def event_generator():
            """Generate streaming events."""
            try:
                full_response = ""
                with llm_stream_duration_seconds.labels(model=agent.model_name).time():
                    # Use stateless streaming - no database dependency
                    async for chunk in agent.get_stateless_stream_response(chat_request.messages):
                        full_response += chunk
                        response = StreamResponse(content=chunk, done=False)
                        yield f"data: {json.dumps(response.model_dump())}\n\n"

                # Print assistant response to console
                print(f"\n🤖 Assistant (Stream): {full_response[:100]}...\n{'-'*50}")
                
                # Log response summary
                logger.info(
                    "stream_chat_completed",
                    response_length=len(full_response),
                    response_preview=full_response[:100] + "..." if len(full_response) > 100 else full_response
                )

                # Send final message indicating completion
                final_response = StreamResponse(content="", done=True)
                yield f"data: {json.dumps(final_response.model_dump())}\n\n"

            except Exception as e:
                logger.error(
                    "stream_chat_request_failed",
                    error=str(e),
                    exc_info=True,
                )
                error_response = StreamResponse(content=str(e), done=True)
                yield f"data: {json.dumps(error_response.model_dump())}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error("stream_chat_request_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/opensearch", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def opensearch_chat(
    request: Request,
    chat_request: ChatRequest,
    authorized: bool = Depends(require_api_key),
):
    """Process a chat request specifically for OpenSearch integration.
    
    Returns a simplified response without the thinking process or JSON formatting.
    """
    try:
        # Extract the user message
        user_message = ""
        if chat_request.messages and len(chat_request.messages) > 0:
            last_message = chat_request.messages[-1]
            if last_message.role == "user":
                user_message = last_message.content

        # Log request
        logger.info(
            "opensearch_request_received",
            message_count=len(chat_request.messages),
            user_message=user_message,
        )

        # Get stateless response
        result = await agent.get_stateless_response(chat_request.messages)
        
        # Process the response to extract only the final answer
        clean_response = ""
        if result and len(result) > 0:
            content = result[0].content if hasattr(result[0], 'content') else ""
            
            # Try to extract just the final answer from JSON format
            import re
            import json
            
            # Look for JSON blocks in the response
            json_pattern = r'```json\s*(.*?)\s*```'
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            
            if json_matches:
                try:
                    json_obj = json.loads(json_matches[0])
                    if "final_answer" in json_obj:
                        clean_response = json_obj["final_answer"]
                    else:
                        # For tool usage responses, give a helpful message
                        clean_response = "I need to gather more information to answer your question completely."
                except:
                    clean_response = content
            else:
                clean_response = content
        
        # If we couldn't extract a clean response, use the original
        if not clean_response:
            clean_response = content
            
        # Create a new message with the clean response
        clean_message = Message(role="assistant", content=clean_response)
        
        return ChatResponse(messages=[clean_message])
    except Exception as e:
        logger.error("opensearch_request_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Simple endpoint to clear the database cache if needed
@router.delete("/cache")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def clear_chat_cache(
    request: Request,
    authorized: bool = Depends(require_api_key),
):
    """Clear any cached chat history from the database."""
    try:
        session_id = "wazuh-api"
        await agent.clear_chat_history(session_id)
        return {"message": "Chat cache cleared successfully"}
    except Exception as e:
        logger.error("clear_cache_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))