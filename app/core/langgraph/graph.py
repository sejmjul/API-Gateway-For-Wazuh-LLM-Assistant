"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""


import uuid
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Literal,
    Optional,
)

from asgiref.sync import sync_to_async
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import (
    END,
    StateGraph,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from openai import OpenAIError
from psycopg_pool import AsyncConnectionPool

from app.core.config import (
    Environment,
    settings,
)
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.metrics import llm_inference_duration_seconds
from app.core.prompts import SYSTEM_PROMPT
from app.schemas import (
    GraphState,
    Message,
)
from app.utils import (
    dump_messages,
    prepare_messages,
)


class LangGraphAgent:
    """Stateless LangGraph Agent for Wazuh integration.
    
    This agent processes each request independently. The client (Wazuh)
    sends the full conversation history with each request, so no database
    persistence is needed. Langfuse is used solely for observability.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with LLM only - no database."""
        provider = settings.MODEL_PROVIDER
        if provider == "ollama":
            self.llm = ChatOllama(
                model=settings.llm_model,
                base_url=settings.LLM_BASE_URL,
                temperature=settings.DEFAULT_LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )
            self.model_name = settings.llm_model
        elif provider == "openai":
            self.llm = ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                temperature=settings.DEFAULT_LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )
            self.model_name = settings.llm_model
        else:
            raise ValueError(f"Unsupported MODEL_PROVIDER: {provider}")

        self.tools_by_name = {tool.name: tool for tool in tools}
        # Bind tools to LLM
        self.llm = self.llm.bind_tools(tools)
        
        # Create stateless graph immediately
        self._graph = self._create_stateless_graph()
        
        logger.info(
            "llm_initialized",
            model_name=self.model_name,
            provider=provider,
            has_tools=len(self.tools_by_name),
            mode="stateless"
        )

    def _create_stateless_graph(self) -> CompiledStateGraph:
        """Create a stateless graph with no checkpointing.
        
        Returns:
            CompiledStateGraph: The compiled graph ready for use.
        """
        graph_builder = StateGraph(GraphState)
        graph_builder.add_node("chat", self._chat)
        graph_builder.add_node("tool_call", self._tool_call)
        graph_builder.add_conditional_edges(
            "chat",
            self._should_continue,
            {"continue": "tool_call", "end": END},
        )
        graph_builder.add_edge("tool_call", "chat")
        graph_builder.set_entry_point("chat")
        graph_builder.set_finish_point("chat")
        
        # No checkpointer - pure stateless operation
        return graph_builder.compile(name=f"{settings.PROJECT_NAME} Stateless Agent")

    async def _chat(self, state: GraphState) -> dict:
        """Process the chat state and generate a response.

        Args:
            state (GraphState): The current state of the conversation.

        Returns:
            dict: Updated state with new messages.
        """
        messages = prepare_messages(state.messages, self.llm, SYSTEM_PROMPT)

        llm_calls_num = 0

        # Configure retry attempts based on environment
        max_retries = settings.MAX_LLM_CALL_RETRIES

        for attempt in range(max_retries):
            try:
                # Use self.model_name instead of self.llm.model_name
                with llm_inference_duration_seconds.labels(model=self.model_name).time():
                    generated_state = {"messages": [await self.llm.ainvoke(dump_messages(messages))]}
                logger.info(
                    "llm_response_generated",
                    session_id=state.session_id,
                    llm_calls_num=llm_calls_num + 1,
                    model=settings.llm_model,
                    environment=settings.ENVIRONMENT.value,
                )
                return generated_state
            except OpenAIError as e:
                logger.error(
                    "llm_call_failed",
                    llm_calls_num=llm_calls_num,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                    environment=settings.ENVIRONMENT.value,
                )
                llm_calls_num += 1

                # In production, we might want to fall back to a more reliable model
                if settings.ENVIRONMENT == Environment.PRODUCTION and attempt == max_retries - 2:
                    fallback_model = "gpt-4o"
                    logger.warning(
                        "using_fallback_model", model=fallback_model, environment=settings.ENVIRONMENT.value
                    )
                    # Update the model_name attribute instead of trying to set it on llm
                    self.model_name = fallback_model
                    # You may need to recreate the LLM here if you need to actually use the fallback model
                    self.llm = ChatOpenAI(
                        model=fallback_model,
                        api_key=settings.OPENAI_API_KEY,
                        base_url=settings.OPENAI_BASE_URL,
                        temperature=settings.DEFAULT_LLM_TEMPERATURE,
                        max_tokens=settings.MAX_TOKENS,
                    )

                continue

        raise Exception(f"Failed to get a response from the LLM after {max_retries} attempts")

    # Define our tool node
    async def _tool_call(self, state: GraphState) -> GraphState:
        """Process tool calls from the last message.

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            Dict with updated messages containing tool responses.
        """
        outputs = []
        for tool_call in state.messages[-1].tool_calls:
            tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    def _should_continue(self, state: GraphState) -> Literal["end", "continue"]:
        """Determine if the agent should continue or end based on the last message.

        Args:
            state: The current agent state containing messages.

        Returns:
            Literal["end", "continue"]: "end" if there are no tool calls, "continue" otherwise.
        """
        messages = state.messages
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    async def get_response(
        self,
        messages: list[Message],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Get a response from the LLM (stateless).
        
        Args:
            messages: The full conversation history from the client.
            session_id: Optional ID for Langfuse tracking only.
            user_id: Optional user ID for Langfuse tracking only.
            
        Returns:
            list[dict]: The assistant's response messages.
        """
        # Generate session ID if not provided (for Langfuse tracking)
        if not session_id:
            session_id = f"wazuh-{uuid.uuid4()}"
        
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [
                CallbackHandler()  # No parameters needed - uses environment variables
            ],
            "metadata": {
                "user_id": user_id or "anonymous",
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
            },
            "tags": [settings.ENVIRONMENT.value, "wazuh"],
        }
        
        try:
            response = await self._graph.ainvoke(
                {"messages": dump_messages(messages), "session_id": session_id},
                config
            )
            
            logger.info(
                "response_generated",
                session_id=session_id,
                message_count=len(response.get("messages", [])),
            )
            
            return self.__process_messages(response["messages"])
            
        except Exception as e:
            logger.error("response_generation_failed", error=str(e), session_id=session_id, exc_info=True)
            raise e

    async def get_stream_response(
        self,
        messages: list[Message],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Get a streaming response from the LLM (stateless).
        
        Args:
            messages: The full conversation history from the client.
            session_id: Optional ID for Langfuse tracking only.
            user_id: Optional user ID for Langfuse tracking only.
            
        Yields:
            str: Tokens of the LLM response.
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = f"wazuh-{uuid.uuid4()}"
        
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [
                CallbackHandler()  # No parameters needed
            ],
            "metadata": {
                "user_id": user_id or "anonymous", 
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
            },
            "tags": [settings.ENVIRONMENT.value, "wazuh"],
        }
        
        try:
            async for token, _ in self._graph.astream(
                {"messages": dump_messages(messages), "session_id": session_id},
                config,
                stream_mode="messages"
            ):
                try:
                    if hasattr(token, "content") and token.content:
                        yield token.content
                except (StopIteration, GeneratorExit):
                    break
                    
        except Exception as e:
            logger.error("stream_response_failed", error=str(e), session_id=session_id, exc_info=True)
            raise e

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        """Process LangChain messages to API format."""
        openai_style_messages = convert_to_openai_messages(messages)
        return [
            Message(**message)
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]
