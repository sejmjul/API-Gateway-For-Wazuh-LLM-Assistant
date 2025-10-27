"""Wazuh log query tool for LangGraph."""

from typing import Optional
from langchain_core.tools import tool
from app.core.logging import logger

@tool
async def query_wazuh_logs(
    query: str,
    time_range: Optional[str] = "24h",
    max_results: int = 100
) -> str:
    """Query Wazuh logs for security events and alerts.
    
    Args:
        query: The search query (e.g., "rule.level:>=10", "agent.name:web-server")
        time_range: Time range for the search (e.g., "24h", "7d", "30d")
        max_results: Maximum number of results to return
        
    Returns:
        str: JSON string with matching log entries
    """
    try:
        # TODO: Implement actual Wazuh API integration
        # Example structure:
        # import requests
        # from app.core.config import settings
        # 
        # response = requests.post(
        #     f"{settings.WAZUH_API_URL}/events",
        #     headers={"Authorization": f"Bearer {settings.WAZUH_API_KEY}"},
        #     json={"query": query, "time_range": time_range, "size": max_results}
        # )
        # return response.json()
        
        logger.info("wazuh_logs_queried", query=query, time_range=time_range)
        
        # Mock response for now
        return f"Searched Wazuh logs for: {query} (last {time_range}). Found 0 results. [TODO: Implement Wazuh API integration]"
        
    except Exception as e:
        logger.error("wazuh_query_failed", error=str(e))
        return f"Error querying Wazuh logs: {str(e)}"