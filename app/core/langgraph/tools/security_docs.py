"""Security documentation search tool."""

from langchain_core.tools import tool
from app.core.logging import logger

@tool
async def search_security_docs(query: str, max_results: int = 5) -> str:
    """Search Wazuh documentation and security knowledge base.
    
    Args:
        query: Search query for documentation
        max_results: Maximum number of results to return
        
    Returns:
        str: Relevant documentation excerpts
    """
    try:
        # TODO: Implement documentation search
        # Options:
        # 1. Use DuckDuckGo with site:wazuh.com filter
        # 2. Implement vector search over Wazuh docs
        # 3. Use Wazuh's official documentation API
        
        logger.info("security_docs_searched", query=query)
        
        # For now, use the existing DuckDuckGo tool with a filter
        from app.core.langgraph.tools.duckduckgo_search import duckduckgo_search_tool
        
        search_query = f"site:documentation.wazuh.com {query}"
        results = await duckduckgo_search_tool.ainvoke(search_query)
        
        return f"Documentation search results for '{query}':\n{results}"
        
    except Exception as e:
        logger.error("docs_search_failed", error=str(e))
        return f"Error searching documentation: {str(e)}"