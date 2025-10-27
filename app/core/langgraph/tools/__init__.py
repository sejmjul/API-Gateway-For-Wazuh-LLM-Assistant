"""LangGraph tools for enhanced language model capabilities."""

from langchain_core.tools.base import BaseTool

from .duckduckgo_search import duckduckgo_search_tool
from .wazuh import query_wazuh_logs
from .netbox import fetch_netbox_info
from .security_docs import search_security_docs

tools: list[BaseTool] = [
    duckduckgo_search_tool,
    query_wazuh_logs,
    fetch_netbox_info,
    search_security_docs,
]

__all__ = [
    "tools",
    "duckduckgo_search_tool",
    "query_wazuh_logs",
    "fetch_netbox_info",
    "search_security_docs",
]