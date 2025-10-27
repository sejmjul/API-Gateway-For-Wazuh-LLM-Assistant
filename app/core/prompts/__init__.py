"""System prompt loader for LangGraph agent."""
import os
from datetime import datetime
from app.core.config import settings

def load_system_prompt() -> str:
    """Load and format the system prompt from markdown file."""
    prompt_path = os.path.join(os.path.dirname(__file__), "system.md")
    with open(prompt_path, "r") as f:
        content = f.read()
        
    # Use replace instead of format to avoid conflicts with JSON braces
    content = content.replace("{agent_name}", settings.PROJECT_NAME + " Agent")
    content = content.replace("{current_date_and_time}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    return content

SYSTEM_PROMPT = load_system_prompt()

__all__ = ["SYSTEM_PROMPT"]
