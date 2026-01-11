"""
API dependencies for dependency injection.

Provides shared instances and utilities for API endpoints.
"""

from functools import lru_cache
from src.marketing_agents.core.prompt_manager import PromptManager


# Singleton instance of PromptManager
_prompt_manager_instance = None


def get_prompt_manager() -> PromptManager:
    """
    Get or create a singleton PromptManager instance.

    Returns:
        Shared PromptManager instance
    """
    global _prompt_manager_instance

    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager(prompts_dir="config/prompts")

    return _prompt_manager_instance
