"""
API dependencies for dependency injection.

Provides shared instances and utilities for API endpoints.
"""

from functools import lru_cache
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.marketing_agents.core.prompt_manager import PromptManager
from src.marketing_agents.core import OrchestratorAgent, MessageBus
from src.marketing_agents.agents import (
    MarketingStrategyAgent,
    CustomerSupportAgent,
    AnalyticsEvaluationAgent,
    FeedbackLearningAgent,
)
from src.marketing_agents.memory import create_memory_manager
from config.settings import get_settings

# Singleton instances
_prompt_manager_instance = None
_orchestrator_instance = None
_memory_manager_instance = None
_message_bus_instance = None

# Security
security = HTTPBearer(auto_error=False)


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


def get_message_bus() -> MessageBus:
    """
    Get or create a singleton MessageBus instance.

    Returns:
        Shared MessageBus instance
    """
    global _message_bus_instance

    if _message_bus_instance is None:
        _message_bus_instance = MessageBus()

    return _message_bus_instance


def get_memory_manager():
    """
    Get or create a singleton MemoryManager instance.

    Returns:
        Shared MemoryManager instance
    """
    global _memory_manager_instance

    if _memory_manager_instance is None:
        try:
            _memory_manager_instance = create_memory_manager()
        except Exception as e:
            # Log error but don't fail - some features may work without memory manager
            import logging

            logging.warning(f"Failed to create memory manager: {e}")
            _memory_manager_instance = None

    return _memory_manager_instance


def get_orchestrator() -> OrchestratorAgent:
    """
    Get or create a singleton OrchestratorAgent instance with all agents registered.

    Returns:
        Shared OrchestratorAgent instance
    """
    global _orchestrator_instance

    if _orchestrator_instance is None:
        settings = get_settings()

        # Get shared dependencies
        message_bus = get_message_bus()
        memory_manager = get_memory_manager()
        prompt_manager = get_prompt_manager()

        # Get orchestrator config
        orchestrator_config = settings.get_agent_config("orchestrator") or {
            "handoff_timeout": settings.system.handoff_timeout,
            "max_retries": settings.system.max_retries,
        }

        # Initialize orchestrator
        _orchestrator_instance = OrchestratorAgent(
            config=orchestrator_config,
            memory_manager=memory_manager,
            message_bus=message_bus,
        )

        # Register specialized agents
        agents_to_register = [
            (
                "marketing_strategy",
                MarketingStrategyAgent,
                "Marketing Strategy Agent",
            ),
            (
                "customer_support",
                CustomerSupportAgent,
                "Customer Support Agent",
            ),
            (
                "analytics_evaluation",
                AnalyticsEvaluationAgent,
                "Analytics Evaluation Agent",
            ),
            (
                "feedback_learning",
                FeedbackLearningAgent,
                "Feedback Learning Agent",
            ),
        ]

        for agent_id, agent_class, agent_name in agents_to_register:
            try:
                agent_config = settings.get_agent_config(agent_id) or {
                    "capabilities": []
                }
                agent = agent_class(
                    config=agent_config,
                    memory_manager=memory_manager,
                    message_bus=message_bus,
                    prompt_manager=prompt_manager,
                )
                _orchestrator_instance.register_agent(agent)
            except Exception as e:
                import logging

                logging.error(f"Failed to register {agent_name}: {e}")

    return _orchestrator_instance


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> str:
    """
    Verify API key from Authorization header.

    Args:
        credentials: Bearer token credentials

    Returns:
        API key if valid

    Raises:
        HTTPException: If authentication fails
    """
    settings = get_settings()

    # Skip authentication if not required
    if not settings.security.authentication_required:
        return "no-auth"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # In production, validate against stored API keys
    # For now, accept any non-empty token
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials
