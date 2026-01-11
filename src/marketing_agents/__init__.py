"""
Enterprise Marketing AI Agents - Multi-Agent System.

A professional, production-ready multi-agent AI system for marketing automation,
featuring specialized agents, orchestration, memory management, and continuous learning.
"""

__version__ = "0.1.0"
__author__ = "Srikanth Gali"

from .core import (
    BaseAgent,
    AgentStatus,
    HandoffRequest,
    OrchestratorAgent,
    HandoffManager,
    MessageBus,
)

from .agents import (
    MarketingStrategyAgent,
    CustomerSupportAgent,
    AnalyticsEvaluationAgent,
    FeedbackLearningAgent,
)

__all__ = [
    # Core
    "BaseAgent",
    "AgentStatus",
    "HandoffRequest",
    "OrchestratorAgent",
    "HandoffManager",
    "MessageBus",
    # Agents
    "MarketingStrategyAgent",
    "CustomerSupportAgent",
    "AnalyticsEvaluationAgent",
    "FeedbackLearningAgent",
]
