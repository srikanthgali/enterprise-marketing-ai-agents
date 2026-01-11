"""
Specialized marketing agents for the multi-agent system.

Provides domain-specific agents for marketing strategy, customer support,
analytics, and continuous learning.
"""

from .marketing_strategy import MarketingStrategyAgent
from .customer_support import CustomerSupportAgent
from .analytics_evaluation import AnalyticsEvaluationAgent
from .feedback_learning import FeedbackLearningAgent

__all__ = [
    "MarketingStrategyAgent",
    "CustomerSupportAgent",
    "AnalyticsEvaluationAgent",
    "FeedbackLearningAgent",
]
