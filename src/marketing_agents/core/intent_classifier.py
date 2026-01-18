"""
LLM-Based Intent Classifier for Multi-Agent System.

Uses an LLM to classify user intents and extract structured parameters
from natural language queries.
"""

from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
import json
import logging

from src.marketing_agents.utils import get_logger


class AgentIntent(str, Enum):
    """Supported agent intents."""

    CAMPAIGN_CREATION = "campaign_creation"
    MARKET_ANALYSIS = "market_analysis"
    CONTENT_STRATEGY = "content_strategy"
    CUSTOMER_SUPPORT = "customer_support"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    FEEDBACK_ANALYSIS = "feedback_analysis"
    SYSTEM_OPTIMIZATION = "system_optimization"
    GENERAL_INQUIRY = "general_inquiry"


class IntentClassification(BaseModel):
    """Structured output from intent classification."""

    intent: AgentIntent = Field(description="The classified intent")
    confidence: float = Field(description="Confidence score 0-1", ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation of classification")

    # Extracted entities (optional, depends on intent)
    entities: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted parameters from the query"
    )

    # Suggested agent
    target_agent: str = Field(description="Agent that should handle this intent")


class IntentClassifier:
    """
    LLM-based intent classifier that determines which agent should handle a query.

    Uses structured output to ensure consistent classification and parameter extraction.
    Reads configuration from agents_config.yaml for model settings.
    """

    SYSTEM_PROMPT = """You are an intent classification system for a multi-agent marketing AI platform.

Your job is to analyze user queries and classify them into one of these intents:

1. **CAMPAIGN_CREATION**: User wants to create/plan a marketing campaign
   - Extract: campaign_name, budget, duration, objectives, target_audience
   - Examples: "Create a Q1 campaign", "Plan a social media campaign for product launch"
   - Examples: "I need to create a marketing campaign to promote our new payment processing feature"

2. **MARKET_ANALYSIS**: User wants market research, competitive analysis, or positioning
   - Extract: market_segment, competitors, focus_areas
   - Examples: "Analyze the fintech market", "Who are our main competitors?"
   - Examples: "What positioning strategy should we use to stand out in the payment processing market?"

3. **CONTENT_STRATEGY**: User wants content planning, calendars, or messaging
   - Extract: content_types, channels, themes, duration
   - Examples: "Create a content calendar", "Plan blog posts for next month"

4. **CUSTOMER_SUPPORT**: User has questions, needs help, or reports issues
   - Extract: issue_type, urgency, customer_id (if mentioned)
   - Examples: "How do I integrate the API?", "Our checkout is throwing a 400 error"
   - Examples: "How do I implement webhooks?", "Customer is having payment issues"
   - Note: For performance/metrics issues, use PERFORMANCE_ANALYTICS instead

5. **PERFORMANCE_ANALYTICS**: User wants metrics, reports, data analysis, or investigating performance issues
   - Extract: report_type, metrics, date_range, filters
   - Examples: "Generate monthly report", "Show me conversion funnel", "What's our ROI?"
   - Examples: "Customer satisfaction scores dropped 15% this month. What's causing this?"
   - Examples: "Our conversion rates are falling. Need help optimizing our approach."
   - Examples: "Show me the conversion funnel for our checkout process"

6. **FEEDBACK_ANALYSIS**: User wants to analyze feedback, ratings, or improve predictions
   - Extract: feedback_source, time_range, analysis_type, rating (if provided)
   - Examples: "How well is the campaign performing?", "Analyze customer feedback"
   - Examples: "Rate the quality: 2/5 stars. It was too generic."
   - Examples: "Multiple agents are reporting that mobile checkout has issues"
   - Examples: "Recommend improvements for our campaign performance"
   - Examples: "Are our conversion rate predictions accurate? How can we improve them?"

7. **SYSTEM_OPTIMIZATION**: User wants to improve system, learn from data, or optimize
   - Extract: optimization_target, focus_area
   - Examples: "Optimize our marketing strategy", "Learn from past campaigns"

8. **GENERAL_INQUIRY**: General questions that don't fit other categories
   - Default fallback

**Important:**
- Classify with high confidence (>0.8) when intent is clear
- Extract ALL relevant entities from the query (don't use placeholders!)
- If entities are missing but needed, note them in reasoning
- For queries about declining metrics, performance issues, or data analysis (like "satisfaction scores dropped", "conversion rates falling"), always classify as PERFORMANCE_ANALYTICS
- For queries that mention specific metrics (conversion rate, ROI, CTR) AND ask for recommendations, classify as PERFORMANCE_ANALYTICS first (analytics provides data, then can handoff to feedback_learning for recommendations)
- For technical integration or how-to questions, classify as CUSTOMER_SUPPORT
- For investigation and recommendation requests (without specific metrics), classify as FEEDBACK_ANALYSIS
- Map to the appropriate target_agent:
  - campaign_creation → marketing_strategy
  - market_analysis → marketing_strategy
  - content_strategy → marketing_strategy
  - customer_support → customer_support
  - performance_analytics → analytics_evaluation
  - feedback_analysis → feedback_learning
  - system_optimization → feedback_learning
  - general_inquiry → customer_support (default)

Respond with valid JSON following the IntentClassification schema."""

    def __init__(
        self, llm: Optional[ChatOpenAI] = None, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize intent classifier.

        Args:
            llm: Language model instance (e.g., ChatOpenAI). If None, creates from config.
            config: Configuration dict from agents_config.yaml orchestrator.intent_classifier
                   If None, uses default values.
        """
        self.logger = get_logger(__name__)

        # Load config or use defaults
        if config is None:
            config = {
                "enabled": True,
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "confidence_threshold": 0.7,
                "fallback_agent": "customer_support",
            }

        self.config = config
        self.enabled = config.get("enabled", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.fallback_agent = config.get("fallback_agent", "customer_support")

        if llm is None:
            # Create LLM from config
            model_name = config.get("model", "gpt-4o-mini")
            temperature = config.get("temperature", 0.1)
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
            self.logger.info(
                f"IntentClassifier initialized with model={model_name}, "
                f"temperature={temperature}, confidence_threshold={self.confidence_threshold}"
            )
        else:
            self.llm = llm
            self.logger.info("IntentClassifier initialized with provided LLM")

        self.parser = PydanticOutputParser(pydantic_object=IntentClassification)

    async def classify(self, user_message: str) -> IntentClassification:
        """
        Classify user intent from natural language message.

        Args:
            user_message: The user's raw message

        Returns:
            IntentClassification with intent, confidence, entities, and target agent
        """
        try:
            # Create prompt with output format instructions
            format_instructions = self.parser.get_format_instructions()

            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""User message: "{user_message}"

{format_instructions}

Classify this message and extract all relevant entities."""
                ),
            ]

            # Get LLM response
            self.logger.info(f"Classifying intent for: {user_message[:100]}...")
            response = await self.llm.ainvoke(messages)

            # Parse structured output
            classification = self.parser.parse(response.content)

            self.logger.info(
                f"Intent classified: {classification.intent} "
                f"(confidence: {classification.confidence:.2f}) → {classification.target_agent}"
            )

            return classification

        except Exception as e:
            self.logger.error(f"Error classifying intent: {e}", exc_info=True)

            # Return fallback classification
            return IntentClassification(
                intent=AgentIntent.GENERAL_INQUIRY,
                confidence=0.5,
                reasoning=f"Failed to classify intent: {str(e)}. Defaulting to general inquiry.",
                entities={},
                target_agent="customer_support",
            )

    def get_task_type(self, intent: AgentIntent) -> str:
        """
        Map intent to orchestrator task_type.

        Args:
            intent: Classified intent

        Returns:
            Task type string for orchestrator
        """
        intent_to_task = {
            AgentIntent.CAMPAIGN_CREATION: "campaign_launch",
            AgentIntent.MARKET_ANALYSIS: "marketing_strategy",  # Maps to marketing agent
            AgentIntent.CONTENT_STRATEGY: "content_strategy",
            AgentIntent.CUSTOMER_SUPPORT: "customer_support",
            AgentIntent.PERFORMANCE_ANALYTICS: "analytics",
            AgentIntent.FEEDBACK_ANALYSIS: "feedback_learning",
            AgentIntent.SYSTEM_OPTIMIZATION: "system_improvement",
            AgentIntent.GENERAL_INQUIRY: "customer_support",
        }
        return intent_to_task.get(intent, "customer_support")
