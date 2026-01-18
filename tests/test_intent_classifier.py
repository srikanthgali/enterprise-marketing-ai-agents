"""
Unit tests for IntentClassifier.

Tests intent classification accuracy and entity extraction.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.messages import AIMessage

from src.marketing_agents.core.intent_classifier import (
    IntentClassifier,
    AgentIntent,
    IntentClassification,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def intent_classifier(mock_llm):
    """Create IntentClassifier with mock LLM."""
    return IntentClassifier(llm=mock_llm)


@pytest.mark.asyncio
async def test_campaign_creation_intent(intent_classifier, mock_llm):
    """Test classification of campaign creation queries."""
    # Mock LLM response
    mock_response = AIMessage(
        content="""{
        "intent": "campaign_creation",
        "confidence": 0.95,
        "reasoning": "User wants to create a marketing campaign",
        "entities": {
            "campaign_name": "Q1 Product Launch",
            "budget": 50000,
            "target_audience": "small businesses"
        },
        "target_agent": "marketing_strategy"
    }"""
    )
    mock_llm.ainvoke.return_value = mock_response

    result = await intent_classifier.classify(
        "Create a marketing campaign for Q1 product launch with $50k budget targeting small businesses"
    )

    assert result.intent == AgentIntent.CAMPAIGN_CREATION
    assert result.confidence >= 0.7
    assert result.target_agent == "marketing_strategy"
    assert "campaign_name" in result.entities
    assert result.entities["budget"] == 50000


@pytest.mark.asyncio
async def test_analytics_intent(intent_classifier, mock_llm):
    """Test classification of analytics queries."""
    mock_response = AIMessage(
        content="""{
        "intent": "performance_analytics",
        "confidence": 0.92,
        "reasoning": "User wants to generate a performance report",
        "entities": {
            "report_type": "monthly",
            "date_range": "December"
        },
        "target_agent": "analytics_evaluation"
    }"""
    )
    mock_llm.ainvoke.return_value = mock_response

    result = await intent_classifier.classify(
        "Generate a monthly performance report for December"
    )

    assert result.intent == AgentIntent.PERFORMANCE_ANALYTICS
    assert result.target_agent == "analytics_evaluation"


@pytest.mark.asyncio
async def test_customer_support_intent(intent_classifier, mock_llm):
    """Test classification of customer support queries."""
    mock_response = AIMessage(
        content="""{
        "intent": "customer_support",
        "confidence": 0.88,
        "reasoning": "User has a technical issue with checkout",
        "entities": {
            "issue_type": "technical_error",
            "urgency": "high"
        },
        "target_agent": "customer_support"
    }"""
    )
    mock_llm.ainvoke.return_value = mock_response

    result = await intent_classifier.classify(
        "Our checkout is throwing a 400 error when customers try to pay"
    )

    assert result.intent == AgentIntent.CUSTOMER_SUPPORT
    assert result.target_agent == "customer_support"


@pytest.mark.asyncio
async def test_feedback_analysis_intent(intent_classifier, mock_llm):
    """Test classification of feedback analysis queries."""
    mock_response = AIMessage(
        content="""{
        "intent": "feedback_analysis",
        "confidence": 0.90,
        "reasoning": "User wants to rate and provide feedback on campaign quality",
        "entities": {
            "rating": 2,
            "max_rating": 5,
            "feedback": "too generic"
        },
        "target_agent": "feedback_learning"
    }"""
    )
    mock_llm.ainvoke.return_value = mock_response

    result = await intent_classifier.classify(
        "Rate the campaign strategy: 2/5 stars. It was too generic."
    )

    assert result.intent == AgentIntent.FEEDBACK_ANALYSIS
    assert result.target_agent == "feedback_learning"


@pytest.mark.asyncio
async def test_task_type_mapping(intent_classifier):
    """Test intent to task_type mapping."""
    assert (
        intent_classifier.get_task_type(AgentIntent.CAMPAIGN_CREATION)
        == "campaign_launch"
    )
    assert (
        intent_classifier.get_task_type(AgentIntent.MARKET_ANALYSIS)
        == "market_analysis"
    )
    assert (
        intent_classifier.get_task_type(AgentIntent.CUSTOMER_SUPPORT)
        == "customer_support"
    )
    assert (
        intent_classifier.get_task_type(AgentIntent.PERFORMANCE_ANALYTICS)
        == "analytics"
    )
    assert (
        intent_classifier.get_task_type(AgentIntent.FEEDBACK_ANALYSIS)
        == "feedback_learning"
    )


@pytest.mark.asyncio
async def test_error_handling(intent_classifier, mock_llm):
    """Test that classifier handles errors gracefully."""
    # Mock LLM to raise an exception
    mock_llm.ainvoke.side_effect = Exception("LLM API error")

    result = await intent_classifier.classify("Test message")

    # Should return fallback classification
    assert result.intent == AgentIntent.GENERAL_INQUIRY
    assert result.confidence == 0.5
    assert result.target_agent == "customer_support"
    assert "Failed to classify" in result.reasoning


@pytest.mark.asyncio
async def test_entity_extraction(intent_classifier, mock_llm):
    """Test that entities are correctly extracted."""
    mock_response = AIMessage(
        content="""{
        "intent": "campaign_creation",
        "confidence": 0.93,
        "reasoning": "User wants to create a specific campaign",
        "entities": {
            "campaign_name": "Summer Sale",
            "budget": 25000,
            "duration": 6,
            "objectives": ["awareness", "conversions"],
            "target_audience": "millennials"
        },
        "target_agent": "marketing_strategy"
    }"""
    )
    mock_llm.ainvoke.return_value = mock_response

    result = await intent_classifier.classify(
        "Create a campaign called 'Summer Sale' with $25,000 budget for 6 weeks targeting millennials"
    )

    assert result.entities["campaign_name"] == "Summer Sale"
    assert result.entities["budget"] == 25000
    assert result.entities["duration"] == 6
    assert "millennials" in result.entities["target_audience"]
