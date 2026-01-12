"""
Sentiment Analysis Tool for customer support interactions.

Provides emotion detection, urgency assessment, and tone analysis
for customer messages using LLM-based classification.
"""

import logging
from typing import Dict, List, Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import get_settings

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analysis tool for customer support messages.

    Capabilities:
    - Sentiment classification (positive/negative/neutral)
    - Emotion detection (frustrated, angry, satisfied, confused, etc.)
    - Urgency level assessment
    - Confidence scoring
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize sentiment analyzer.

        Args:
            model_name: LLM model to use for analysis
        """
        self.settings = get_settings()
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,  # Deterministic for consistency
        )

        # Sentiment analysis prompt
        self.system_prompt = """You are a sentiment analysis expert for customer support.

Analyze customer messages and provide:
1. Overall sentiment: positive, negative, or neutral
2. Detected emotions: List of emotions (frustrated, angry, satisfied, confused, anxious, etc.)
3. Urgency level: low, medium, high, critical
4. Confidence score: 0.0 to 1.0

Consider:
- Word choice and tone
- Punctuation and capitalization patterns
- Problem severity indicators
- Time-sensitive language

Respond in JSON format:
{
    "sentiment": "positive|negative|neutral",
    "emotions": ["emotion1", "emotion2"],
    "urgency_level": "low|medium|high|critical",
    "confidence": 0.85,
    "reasoning": "Brief explanation"
}"""

    async def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of customer message.

        Args:
            text: Customer message text

        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for sentiment analysis")
            return {
                "sentiment": "neutral",
                "emotions": [],
                "urgency_level": "low",
                "confidence": 0.0,
                "reasoning": "Empty input",
            }

        try:
            # Prepare messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Analyze this customer message:\n\n{text}"),
            ]

            # Get LLM response
            response = await self.llm.ainvoke(messages)

            # Parse JSON response
            import json

            result = json.loads(response.content)

            logger.info(
                f"Sentiment analysis complete: {result['sentiment']} "
                f"(confidence: {result['confidence']:.2f})"
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback to rule-based analysis
            return self._rule_based_analysis(text)

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                "sentiment": "neutral",
                "emotions": [],
                "urgency_level": "medium",
                "confidence": 0.0,
                "reasoning": f"Analysis error: {str(e)}",
            }

    def _rule_based_analysis(self, text: str) -> Dict[str, Any]:
        """
        Fallback rule-based sentiment analysis.

        Args:
            text: Customer message text

        Returns:
            Basic sentiment analysis results
        """
        text_lower = text.lower()

        # Sentiment keywords
        positive_words = {"thank", "great", "excellent", "appreciate", "happy", "love"}
        negative_words = {
            "issue",
            "problem",
            "broken",
            "error",
            "fail",
            "bug",
            "urgent",
        }

        # Count matches
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Determine sentiment
        if positive_count > negative_count:
            sentiment = "positive"
            emotions = ["satisfied"]
        elif negative_count > positive_count:
            sentiment = "negative"
            emotions = ["frustrated"]
        else:
            sentiment = "neutral"
            emotions = []

        # Urgency detection
        urgent_keywords = [
            "urgent",
            "asap",
            "immediately",
            "critical",
            "emergency",
            "!!!",
        ]
        urgency_level = (
            "high" if any(kw in text_lower for kw in urgent_keywords) else "medium"
        )

        return {
            "sentiment": sentiment,
            "emotions": emotions,
            "urgency_level": urgency_level,
            "confidence": 0.6,  # Lower confidence for rule-based
            "reasoning": "Fallback rule-based analysis",
        }

    def get_tone_recommendation(self, sentiment_result: Dict[str, Any]) -> str:
        """
        Get recommended tone for response based on sentiment.

        Args:
            sentiment_result: Result from analyze()

        Returns:
            Recommended tone description
        """
        sentiment = sentiment_result.get("sentiment", "neutral")
        urgency = sentiment_result.get("urgency_level", "medium")
        emotions = sentiment_result.get("emotions", [])

        if sentiment == "negative":
            if urgency in ["high", "critical"]:
                return "empathetic_urgent"
            elif "angry" in emotions:
                return "empathetic_apologetic"
            elif "frustrated" in emotions:
                return "empathetic_supportive"
            else:
                return "empathetic_professional"
        elif sentiment == "positive":
            return "friendly_professional"
        else:
            return "neutral_professional"
