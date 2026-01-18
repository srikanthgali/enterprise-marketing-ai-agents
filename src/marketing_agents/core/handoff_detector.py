"""
LLM-driven handoff detection for multi-agent collaboration.

Determines when an agent should hand off to another specialized agent
based on contextual understanding rather than keyword matching.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from enum import Enum
import logging
import json


class HandoffDecision(BaseModel):
    """Structured handoff decision from LLM."""

    handoff_required: bool = Field(
        description="Whether a handoff to another agent is needed"
    )
    target_agent: Optional[str] = Field(
        default=None, description="Target agent name if handoff is required"
    )
    reason: str = Field(description="Clear explanation for the handoff decision")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the handoff decision (0.0-1.0)"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context for the receiving agent"
    )


class HandoffDetector:
    """
    LLM-based handoff detection with contextual reasoning.

    Uses language understanding to determine when an agent should
    hand off a task to a more specialized agent based on the query,
    current agent's capabilities, and available target agents.

    Reads configuration from agents_config.yaml for model settings.
    """

    HANDOFF_PROMPT = """You are a routing expert for a multi-agent marketing system. Your job is to determine if the current agent should hand off the task to another specialized agent.

**Current Agent**: {current_agent}
**Available Agents**:
- marketing_strategy: Creates campaigns, brand positioning, market analysis, content strategy
- customer_support: Handles technical questions, integration help, troubleshooting
- analytics_evaluation: Analyzes metrics, generates reports, identifies trends
- feedback_learning: Learns from feedback, improves predictions, provides recommendations

**User Query**: {user_message}

**Current Agent's Analysis**:
{agent_analysis}

**Handoff Guidelines**:

1. **Stay in Current Agent** if:
   - Query is within current agent's expertise
   - Current agent has already provided good analysis/answer
   - No other agent would add significant value
   - Analytics has metrics/data ready and user just wants to see them
   - User asks "show me", "what is", "how much" - these are data queries, not recommendation requests

2. **Feedback Learning STAYS for**:
   - "Investigate issues" or "recommend improvements" - this is feedback_learning's core expertise
   - "Multiple agents reporting X" - investigating patterns and providing recommendations
   - "How can we improve Y" - recommendations based on feedback analysis
   - "Analyze why Z is happening" - pattern detection and learning from data
   - User wants recommendations, improvements, or investigation of issues

3. **Customer Support ONLY for**:
   - HOW TO implement/configure something (API setup, webhook configuration)
   - Specific error messages or technical failures ("Error 500", "API key invalid")
   - Integration steps and technical documentation questions
   - NOT for investigating reported issues or providing recommendations

4. **Handoff from Analytics → Feedback Learning** if:
   - User EXPLICITLY asks for "recommendations" or "how to improve" AFTER seeing metrics
   - User wants to learn from patterns or improve predictions
   - User says "now recommend improvements" or "what should we do about this"
   - DO NOT handoff if user just wants to see/analyze data without asking for recommendations

5. **Handoff from Analytics → Marketing Strategy** if:
   - User asks for strategic advice or positioning after seeing data
   - Query about optimizing approach or standing out in market
   - User wants actionable marketing strategy based on performance

6. **Handoff from Marketing → Analytics** if:
   - User asks for data/metrics/performance analysis
   - Query needs quantitative analysis
   - Needs historical data or trend analysis

7. **NO Handoff** for:
   - Performance analysis questions (Analytics handles these)
   - "What's causing X metric to drop?" (Analytics answers this)
   - General inquiries that current agent can answer
   - Feedback_learning investigating issues or providing recommendations

**Important**:
- Don't handoff just because keywords like "satisfaction" appear - consider the FULL CONTEXT
- If analytics agent has metrics/data to answer "what's causing this", NO handoff needed
- Only handoff when specialized expertise is clearly needed
- **NEVER handoff to the same agent** - if current agent can handle it, set handoff_required to false

Respond with JSON matching this schema:
{{
    "handoff_required": true/false,
    "target_agent": "agent_name" or null,
    "reason": "clear explanation",
    "confidence": 0.0-1.0,
    "context": {{"key": "value"}}
}}"""

    def __init__(
        self, llm: Optional[ChatOpenAI] = None, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize handoff detector.

        Args:
            llm: Language model for handoff detection. If None, creates from config.
            config: Configuration dict from agents_config.yaml orchestrator.handoff_detector
                   If None, uses default values.
        """
        self.logger = logging.getLogger(__name__)

        # Load config or use defaults
        if config is None:
            config = {
                "enabled": True,
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "confidence_threshold": 0.8,
                "max_handoffs": 3,
            }

        self.config = config
        self.enabled = config.get("enabled", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.max_handoffs = config.get("max_handoffs", 3)

        if llm is None:
            # Create LLM from config
            model_name = config.get("model", "gpt-4o-mini")
            temperature = config.get("temperature", 0.1)
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
            self.logger.info(
                f"HandoffDetector initialized with model={model_name}, "
                f"temperature={temperature}, confidence_threshold={self.confidence_threshold}"
            )
        else:
            self.llm = llm
            self.logger.info("HandoffDetector initialized with provided LLM")

    async def detect_handoff(
        self,
        current_agent: str,
        user_message: str,
        agent_analysis: Dict[str, Any],
        available_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detect if handoff is needed using LLM reasoning.

        Args:
            current_agent: Name of current agent
            user_message: Original user query
            agent_analysis: Current agent's analysis/result
            available_agents: List of agents that can receive handoffs

        Returns:
            Handoff decision dictionary
        """
        try:
            # Format agent analysis for prompt
            analysis_summary = self._format_analysis(agent_analysis)

            # Build prompt
            prompt = self.HANDOFF_PROMPT.format(
                current_agent=current_agent,
                user_message=user_message,
                agent_analysis=analysis_summary,
            )

            self.logger.debug(
                f"Detecting handoff for {current_agent}: {user_message[:100]}..."
            )

            # Get LLM decision
            response = await self.llm.ainvoke(prompt)
            decision_text = response.content.strip()

            # Parse JSON response
            # Handle markdown code blocks if present
            if "```json" in decision_text:
                decision_text = (
                    decision_text.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in decision_text:
                decision_text = decision_text.split("```")[1].split("```")[0].strip()

            decision_dict = json.loads(decision_text)

            # Validate and structure
            context_dict = decision_dict.get("context", {})
            # CRITICAL: Always preserve the original user message in the context
            # to prevent it from being lost during handoffs
            if "message" not in context_dict:
                context_dict["message"] = user_message

            handoff_info = {
                "handoff_required": decision_dict.get("handoff_required", False),
                "target_agent": decision_dict.get("target_agent"),
                "handoff_reason": decision_dict.get("reason", ""),
                "confidence": decision_dict.get("confidence", 0.0),
                "context": context_dict,
            }

            # Critical: Prevent self-handoff (agent handing off to itself)
            if (
                handoff_info["handoff_required"]
                and handoff_info["target_agent"] == current_agent
            ):
                self.logger.warning(
                    f"Prevented self-handoff: {current_agent} → {handoff_info['target_agent']}. "
                    f"Agent can handle this request directly."
                )
                handoff_info["handoff_required"] = False
                handoff_info["target_agent"] = None
                handoff_info["handoff_reason"] = "self_handoff_prevented"

            if handoff_info["handoff_required"]:
                self.logger.info(
                    f"Handoff detected: {current_agent} → {handoff_info['target_agent']} "
                    f"(confidence: {handoff_info['confidence']:.2f})"
                )
                self.logger.info(f"Reason: {handoff_info['handoff_reason']}")
            else:
                self.logger.info(f"No handoff needed for {current_agent}")

            return handoff_info

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse handoff decision JSON: {e}")
            self.logger.error(f"Response was: {decision_text[:500]}")
            return {
                "handoff_required": False,
                "target_agent": None,
                "handoff_reason": "parsing_error",
                "confidence": 0.0,
                "context": {},
            }
        except Exception as e:
            self.logger.error(f"Handoff detection failed: {e}", exc_info=True)
            return {
                "handoff_required": False,
                "target_agent": None,
                "handoff_reason": "detection_error",
                "confidence": 0.0,
                "context": {},
            }

    def _format_analysis(self, agent_analysis: Dict[str, Any]) -> str:
        """
        Format agent analysis for prompt.

        Args:
            agent_analysis: Agent's result/analysis

        Returns:
            Formatted string summary
        """
        if not agent_analysis:
            return "No analysis available yet."

        # Extract key information
        summary_parts = []

        # Check for common result keys
        if "metrics" in agent_analysis:
            metrics = agent_analysis["metrics"]
            if isinstance(metrics, dict):
                summary_parts.append(f"Metrics analyzed: {len(metrics)} data points")

        if "strategy" in agent_analysis:
            summary_parts.append("Campaign strategy generated")

        if "learning_results" in agent_analysis:
            summary_parts.append("Learning analysis completed")

        if "response" in agent_analysis:
            response = str(agent_analysis["response"])[:200]
            summary_parts.append(f"Response: {response}...")

        if "success" in agent_analysis:
            summary_parts.append(f"Success: {agent_analysis['success']}")

        if "error" in agent_analysis:
            summary_parts.append(f"Error: {agent_analysis['error']}")

        # If no specific keys, provide general info
        if not summary_parts:
            summary_parts.append(f"Analysis type: {type(agent_analysis).__name__}")
            if isinstance(agent_analysis, dict):
                summary_parts.append(f"Keys: {list(agent_analysis.keys())[:5]}")

        return " | ".join(summary_parts)
