#!/usr/bin/env python3
"""Test LLM-driven handoff detection."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.marketing_agents.core.handoff_detector import HandoffDetector
from langchain_openai import ChatOpenAI


async def test_handoffs():
    """Test various handoff scenarios."""

    # Initialize detector
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    detector = HandoffDetector(llm=llm)

    # Test scenarios
    scenarios = [
        {
            "name": "Satisfaction scores query (should stay in analytics)",
            "current_agent": "analytics_evaluation",
            "message": "Our satisfaction scores dropped 15% last month. What happened?",
            "analysis": {"metrics_found": True, "trend": "declining"},
        },
        {
            "name": "Optimization request from analytics (should handoff to feedback_learning)",
            "current_agent": "analytics_evaluation",
            "message": "Based on these metrics, recommend improvements",
            "analysis": {"performance_issues": True},
        },
        {
            "name": "Campaign issue from support (should handoff to marketing)",
            "current_agent": "customer_support",
            "message": "The Black Friday discount code isn't working",
            "analysis": {"sentiment": "frustrated", "urgency": "high"},
        },
        {
            "name": "Strategy validation from marketing (should handoff to analytics)",
            "current_agent": "marketing_strategy",
            "message": "Analyze the performance of our email campaigns",
            "analysis": {"strategy_proposed": True},
        },
    ]

    print("=" * 80)
    print("Testing LLM-Driven Handoff Detection")
    print("=" * 80)

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Current Agent: {scenario['current_agent']}")
        print(f"   User Message: {scenario['message']}")

        result = await detector.detect_handoff(
            current_agent=scenario["current_agent"],
            user_message=scenario["message"],
            agent_analysis=scenario["analysis"],
        )

        if result.get("handoff_required"):
            print(f"   ✓ HANDOFF TO: {result['target_agent']}")
            print(f"   Reason: {result.get('reason', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
        else:
            print(f"   ✓ NO HANDOFF (staying in {scenario['current_agent']})")

        print()

    print("=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_handoffs())
