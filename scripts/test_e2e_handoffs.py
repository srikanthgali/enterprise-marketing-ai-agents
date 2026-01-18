#!/usr/bin/env python3
"""End-to-end test of agent handoffs with LLM detection."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.marketing_agents.orchestrator import AgentOrchestrator
from config.settings import get_settings


async def test_e2e_handoffs():
    """Test real agent-to-agent handoffs."""

    settings = get_settings()
    orchestrator = AgentOrchestrator(settings)

    test_cases = [
        {
            "name": "Analytics Query - Satisfaction Scores",
            "message": "Our satisfaction scores dropped 15% last month",
            "expected_agent": "analytics_evaluation",
            "should_handoff": False,
        },
        {
            "name": "Marketing Strategy Request",
            "message": "Create a campaign for Black Friday",
            "expected_agent": "marketing_strategy",
            "should_handoff": False,
        },
        {
            "name": "Customer Support Ticket",
            "message": "My order hasn't arrived yet",
            "expected_agent": "customer_support",
            "should_handoff": False,
        },
    ]

    print("\n" + "=" * 80)
    print("End-to-End Handoff Testing")
    print("=" * 80 + "\n")

    for i, test in enumerate(test_cases, 1):
        print(f"{i}. {test['name']}")
        print(f"   Query: {test['message']}")

        try:
            result = await orchestrator.process_request(
                {
                    "message": test["message"],
                    "conversation_id": f"test_e2e_{i}",
                }
            )

            agent_used = result.get("agent")
            handoff_occurred = result.get("handoff_required", False)

            print(f"   Agent Used: {agent_used}")
            print(f"   Handoff: {handoff_occurred}")

            if agent_used == test["expected_agent"]:
                print(f"   ✅ CORRECT ROUTING")
            else:
                print(f"   ❌ WRONG AGENT (expected {test['expected_agent']})")

            if handoff_occurred != test["should_handoff"]:
                if handoff_occurred:
                    print(
                        f"   → Handoff to: {result.get('handoff_info', {}).get('target_agent')}"
                    )
                else:
                    print(f"   ✓ No unnecessary handoff")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")

        print()

    print("=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_e2e_handoffs())
