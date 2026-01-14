#!/usr/bin/env python3
"""
Test script to verify Customer Support Agent handoff functionality.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.marketing_agents.agents.customer_support import CustomerSupportAgent
from src.marketing_agents.memory.memory_manager import MemoryManager
from config.settings import get_settings


async def test_handoff_scenarios():
    """Test all Customer Support Agent handoff scenarios."""

    # Initialize agent
    settings = get_settings()
    memory_manager = MemoryManager()

    agent_config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.5,
        },
        "handoff_rules": [
            {
                "trigger": "strategic_insight_found",
                "target": "marketing_strategy",
                "conditions": ["customer_pattern", "campaign_issue"],
            },
            {
                "trigger": "analytics_required",
                "target": "analytics_evaluation",
                "conditions": ["sentiment_trends", "feedback_analysis"],
            },
            {
                "trigger": "system_improvement_identified",
                "target": "feedback_learning",
                "conditions": ["recurring_issues", "process_gaps"],
            },
        ],
    }

    agent = CustomerSupportAgent(config=agent_config, memory_manager=memory_manager)

    # Test scenarios from CUSTOMER_SUPPORT_AGENT_TEST_GUIDE.md
    test_scenarios = [
        {
            "name": "Campaign Issue → Marketing Strategy",
            "message": "I signed up through your Black Friday promotion but didn't receive the 20% discount code mentioned in the email.",
            "expected_target": "marketing_strategy",
        },
        {
            "name": "Customer Insight → Marketing Strategy",
            "message": "Multiple clients are asking if you support cryptocurrency payments. Any plans for that?",
            "expected_target": "marketing_strategy",
        },
        {
            "name": "Trend Analysis → Analytics",
            "message": "We've noticed checkout abandonment increased this month. Can you help analyze why?",
            "expected_target": "analytics_evaluation",
        },
        {
            "name": "Sentiment Analysis → Analytics",
            "message": "Can you tell me what customers are generally saying about the new checkout experience?",
            "expected_target": "analytics_evaluation",
        },
        {
            "name": "Recurring Issue → Feedback Learning",
            "message": "This is the third time this month I've had to contact support about webhook failures. This needs to be fixed permanently.",
            "expected_target": "feedback_learning",
        },
        {
            "name": "Knowledge Gap → Feedback Learning",
            "message": "I searched your documentation but couldn't find anything about handling disputed chargebacks.",
            "expected_target": "feedback_learning",
        },
        {
            "name": "Standard Support Query (No Handoff)",
            "message": "How do I reset my password?",
            "expected_target": None,
        },
    ]

    results = []

    print("\n" + "=" * 80)
    print("CUSTOMER SUPPORT AGENT HANDOFF TESTS")
    print("=" * 80)

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'─' * 80}")
        print(f"TEST {i}: {scenario['name']}")
        print(f"{'─' * 80}")
        print(f"Message: {scenario['message']}")
        print(f"Expected Target: {scenario['expected_target'] or 'No handoff'}")

        test_input = {
            "type": "inquiry",
            "message": scenario["message"],
        }

        try:
            result = await agent.process(test_input)

            handoff_required = result.get("handoff_required", False)
            target_agent = result.get("target_agent")
            handoff_reason = result.get("handoff_reason")
            is_final = result.get("is_final", True)

            print(f"\n✓ Processing completed")
            print(f"  Success: {result.get('success')}")
            print(f"  Is Final: {is_final}")
            print(f"  Handoff Required: {handoff_required}")

            if handoff_required:
                print(f"  Target Agent: {target_agent}")
                print(f"  Reason: {handoff_reason}")
                context = result.get("context", {})
                print(
                    f"  Context Type: {context.get('issue_type', context.get('insight_type', context.get('analysis_type', context.get('pattern_type'))))}"
                )

            # Check if result matches expectation
            passed = False
            if scenario["expected_target"] is None:
                # Should NOT trigger handoff
                passed = not handoff_required and is_final
                status = "✓ PASS" if passed else "✗ FAIL"
            else:
                # Should trigger handoff to specific agent
                passed = (
                    handoff_required
                    and target_agent == scenario["expected_target"]
                    and not is_final
                )
                status = "✓ PASS" if passed else "✗ FAIL"

            print(f"\n{status}")

            if not passed and scenario["expected_target"]:
                print(f"  Expected: handoff to {scenario['expected_target']}")
                print(
                    f"  Got: {'handoff to ' + target_agent if handoff_required else 'no handoff'}"
                )
            elif not passed:
                print(f"  Expected: no handoff")
                print(f"  Got: handoff to {target_agent}")

            results.append(
                {
                    "scenario": scenario["name"],
                    "passed": passed,
                    "expected": scenario["expected_target"],
                    "actual": target_agent if handoff_required else None,
                }
            )

        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            results.append(
                {
                    "scenario": scenario["name"],
                    "passed": False,
                    "expected": scenario["expected_target"],
                    "actual": f"Error: {e}",
                }
            )

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\nPassed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ Some tests failed:")
        for r in results:
            if not r["passed"]:
                print(
                    f"  - {r['scenario']}: expected {r['expected']}, got {r['actual']}"
                )

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(test_handoff_scenarios())
    sys.exit(0 if success else 1)
