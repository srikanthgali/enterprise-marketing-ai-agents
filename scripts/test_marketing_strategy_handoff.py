#!/usr/bin/env python3
"""
Test script to verify Marketing Strategy Agent handoff functionality.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.marketing_agents.agents.marketing_strategy import MarketingStrategyAgent
from src.marketing_agents.memory.memory_manager import MemoryManager
from config.settings import get_settings


async def test_handoff_scenarios():
    """Test all Marketing Strategy Agent handoff scenarios."""

    # Initialize agent
    settings = get_settings()
    memory_manager = MemoryManager()

    agent_config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.7,
        },
        "handoff_rules": [
            {
                "trigger": "strategy_validation_needed",
                "target": "analytics_evaluation",
                "conditions": ["performance_forecast", "feasibility_analysis"],
            },
            {
                "trigger": "customer_insights_needed",
                "target": "customer_support",
                "conditions": ["customer_feedback", "pain_points"],
            },
            {
                "trigger": "optimization_needed",
                "target": "feedback_learning",
                "conditions": ["strategy_improvement", "past_performance"],
            },
        ],
    }

    agent = MarketingStrategyAgent(config=agent_config, memory_manager=memory_manager)

    # Test scenarios from MARKETING_AGENT_TEST_GUIDE.md
    test_scenarios = [
        {
            "name": "Strategy Validation → Analytics",
            "message": "I created a campaign plan last month. Can you analyze if it's performing well?",
            "expected_target": "analytics_evaluation",
        },
        {
            "name": "Forecast Request → Analytics",
            "message": "If we increase our content marketing budget by 20%, what impact can we expect?",
            "expected_target": "analytics_evaluation",
        },
        {
            "name": "Customer Feedback → Customer Support",
            "message": "What are customers saying about our checkout experience? I want to incorporate this into our messaging.",
            "expected_target": "customer_support",
        },
        {
            "name": "Pain Point Discovery → Customer Support",
            "message": "What problems are our enterprise customers experiencing with onboarding?",
            "expected_target": "customer_support",
        },
        {
            "name": "Strategy Optimization → Feedback Learning",
            "message": "My last three campaigns underperformed. How can I improve?",
            "expected_target": "feedback_learning",
        },
        {
            "name": "A/B Test Design → Feedback Learning",
            "message": "I want to test different messaging approaches for our enterprise segment. Help me set up experiments.",
            "expected_target": "feedback_learning",
        },
        {
            "name": "Standard Strategy Request (No Handoff)",
            "message": "Create a Q2 marketing campaign for our API billing product targeting enterprise developers.",
            "expected_target": None,
        },
    ]

    results = []

    print("\n" + "=" * 80)
    print("MARKETING STRATEGY AGENT HANDOFF TESTS")
    print("=" * 80)

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'─' * 80}")
        print(f"TEST {i}: {scenario['name']}")
        print(f"{'─' * 80}")
        print(f"Message: {scenario['message']}")
        print(f"Expected Target: {scenario['expected_target'] or 'No handoff'}")

        test_input = {
            "type": "campaign_strategy",
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
                    f"  Context Type: {context.get('analysis_type', context.get('insight_type', context.get('optimization_type')))}"
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
            import traceback

            traceback.print_exc()
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
