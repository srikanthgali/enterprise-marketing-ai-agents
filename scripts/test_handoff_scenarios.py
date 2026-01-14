#!/usr/bin/env python3
"""
Test handoff scenarios with scenario-specific synthetic data.

Tests the complete flow from user query through analytics agent to handoff detection.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.marketing_agents.agents.analytics_evaluation import AnalyticsEvaluationAgent
from src.marketing_agents.memory.memory_manager import MemoryManager


async def test_scenario(
    scenario_name: str,
    user_query: str,
    expected_target: str = None,
    should_handoff: bool = True,
):
    """Test a specific handoff scenario."""
    print(f"\n{'='*70}")
    print(f"TEST: {scenario_name}")
    print(f"{'='*70}")
    print(f"Query: {user_query}")

    # Initialize agent
    memory_manager = MemoryManager()

    agent_config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.2,
        },
        "handoff_rules": [
            {
                "trigger": "strategic_pivot_needed",
                "target": "marketing_strategy",
                "conditions": ["performance_decline", "new_opportunity"],
            },
            {
                "trigger": "learning_opportunity",
                "target": "feedback_learning",
                "conditions": ["performance_pattern", "optimization_potential"],
            },
            {
                "trigger": "customer_issue_detected",
                "target": "customer_support",
                "conditions": ["satisfaction_drop", "churn_risk"],
            },
        ],
    }

    agent = AnalyticsEvaluationAgent(config=agent_config, memory_manager=memory_manager)

    # Create input that simulates Gradio
    test_input = {
        "type": "performance_report",
        "message": user_query,
        "time_range": "365d",  # Get all synthetic data
    }

    print("\nProcessing...")
    result = await agent.process(test_input)

    print(f"\n📊 Result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Is Final: {result.get('is_final')}")
    print(f"  Handoff Required: {result.get('handoff_required', False)}")

    # Check metrics
    if result.get("analytics", {}).get("metrics"):
        metrics = result["analytics"]["metrics"]
        campaign_metrics = metrics.get("campaign_metrics", {})
        print(f"\n📈 Campaign Metrics:")
        print(f"  ROI: {campaign_metrics.get('roi', 0):.1f}%")
        print(f"  CTR: {campaign_metrics.get('ctr', 0):.2f}%")
        print(f"  Conversion Rate: {campaign_metrics.get('conversion_rate', 0):.2f}%")

    # Check handoff
    passed = True
    if should_handoff:
        if result.get("handoff_required"):
            target = result.get("target_agent")
            reason = result.get("handoff_reason")
            print(f"\n✅ HANDOFF DETECTED!")
            print(f"  Target: {target}")
            print(f"  Reason: {reason}")

            if expected_target and target != expected_target:
                print(f"  ⚠️  Expected {expected_target}, got {target}")
                passed = False
        else:
            print(f"\n❌ EXPECTED HANDOFF BUT NONE OCCURRED")
            passed = False
    else:
        if result.get("handoff_required"):
            print(f"\n❌ UNEXPECTED HANDOFF to {result.get('target_agent')}")
            passed = False
        else:
            print(f"\n✅ CORRECT: No handoff (as expected)")

    return passed


async def main():
    """Run all handoff scenario tests."""
    print("\n" + "=" * 70)
    print("  Analytics Agent Handoff Scenario Tests")
    print("  Testing with scenario-specific synthetic data")
    print("=" * 70)

    results = []

    # Test 1: Performance Decline → Marketing Strategy
    results.append(
        await test_scenario(
            "Performance Decline Requires Strategic Pivot",
            "Our paid social campaigns have declining ROI for 3 months straight. Help.",
            expected_target="marketing_strategy",
            should_handoff=True,
        )
    )

    # Test 2: Pattern Discovery → Feedback Learning
    results.append(
        await test_scenario(
            "Video Content Pattern Discovery",
            "I notice that campaigns with video content consistently outperform static images by 40%. Can we systematize this learning?",
            expected_target="feedback_learning",
            should_handoff=True,
        )
    )

    # Test 3: Customer Satisfaction → Customer Support
    results.append(
        await test_scenario(
            "Customer Satisfaction Drop",
            "Customer satisfaction scores dropped 15% this month. What's causing this?",
            expected_target="customer_support",
            should_handoff=True,
        )
    )

    # Test 4: Prediction Accuracy → Feedback Learning
    results.append(
        await test_scenario(
            "Prediction Accuracy Improvement",
            "Are our conversion rate predictions accurate? How can we improve them?",
            expected_target="feedback_learning",
            should_handoff=True,
        )
    )

    # Test 5: Normal Analytics → No Handoff
    results.append(
        await test_scenario(
            "Normal Analytics Request",
            "Show me the campaign metrics for last month",
            should_handoff=False,
        )
    )

    # Test 6: Strategic Recommendations → Marketing Strategy
    results.append(
        await test_scenario(
            "Strategic Recommendations Request",
            "Analyze our campaign performance and recommend improvements",
            expected_target="marketing_strategy",
            should_handoff=True,
        )
    )

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("\nHandoff system is working correctly with scenario-specific data!")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        print("\nCheck the scenarios and handoff detection logic.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
