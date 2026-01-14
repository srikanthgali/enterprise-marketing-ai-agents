#!/usr/bin/env python3
"""
Test script to verify Feedback & Learning Agent handoff functionality.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.marketing_agents.agents.feedback_learning import FeedbackLearningAgent
from src.marketing_agents.memory.memory_manager import MemoryManager
from config.settings import get_settings


async def test_handoff_scenarios():
    """Test all Feedback & Learning Agent handoff scenarios."""

    # Initialize agent
    settings = get_settings()
    memory_manager = MemoryManager()

    agent_config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.4,
        },
        "handoff_rules": [
            {
                "trigger": "system_update_ready",
                "target": "orchestrator",
                "conditions": ["improvement_validated", "change_approved"],
            },
            {
                "trigger": "strategic_learning",
                "target": "marketing_strategy",
                "conditions": ["new_pattern_learned", "best_practice_identified"],
            },
            {
                "trigger": "analysis_needed",
                "target": "analytics_evaluation",
                "conditions": ["learning_validation", "impact_measurement"],
            },
        ],
    }

    agent = FeedbackLearningAgent(config=agent_config, memory_manager=memory_manager)

    # Test scenarios from FEEDBACK_LEARNING_AGENT_TEST_GUIDE.md
    test_scenarios = [
        {
            "name": "System-Wide Configuration → Orchestrator",
            "message": "Based on performance data, I recommend increasing all agents' context windows by 50%.",
            "expected_target": "orchestrator",
        },
        {
            "name": "Critical Issue → Orchestrator",
            "message": "All agents are responding 3x slower than normal. This needs immediate attention.",
            "expected_target": "orchestrator",
        },
        {
            "name": "Strategic Pattern → Marketing Strategy",
            "message": "Analysis shows enterprise customers respond 3x better to case study content than feature lists. This should inform strategy.",
            "expected_target": "marketing_strategy",
        },
        {
            "name": "Optimization Opportunity → Marketing Strategy",
            "message": "We're underutilizing LinkedIn which has 2x better ROI than other channels for B2B.",
            "expected_target": "marketing_strategy",
        },
        {
            "name": "Learning Validation → Analytics",
            "message": "Before implementing this improvement, I need deeper analysis of the performance impact.",
            "expected_target": "analytics_evaluation",
        },
        {
            "name": "Impact Measurement → Analytics",
            "message": "Measure the actual impact of the optimizations we implemented last month.",
            "expected_target": "analytics_evaluation",
        },
        {
            "name": "Standard Learning Query (No Handoff)",
            "message": "The email campaign from last week had a 45% open rate. What should we learn from this?",
            "expected_target": None,
        },
    ]

    results = []

    print("\n" + "=" * 80)
    print("FEEDBACK & LEARNING AGENT HANDOFF TESTS")
    print("=" * 80)

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'─' * 80}")
        print(f"TEST {i}: {scenario['name']}")
        print(f"{'─' * 80}")
        print(f"Message: {scenario['message']}")
        print(f"Expected Target: {scenario['expected_target'] or 'No handoff'}")

        test_input = {
            "type": "analyze_feedback",
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
                    f"  Context Type: {context.get('update_type', context.get('learning_type', context.get('analysis_type')))}"
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
