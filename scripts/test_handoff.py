#!/usr/bin/env python3
"""
Test script to verify analytics agent handoff functionality.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.marketing_agents.agents.analytics_evaluation import AnalyticsEvaluationAgent
from src.marketing_agents.memory.memory_manager import MemoryManager
from config.settings import get_settings


async def test_handoff_to_marketing_strategy():
    """Test handoff trigger when user requests strategic improvements."""
    print("\n" + "=" * 70)
    print("TEST 1: Handoff to Marketing Strategy Agent")
    print("=" * 70)

    # Initialize agent
    settings = get_settings()
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

    # Test input that should trigger handoff
    test_input = {
        "type": "performance_report",
        "message": "Analyze our campaign performance and recommend improvements",
        "time_range": "30d",
    }

    print(f"\nInput: {test_input['message']}")
    print("\nProcessing...")

    result = await agent.process(test_input)

    print(f"\nSuccess: {result.get('success')}")
    print(f"Is Final: {result.get('is_final')}")
    print(f"Handoff Required: {result.get('handoff_required', False)}")

    if result.get("handoff_required"):
        print(f"✓ HANDOFF TRIGGERED!")
        print(f"  Target Agent: {result.get('target_agent')}")
        print(f"  Reason: {result.get('handoff_reason')}")
        print(f"  Context: {result.get('context', {}).get('analysis_type')}")
    else:
        print("✗ No handoff detected")

    # Check should_handoff method
    print("\nChecking should_handoff method...")
    handoff_request = agent.should_handoff(result)

    if handoff_request:
        print(f"✓ should_handoff returned: {handoff_request.to_agent}")
    else:
        print("✗ should_handoff returned None")

    return result.get("handoff_required", False)


async def test_handoff_to_feedback_learning():
    """Test handoff trigger for learning opportunity."""
    print("\n" + "=" * 70)
    print("TEST 2: Handoff to Feedback & Learning Agent")
    print("=" * 70)

    # Initialize agent
    memory_manager = MemoryManager()

    agent_config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.2,
        },
        "handoff_rules": [
            {
                "trigger": "learning_opportunity",
                "target": "feedback_learning",
                "conditions": ["performance_pattern", "optimization_potential"],
            }
        ],
    }

    agent = AnalyticsEvaluationAgent(config=agent_config, memory_manager=memory_manager)

    # Test input that should trigger handoff
    test_input = {
        "type": "performance_report",
        "message": "I notice campaigns with video content consistently outperform static images. Can we systematize this learning?",
        "time_range": "30d",
    }

    print(f"\nInput: {test_input['message']}")
    print("\nProcessing...")

    result = await agent.process(test_input)

    print(f"\nSuccess: {result.get('success')}")
    print(f"Is Final: {result.get('is_final')}")
    print(f"Handoff Required: {result.get('handoff_required', False)}")

    if result.get("handoff_required"):
        print(f"✓ HANDOFF TRIGGERED!")
        print(f"  Target Agent: {result.get('target_agent')}")
        print(f"  Reason: {result.get('handoff_reason')}")
    else:
        print("✗ No handoff detected")

    return result.get("handoff_required", False)


async def test_handoff_to_customer_support():
    """Test handoff trigger for customer satisfaction issues."""
    print("\n" + "=" * 70)
    print("TEST 3: Handoff to Customer Support Agent")
    print("=" * 70)

    # Initialize agent
    memory_manager = MemoryManager()

    agent_config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.2,
        },
        "handoff_rules": [
            {
                "trigger": "customer_issue_detected",
                "target": "customer_support",
                "conditions": ["satisfaction_drop", "churn_risk"],
            }
        ],
    }

    agent = AnalyticsEvaluationAgent(config=agent_config, memory_manager=memory_manager)

    # Test input that should trigger handoff
    test_input = {
        "type": "performance_report",
        "message": "Customer satisfaction scores dropped 15% this month. What's causing this?",
        "time_range": "30d",
    }

    print(f"\nInput: {test_input['message']}")
    print("\nProcessing...")

    result = await agent.process(test_input)

    print(f"\nSuccess: {result.get('success')}")
    print(f"Is Final: {result.get('is_final')}")
    print(f"Handoff Required: {result.get('handoff_required', False)}")

    if result.get("handoff_required"):
        print(f"✓ HANDOFF TRIGGERED!")
        print(f"  Target Agent: {result.get('target_agent')}")
        print(f"  Reason: {result.get('handoff_reason')}")
    else:
        print("✗ No handoff detected")

    return result.get("handoff_required", False)


async def test_no_handoff_for_normal_report():
    """Test that normal analytics requests don't trigger handoffs."""
    print("\n" + "=" * 70)
    print("TEST 4: No Handoff for Normal Analytics Report")
    print("=" * 70)

    # Initialize agent
    memory_manager = MemoryManager()

    agent_config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.2,
        },
        "handoff_rules": [],
    }

    agent = AnalyticsEvaluationAgent(config=agent_config, memory_manager=memory_manager)

    # Test input that should NOT trigger handoff
    test_input = {
        "type": "performance_report",
        "message": "Show me the campaign metrics for last month",
        "time_range": "30d",
    }

    print(f"\nInput: {test_input['message']}")
    print("\nProcessing...")

    result = await agent.process(test_input)

    print(f"\nSuccess: {result.get('success')}")
    print(f"Is Final: {result.get('is_final')}")
    print(f"Handoff Required: {result.get('handoff_required', False)}")

    if not result.get("handoff_required"):
        print(f"✓ CORRECT: No handoff for normal report")
    else:
        print(f"✗ UNEXPECTED HANDOFF!")
        print(f"  Target Agent: {result.get('target_agent')}")

    return not result.get("handoff_required", False)


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  Analytics Agent Handoff Tests")
    print("=" * 70)

    results = []

    try:
        results.append(await test_handoff_to_marketing_strategy())
        results.append(await test_handoff_to_feedback_learning())
        results.append(await test_handoff_to_customer_support())
        results.append(await test_no_handoff_for_normal_report())
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("✓ ALL TESTS PASSED!")
        return True
    else:
        print(f"✗ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
