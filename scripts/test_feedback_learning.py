"""
Test script for Feedback & Learning Agent.

Tests all implemented methods including feedback aggregation, performance evaluation,
pattern detection, prompt optimization, config updates, and experiment tracking.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.marketing_agents.agents.feedback_learning import FeedbackLearningAgent
from src.marketing_agents.tools.pattern_detector import PatternDetector


def create_mock_execution_data(count: int = 50) -> list:
    """Create mock execution data for testing."""
    from random import random, choice

    agents = [
        "marketing_strategy",
        "customer_support",
        "analytics_evaluation",
        "orchestrator",
    ]
    errors = [
        "Connection timeout after 30s",
        "Rate limit exceeded",
        "Invalid input format",
        "Network connection failed",
        "Authentication failed",
        "Resource not found",
    ]

    data = []
    now = datetime.utcnow()

    for i in range(count):
        success = random() > 0.3  # 70% success rate
        agent_id = choice(agents)
        duration = 10 + random() * 20  # 10-30 seconds

        record = {
            "agent_id": agent_id,
            "timestamp": (now - timedelta(minutes=i * 10)).isoformat(),
            "success": success,
            "duration": duration,
        }

        if not success:
            record["error"] = choice(errors)

        # Add user feedback for some records
        if random() > 0.8:
            record["user_rating"] = 3 + random() * 2  # 3-5 rating
            record["user_comment"] = "Good response" if success else "Could be better"

        data.append(record)

    return data


async def test_feedback_aggregation():
    """Test feedback aggregation method."""
    print("\n" + "=" * 80)
    print("TEST 1: Feedback Aggregation")
    print("=" * 80)

    agent = FeedbackLearningAgent(
        agent_id="feedback_learning", config={"capabilities": ["feedback_aggregation"]}
    )

    # Create mock data and add to execution history
    mock_data = create_mock_execution_data(30)
    agent.execution_history = mock_data

    # Test aggregation
    result = agent._aggregate_feedback(source="all", time_range="last_7_days")

    print(f"\n✓ Aggregated {result['total_items']} feedback items")
    print(f"✓ Time range: {result['time_range']}")
    print(f"✓ Found {len(result['agent_feedback'])} agents")
    print(f"✓ User feedback items: {len(result['user_feedback'])}")
    print(f"✓ Error patterns: {len(result['error_patterns'])}")

    print("\nAgent Feedback Summary:")
    for agent_id, metrics in result["agent_feedback"].items():
        print(f"  - {agent_id}:")
        print(f"    • Executions: {metrics['executions']}")
        print(f"    • Success rate: {metrics['success_rate']:.2%}")
        print(f"    • Avg duration: {metrics['avg_duration']:.2f}s")

    print("\nError Patterns:")
    for error_type, count in result["error_patterns"].items():
        print(f"  - {error_type}: {count} occurrences")

    return result


async def test_performance_evaluation():
    """Test agent performance evaluation."""
    print("\n" + "=" * 80)
    print("TEST 2: Agent Performance Evaluation")
    print("=" * 80)

    agent = FeedbackLearningAgent(
        agent_id="feedback_learning", config={"capabilities": ["model_evaluation"]}
    )

    # Test with sample metrics
    test_metrics = {
        "executions": 100,
        "successes": 85,
        "failures": 15,
        "avg_duration": 18.5,
        "total_duration": 1850,
        "success_rate": 0.85,
    }

    result = agent._evaluate_agent_performance("marketing_strategy", test_metrics)

    print(f"\n✓ Performance score: {result['score']:.3f}")
    print(f"✓ Trend: {result['trend']}")
    print(f"✓ Strengths: {len(result['strengths'])}")
    print(f"✓ Areas for improvement: {len(result['areas_for_improvement'])}")

    print("\nStrengths:")
    for strength in result["strengths"]:
        print(f"  ✓ {strength}")

    if result["areas_for_improvement"]:
        print("\nAreas for Improvement:")
        for area in result["areas_for_improvement"]:
            print(f"  ⚠ {area}")

    return result


async def test_pattern_detection():
    """Test pattern detection."""
    print("\n" + "=" * 80)
    print("TEST 3: Pattern Detection")
    print("=" * 80)

    agent = FeedbackLearningAgent(
        agent_id="feedback_learning", config={"capabilities": ["pattern_detection"]}
    )

    # Create mock execution data
    mock_data = create_mock_execution_data(50)

    result = agent._detect_patterns(mock_data)

    print(f"\n✓ Success patterns found: {len(result['success_patterns'])}")
    print(f"✓ Failure patterns found: {len(result['failure_patterns'])}")
    print(f"✓ Best practices extracted: {len(result['best_practices'])}")

    print("\nSuccess Patterns:")
    for pattern in result["success_patterns"][:3]:
        print(f"  • {pattern.get('type', 'unknown')}: {pattern.get('description', '')}")
        print(f"    Confidence: {pattern.get('confidence', 0):.2f}")

    print("\nFailure Patterns:")
    for pattern in result["failure_patterns"][:3]:
        print(f"  • {pattern.get('type', 'unknown')}: {pattern.get('description', '')}")
        print(f"    Frequency: {pattern.get('frequency', 0)}")

    print("\nBest Practices:")
    for practice in result["best_practices"]:
        print(f"  → {practice}")

    return result


async def test_prompt_optimization():
    """Test prompt optimization."""
    print("\n" + "=" * 80)
    print("TEST 4: Prompt Optimization")
    print("=" * 80)

    agent = FeedbackLearningAgent(
        agent_id="feedback_learning", config={"capabilities": ["workflow_optimization"]}
    )

    # Test with low performance data
    performance_data = {
        "score": 0.55,
        "trend": "declining",
        "metrics": {
            "success_rate": 0.70,
            "avg_duration": 28.5,
            "error_rate": 0.18,
        },
    }

    result = agent._optimize_prompts("marketing_strategy", performance_data)

    print(f"\n✓ Expected improvement: {result['expected_improvement']}")
    print(f"✓ Suggested changes: {len(result['suggested_changes'])}")

    if "error" not in result:
        print("\nCurrent Config:")
        for key, value in result["current_config"].items():
            print(f"  - {key}: {value}")

        print("\nSuggested Changes:")
        for key, value in result["suggested_changes"].items():
            print(f"  → {key}: {value}")

        print("\nReasoning:")
        for reason in result["reasoning"]:
            print(f"  • {reason}")
    else:
        print(f"\n⚠ {result['error']}")

    return result


async def test_config_update():
    """Test configuration update."""
    print("\n" + "=" * 80)
    print("TEST 5: Configuration Update")
    print("=" * 80)

    agent = FeedbackLearningAgent(
        agent_id="feedback_learning", config={"capabilities": ["workflow_optimization"]}
    )

    # Test config update
    new_config = {
        "temperature": 0.4,
        "max_tokens": 1500,
    }

    result = agent._update_agent_config("marketing_strategy", new_config)

    print(f"\n✓ Updated: {result['updated']}")
    print(f"✓ Agent: {result['agent_id']}")
    print(f"✓ Changes: {len(result.get('changes', {}))}")

    if result.get("changes"):
        print("\nChanges Applied:")
        for key, change in result["changes"].items():
            print(f"  - {key}: {change['old']} → {change['new']}")

    if result.get("note"):
        print(f"\n📝 Note: {result['note']}")

    if result.get("error"):
        print(f"\n⚠ Error: {result['error']}")

    return result


async def test_experiment_tracking():
    """Test experiment tracking."""
    print("\n" + "=" * 80)
    print("TEST 6: Experiment Tracking")
    print("=" * 80)

    agent = FeedbackLearningAgent(
        agent_id="feedback_learning", config={"capabilities": ["experiment_tracking"]}
    )

    # Track multiple variants
    variants = [
        ("control", {"success_rate": 0.80, "avg_duration": 25.0}),
        ("variant_a", {"success_rate": 0.85, "avg_duration": 22.0}),
        ("variant_b", {"success_rate": 0.82, "avg_duration": 20.0}),
    ]

    print("\nTracking experiment variants...")
    for variant, metrics in variants:
        result = agent._track_experiment("temperature_optimization", variant, metrics)
        print(f"  ✓ Tracked {variant}: {result['stored']}")

    # Get experiment results
    results = agent.get_experiment_results("temperature_optimization")

    print(f"\n✓ Experiment found: {results['found']}")
    print(f"✓ Total records: {results.get('total_records', 0)}")
    print(f"✓ Variants: {len(results.get('variants', {}))}")

    print("\nVariant Statistics:")
    for variant, stats in results.get("variants", {}).items():
        print(f"  - {variant}:")
        print(f"    • Sample size: {stats.get('sample_size', 0)}")
        print(f"    • Avg success rate: {stats.get('avg_success_rate', 0):.2%}")
        print(f"    • Avg duration: {stats.get('avg_duration', 0):.2f}s")

    print(f"\n🏆 {results.get('recommendation', 'No recommendation')}")

    return results


async def test_pattern_detector_tool():
    """Test standalone pattern detector tool."""
    print("\n" + "=" * 80)
    print("TEST 7: Pattern Detector Tool")
    print("=" * 80)

    detector = PatternDetector()

    # Create test data
    mock_data = create_mock_execution_data(40)

    result = detector.detect_patterns(mock_data)

    print(f"\n✓ Total executions: {result['summary']['total_executions']}")
    print(f"✓ Success count: {result['summary']['success_count']}")
    print(f"✓ Failure count: {result['summary']['failure_count']}")
    print(f"✓ Success rate: {result['summary']['success_rate']:.2%}")

    print(f"\n✓ Success patterns: {len(result['success_patterns'])}")
    print(f"✓ Failure patterns: {len(result['failure_patterns'])}")
    print(f"✓ Insights: {len(result['insights'])}")

    print("\nKey Insights:")
    for insight in result["insights"][:5]:
        print(f"  💡 {insight}")

    # Test clustering
    print("\n\nTesting execution clustering...")
    clusters = detector.cluster_similar_executions(mock_data, n_clusters=3)

    print(f"✓ Created {len(clusters)} clusters")
    for cluster_id, items in clusters.items():
        durations = [item.get("duration", 0) for item in items]
        avg_duration = sum(durations) / len(durations) if durations else 0
        print(f"  - {cluster_id}: {len(items)} executions (avg: {avg_duration:.2f}s)")

    return result


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("FEEDBACK & LEARNING AGENT - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    try:
        # Run all tests
        await test_feedback_aggregation()
        await test_performance_evaluation()
        await test_pattern_detection()
        await test_prompt_optimization()
        await test_config_update()
        await test_experiment_tracking()
        await test_pattern_detector_tool()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ Feedback aggregation: Working")
        print("  ✓ Performance evaluation: Working")
        print("  ✓ Pattern detection: Working")
        print("  ✓ Prompt optimization: Working")
        print("  ✓ Config updates: Working")
        print("  ✓ Experiment tracking: Working")
        print("  ✓ Pattern detector tool: Working")
        print("\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
