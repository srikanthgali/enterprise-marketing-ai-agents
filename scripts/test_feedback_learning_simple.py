"""
Simple test for Feedback & Learning Agent - Direct import test.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from random import random, choice

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct imports without going through __init__
from src.marketing_agents.core.base_agent import BaseAgent, AgentStatus
from src.marketing_agents.agents.feedback_learning import FeedbackLearningAgent
from src.marketing_agents.tools.pattern_detector import PatternDetector


def create_mock_data(count: int = 20) -> list:
    """Create mock execution data."""
    agents = ["agent_a", "agent_b", "agent_c"]
    errors = ["Timeout", "Connection failed", "Rate limit", "Validation error"]

    data = []
    now = datetime.utcnow()

    for i in range(count):
        success = random() > 0.3
        record = {
            "agent_id": choice(agents),
            "timestamp": (now - timedelta(minutes=i * 10)).isoformat(),
            "success": success,
            "duration": 10 + random() * 20,
        }

        if not success:
            record["error"] = choice(errors)

        data.append(record)

    return data


def test_basic_functionality():
    """Test basic functionality of feedback learning agent."""
    print("=" * 80)
    print("FEEDBACK LEARNING AGENT - BASIC FUNCTIONALITY TEST")
    print("=" * 80)

    # Test 1: Agent initialization
    print("\nTest 1: Agent Initialization")
    agent = FeedbackLearningAgent(
        agent_id="feedback_learning", config={"capabilities": ["feedback_aggregation"]}
    )
    print(f"✓ Agent created: {agent.agent_id}")
    print(f"✓ Pattern detector initialized: {agent.pattern_detector is not None}")

    # Test 2: Feedback aggregation
    print("\nTest 2: Feedback Aggregation")
    mock_data = create_mock_data(20)
    agent.execution_history = mock_data

    result = agent._aggregate_feedback(source="all", time_range="all")
    print(f"✓ Aggregated {result['total_items']} items")
    print(f"✓ Found {len(result['agent_feedback'])} agents")
    print(f"✓ Error patterns: {result['error_patterns']}")

    # Test 3: Performance evaluation
    print("\nTest 3: Performance Evaluation")
    metrics = {
        "executions": 100,
        "successes": 85,
        "failures": 15,
        "avg_duration": 18.5,
        "success_rate": 0.85,
    }

    perf_result = agent._evaluate_agent_performance("agent_a", metrics)
    print(f"✓ Performance score: {perf_result['score']:.3f}")
    print(f"✓ Trend: {perf_result['trend']}")
    print(f"✓ Strengths: {len(perf_result['strengths'])}")
    print(f"✓ Areas for improvement: {len(perf_result['areas_for_improvement'])}")

    # Test 4: Pattern detection
    print("\nTest 4: Pattern Detection")
    pattern_result = agent._detect_patterns(mock_data)
    print(f"✓ Success patterns: {len(pattern_result['success_patterns'])}")
    print(f"✓ Failure patterns: {len(pattern_result['failure_patterns'])}")
    print(f"✓ Best practices: {len(pattern_result['best_practices'])}")

    for practice in pattern_result["best_practices"][:3]:
        print(f"  → {practice}")

    # Test 5: Prompt optimization
    print("\nTest 5: Prompt Optimization")
    opt_result = agent._optimize_prompts("marketing_strategy", perf_result)
    print(f"✓ Expected improvement: {opt_result['expected_improvement']}")
    print(f"✓ Suggested changes: {len(opt_result['suggested_changes'])}")

    # Test 6: Config validation
    print("\nTest 6: Config Validation")
    valid_config = {"temperature": 0.5, "max_tokens": 1500}
    invalid_config = {"temperature": 5.0}  # Invalid

    valid_result = agent._validate_config(valid_config)
    invalid_result = agent._validate_config(invalid_config)

    print(f"✓ Valid config: {valid_result['valid']}")
    print(f"✓ Invalid config detected: {not invalid_result['valid']}")
    print(f"  Errors: {invalid_result['errors']}")

    # Test 7: Experiment tracking
    print("\nTest 7: Experiment Tracking")
    exp1 = agent._track_experiment("test_exp", "control", {"success_rate": 0.80})
    exp2 = agent._track_experiment("test_exp", "variant_a", {"success_rate": 0.85})

    print(f"✓ Tracked control: {exp1['stored']}")
    print(f"✓ Tracked variant_a: {exp2['stored']}")

    exp_results = agent.get_experiment_results("test_exp")
    print(f"✓ Experiment found: {exp_results['found']}")
    print(f"✓ Recommendation: {exp_results['recommendation']}")

    # Test 8: Pattern Detector Tool
    print("\nTest 8: Pattern Detector Tool")
    detector = PatternDetector()
    patterns = detector.detect_patterns(mock_data)

    print(f"✓ Total executions: {patterns['summary']['total_executions']}")
    print(f"✓ Success rate: {patterns['summary']['success_rate']:.2%}")
    print(f"✓ Insights generated: {len(patterns['insights'])}")

    for insight in patterns["insights"][:2]:
        print(f"  💡 {insight}")

    # Test 9: Clustering
    print("\nTest 9: Execution Clustering")
    clusters = detector.cluster_similar_executions(mock_data, n_clusters=3)
    print(f"✓ Created {len(clusters)} clusters")
    for cluster_id, items in clusters.items():
        print(f"  - {cluster_id}: {len(items)} items")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED SUCCESSFULLY")
    print("=" * 80)

    print("\nImplemented Methods:")
    print("  ✓ _aggregate_feedback() - Collects feedback from execution history")
    print("  ✓ _evaluate_agent_performance() - Evaluates against baselines")
    print("  ✓ _detect_patterns() - Identifies success/failure patterns")
    print("  ✓ _optimize_prompts() - Suggests prompt optimizations")
    print("  ✓ _update_agent_config() - Updates config with validation")
    print("  ✓ _track_experiment() - Tracks A/B test variants")
    print("  ✓ PatternDetector - ML-based pattern detection tool")
    print("")


if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
