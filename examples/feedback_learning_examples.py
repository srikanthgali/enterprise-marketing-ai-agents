"""
Quick Reference: Feedback & Learning Agent Tools

This module provides quick examples for using the feedback learning agent's
learning and optimization tools.
"""

# ==============================================================================
# 1. FEEDBACK AGGREGATION
# ==============================================================================

from src.marketing_agents.agents.feedback_learning import FeedbackLearningAgent

agent = FeedbackLearningAgent("feedback_learning", config={})

# Aggregate all feedback from last 7 days
feedback = agent._aggregate_feedback(source="all", time_range="last_7_days")

print(f"Total items: {feedback['total_items']}")
print(f"Agents analyzed: {len(feedback['agent_feedback'])}")
print(f"Error patterns: {feedback['error_patterns']}")

# Access agent-specific feedback
for agent_id, metrics in feedback["agent_feedback"].items():
    print(f"{agent_id}: {metrics['success_rate']:.2%} success rate")

# ==============================================================================
# 2. PERFORMANCE EVALUATION
# ==============================================================================

# Prepare metrics
metrics = {
    "executions": 100,
    "successes": 85,
    "failures": 15,
    "avg_duration": 18.5,
    "success_rate": 0.85,
}

# Evaluate performance
result = agent._evaluate_agent_performance("marketing_strategy", metrics)

print(f"Performance Score: {result['score']}")
print(f"Trend: {result['trend']}")

# Show strengths
for strength in result["strengths"]:
    print(f"✓ {strength}")

# Show areas for improvement
for area in result["areas_for_improvement"]:
    print(f"⚠ {area}")

# ==============================================================================
# 3. PATTERN DETECTION
# ==============================================================================

# Detect patterns in execution data
patterns = agent._detect_patterns(execution_data)

print(f"Success patterns: {len(patterns['success_patterns'])}")
print(f"Failure patterns: {len(patterns['failure_patterns'])}")

# Extract best practices
for practice in patterns["best_practices"]:
    print(f"→ {practice}")

# Analyze specific patterns
for pattern in patterns["success_patterns"]:
    print(f"{pattern['type']}: {pattern['description']}")
    print(f"  Confidence: {pattern['confidence']:.2f}")

# ==============================================================================
# 4. PROMPT OPTIMIZATION
# ==============================================================================

# Prepare performance data
performance_data = {
    "score": 0.55,
    "trend": "declining",
    "metrics": {
        "success_rate": 0.70,
        "avg_duration": 28.5,
        "error_rate": 0.18,
    },
}

# Get optimization suggestions
optimization = agent._optimize_prompts("marketing_strategy", performance_data)

print(f"Expected improvement: {optimization['expected_improvement']}")

# Review suggested changes
for key, value in optimization["suggested_changes"].items():
    print(f"{key}: {value}")

# Review reasoning
for reason in optimization["reasoning"]:
    print(f"• {reason}")

# ==============================================================================
# 5. CONFIG UPDATES
# ==============================================================================

# Validate config before updating
new_config = {
    "temperature": 0.4,
    "max_tokens": 1500,
}

validation = agent._validate_config(new_config)

if validation["valid"]:
    # Update config
    result = agent._update_agent_config("marketing_strategy", new_config)

    if result["updated"]:
        print("Config updated successfully!")
        for key, change in result["changes"].items():
            print(f"{key}: {change['old']} → {change['new']}")
    else:
        print(f"Update failed: {result.get('error')}")
else:
    print(f"Validation errors: {validation['errors']}")

# ==============================================================================
# 6. EXPERIMENT TRACKING
# ==============================================================================

# Track multiple variants
variants = [
    ("control", {"success_rate": 0.80, "avg_duration": 25.0}),
    ("variant_a", {"success_rate": 0.85, "avg_duration": 22.0}),
    ("variant_b", {"success_rate": 0.82, "avg_duration": 20.0}),
]

for variant, metrics in variants:
    result = agent._track_experiment("temperature_test", variant, metrics)
    print(f"Tracked {variant}: {result['stored']}")

# Get experiment results
results = agent.get_experiment_results("temperature_test")

if results["found"]:
    print(f"Total records: {results['total_records']}")

    # Compare variants
    for variant, stats in results["variants"].items():
        print(f"{variant}:")
        print(f"  Sample size: {stats['sample_size']}")
        print(f"  Success rate: {stats.get('avg_success_rate', 0):.2%}")
        print(f"  Duration: {stats.get('avg_duration', 0):.2f}s")

    # Get recommendation
    print(f"\n{results['recommendation']}")

# ==============================================================================
# 7. PATTERN DETECTOR TOOL (Standalone)
# ==============================================================================

from src.marketing_agents.tools.pattern_detector import PatternDetector

detector = PatternDetector()

# Detect patterns
patterns = detector.detect_patterns(execution_data)

print(f"Total executions: {patterns['summary']['total_executions']}")
print(f"Success rate: {patterns['summary']['success_rate']:.2%}")

# Get insights
for insight in patterns["insights"]:
    print(f"💡 {insight}")

# Cluster similar executions
clusters = detector.cluster_similar_executions(execution_data, n_clusters=3)

for cluster_id, items in clusters.items():
    avg_duration = sum(i["duration"] for i in items) / len(items)
    print(f"{cluster_id}: {len(items)} executions, avg {avg_duration:.2f}s")

# ==============================================================================
# 8. COMPLETE WORKFLOW EXAMPLE
# ==============================================================================


def analyze_and_optimize_agent(agent_id: str):
    """Complete workflow for analyzing and optimizing an agent."""

    # Step 1: Aggregate feedback
    feedback = agent._aggregate_feedback(source="all", time_range="last_7_days")
    agent_metrics = feedback["agent_feedback"].get(agent_id)

    if not agent_metrics:
        print(f"No data found for agent: {agent_id}")
        return

    print(f"\n{'='*60}")
    print(f"Analysis for: {agent_id}")
    print(f"{'='*60}")

    # Step 2: Evaluate performance
    evaluation = agent._evaluate_agent_performance(agent_id, agent_metrics)

    print(f"\nPerformance Score: {evaluation['score']:.3f}")
    print(f"Trend: {evaluation['trend']}")

    # Step 3: Detect patterns
    execution_data = [
        r for r in agent.execution_history if r.get("agent_id") == agent_id
    ]
    patterns = agent._detect_patterns(execution_data)

    print(f"\nPatterns detected:")
    print(f"  Success: {len(patterns['success_patterns'])}")
    print(f"  Failure: {len(patterns['failure_patterns'])}")

    # Step 4: Generate recommendations
    if evaluation["score"] < 0.7 or evaluation["trend"] == "declining":
        print("\n⚠️ Optimization needed!")

        optimization = agent._optimize_prompts(agent_id, evaluation)

        if optimization["suggested_changes"]:
            print(
                f"\nSuggested changes ({optimization['expected_improvement']} improvement):"
            )
            for key, value in optimization["suggested_changes"].items():
                print(f"  → {key}: {value}")

        # Optionally apply changes
        # result = agent._update_agent_config(agent_id, optimization['suggested_changes'])
    else:
        print("\n✓ Agent performing well!")

    # Step 5: Show best practices
    if patterns["best_practices"]:
        print("\nBest practices:")
        for practice in patterns["best_practices"]:
            print(f"  • {practice}")


# Use the workflow
# analyze_and_optimize_agent("marketing_strategy")

# ==============================================================================
# 9. ERROR CATEGORIZATION
# ==============================================================================

# The agent automatically categorizes errors into these types:
ERROR_TYPES = {
    "timeout_error": "Connection or operation timeout",
    "network_error": "Network or connection issues",
    "rate_limit_error": "API rate limit exceeded",
    "auth_error": "Authentication or authorization failure",
    "validation_error": "Input validation or format error",
    "not_found_error": "Resource not found (404)",
    "memory_error": "Out of memory issues",
    "unknown_error": "Uncategorized errors",
}

# View error distribution from feedback
feedback = agent._aggregate_feedback(source="all", time_range="last_7_days")
for error_type, count in feedback["error_patterns"].items():
    description = ERROR_TYPES.get(error_type, "Unknown")
    print(f"{error_type}: {count} occurrences - {description}")

# ==============================================================================
# 10. PERFORMANCE BASELINES
# ==============================================================================

# Default baselines (can be customized via config)
BASELINES = {
    "response_time": 30.0,  # Max 30 seconds
    "success_rate": 0.85,  # Min 85% success
    "handoff_accuracy": 0.90,  # Min 90% handoff accuracy
    "user_satisfaction": 0.80,  # Min 80% satisfaction
}

# Customize baselines
agent.performance_baselines = {
    "response_time": 20.0,  # Stricter: 20 seconds
    "success_rate": 0.90,  # Stricter: 90% success
    "handoff_accuracy": 0.95,  # Stricter: 95% handoff
    "user_satisfaction": 0.85,  # Stricter: 85% satisfaction
}
