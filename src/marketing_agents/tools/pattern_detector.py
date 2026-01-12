"""
Pattern Detector - ML-based pattern detection for agent execution data.

Identifies patterns in successful and failed executions to extract insights
and best practices for system improvement.
"""

from typing import Dict, List, Any
from collections import Counter, defaultdict
import statistics
import logging


class PatternDetector:
    """Detects patterns in agent execution data using rule-based and statistical methods."""

    def __init__(self):
        """Initialize pattern detector."""
        self.logger = logging.getLogger(__name__)
        self.min_pattern_frequency = 3  # Minimum occurrences to consider a pattern
        self.confidence_threshold = 0.7

    def detect_patterns(self, execution_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in execution data.

        Args:
            execution_data: List of execution records with success/failure data

        Returns:
            dict: Detected patterns with confidence scores
        """
        if not execution_data:
            return {
                "success_patterns": [],
                "failure_patterns": [],
                "insights": [],
            }

        # Separate successes and failures
        successes = [e for e in execution_data if e.get("success", False)]
        failures = [e for e in execution_data if not e.get("success", False)]

        success_patterns = self._analyze_successes(successes)
        failure_patterns = self._analyze_failures(failures)
        insights = self._generate_insights(success_patterns, failure_patterns)

        return {
            "success_patterns": success_patterns,
            "failure_patterns": failure_patterns,
            "insights": insights,
            "summary": {
                "total_executions": len(execution_data),
                "success_count": len(successes),
                "failure_count": len(failures),
                "success_rate": (
                    len(successes) / len(execution_data) if execution_data else 0
                ),
            },
        }

    def _analyze_successes(
        self, successes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze successful executions for patterns."""
        patterns = []

        if len(successes) < self.min_pattern_frequency:
            return patterns

        # Pattern 1: Agent performance
        agent_counts = Counter([e.get("agent_id") for e in successes])
        for agent_id, count in agent_counts.most_common(5):
            if count >= self.min_pattern_frequency:
                patterns.append(
                    {
                        "type": "high_performing_agent",
                        "agent_id": agent_id,
                        "frequency": count,
                        "confidence": min(0.95, count / len(successes)),
                        "description": f"Agent {agent_id} has consistently successful executions",
                    }
                )

        # Pattern 2: Duration patterns
        durations = [e.get("duration", 0) for e in successes if "duration" in e]
        if len(durations) > self.min_pattern_frequency:
            avg_duration = statistics.mean(durations)
            std_duration = statistics.stdev(durations) if len(durations) > 1 else 0

            patterns.append(
                {
                    "type": "optimal_duration",
                    "avg_duration": avg_duration,
                    "std_duration": std_duration,
                    "frequency": len(durations),
                    "confidence": 0.85,
                    "description": f"Successful executions average {avg_duration:.2f}s (±{std_duration:.2f}s)",
                }
            )

        # Pattern 3: Time-based patterns
        time_patterns = self._analyze_time_patterns(successes)
        patterns.extend(time_patterns)

        return patterns

    def _analyze_failures(self, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze failed executions for patterns."""
        patterns = []

        if len(failures) < self.min_pattern_frequency:
            return patterns

        # Pattern 1: Error type clustering
        error_types = []
        for e in failures:
            error_msg = str(e.get("error", "unknown"))
            error_type = self._categorize_error(error_msg)
            error_types.append(error_type)

        error_counts = Counter(error_types)
        for error_type, count in error_counts.most_common():
            if count >= self.min_pattern_frequency:
                patterns.append(
                    {
                        "type": "error_pattern",
                        "error_type": error_type,
                        "frequency": count,
                        "confidence": min(0.95, count / len(failures)),
                        "description": f"{error_type} occurred {count} times",
                    }
                )

        # Pattern 2: Agent failure patterns
        agent_counts = Counter([e.get("agent_id") for e in failures])
        for agent_id, count in agent_counts.most_common(3):
            if count >= self.min_pattern_frequency:
                patterns.append(
                    {
                        "type": "high_failure_agent",
                        "agent_id": agent_id,
                        "frequency": count,
                        "confidence": min(0.9, count / len(failures)),
                        "description": f"Agent {agent_id} has recurring failures",
                    }
                )

        # Pattern 3: Duration outliers (timeouts)
        durations = [e.get("duration", 0) for e in failures if "duration" in e]
        if len(durations) > self.min_pattern_frequency:
            avg_duration = statistics.mean(durations)

            # Check for timeout pattern (very long durations)
            timeout_threshold = avg_duration * 2
            timeouts = [d for d in durations if d > timeout_threshold]

            if len(timeouts) >= self.min_pattern_frequency:
                patterns.append(
                    {
                        "type": "timeout_pattern",
                        "frequency": len(timeouts),
                        "avg_timeout_duration": statistics.mean(timeouts),
                        "confidence": 0.9,
                        "description": f"{len(timeouts)} failures likely due to timeouts",
                    }
                )

        return patterns

    def _analyze_time_patterns(
        self, executions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze time-based patterns in executions."""
        patterns = []

        # Group by hour of day
        hourly_counts = defaultdict(int)
        for e in executions:
            timestamp = e.get("timestamp")
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        from datetime import datetime

                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    else:
                        dt = timestamp

                    hour = dt.hour
                    hourly_counts[hour] += 1
                except Exception:
                    continue

        if hourly_counts:
            # Find peak hours
            sorted_hours = sorted(
                hourly_counts.items(), key=lambda x: x[1], reverse=True
            )
            if sorted_hours and sorted_hours[0][1] >= self.min_pattern_frequency:
                peak_hour, peak_count = sorted_hours[0]
                patterns.append(
                    {
                        "type": "peak_hour_pattern",
                        "peak_hour": peak_hour,
                        "frequency": peak_count,
                        "confidence": 0.75,
                        "description": f"Peak activity at hour {peak_hour} with {peak_count} executions",
                    }
                )

        return patterns

    def _categorize_error(self, error_msg: str) -> str:
        """Categorize error messages into types."""
        error_msg_lower = error_msg.lower()

        if "timeout" in error_msg_lower:
            return "timeout_error"
        elif "connection" in error_msg_lower or "network" in error_msg_lower:
            return "network_error"
        elif "rate limit" in error_msg_lower:
            return "rate_limit_error"
        elif "authentication" in error_msg_lower or "unauthorized" in error_msg_lower:
            return "auth_error"
        elif "validation" in error_msg_lower or "invalid" in error_msg_lower:
            return "validation_error"
        elif "not found" in error_msg_lower or "404" in error_msg_lower:
            return "not_found_error"
        elif "memory" in error_msg_lower or "oom" in error_msg_lower:
            return "memory_error"
        else:
            return "unknown_error"

    def _generate_insights(
        self,
        success_patterns: List[Dict[str, Any]],
        failure_patterns: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate actionable insights from patterns."""
        insights = []

        # From success patterns
        for pattern in success_patterns:
            if pattern["type"] == "high_performing_agent":
                insights.append(
                    f"Consider replicating strategies from agent {pattern['agent_id']} "
                    f"which has {pattern['frequency']} successful executions"
                )
            elif pattern["type"] == "optimal_duration":
                insights.append(
                    f"Optimal execution time is around {pattern['avg_duration']:.2f}s - "
                    f"use this as a benchmark for other agents"
                )

        # From failure patterns
        for pattern in failure_patterns:
            if pattern["type"] == "error_pattern":
                error_type = pattern["error_type"]
                if error_type == "timeout_error":
                    insights.append(
                        "Implement retry logic with exponential backoff to handle timeout errors"
                    )
                elif error_type == "network_error":
                    insights.append(
                        "Add connection pooling and circuit breakers for better network reliability"
                    )
                elif error_type == "rate_limit_error":
                    insights.append(
                        "Implement rate limiting and request throttling to prevent API limit errors"
                    )
                elif error_type == "validation_error":
                    insights.append(
                        "Strengthen input validation and add schema checking"
                    )
            elif pattern["type"] == "high_failure_agent":
                insights.append(
                    f"Agent {pattern['agent_id']} needs attention - "
                    f"investigate and optimize its configuration"
                )

        return insights

    def cluster_similar_executions(
        self, executions: List[Dict[str, Any]], n_clusters: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Cluster similar executions using simple rule-based approach.

        For MVP, we use simple duration-based clustering.
        In production, this could use KMeans or DBSCAN on vectorized features.
        """
        if not executions or len(executions) < n_clusters:
            return {"cluster_0": executions}

        # Extract durations
        durations = [(i, e.get("duration", 0)) for i, e in enumerate(executions)]
        durations.sort(key=lambda x: x[1])

        # Simple equal-frequency clustering
        cluster_size = len(durations) // n_clusters
        clusters = {}

        for i in range(n_clusters):
            start_idx = i * cluster_size
            end_idx = start_idx + cluster_size if i < n_clusters - 1 else len(durations)

            cluster_items = [executions[idx] for idx, _ in durations[start_idx:end_idx]]
            clusters[f"cluster_{i}"] = cluster_items

        return clusters
