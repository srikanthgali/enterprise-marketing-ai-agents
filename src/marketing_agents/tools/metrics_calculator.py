"""
Metrics Calculator - Reusable metric computation functions for analytics.

Provides standardized calculation methods for campaign, agent, and system metrics.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import statistics
from collections import defaultdict


def calculate_campaign_metrics(execution_data: List[Dict]) -> Dict[str, float]:
    """
    Calculate campaign performance metrics from execution history.

    Args:
        execution_data: List of execution records with campaign data

    Returns:
        Dictionary containing campaign metrics:
        - ctr: Click-through rate (clicks / impressions)
        - conversion_rate: Conversion rate (conversions / clicks)
        - engagement_rate: Engagement rate ((likes + shares + comments) / impressions)
        - roi: Return on investment ((revenue - cost) / cost * 100)
        - avg_cpc: Average cost per click
        - avg_cpa: Average cost per acquisition
    """
    total_impressions = 0
    total_clicks = 0
    total_conversions = 0
    total_engagements = 0
    total_cost = 0
    total_revenue = 0

    for record in execution_data:
        result = record.get("result", {})
        metrics = result.get("metrics", {})

        total_impressions += metrics.get("impressions", 0)
        total_clicks += metrics.get("clicks", 0)
        total_conversions += metrics.get("conversions", 0)
        total_engagements += (
            metrics.get("likes", 0)
            + metrics.get("shares", 0)
            + metrics.get("comments", 0)
        )
        total_cost += metrics.get("cost", 0)
        total_revenue += metrics.get("revenue", 0)

    # Calculate rates with zero-division protection
    ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    conversion_rate = (
        (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
    )
    engagement_rate = (
        (total_engagements / total_impressions * 100) if total_impressions > 0 else 0
    )
    roi = ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
    avg_cpc = (total_cost / total_clicks) if total_clicks > 0 else 0
    avg_cpa = (total_cost / total_conversions) if total_conversions > 0 else 0

    return {
        "ctr": round(ctr, 2),
        "conversion_rate": round(conversion_rate, 2),
        "engagement_rate": round(engagement_rate, 2),
        "roi": round(roi, 2),
        "avg_cpc": round(avg_cpc, 2),
        "avg_cpa": round(avg_cpa, 2),
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "total_conversions": total_conversions,
        "total_cost": round(total_cost, 2),
        "total_revenue": round(total_revenue, 2),
    }


def calculate_agent_metrics(execution_data: List[Dict]) -> Dict[str, Any]:
    """
    Calculate agent performance metrics from execution history.

    Args:
        execution_data: List of agent execution records

    Returns:
        Dictionary containing agent metrics:
        - avg_response_time: Average time to complete tasks (seconds)
        - success_rate: Percentage of successful executions
        - handoff_rate: Percentage of executions requiring handoffs
        - error_rate: Percentage of failed executions
        - agent_breakdown: Per-agent statistics
    """
    total_executions = len(execution_data)
    if total_executions == 0:
        return {
            "avg_response_time": 0,
            "success_rate": 0,
            "handoff_rate": 0,
            "error_rate": 0,
            "agent_breakdown": {},
        }

    response_times = []
    successful_count = 0
    handoff_count = 0
    error_count = 0
    agent_stats = defaultdict(lambda: {"count": 0, "success": 0, "errors": 0})

    for record in execution_data:
        agent_id = record.get("agent_id", "unknown")
        status = record.get("status", "unknown")

        # Calculate response time
        if record.get("started_at") and record.get("completed_at"):
            try:
                started = record["started_at"]
                completed = record["completed_at"]
                if isinstance(started, str):
                    started = datetime.fromisoformat(started.replace("Z", "+00:00"))
                if isinstance(completed, str):
                    completed = datetime.fromisoformat(completed.replace("Z", "+00:00"))

                # Only calculate if both are valid datetime objects
                if (
                    started
                    and completed
                    and isinstance(started, datetime)
                    and isinstance(completed, datetime)
                ):
                    response_time = (completed - started).total_seconds()
                    response_times.append(response_time)
            except (ValueError, TypeError, AttributeError) as e:
                # Skip records with invalid timestamps
                pass

        # Count statuses
        if status == "completed":
            successful_count += 1
            agent_stats[agent_id]["success"] += 1
        elif status == "failed" or status == "error":
            error_count += 1
            agent_stats[agent_id]["errors"] += 1

        # Check for handoffs
        result = record.get("result", {})
        if result.get("handoff_required") or result.get("target_agent"):
            handoff_count += 1

        agent_stats[agent_id]["count"] += 1

    avg_response_time = statistics.mean(response_times) if response_times else 0
    success_rate = successful_count / total_executions * 100
    handoff_rate = handoff_count / total_executions * 100
    error_rate = error_count / total_executions * 100

    # Calculate per-agent metrics
    agent_breakdown = {}
    for agent_id, stats in agent_stats.items():
        agent_breakdown[agent_id] = {
            "executions": stats["count"],
            "success_rate": (
                round(stats["success"] / stats["count"] * 100, 2)
                if stats["count"] > 0
                else 0
            ),
            "error_rate": (
                round(stats["errors"] / stats["count"] * 100, 2)
                if stats["count"] > 0
                else 0
            ),
        }

    return {
        "avg_response_time": round(avg_response_time, 2),
        "success_rate": round(success_rate, 2),
        "handoff_rate": round(handoff_rate, 2),
        "error_rate": round(error_rate, 2),
        "total_executions": total_executions,
        "agent_breakdown": agent_breakdown,
    }


def calculate_system_metrics(
    execution_data: List[Dict], time_window: int = 3600
) -> Dict[str, Any]:
    """
    Calculate system-level performance metrics.

    Args:
        execution_data: List of execution records
        time_window: Time window in seconds for throughput calculation (default: 1 hour)

    Returns:
        Dictionary containing system metrics:
        - throughput: Executions per hour
        - error_rate: Percentage of failed executions
        - latency_p50: 50th percentile latency (median)
        - latency_p95: 95th percentile latency
        - latency_p99: 99th percentile latency
        - avg_latency: Average latency
    """
    if not execution_data:
        return {
            "throughput": 0,
            "error_rate": 0,
            "latency_p50": 0,
            "latency_p95": 0,
            "latency_p99": 0,
            "avg_latency": 0,
        }

    latencies = []
    error_count = 0

    # Calculate latencies and errors
    for record in execution_data:
        if record.get("started_at") and record.get("completed_at"):
            try:
                started = record["started_at"]
                completed = record["completed_at"]

                if isinstance(started, str):
                    started = datetime.fromisoformat(started.replace("Z", "+00:00"))
                if isinstance(completed, str):
                    completed = datetime.fromisoformat(completed.replace("Z", "+00:00"))

                # Only calculate if both are valid datetime objects
                if (
                    started
                    and completed
                    and isinstance(started, datetime)
                    and isinstance(completed, datetime)
                ):
                    latency = (completed - started).total_seconds()
                    latencies.append(latency)
            except (ValueError, TypeError, AttributeError):
                # Skip records with invalid timestamps
                pass

        status = record.get("status", "")
        if status in ["failed", "error"]:
            error_count += 1

    # Calculate throughput (executions per hour)
    if execution_data:
        try:
            first_time = execution_data[0].get("started_at")
            last_time = execution_data[-1].get(
                "completed_at", execution_data[-1].get("started_at")
            )

            if isinstance(first_time, str):
                first_time = datetime.fromisoformat(first_time.replace("Z", "+00:00"))
            if isinstance(last_time, str):
                last_time = datetime.fromisoformat(last_time.replace("Z", "+00:00"))

            # Only calculate if both are valid datetime objects
            if (
                first_time
                and last_time
                and isinstance(first_time, datetime)
                and isinstance(last_time, datetime)
            ):
                time_span = (last_time - first_time).total_seconds()
            else:
                time_span = 0
        except (ValueError, TypeError, AttributeError):
            time_span = 0
        throughput = (len(execution_data) / time_span * 3600) if time_span > 0 else 0
    else:
        throughput = 0

    # Calculate percentiles
    if latencies:
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        p50 = sorted_latencies[int(n * 0.50)] if n > 0 else 0
        p95 = sorted_latencies[int(n * 0.95)] if n > 0 else 0
        p99 = sorted_latencies[int(n * 0.99)] if n > 0 else 0
        avg_latency = statistics.mean(latencies)
    else:
        p50 = p95 = p99 = avg_latency = 0

    error_rate = (error_count / len(execution_data) * 100) if execution_data else 0

    return {
        "throughput": round(throughput, 2),
        "error_rate": round(error_rate, 2),
        "latency_p50": round(p50, 3),
        "latency_p95": round(p95, 3),
        "latency_p99": round(p99, 3),
        "avg_latency": round(avg_latency, 3),
        "total_executions": len(execution_data),
        "total_errors": error_count,
    }


def calculate_time_series_metrics(
    execution_data: List[Dict], metric_key: str, interval: str = "hour"
) -> List[Dict[str, Any]]:
    """
    Calculate metrics over time for trend analysis.

    Args:
        execution_data: List of execution records
        metric_key: Key of the metric to track over time
        interval: Time interval for grouping ('hour', 'day', 'week')

    Returns:
        List of time-series data points with timestamp and value
    """
    time_series = defaultdict(list)

    for record in execution_data:
        timestamp = record.get("started_at")
        if not timestamp:
            continue

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Group by interval
        if interval == "hour":
            key = timestamp.replace(minute=0, second=0, microsecond=0)
        elif interval == "day":
            key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif interval == "week":
            key = timestamp - timedelta(days=timestamp.weekday())
            key = key.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            key = timestamp

        # Extract metric value
        result = record.get("result", {})
        metrics = result.get("metrics", {})
        value = metrics.get(metric_key, 0)

        time_series[key].append(value)

    # Aggregate values for each time period
    result = []
    for timestamp, values in sorted(time_series.items()):
        result.append(
            {
                "timestamp": timestamp.isoformat(),
                "value": sum(values),
                "count": len(values),
                "avg": statistics.mean(values) if values else 0,
            }
        )

    return result


def calculate_comparison_metrics(
    current_data: List[Dict], previous_data: List[Dict]
) -> Dict[str, Any]:
    """
    Compare metrics between two time periods.

    Args:
        current_data: Execution data from current period
        previous_data: Execution data from previous period

    Returns:
        Dictionary with comparison metrics including percent changes
    """
    current_campaign = calculate_campaign_metrics(current_data)
    previous_campaign = calculate_campaign_metrics(previous_data)

    current_agent = calculate_agent_metrics(current_data)
    previous_agent = calculate_agent_metrics(previous_data)

    def percent_change(current: float, previous: float) -> float:
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        return (current - previous) / previous * 100

    return {
        "campaign": {
            "ctr_change": round(
                percent_change(current_campaign["ctr"], previous_campaign["ctr"]), 2
            ),
            "conversion_rate_change": round(
                percent_change(
                    current_campaign["conversion_rate"],
                    previous_campaign["conversion_rate"],
                ),
                2,
            ),
            "roi_change": round(
                percent_change(current_campaign["roi"], previous_campaign["roi"]), 2
            ),
        },
        "agent": {
            "response_time_change": round(
                percent_change(
                    current_agent["avg_response_time"],
                    previous_agent["avg_response_time"],
                ),
                2,
            ),
            "success_rate_change": round(
                percent_change(
                    current_agent["success_rate"], previous_agent["success_rate"]
                ),
                2,
            ),
        },
        "current_period": {
            "executions": len(current_data),
            "campaign_metrics": current_campaign,
            "agent_metrics": current_agent,
        },
        "previous_period": {
            "executions": len(previous_data),
            "campaign_metrics": previous_campaign,
            "agent_metrics": previous_agent,
        },
    }
