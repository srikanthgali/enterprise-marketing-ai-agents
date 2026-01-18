"""
Response formatter for agent results.

Formats agent execution results into user-friendly messages.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def format_agent_response(final_result: Dict[str, Any], agent_id: str) -> str:
    """
    Format agent result into a user-friendly response message.

    Args:
        final_result: The agent's execution result
        agent_id: ID of the agent that produced the result

    Returns:
        Formatted response string
    """
    if not final_result or not isinstance(final_result, dict):
        return "Processing complete."

    # Strategy 1: Check for direct 'response' key (customer support, general)
    if "response" in final_result:
        return str(final_result["response"])

    # Strategy 2: Check for nested 'details' dict with 'response'
    if "details" in final_result:
        details = final_result["details"]
        if isinstance(details, dict) and "response" in details:
            return str(details["response"])

    # Strategy 3: Format based on agent-specific result structures

    # Marketing Strategy Agent
    if "strategy" in final_result:
        return format_marketing_strategy(final_result["strategy"])

    # Analytics Agent
    if "analytics" in final_result:
        return format_analytics_result(final_result["analytics"])

    # Feedback Learning Agent
    if "learning_results" in final_result:
        return format_learning_results(final_result["learning_results"])

    # Fallback: Use summary if available
    if "summary" in final_result:
        return str(final_result["summary"])

    # Last resort: stringify the result
    return "Processing complete. Results available in metadata."


def format_marketing_strategy(strategy: Any) -> str:
    """Format marketing strategy results."""
    if not isinstance(strategy, dict):
        return f"Marketing strategy: {str(strategy)}"

    lines = []
    lines.append("# Marketing Strategy")
    lines.append("")

    # Campaign name
    if strategy.get("campaign_name"):
        lines.append(f"## {strategy['campaign_name']}")
        lines.append("")

    # Objectives
    if strategy.get("objectives"):
        lines.append("### 🎯 Campaign Objectives")
        lines.append("")
        objectives = strategy["objectives"]
        if isinstance(objectives, list):
            for obj in objectives:
                lines.append(f"- {obj}")
        lines.append("")

    # Target Audience
    if strategy.get("target_audience"):
        lines.append("### 👥 Target Audience")
        lines.append("")
        audience = strategy["target_audience"]
        if isinstance(audience, dict):
            segments = audience.get("segments", [])
            for i, segment in enumerate(segments[:3], 1):  # Show top 3
                if isinstance(segment, dict):
                    lines.append(f"**Segment {i}:** {segment.get('name', 'Unnamed')}")
                    if segment.get("description"):
                        lines.append(f"- {segment['description']}")
        lines.append("")

    # Channels
    if strategy.get("channels"):
        lines.append("### 📢 Marketing Channels")
        lines.append("")
        channels = strategy["channels"]
        if isinstance(channels, dict):
            # Handle both structures:
            # 1. {"channels": {"channels": [list]}} - LLM response
            # 2. {"channels": {"social_media": {}, "email": {}}} - default response
            channel_list = channels.get("channels", channels)

            if isinstance(channel_list, list):
                # LLM response: list of channel dicts
                for channel in channel_list[:5]:
                    if isinstance(channel, dict):
                        name = channel.get("channel", "Unknown")
                        allocation = channel.get("allocation", 0)
                        lines.append(f"- **{name}**: ${allocation:,.0f}")
            elif isinstance(channel_list, dict):
                # Default response: dict of channel_name -> channel_details
                for channel_name, channel_info in list(channel_list.items())[:5]:
                    if isinstance(channel_info, dict):
                        priority = channel_info.get("priority", "N/A")
                        rationale = channel_info.get("rationale", "")
                        lines.append(f"- **{channel_name}** (Priority: {priority})")
                        if rationale:
                            lines.append(f"  - {rationale}")
        lines.append("")

    # KPIs
    if strategy.get("kpis"):
        lines.append("### 📊 Key Performance Indicators")
        lines.append("")
        kpis = strategy["kpis"]
        if isinstance(kpis, list):
            for kpi in kpis[:5]:  # Show top 5
                if isinstance(kpi, dict):
                    metric = kpi.get("metric", "Unknown")
                    target = kpi.get("target", "N/A")
                    lines.append(f"- **{metric}:** {target}")
        lines.append("")

    return "\n".join(lines)


def format_analytics_result(analytics: Any) -> str:
    """Format analytics results."""
    if not isinstance(analytics, dict):
        return f"Analytics report: {str(analytics)}"

    lines = []
    lines.append("# 📊 Analytics Report")
    lines.append("")

    # Check for pre-formatted report
    if "report" in analytics:
        report = analytics["report"]
        if isinstance(report, str):
            return report  # Already formatted
        elif isinstance(report, dict) and "report_content" in report:
            return str(report["report_content"])

    # Format metrics if available
    if "metrics" in analytics:
        metrics = analytics["metrics"]
        if isinstance(metrics, dict) and "campaign_metrics" in metrics:
            cm = metrics["campaign_metrics"]
            lines.append("## Campaign Performance")
            lines.append("")
            lines.append(f"- **CTR:** {cm.get('ctr', 0):.2f}%")
            lines.append(f"- **Conversion Rate:** {cm.get('conversion_rate', 0):.2f}%")
            lines.append(f"- **ROI:** {cm.get('roi', 0):.2f}%")
            lines.append(f"- **Total Impressions:** {cm.get('total_impressions', 0):,}")
            lines.append("")

    if not lines or len(lines) <= 2:
        return "Analytics report generated. No detailed metrics available."

    return "\n".join(lines)


def format_learning_results(learning: Any) -> str:
    """Format feedback learning results."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Formatting learning results: {learning}")

    if not isinstance(learning, dict):
        return f"Learning analysis: {str(learning)}"

    lines = []
    request_type = learning.get("request_type", "analysis")
    feedback_type = learning.get("feedback_type", "")

    # Different formats based on feedback type
    if feedback_type == "user_rating":
        lines.append("# 💬 User Feedback Analysis")
        lines.append("")
        analysis = learning.get("analysis", {})
        rating = analysis.get("rating")
        if rating:
            stars = "⭐" * rating + "☆" * (5 - rating)
            lines.append(f"## Rating: {rating}/5 {stars}")
            lines.append("")

    elif feedback_type == "performance_evaluation":
        agent_name = learning.get("agent_name", "Agent")
        lines.append(f"# 📊 {agent_name.replace('_', ' ').title()} Performance Report")
        lines.append("")

    elif request_type == "prediction_improvement":
        lines.append("# 🎯 Prediction Improvement Analysis")
        lines.append("")

    else:
        lines.append("# 📚 Learning & Feedback Analysis")
        lines.append("")

    # Add feedback summary if available
    summary = learning.get("feedback_summary", {})
    if summary:
        lines.append("## 📊 Feedback Summary")
        lines.append("")
        total = summary.get("total_items", 0)
        time_range = summary.get("time_range", "recent")
        agents = summary.get("agents_analyzed", 0)
        lines.append(f"- **Total Feedback Items:** {total}")
        lines.append(f"- **Time Range:** {time_range}")
        lines.append(f"- **Agents Analyzed:** {agents}")
        lines.append("")

    # Add recommendations
    recommendations = learning.get("recommendations", [])
    improvements = learning.get("improvements", [])
    all_recs = recommendations + improvements

    if all_recs:
        lines.append(f"## 💡 {len(all_recs)} Recommendations")
        lines.append("")
        for i, rec in enumerate(all_recs[:5], 1):  # Show top 5
            if isinstance(rec, dict):
                action = rec.get("action", "Action")
                priority = rec.get("priority", "Medium")
                lines.append(f"### {i}. {action}")
                lines.append(f"**Priority:** {priority}")
                details = rec.get("details", "")
                if details:
                    lines.append(f"\n{details}")
                lines.append("")
            elif isinstance(rec, str):
                lines.append(f"{i}. {rec}")
        lines.append("")

    # Add next steps
    next_steps = learning.get("next_steps", [])
    if next_steps:
        lines.append("## 📋 Next Steps")
        lines.append("")
        for step in next_steps[:5]:  # Show top 5
            lines.append(f"- {step}")
        lines.append("")

    if not lines or len(lines) <= 2:
        return f"Learning analysis completed: {request_type}"

    return "\n".join(lines)
