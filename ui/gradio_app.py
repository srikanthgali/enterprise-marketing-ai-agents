"""
Gradio Conversational UI for Enterprise Marketing AI Agents.

Interactive chat interface for multi-agent system with:
- Real-time agent conversations
- Agent handoff visualization
- Intermediate results display
- Conversation history management
- Export capabilities
"""

import gradio as gr
import httpx
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings

# Initialize settings
settings = get_settings()

# API Configuration
API_BASE_URL = f"http://localhost:{settings.security.fastapi_host.split(':')[-1] if hasattr(settings.security, 'fastapi_host') else '8000'}/api/v1"
TIMEOUT = 300.0

# State management
conversation_states: Dict[str, Dict[str, Any]] = {}


class AgentChatInterface:
    """Main chat interface for agent interactions."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        self.current_workflow_id = None

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    def format_message_html(
        self, role: str, content: str, metadata: Optional[Dict] = None
    ) -> str:
        """Format message with syntax highlighting and metadata."""
        # Detect code blocks and JSON
        if "```" in content:
            # Already has code formatting
            formatted_content = content
        elif content.strip().startswith("{") or content.strip().startswith("["):
            # Try to parse as JSON
            try:
                parsed = json.loads(content)
                formatted_content = f"```json\n{json.dumps(parsed, indent=2)}\n```"
            except:
                formatted_content = content
        else:
            formatted_content = content

        # Add metadata if present
        if metadata:
            meta_str = "\n\n---\n"
            if "agent" in metadata:
                meta_str += f"**Agent:** {metadata['agent']}\n"
            if "timestamp" in metadata:
                meta_str += f"**Time:** {metadata['timestamp']}\n"
            if "workflow_id" in metadata:
                meta_str += f"**Workflow ID:** {metadata['workflow_id'][:8]}...\n"
            formatted_content += meta_str

        return formatted_content

    def detect_intent_and_route(
        self, message: str, agent_choice: str
    ) -> Tuple[str, str]:
        """
        Detect user intent and route to appropriate workflow.

        Returns:
            Tuple of (workflow_type, agent_name)
        """
        message_lower = message.lower()

        # Auto routing
        if agent_choice == "Auto":
            # Feedback/Learning keywords - Check FIRST (highest priority for learning queries)
            # Performance evaluation queries
            if any(
                keyword in message_lower
                for keyword in [
                    "how well is",
                    "how is the",
                    "agent performing",
                    "agent performance",
                    "performance of",
                    "evaluate agent",
                    "agent evaluation",
                ]
            ):
                return "feedback_learning", "Feedback Learning Agent"

            # Learning from data queries
            elif any(
                keyword in message_lower
                for keyword in [
                    "what should we learn",
                    "what can we learn",
                    "what did we learn",
                    "learn from this",
                    "learn from",
                    "takeaways from",
                    "lessons from",
                ]
            ):
                return "feedback_learning", "Feedback Learning Agent"

            # Issue investigation queries
            elif any(
                keyword in message_lower
                for keyword in [
                    "multiple agents",
                    "several agents",
                    "agents are reporting",
                    "agents report",
                    "investigate and",
                    "investigate the",
                    "recurring issue",
                    "recurring problem",
                    "keeps happening",
                    "happening again",
                ]
            ):
                return "feedback_learning", "Feedback Learning Agent"

            # User feedback and ratings
            elif any(
                keyword in message_lower
                for keyword in [
                    "rating",
                    "stars",
                    "/5",
                    "quality of the",
                    "rate the",
                    "too generic",
                    "not specific enough",
                    "feedback on",
                ]
            ):
                return "feedback_learning", "Feedback Learning Agent"

            # System improvement queries
            elif any(
                keyword in message_lower
                for keyword in [
                    "improve prediction",
                    "improve forecast",
                    "improve accuracy",
                    "optimize system",
                    "system improvement",
                    "how can we improve",
                    "ways to improve",
                ]
            ):
                return "feedback_learning", "Feedback Learning Agent"

            # Analytics keywords - Check for pure analysis/reporting requests
            elif any(
                keyword in message_lower
                for keyword in [
                    "analyze performance",
                    "show me metrics",
                    "generate report",
                    "create dashboard",
                    "create a report",
                    "view statistics",
                    "display data",
                    "calculate roi",
                    "measure conversion",
                    "track funnel",
                    "customer journey map",
                    "attribution model",
                    "a/b test results",
                    "experiment results",
                ]
            ):
                return "analytics", "Analytics Agent"

            # Campaign/Strategy keywords - Check for campaign creation/planning
            elif any(
                keyword in message_lower
                for keyword in [
                    "launch a campaign",
                    "launch campaign",
                    "create a campaign",
                    "create campaign",
                    "new campaign",
                    "plan a campaign",
                    "plan campaign",
                    "campaign strategy",
                    "marketing strategy",
                    "content calendar",
                    "editorial calendar",
                    "content plan",
                    "email sequence",
                    "nurture sequence",
                    "drip campaign",
                    "market research",
                    "competitive analysis",
                    "go-to-market",
                    "gtm strategy",
                ]
            ):
                return "campaign_launch", "Marketing Strategy Agent"

            # Support/KB keywords - Help and documentation queries
            elif any(
                keyword in message_lower
                for keyword in [
                    "how do i",
                    "how does",
                    "how to",
                    "what is",
                    "explain how",
                    "documentation for",
                    "help me with",
                    "support ticket",
                    "customer issue with",
                ]
            ):
                return "customer_support", "Customer Support Agent"

            # Default to feedback learning for improvement-related queries
            elif any(
                keyword in message_lower
                for keyword in [
                    "improve",
                    "better",
                    "enhance",
                    "optimize",
                ]
            ):
                return "feedback_learning", "Feedback Learning Agent"

            # Default to support for questions
            else:
                return "customer_support", "Customer Support Agent"

        # Manual routing
        else:
            agent_mapping = {
                "Marketing Strategy": ("campaign_launch", "Marketing Strategy Agent"),
                "Customer Support": ("customer_support", "Customer Support Agent"),
                "Analytics": ("analytics", "Analytics Agent"),
                "Feedback Learning": ("feedback_learning", "Feedback Learning Agent"),
            }
            return agent_mapping.get(
                agent_choice, ("customer_support", "Customer Support Agent")
            )

    async def poll_workflow_status(
        self, workflow_id: str, max_attempts: int = 60
    ) -> Dict[str, Any]:
        """
        Poll workflow status until completion.

        Args:
            workflow_id: Workflow ID to poll
            max_attempts: Maximum polling attempts

        Returns:
            Final workflow result
        """
        for _ in range(max_attempts):
            try:
                response = await self.client.get(
                    f"{API_BASE_URL}/workflows/status/{workflow_id}"
                )
                if response.status_code == 200:
                    data = response.json()

                    if data.get("status") == "completed":
                        return data
                    elif data.get("status") == "failed":
                        raise Exception(
                            f"Workflow failed: {data.get('error', 'Unknown error')}"
                        )

                await asyncio.sleep(2)  # Poll every 2 seconds
            except httpx.HTTPError as e:
                await asyncio.sleep(2)
                continue

        raise TimeoutError("Workflow did not complete in time")

    async def stream_response(
        self,
        message: str,
        agent_choice: str,
        history: List[Dict[str, str]],
        conversation_id: str,
    ):
        """
        Stream agent response with intermediate updates.

        Yields:
            Tuple of (history, status, metrics)
        """
        try:
            # Detect workflow and agent
            workflow_type, agent_name = self.detect_intent_and_route(
                message, agent_choice
            )

            # Add user message to history (Gradio 6.0 format)
            history = history + [{"role": "user", "content": message}]
            yield history, f"🔄 Routing to {agent_name}...", {}

            await asyncio.sleep(0.5)

            # Create workflow payload
            workflow_id = str(uuid.uuid4())
            self.current_workflow_id = workflow_id

            # Prepare request based on workflow type
            if workflow_type == "campaign_launch":
                payload = {
                    "campaign_name": "AI-Generated Campaign",
                    "objectives": ["awareness", "leads"],
                    "target_audience": "Business decision makers",
                    "budget": 10000.0,
                    "duration_weeks": 8,
                    "additional_context": {"user_query": message},
                }
                endpoint = f"{API_BASE_URL}/workflows/campaign-launch"

            elif workflow_type == "customer_support":
                payload = {
                    "inquiry": message,
                    "customer_id": f"user_{conversation_id[:8]}",
                    "urgency": "normal",
                }
                endpoint = f"{API_BASE_URL}/workflows/customer-support"

            elif workflow_type == "analytics":
                payload = {
                    "report_type": "campaign_performance",
                    "date_range": {"start": "2026-01-01", "end": "2026-01-31"},
                    "metrics": ["conversion_rate", "roi", "engagement"],
                    "filters": {"user_query": message},
                }
                endpoint = f"{API_BASE_URL}/workflows/analytics"

            elif workflow_type == "feedback_learning":
                # Determine request type based on message content
                # Check in priority order (most specific first)
                request_type = "analyze_feedback"
                message_lower = message.lower()

                # Issue investigation - CHECK FIRST (highest priority)
                if any(
                    keyword in message_lower
                    for keyword in [
                        "investigate",
                        "issues",
                        "recurring",
                        "pattern",
                        "multiple agents",
                        "several agents",
                    ]
                ):
                    request_type = "detect_patterns"
                # Performance evaluation
                elif any(
                    keyword in message_lower
                    for keyword in [
                        "how well is",
                        "agent performing",
                        "performance of",
                        "agent performance",
                    ]
                ):
                    request_type = "analyze_feedback"
                # User ratings/feedback
                elif any(
                    keyword in message_lower
                    for keyword in ["rate", "rating", "stars", "quality", "/5"]
                ):
                    request_type = "analyze_feedback"
                # Prediction/accuracy improvement (LAST - least priority)
                elif any(
                    keyword in message_lower
                    for keyword in [
                        "improve prediction",
                        "prediction accuracy",
                        "forecast accuracy",
                    ]
                ):
                    request_type = "prediction_improvement"

                payload = {
                    "message": message,
                    "request_type": request_type,
                    "time_range": "last_7_days",
                    "context": {
                        "source": "user_feedback",
                        "conversation_id": conversation_id,
                    },
                }
                endpoint = f"{API_BASE_URL}/workflows/feedback-learning"

            else:
                # Fallback to customer support
                payload = {
                    "inquiry": message,
                    "customer_id": f"user_{conversation_id[:8]}",
                    "urgency": "normal",
                }
                endpoint = f"{API_BASE_URL}/workflows/customer-support"

            # Update status
            yield history, f"⚙️ Processing with {agent_name}...", {}

            # Make API request (uses default timeout of 300s)
            response = await self.client.post(endpoint, json=payload)

            if response.status_code == 200:
                result = response.json()
                workflow_id = result.get("workflow_id")

                # Update status
                yield history, f"⏳ Workflow started (ID: {workflow_id[:8]}...)...", {}

                # Poll for completion
                intermediate_count = 0
                while (
                    intermediate_count < 150
                ):  # Max 150 polls = 300 seconds (5 minutes)
                    try:
                        status_response = await self.client.get(
                            f"{API_BASE_URL}/workflows/{workflow_id}"
                        )

                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            current_status = status_data.get("status")

                            if current_status == "completed":
                                # Get final result
                                result_response = await self.client.get(
                                    f"{API_BASE_URL}/workflows/{workflow_id}/results"
                                )

                                if result_response.status_code == 200:
                                    final_result = result_response.json()

                                    # API returns results key, not result
                                    result_data = final_result.get(
                                        "results", final_result.get("result", {})
                                    )

                                    # Check if we need to unwrap further
                                    if isinstance(result_data, dict):
                                        if "final_result" in result_data:
                                            result_data = result_data["final_result"]
                                        elif "result" in result_data:
                                            result_data = result_data["result"]

                                    # Format response
                                    response_text = self.format_workflow_result(
                                        result_data, workflow_type
                                    )

                                    # Add agent handoffs if present
                                    if "agent_transitions" in result_data:
                                        handoff_text = "\n\n### Agent Handoffs:\n"
                                        for transition in result_data[
                                            "agent_transitions"
                                        ]:
                                            handoff_text += f"- {transition.get('from_agent', 'Start')} → {transition.get('to_agent', 'End')}\n"
                                        response_text += handoff_text

                                    # Update history with final response (Gradio 6.0 format)
                                    history = history + [
                                        {"role": "assistant", "content": response_text}
                                    ]

                                    # Extract metrics
                                    metrics = {
                                        "workflow_id": workflow_id,
                                        "duration": result_data.get("duration", "N/A"),
                                        "agent_count": len(
                                            result_data.get("agent_transitions", [])
                                        ),
                                        "status": "completed",
                                    }

                                    yield history, "✅ Complete", metrics
                                    return

                            elif current_status == "failed":
                                error_msg = status_data.get("error", "Unknown error")
                                history = history + [
                                    {
                                        "role": "assistant",
                                        "content": f"❌ Error: {error_msg}",
                                    }
                                ]
                                yield history, "❌ Failed", {"error": error_msg}
                                return

                            elif current_status == "in_progress":
                                # Show intermediate progress
                                agent_info = status_data.get(
                                    "current_agent", "Processing"
                                )
                                yield history, f"⚙️ {agent_info}...", {}

                    except httpx.HTTPError:
                        pass

                    await asyncio.sleep(2)
                    intermediate_count += 1

                # Timeout
                history = history + [
                    {
                        "role": "assistant",
                        "content": "⏱️ Workflow is taking longer than expected. Please check status later.",
                    }
                ]
                yield history, "⏱️ Timeout", {"workflow_id": workflow_id}

            else:
                error_msg = f"API Error: {response.status_code}"
                try:
                    error_detail = response.json().get("detail", error_msg)
                    error_msg = error_detail
                except:
                    pass

                history = history + [
                    {"role": "assistant", "content": f"❌ {error_msg}"}
                ]
                yield history, "❌ Error", {"error": error_msg}

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history = history + [{"role": "assistant", "content": f"❌ {error_msg}"}]
            yield history, "❌ Error", {"error": str(e)}

    def format_workflow_result(self, result: Dict[str, Any], workflow_type: str) -> str:
        """Format workflow result for display."""
        # Debug logging
        print(
            f"[DEBUG] format_workflow_result called with workflow_type={workflow_type}"
        )
        print(f"[DEBUG] result type: {type(result)}")
        print(
            f"[DEBUG] result keys: {result.keys() if isinstance(result, dict) else 'N/A'}"
        )
        if isinstance(result, dict) and "analytics" in result:
            print(
                f"[DEBUG] Analytics found! Keys in analytics: {result['analytics'].keys() if isinstance(result['analytics'], dict) else 'N/A'}"
            )

        if not result:
            return "No result available."

        formatted = ""

        # Handle marketing strategy results
        if "strategy" in result:
            strategy = result["strategy"]
            formatted += "# Marketing Strategy\n\n"

            # Campaign name
            if strategy.get("campaign_name"):
                formatted += f"## {strategy['campaign_name']}\n\n"

            # Objectives
            if strategy.get("objectives"):
                formatted += "### 🎯 Campaign Objectives\n\n"
                objectives = strategy["objectives"]
                if isinstance(objectives, list):
                    for obj in objectives:
                        formatted += f"- {obj}\n"
                else:
                    formatted += str(objectives)
                formatted += "\n\n"

            # Target Audience
            if strategy.get("target_audience"):
                formatted += "### 👥 Target Audience\n\n"
                audience = strategy["target_audience"]
                if isinstance(audience, dict):
                    segments = audience.get("segments", [])
                    if segments:
                        for i, segment in enumerate(segments, 1):
                            if isinstance(segment, dict):
                                formatted += f"**Segment {i}:** {segment.get('name', 'Unnamed')}\n"
                                if segment.get("description"):
                                    formatted += f"- {segment['description']}\n"
                                if segment.get("size"):
                                    formatted += f"- Size: {segment['size']}\n"
                            else:
                                formatted += f"- {segment}\n"
                    else:
                        formatted += str(audience)
                else:
                    formatted += str(audience)
                formatted += "\n\n"

            # Channels
            if strategy.get("channels"):
                formatted += "### 📢 Marketing Channels\n\n"
                channels = strategy["channels"]
                if isinstance(channels, dict):
                    channel_list = channels.get("channels", [])
                    for channel in channel_list:
                        if isinstance(channel, dict):
                            formatted += f"**{channel.get('channel', 'Unknown')}**\n"
                            if channel.get("allocation"):
                                formatted += (
                                    f"- Budget: ${channel['allocation']:,.0f}\n"
                                )
                            if channel.get("rationale"):
                                formatted += f"- {channel['rationale']}\n"
                        else:
                            formatted += f"- {channel}\n"
                elif isinstance(channels, list):
                    for channel in channels:
                        formatted += f"- {channel}\n"
                else:
                    formatted += str(channels)
                formatted += "\n\n"

            # Budget Allocation
            if strategy.get("budget_allocation"):
                formatted += "### 💰 Budget Allocation\n\n"
                budget = strategy["budget_allocation"]
                if isinstance(budget, dict):
                    total = budget.get("total_budget", 0)
                    formatted += f"**Total Budget:** ${total:,.0f}\n\n"
                    allocations = budget.get("allocations", [])
                    for alloc in allocations:
                        if isinstance(alloc, dict):
                            formatted += f"- **{alloc.get('channel', 'Unknown')}:** ${alloc.get('amount', 0):,.0f} ({alloc.get('percentage', 0):.0f}%)\n"
                formatted += "\n\n"

            # Content Strategy
            if strategy.get("content_strategy"):
                formatted += "### 📝 Content Strategy\n\n"
                content = strategy["content_strategy"]
                if isinstance(content, dict):
                    items = content.get("items", [])
                    formatted += (
                        f"**Duration:** {content.get('duration_weeks', 0)} weeks\n\n"
                    )
                    if items:
                        formatted += "**Content Calendar:**\n"
                        for item in items[:6]:  # Show first 6 items
                            if isinstance(item, dict):
                                formatted += f"- Week {item.get('week', '?')}: {item.get('theme', 'N/A')} - {item.get('format', 'N/A')}\n"
                        if len(items) > 6:
                            formatted += f"\n...and {len(items) - 6} more items\n"
                formatted += "\n\n"

            # KPIs
            if strategy.get("kpis"):
                formatted += "### 📊 Key Performance Indicators\n\n"
                kpis = strategy["kpis"]
                if isinstance(kpis, list):
                    for kpi in kpis:
                        if isinstance(kpi, dict):
                            formatted += f"- **{kpi.get('metric', 'Unknown')}:** {kpi.get('target', 'N/A')}\n"
                        else:
                            formatted += f"- {kpi}\n"
                formatted += "\n\n"

            # Recommendations
            if strategy.get("recommendations"):
                formatted += "### 💡 Recommendations\n\n"
                recommendations = strategy["recommendations"]
                if isinstance(recommendations, list):
                    for rec in recommendations:
                        formatted += f"- {rec}\n"
                else:
                    formatted += str(recommendations)
                formatted += "\n\n"

        # Analytics results
        elif "analytics" in result:
            analytics = result["analytics"]

            # Check if there's a report (it's a string with markdown content)
            if isinstance(analytics, dict) and "report" in analytics:
                report = analytics["report"]
                # Report can be either a string or a dict with report_content
                if isinstance(report, str):
                    # Use the string directly (it's already markdown)
                    formatted += report
                    formatted += "\n\n"
                elif isinstance(report, dict) and "report_content" in report:
                    # Use the pre-formatted markdown report
                    formatted += report["report_content"]
                    formatted += "\n\n"
                else:
                    # Fallback: Format analytics data manually
                    formatted += "# 📊 Analytics Report\n\n"

                    # Add metrics if available
                    if "metrics" in analytics:
                        metrics = analytics["metrics"]
                        if isinstance(metrics, dict):
                            # Campaign metrics
                            if "campaign_metrics" in metrics:
                                formatted += "## Campaign Performance\n\n"
                                cm = metrics["campaign_metrics"]
                                formatted += f"- **CTR:** {cm.get('ctr', 0):.2f}%\n"
                                formatted += f"- **Conversion Rate:** {cm.get('conversion_rate', 0):.2f}%\n"
                                formatted += f"- **ROI:** {cm.get('roi', 0):.2f}%\n"
                                formatted += f"- **Total Impressions:** {cm.get('total_impressions', 0):,}\n"
                                formatted += f"- **Total Conversions:** {cm.get('total_conversions', 0):,}\n\n"

                    # Add insights if available from report dict
                    if isinstance(report, dict):
                        insights = report.get("insights", [])
                        if insights:
                            formatted += "## 💡 Key Insights\n\n"
                            for insight in insights:
                                formatted += f"- {insight}\n"
                            formatted += "\n"
            else:
                # Analytics exists but no proper report structure
                formatted += "# 📊 Analytics Report\n\n"
                formatted += str(analytics) + "\n\n"

        # Learning/Feedback results (from feedback_learning agent)
        elif "learning_results" in result:
            learning = result["learning_results"]
            feedback_type = learning.get("feedback_type", "")
            request_type = learning.get("request_type", "")

            # Performance Evaluation
            if feedback_type == "performance_evaluation":
                agent_name = learning.get("agent_name", "Agent")
                formatted += f"# 📊 {agent_name.replace('_', ' ').title()} Performance Report\n\n"

                analysis = learning.get("analysis", {})
                if analysis:
                    status = analysis.get("status", "Unknown")
                    formatted += f"## Current Status: {status}\n\n"

                    metrics = analysis.get("metrics", {})
                    if metrics:
                        formatted += "### 📈 Performance Metrics\n\n"
                        for key, value in metrics.items():
                            formatted += (
                                f"- **{key.replace('_', ' ').title()}:** {value}\n"
                            )
                        formatted += "\n"

                    if "performance_score" in analysis:
                        formatted += f"**Overall Score:** {analysis['performance_score']:.2f}/1.0\n"
                        formatted += (
                            f"**Trend:** {analysis.get('trend', 'stable').title()}\n\n"
                        )

                # Recommendations
                recommendations = learning.get("recommendations", [])
                if recommendations:
                    formatted += f"## 💡 {len(recommendations)} Recommendations\n\n"
                    for i, rec in enumerate(recommendations, 1):
                        formatted += f"### {i}. {rec.get('action', 'Action')}\n"
                        formatted += f"**Priority:** {rec.get('priority', 'Medium')} | **Category:** {rec.get('category', 'General')}\n\n"
                        formatted += f"{rec.get('details', '')}\n\n"
                        formatted += f"*Expected Impact:* {rec.get('expected_impact', 'N/A')}\n\n"

            # User Rating/Feedback
            elif feedback_type == "user_rating":
                formatted += "# 💬 User Feedback Analysis\n\n"

                analysis = learning.get("analysis", {})
                if analysis:
                    rating = analysis.get("rating")
                    if rating:
                        stars = "⭐" * rating + "☆" * (5 - rating)
                        formatted += f"## Rating: {rating}/5 {stars}\n\n"

                    sentiment = analysis.get("sentiment", "").title()
                    subject = (
                        analysis.get("subject", "system").replace("_", " ").title()
                    )
                    formatted += f"**Subject:** {subject}\n"
                    formatted += f"**Sentiment:** {sentiment}\n\n"

                    issues = analysis.get("issues_mentioned", [])
                    if issues:
                        formatted += "**Issues Mentioned:**\n"
                        for issue in issues:
                            formatted += f"- {issue}\n"
                        formatted += "\n"

                # Recommendations
                recommendations = learning.get("recommendations", [])
                if recommendations:
                    formatted += f"## 💡 {len(recommendations)} Improvement Actions\n\n"
                    for i, rec in enumerate(recommendations, 1):
                        formatted += f"### {i}. {rec.get('action', 'Action')}\n"
                        formatted += f"**Priority:** {rec.get('priority', 'Medium')} | **Category:** {rec.get('category', 'General')}\n\n"
                        formatted += f"{rec.get('details', '')}\n\n"
                        formatted += f"*Expected Impact:* {rec.get('expected_impact', 'N/A')}\n\n"

            # Recurring Issue / Pattern Detection
            elif (
                feedback_type == "recurring_issue" or request_type == "detect_patterns"
            ):
                formatted += "# 🔍 Issue Investigation Report\n\n"

                analysis = learning.get("analysis", {})
                if analysis:
                    formatted += "## 📊 Issue Analysis\n\n"
                    formatted += f"- **Affected Area:** {analysis.get('affected_area', 'Unknown').replace('_', ' ').title()}\n"
                    formatted += f"- **Severity:** {analysis.get('severity', 'Unknown').title()}\n"
                    formatted += f"- **Multiple Reports:** {'Yes' if analysis.get('multiple_reports') else 'No'}\n\n"

                    issues = analysis.get("issues_identified", [])
                    if issues:
                        formatted += "**Issues Identified:**\n"
                        for issue in issues:
                            formatted += f"- {issue}\n"
                        formatted += "\n"

                # Recommendations
                recommendations = learning.get("recommendations", [])
                if recommendations:
                    formatted += f"## 💡 {len(recommendations)} Recommended Actions\n\n"
                    for i, rec in enumerate(recommendations, 1):
                        formatted += f"### {i}. {rec.get('action', 'Action')}\n"
                        formatted += f"**Priority:** {rec.get('priority', 'Medium')} | **Category:** {rec.get('category', 'General')}\n\n"
                        formatted += f"{rec.get('details', '')}\n\n"
                        formatted += f"*Expected Impact:* {rec.get('expected_impact', 'N/A')}\n\n"

                # Next steps
                next_steps = learning.get("next_steps", [])
                if next_steps:
                    formatted += "## 📋 Next Steps\n\n"
                    for step in next_steps:
                        formatted += f"- {step}\n"
                    formatted += "\n"

            # Success Pattern / Learning from Success
            elif feedback_type == "success_pattern" or "key_learnings" in learning:
                formatted += "# 🎓 Learning from Success\n\n"

                analysis = learning.get("analysis", {})
                if analysis:
                    formatted += "## 📊 Success Analysis\n\n"
                    success_type = (
                        analysis.get("success_type", "general")
                        .replace("_", " ")
                        .title()
                    )
                    formatted += f"**Success Type:** {success_type}\n\n"

                    metrics = analysis.get("metrics", {})
                    if metrics:
                        formatted += "**Key Metrics:**\n"
                        for key, value in metrics.items():
                            formatted += f"- {key.replace('_', ' ').title()}: {value}\n"
                        formatted += "\n"

                # Key learnings
                learnings = learning.get("key_learnings", [])
                if learnings:
                    formatted += "## 🎯 Key Learnings\n\n"
                    for learning_item in learnings:
                        formatted += f"- {learning_item}\n"
                    formatted += "\n"

                # Recommendations
                recommendations = learning.get("recommendations", [])
                if recommendations:
                    formatted += f"## 💡 {len(recommendations)} Recommendations\n\n"
                    for i, rec in enumerate(recommendations, 1):
                        formatted += f"### {i}. {rec.get('action', 'Action')}\n"
                        formatted += f"**Priority:** {rec.get('priority', 'Medium')} | **Category:** {rec.get('category', 'General')}\n\n"
                        formatted += f"{rec.get('details', '')}\n\n"
                        formatted += f"*Expected Impact:* {rec.get('expected_impact', 'N/A')}\n\n"

            # Prediction improvement analysis
            elif request_type == "prediction_improvement":
                formatted += "# 🎯 Prediction Improvement Analysis\n\n"

                # Analysis section
                analysis = learning.get("analysis", {})
                if analysis:
                    formatted += "## 📊 Current Status\n\n"
                    formatted += f"**Query:** {analysis.get('query', 'N/A')}\n\n"

                    current_metrics = analysis.get("current_metrics", {})
                    if current_metrics:
                        formatted += "**Current Performance:**\n"
                        for key, value in current_metrics.items():
                            formatted += f"- {key.replace('_', ' ').title()}: {value}\n"
                        formatted += "\n"

                    formatted += f"**Prediction Confidence:** {analysis.get('prediction_confidence', 'Unknown')}\n"
                    formatted += f"**Data Quality:** {analysis.get('data_quality', 'Unknown')}\n\n"

                # Recommendations section
                recommendations = learning.get("recommendations", [])
                if recommendations:
                    formatted += f"## 💡 {len(recommendations)} Recommendations\n\n"
                    for i, rec in enumerate(recommendations, 1):
                        priority = rec.get("priority", "Medium")
                        category = rec.get("category", "General")
                        action = rec.get("action", "")
                        details = rec.get("details", "")
                        impact = rec.get("expected_impact", "")

                        formatted += f"### {i}. {action}\n"
                        formatted += (
                            f"**Priority:** {priority} | **Category:** {category}\n\n"
                        )
                        formatted += f"{details}\n\n"
                        formatted += f"*Expected Impact:* {impact}\n\n"

                # Next steps
                next_steps = learning.get("next_steps", [])
                if next_steps:
                    formatted += "## 📋 Next Steps\n\n"
                    for step in next_steps:
                        formatted += f"- {step}\n"
                    formatted += "\n"

            # Generic feedback/learning results (fallback)
            else:
                formatted += "# 📚 Learning & Feedback Analysis\n\n"

                # Feedback summary
                if "feedback_summary" in learning:
                    summary = learning["feedback_summary"]
                    formatted += "## 📊 Summary\n\n"
                    formatted += (
                        f"- Total Items Analyzed: {summary.get('total_items', 0)}\n"
                    )
                    formatted += f"- Time Range: {summary.get('time_range', 'N/A')}\n"
                    formatted += (
                        f"- Agents Analyzed: {summary.get('agents_analyzed', 0)}\n\n"
                    )

                # Improvements
                if "improvements" in learning:
                    formatted += "## 💡 Improvements\n\n"
                    improvements = learning["improvements"]
                    if isinstance(improvements, list):
                        for improvement in improvements:
                            if isinstance(improvement, dict):
                                formatted += f"- **{improvement.get('action', '')}:** {improvement.get('details', '')}\n"
                            else:
                                formatted += f"- {improvement}\n"
                    formatted += "\n"

                # Agent evaluations
                if "agent_evaluations" in learning:
                    evals = learning["agent_evaluations"]
                    if evals:
                        formatted += "## 🎯 Agent Performance\n\n"
                        for agent_id, evaluation in evals.items():
                            score = evaluation.get("score", 0)
                            trend = evaluation.get("trend", "stable")
                            formatted += (
                                f"**{agent_id}:** Score {score:.2f} ({trend})\n"
                            )
                        formatted += "\n"

        # Extract main response (for customer support)
        elif "response" in result:
            formatted += result["response"] + "\n\n"
        elif "answer" in result:
            formatted += result["answer"] + "\n\n"
        elif "analysis" in result:
            formatted += "### Analysis Results\n\n"
            analysis = result["analysis"]
            if isinstance(analysis, dict):
                for key, value in analysis.items():
                    formatted += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
            else:
                formatted += str(analysis) + "\n\n"

        # Add citations if available (from customer support responses)
        if "citations" in result and result["citations"]:
            formatted += "### 📚 Sources\n\n"
            for citation in result["citations"]:
                if isinstance(citation, dict):
                    source = citation.get("source", "Unknown")
                    relevance = citation.get("relevance", 0)
                    formatted += f"- {source} (relevance: {relevance:.0%})\n"
                else:
                    formatted += f"- {citation}\n"
            formatted += "\n"

        # Add sources if available
        if "sources" in result and result["sources"]:
            formatted += "### 📚 Sources\n\n"
            for i, source in enumerate(result["sources"][:3], 1):
                if isinstance(source, dict):
                    title = source.get("title", source.get("source", f"Source {i}"))
                    url = source.get("url", "")
                    formatted += f"{i}. {title}"
                    if url:
                        formatted += f" - [{url}]({url})"
                    formatted += "\n"
                else:
                    formatted += f"{i}. {source}\n"
            formatted += "\n"

        # Add metrics if available
        if "metrics" in result:
            formatted += "### 📊 Key Metrics\n\n"
            metrics = result["metrics"]
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    formatted += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            formatted += "\n"

        # Add recommendations if available (if not already shown in strategy)
        if "recommendations" in result and "strategy" not in result:
            formatted += "### 💡 Recommendations\n\n"
            recommendations = result["recommendations"]
            if isinstance(recommendations, list):
                for rec in recommendations:
                    formatted += f"- {rec}\n"
            else:
                formatted += str(recommendations) + "\n"
            formatted += "\n"

        return (
            formatted.strip() or "Processing complete. No detailed results available."
        )

    def export_conversation(
        self, history: List[Dict[str, str]], conversation_id: str
    ) -> str:
        """Export conversation to JSON file."""
        export_data = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": [
                {**msg, "timestamp": datetime.now().isoformat()} for msg in history
            ],
        }

        export_path = (
            project_root
            / "data"
            / "conversations"
            / f"conversation_{conversation_id[:8]}.json"
        )
        export_path.parent.mkdir(parents=True, exist_ok=True)

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return str(export_path)


# Initialize chat interface
chat_interface = AgentChatInterface()


# Gradio event handlers
async def handle_message(
    message: str, agent_choice: str, history: List, conversation_id: str
):
    """Handle user message and stream response."""
    if not message.strip():
        # Use yield for async generator consistency
        yield history, "💬 Waiting for input", {}, conversation_id
        return

    # Generate conversation ID if needed
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    # Stream response
    async for updated_history, status, metrics in chat_interface.stream_response(
        message, agent_choice, history, conversation_id
    ):
        yield updated_history, status, metrics, conversation_id


def clear_conversation():
    """Clear conversation history."""
    return [], "", "💬 Ready", {}, str(uuid.uuid4())


def export_conversation_handler(history: List, conversation_id: str):
    """Export conversation to file."""
    if not history:
        return "No conversation to export", ""

    try:
        file_path = chat_interface.export_conversation(history, conversation_id)
        return f"✅ Exported to: {file_path}", file_path
    except Exception as e:
        return f"❌ Export failed: {str(e)}", ""


# Build Gradio Interface
def build_interface():
    """Build Gradio chat interface."""

    with gr.Blocks(title="Enterprise Marketing AI Agents") as demo:
        # State management
        conversation_id_state = gr.State(value=str(uuid.uuid4()))

        # Header
        gr.Markdown(
            """
            # 🤖 Enterprise Marketing AI Agents

            **Conversational interface for multi-agent marketing workflows**

            Ask me to:
            - 📱 Plan a campaign for your product
            - 📚 Search knowledge base (e.g., "How does Stripe handle subscriptions?")
            - 📊 Analyze campaign performance
            - 🎯 Get improvement recommendations
            """
        )

        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Agent Conversation",
                    height=500,
                    show_label=True,
                    avatar_images=(
                        None,
                        "https://api.dicebear.com/7.x/bottts/svg?seed=agent",
                    ),
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Ask me anything about marketing, campaigns, or product questions...",
                        lines=2,
                        scale=4,
                        container=False,
                    )

                with gr.Row():
                    submit = gr.Button("Send", variant="primary", scale=1)
                    clear = gr.Button("Clear", scale=1)
                    export = gr.Button("📥 Export", scale=1)

                # Export status
                export_status = gr.Textbox(label="Export Status", visible=False)
                export_path = gr.Textbox(label="Export Path", visible=False)

            # Sidebar
            with gr.Column(scale=1):
                gr.Markdown("### Settings")

                agent_selector = gr.Dropdown(
                    label="Route to Agent",
                    choices=[
                        "Auto",
                        "Marketing Strategy",
                        "Customer Support",
                        "Analytics",
                        "Feedback Learning",
                    ],
                    value="Auto",
                    info="Auto-routing analyzes your message",
                )

                status = gr.Textbox(
                    label="Status",
                    value="💬 Ready",
                    interactive=False,
                    elem_classes="status-box",
                )

                metrics = gr.JSON(
                    label="Current Metrics",
                    elem_classes="metrics-box",
                )

                gr.Markdown("### Example Queries")
                gr.Examples(
                    examples=[
                        ["Plan a campaign for our new payment API product"],
                        ["How does Stripe handle subscription billing?"],
                        ["Analyze our campaign performance from last month"],
                        ["What improvements can we make to agent performance?"],
                        ["Create a content strategy for developer outreach"],
                    ],
                    inputs=msg,
                )

                gr.Markdown(
                    """
                    ### Tips
                    - Use natural language
                    - Be specific about your needs
                    - Mention products/features for context
                    - Ask follow-up questions
                    """
                )

        # Event handlers
        msg.submit(
            fn=handle_message,
            inputs=[msg, agent_selector, chatbot, conversation_id_state],
            outputs=[chatbot, status, metrics, conversation_id_state],
            queue=True,
        ).then(lambda: "", None, msg)

        submit.click(
            fn=handle_message,
            inputs=[msg, agent_selector, chatbot, conversation_id_state],
            outputs=[chatbot, status, metrics, conversation_id_state],
            queue=True,
        ).then(lambda: "", None, msg)

        clear.click(
            fn=clear_conversation,
            inputs=None,
            outputs=[chatbot, msg, status, metrics, conversation_id_state],
            queue=False,
        )

        export.click(
            fn=export_conversation_handler,
            inputs=[chatbot, conversation_id_state],
            outputs=[export_status, export_path],
            queue=False,
        )

    return demo


# Main entry point
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting Enterprise Marketing AI Agents - Gradio Interface")
    print("=" * 80)
    print(f"\n📡 API Backend: {API_BASE_URL}")
    print(
        f"💡 Make sure FastAPI server is running on port {API_BASE_URL.split(':')[-1].split('/')[0]}"
    )
    print(f"\n🌐 Gradio UI will launch on: http://localhost:7860")
    print("=" * 80 + "\n")

    # Build and launch interface
    demo = build_interface()

    demo.queue(
        max_size=20,  # Max queue size
        default_concurrency_limit=5,  # Max concurrent requests
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public sharing
        # Gradio 6.0: theme and css moved to launch()
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .message-box {font-family: 'Inter', sans-serif;}
        .status-box {font-weight: 600; padding: 10px; border-radius: 5px;}
        .metrics-box {background: #f0f4f8; padding: 10px; border-radius: 5px;}
        """,
    )
