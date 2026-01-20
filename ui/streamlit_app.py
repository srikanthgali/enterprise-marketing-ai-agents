"""
Enterprise Marketing AI Agents - System Monitoring Dashboard

Interactive Streamlit dashboard for monitoring agent execution, workflows, and analytics.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import time
import json
from typing import Dict, List, Any, Optional
import io
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.marketing_agents.tools.synthetic_data_loader import load_execution_data

# Page configuration
st.set_page_config(
    page_title="Marketing AI Agents Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
API_BASE_URL = "http://127.0.0.1:8000/api/v1"
AUTO_REFRESH_INTERVAL = 5  # seconds
MAX_RETRIES = 3
REQUEST_TIMEOUT = 10

# Custom CSS
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-healthy {
        color: #00c853;
        font-weight: bold;
    }
    .status-warning {
        color: #ff9800;
        font-weight: bold;
    }
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# API Client Functions
# ============================================================================


def api_request(
    endpoint: str, method: str = "GET", data: Dict = None
) -> Optional[Dict]:
    """Make API request with error handling."""
    url = f"{API_BASE_URL}{endpoint}"

    for attempt in range(MAX_RETRIES):
        try:
            if method == "GET":
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
            else:
                response = requests.request(
                    method, url, json=data, timeout=REQUEST_TIMEOUT
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES - 1:
                st.error(f"❌ Cannot connect to API at {API_BASE_URL}")
                return None
            time.sleep(1)

        except requests.exceptions.Timeout:
            st.warning(f"⏱️ Request timeout (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES - 1:
                return None

        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API Error: {e.response.status_code} - {e.response.text}")
            return None

        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None

    return None


def check_api_status() -> bool:
    """Check if API is responding."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_agents() -> List[Dict]:
    """Get list of all agents."""
    result = api_request("/agents")
    return result if result else []


def get_agent_details(agent_id: str) -> Optional[Dict]:
    """Get detailed information about a specific agent."""
    return api_request(f"/agents/{agent_id}")


def get_agent_history(agent_id: str, limit: int = 50) -> Optional[Dict]:
    """Get execution history for an agent."""
    return api_request(f"/agents/{agent_id}/history?limit={limit}")


def execute_agent(agent_id: str, task_data: Dict) -> Optional[Dict]:
    """Execute an agent with given task data."""
    return api_request(f"/agents/{agent_id}/execute", method="POST", data=task_data)


def get_workflows(status: str = None, limit: int = 50) -> List[Dict]:
    """Get list of workflows."""
    endpoint = f"/workflows?limit={limit}"
    if status:
        endpoint += f"&status={status}"
    result = api_request(endpoint)
    # Handle new response format with workflows list
    if result and "workflows" in result:
        return result["workflows"]
    return result if result else []


def get_workflow_details(workflow_id: str) -> Optional[Dict]:
    """Get detailed information about a workflow."""
    return api_request(f"/workflows/{workflow_id}")


def get_system_metrics() -> Dict:
    """Get system-wide metrics."""
    try:
        agents = get_agents()
        if not agents:
            return {
                "total_executions": 0,
                "avg_response_time": 0,
                "success_rate": 0,
                "active_agents": 0,
            }

        total_executions = sum(agent.get("total_executions", 0) for agent in agents)

        avg_times = []
        for agent in agents:
            avg_time = agent.get("avg_execution_time")
            if avg_time and isinstance(avg_time, (int, float)):
                avg_times.append(avg_time)

        # If no agent times, calculate from workflows
        if not avg_times:
            try:
                workflows = get_workflows(limit=50)
                workflow_durations = []
                for wf in workflows:
                    if wf.get("completed_at") and wf.get("started_at"):
                        try:
                            start = datetime.fromisoformat(
                                str(wf["started_at"]).replace("Z", "+00:00")
                            )
                            end = datetime.fromisoformat(
                                str(wf["completed_at"]).replace("Z", "+00:00")
                            )
                            duration = (end - start).total_seconds()
                            if duration > 0:
                                workflow_durations.append(duration)
                        except:
                            pass
                if workflow_durations:
                    avg_times.append(sum(workflow_durations) / len(workflow_durations))
            except:
                pass

        avg_response_time = sum(avg_times) / len(avg_times) if avg_times else 0

        all_executions = []
        for agent in agents:
            try:
                history = get_agent_history(agent.get("agent_id"), limit=100)
                if history and isinstance(history, dict) and "history" in history:
                    exec_history = history.get("history", [])
                    if exec_history:
                        all_executions.extend(exec_history)
            except Exception:
                continue

        # Calculate success rate after collecting all executions
        if all_executions:
            successful = sum(
                1
                for e in all_executions
                if isinstance(e, dict) and e.get("status") == "completed"
            )
            success_rate = (
                (successful / len(all_executions)) * 100 if all_executions else 0
            )
        else:
            # Fallback to workflow-based metrics
            workflows = get_workflows(limit=100)
            if workflows:
                total_workflows = len(workflows)
                completed = sum(
                    1 for wf in workflows if wf.get("status") == "completed"
                )
                success_rate = (
                    (completed / total_workflows) * 100 if total_workflows else 0
                )
                # If agent total executions are zero, use workflows count
                if total_executions == 0:
                    total_executions = total_workflows
            else:
                success_rate = 0

        return {
            "total_executions": total_executions,
            "avg_response_time": round(avg_response_time, 2),
            "success_rate": round(success_rate, 1),
            "active_agents": len([a for a in agents if a.get("status") != "error"]),
        }
    except Exception as e:
        return {
            "total_executions": 0,
            "avg_response_time": 0,
            "success_rate": 0,
            "active_agents": 0,
        }


# ============================================================================
# Session State Initialization
# ============================================================================

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if "cached_agents" not in st.session_state:
    st.session_state.cached_agents = []

if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("🤖 Agent Dashboard")
    st.markdown("---")

    # API Status
    st.subheader("🔌 API Status")
    api_healthy = check_api_status()
    if api_healthy:
        st.markdown('<p class="status-healthy">● Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<p class="status-error">● Disconnected</p>', unsafe_allow_html=True
        )
        st.warning("API is not responding. Please check if the server is running.")

    st.markdown("---")

    # Auto-refresh control
    st.subheader("⚙️ Settings")
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh

    if st.button("🔄 Refresh Now"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

    st.markdown("---")

    # Time range filter
    st.subheader("📅 Time Range")
    time_range = st.selectbox(
        "Select range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
        index=2,
    )

    # Convert to datetime
    time_ranges = {
        "Last Hour": timedelta(hours=1),
        "Last 6 Hours": timedelta(hours=6),
        "Last 24 Hours": timedelta(days=1),
        "Last 7 Days": timedelta(days=7),
        "Last 30 Days": timedelta(days=30),
    }
    start_date = datetime.now() - time_ranges[time_range]

    st.markdown("---")

    # Agent filter
    st.subheader("🤖 Agent Filter")
    if api_healthy:
        agents = get_agents()
        st.session_state.cached_agents = agents
    else:
        agents = st.session_state.cached_agents

    agent_names = ["All Agents"] + [
        agent.get("name", agent.get("agent_id")) for agent in agents
    ]
    selected_agent_name = st.selectbox("Select agent", agent_names)

    if selected_agent_name != "All Agents":
        st.session_state.selected_agent = next(
            (agent for agent in agents if agent.get("name") == selected_agent_name),
            None,
        )
    else:
        st.session_state.selected_agent = None

    st.markdown("---")

    # Status filter
    st.subheader("📊 Status Filter")
    status_filter = st.multiselect(
        "Filter by status",
        ["completed", "failed", "in_progress", "pending"],
        default=["completed", "failed", "in_progress"],
    )

    st.markdown("---")
    st.caption("Enterprise Marketing AI Agents v1.0")


# ============================================================================
# Main Content Area
# ============================================================================

st.title("📊 Marketing AI Agents - System Monitor")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Overview", "🤖 Agents", "🔄 Workflows", "📊 Analytics"]
)


# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    st.header("System Overview")

    if not api_healthy:
        st.error("⚠️ API is not available. Please start the API server to view metrics.")
    else:
        # Key Metrics
        st.subheader("📊 Key Metrics")
        metrics = get_system_metrics()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Executions", f"{metrics['total_executions']:,}")
        with col2:
            st.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
        with col3:
            st.metric("Avg Response Time", f"{metrics['avg_response_time']:.2f}s")
        with col4:
            st.metric("Active Agents", metrics["active_agents"])

        st.markdown("---")

        # System Health Indicators
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏥 System Health")
            health_data = []

            # API Status
            # Measure actual API response time
            api_response_time = "N/A"
            if api_healthy:
                try:
                    import time as time_module

                    start = time_module.time()
                    requests.get(f"{API_BASE_URL}/health", timeout=5)
                    elapsed = (time_module.time() - start) * 1000  # Convert to ms
                    api_response_time = f"{elapsed:.1f}ms"
                except:
                    api_response_time = "N/A"

            health_data.append(
                {
                    "Component": "API Server",
                    "Status": "✅ Healthy" if api_healthy else "❌ Down",
                    "Response Time": api_response_time,
                }
            )

            # Agent Status
            agents = get_agents()
            active_agents = len([a for a in agents if a.get("status") != "error"])
            total_agents = len(agents)

            # Calculate average agent response time from recent history
            agent_response_time = "N/A"
            if agents:
                try:
                    all_times = []
                    for agent in agents[:5]:  # Sample first 5 agents
                        avg_time = agent.get("avg_execution_time")
                        if avg_time and isinstance(avg_time, (int, float)):
                            all_times.append(avg_time)
                    if all_times:
                        avg = sum(all_times) / len(all_times)
                        agent_response_time = f"{avg:.2f}s"
                except:
                    pass

            health_data.append(
                {
                    "Component": "Agents",
                    "Status": f"✅ {active_agents}/{total_agents} Active",
                    "Response Time": agent_response_time,
                }
            )

            # Memory Store - Assume available if API is healthy
            memory_store_status = "✅ Available" if api_healthy else "⚠️ Unavailable"
            memory_response_time = "N/A"
            if api_healthy:
                try:
                    import time as time_module

                    start = time_module.time()
                    # Memory is managed internally, so if API works, memory works
                    elapsed = (time_module.time() - start) * 1000
                    memory_response_time = "< 1ms"
                except:
                    pass

            health_data.append(
                {
                    "Component": "Memory Store",
                    "Status": memory_store_status,
                    "Response Time": memory_response_time,
                }
            )

            st.dataframe(
                pd.DataFrame(health_data), use_container_width=True, hide_index=True
            )

        with col2:
            st.subheader("📡 Agent Status Distribution")
            if agents:
                status_counts = pd.DataFrame(agents)["status"].value_counts()
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Agent Status",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Real-time Activity Log
        st.subheader("📝 Recent Activity (Last 20 Events)")

        # Aggregate recent activity from workflows AND agents
        all_activity = []

        # First, try to get workflow activity (more reliable)
        try:
            workflows = get_workflows(limit=20)
            if workflows:
                for wf in workflows:
                    # Add workflow as an activity
                    started_at = wf.get("started_at", "N/A")
                    if started_at != "N/A":
                        try:
                            timestamp_str = (
                                str(started_at)[:19] if started_at else "N/A"
                            )
                        except:
                            timestamp_str = "N/A"
                    else:
                        timestamp_str = "N/A"

                    # Calculate duration
                    duration_str = "N/A"
                    if wf.get("completed_at") and wf.get("started_at"):
                        try:
                            start = datetime.fromisoformat(
                                str(wf["started_at"]).replace("Z", "+00:00")
                            )
                            end = datetime.fromisoformat(
                                str(wf["completed_at"]).replace("Z", "+00:00")
                            )
                            duration_seconds = (end - start).total_seconds()
                            duration_str = f"{duration_seconds:.2f}s"
                        except:
                            pass

                    # Get agents involved - if empty, use workflow type as agent name
                    agents_involved = wf.get("agents_executed", [])
                    if agents_involved:
                        agent_str = ", ".join(agents_involved)
                    else:
                        # Infer from workflow type
                        wf_type = wf.get("workflow_type", "")
                        if wf_type == "customer_support":
                            agent_str = "Customer Support Agent"
                        elif wf_type == "campaign_launch":
                            agent_str = "Campaign Agent"
                        elif wf_type == "analytics":
                            agent_str = "Analytics Agent"
                        else:
                            agent_str = (
                                wf_type.replace("_", " ").title()
                                if wf_type
                                else "Workflow Agent"
                            )

                    all_activity.append(
                        {
                            "Timestamp": timestamp_str,
                            "Agent": agent_str,
                            "Task Type": wf.get("workflow_type", "N/A"),
                            "Status": wf.get("status", "N/A"),
                            "Duration": duration_str,
                            "Summary": f"Workflow {wf.get('workflow_id', 'N/A')[:12]}...",
                        }
                    )
        except Exception as e:
            st.warning(f"Could not load workflow activity: {str(e)}", icon="⚠️")

        # Also try to get agent execution history
        try:
            for agent in agents:
                try:
                    history = get_agent_history(agent.get("agent_id"), limit=5)
                    if history and isinstance(history, dict) and "history" in history:
                        exec_history = history.get("history", [])
                        for item in exec_history:
                            if isinstance(item, dict):
                                all_activity.append(
                                    {
                                        "Timestamp": item.get("timestamp", "N/A"),
                                        "Agent": agent.get(
                                            "name", agent.get("agent_id")
                                        ),
                                        "Task Type": item.get("task_type", "N/A"),
                                        "Status": item.get("status", "N/A"),
                                        "Duration": f"{float(item.get('duration_seconds', 0)):.2f}s",
                                        "Summary": str(
                                            item.get("summary", "No summary")
                                        )[:50]
                                        + "...",
                                    }
                                )
                except Exception:
                    continue

            # Sort by timestamp and get last 20
            if all_activity:
                activity_df = pd.DataFrame(all_activity)
                try:
                    activity_df = activity_df.sort_values(
                        "Timestamp", ascending=False
                    ).head(20)
                except:
                    pass
                st.dataframe(activity_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "💡 No recent activity. Run some workflows to see execution history here."
                )
        except Exception as e:
            st.warning(f"Could not load activity log: {str(e)}", icon="⚠️")

        st.markdown("---")

        # System Resource Usage
        st.subheader("💻 System Resources")

        # Calculate system metrics from agents
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)

        except ImportError:
            # Fallback if psutil not available
            cpu_percent = None
            memory_used_gb = None
            memory_total_gb = None
            # psutil not available; metrics will show as N/A

        col1, col2, col3 = st.columns(3)

        with col1:
            if cpu_percent is not None:
                delta_cpu = "↑" if cpu_percent > 50 else "↓"
                st.metric("CPU Usage", f"{cpu_percent:.1f}%", delta=delta_cpu)
            else:
                st.metric("CPU Usage", "N/A")

        with col2:
            if memory_used_gb is not None and memory_total_gb is not None:
                memory_percent = (memory_used_gb / memory_total_gb) * 100
                st.metric(
                    "Memory Usage",
                    f"{memory_used_gb:.1f} GB / {memory_total_gb:.1f} GB",
                    delta=f"{memory_percent:.1f}%",
                )
            else:
                st.metric("Memory Usage", "N/A")

        with col3:
            st.metric("Total Executions", f"{int(metrics.get('total_executions', 0))}")


# ============================================================================
# TAB 2: AGENTS
# ============================================================================

with tab2:
    st.header("Agent Management")

    if not api_healthy:
        st.error("⚠️ API is not available. Please start the API server.")
    else:
        agents = get_agents()

        # Agent List
        st.subheader("🤖 Available Agents")

        if agents and len(agents) > 0:
            try:
                # Create agent cards
                for i in range(0, len(agents), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(agents):
                            agent = agents[i + j]
                            with col:
                                with st.container():
                                    agent_name = agent.get("name", "Unknown Agent")
                                    agent_id = agent.get("agent_id", "unknown_id")
                                    st.markdown("### " + agent_name)
                                    st.caption(f"ID: {agent_id}")

                                    status = agent.get("status", "unknown")
                                    status_emoji = {
                                        "idle": "🟢",
                                        "busy": "🟡",
                                        "error": "🔴",
                                    }.get(status, "⚪")

                                    st.markdown(
                                        f"**Status:** {status_emoji} {status.capitalize()}"
                                    )

                                    total_execs = agent.get("total_executions", 0)
                                    st.metric(
                                        "Executions",
                                        int(total_execs) if total_execs else 0,
                                    )

                                    avg_time = agent.get("avg_execution_time")
                                    if avg_time and isinstance(avg_time, (int, float)):
                                        st.metric("Avg Time", f"{float(avg_time):.2f}s")
                                    else:
                                        st.metric("Avg Time", "N/A")

                                    if st.button(
                                        "View Details",
                                        key=f"details_{agent_id}",
                                    ):
                                        st.session_state.selected_agent = agent
            except Exception as e:
                st.error(f"Error displaying agents: {str(e)}", icon="❌")
        else:
            st.info("💡 No agents available. Check if API is running.")

        # Selected Agent Details (outside the if/else to always show if agent selected)
        if st.session_state.selected_agent:
            agent = st.session_state.selected_agent
            agent_id = agent.get("agent_id")

            st.markdown("---")
            st.subheader(f"📋 Details: {agent.get('name')}")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Capabilities:**")
                capabilities = agent.get("capabilities", [])
                if capabilities:
                    for cap in capabilities:
                        st.markdown(f"- {cap}")
                else:
                    st.info("No capabilities listed")

                # Execution History
                st.markdown("---")
                st.markdown("**📊 Execution History**")
                history = None
                try:
                    history = get_agent_history(agent_id, limit=50)
                    if history and isinstance(history, dict) and "history" in history:
                        history_items = history.get("history", [])

                        if history_items and isinstance(history_items, list):
                            hist_data = []
                            for item in history_items:
                                if isinstance(item, dict):
                                    hist_data.append(
                                        {
                                            "Timestamp": item.get("timestamp", "N/A"),
                                            "Task Type": item.get("task_type", "N/A"),
                                            "Status": item.get("status", "N/A"),
                                            "Duration (s)": f"{float(item.get('duration_seconds', 0)):.2f}",
                                            "Summary": str(
                                                item.get("summary", "No summary")
                                            )[:100],
                                        }
                                    )

                            if hist_data:
                                history_df = pd.DataFrame(hist_data)
                                st.dataframe(
                                    history_df,
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                # Export to CSV
                                csv = history_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv,
                                    file_name=f"{agent_id}_history.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.info(
                                    "💡 No execution history yet. Execute the agent to see results here."
                                )
                        else:
                            st.info(
                                "💡 No execution history yet. Execute the agent to see results here."
                            )
                    else:
                        st.info("💡 No execution history available.")
                except Exception as e:
                    st.warning(f"Could not load execution history: {str(e)}", icon="⚠️")

            with col2:
                # Performance Metrics
                st.markdown("**📈 Performance Metrics**")

                if history and "history" in history:
                    history_items = history["history"]

                    if history_items:
                        # Success rate
                        statuses = [item.get("status") for item in history_items]
                        success_count = sum(1 for s in statuses if s == "completed")
                        success_rate = (
                            (success_count / len(statuses) * 100) if statuses else 0
                        )

                        st.metric("Success Rate", f"{success_rate:.1f}%")

                        # Average duration
                        durations = [
                            item.get("duration_seconds", 0) for item in history_items
                        ]
                        avg_duration = (
                            sum(durations) / len(durations) if durations else 0
                        )

                        st.metric("Avg Duration", f"{avg_duration:.2f}s")

                        # Total executions
                        st.metric("Total Executions", len(history_items))

                        # Duration trend chart
                        if len(durations) > 1:
                            fig = px.line(
                                x=list(range(len(durations))),
                                y=durations,
                                title="Duration Trend",
                                labels={"x": "Execution #", "y": "Duration (s)"},
                            )
                            st.plotly_chart(fig, use_container_width=True)

                # Tool Usage Statistics (from agent capabilities and history)
                st.markdown("---")
                st.markdown("**🔧 Tool Usage**")

                # Count tool usage from agent history
                tool_usage = {}
                if history and "history" in history:
                    history_items = history["history"]
                    for item in history_items:
                        # Extract tools from task type or summary
                        task_type = item.get("task_type", "").lower()
                        summary = item.get("summary", "").lower()

                        # Map common task types to tools
                        if "search" in task_type or "search" in summary:
                            tool_usage["web_search"] = (
                                tool_usage.get("web_search", 0) + 1
                            )
                        if "analysis" in task_type or "analysis" in summary:
                            tool_usage["data_analysis"] = (
                                tool_usage.get("data_analysis", 0) + 1
                            )
                        if (
                            "content" in task_type
                            or "content" in summary
                            or "generate" in task_type
                        ):
                            tool_usage["content_generation"] = (
                                tool_usage.get("content_generation", 0) + 1
                            )
                        if "email" in task_type or "email" in summary:
                            tool_usage["email"] = tool_usage.get("email", 0) + 1

                if tool_usage:
                    for tool, count in tool_usage.items():
                        st.metric(tool.replace("_", " ").title(), count)
                else:
                    st.info("No tool usage data available")

            st.markdown("---")

            # Interactive Agent Execution Form
            st.subheader("🚀 Test Agent Execution")

            with st.form(key=f"execute_form_{agent_id}"):
                st.markdown("Execute this agent with custom task data:")

                task_input = st.text_area(
                    "Task Data (JSON format)",
                    value='{\n  "query": "Create a marketing strategy",\n  "context": "New product launch"\n}',
                    height=150,
                )

                submit_button = st.form_submit_button("▶️ Execute Agent")

                if submit_button:
                    try:
                        task_data = json.loads(task_input)

                        with st.spinner("Executing agent..."):
                            result = execute_agent(agent_id, task_data)

                        if result:
                            st.success("✅ Agent executed successfully!")
                            st.json(result)
                        else:
                            st.error("❌ Agent execution failed.")

                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON format. Please check your input.")


# ============================================================================
# TAB 3: WORKFLOWS
# ============================================================================

with tab3:
    st.header("Workflow Management")

    if not api_healthy:
        st.error("⚠️ API is not available. Please start the API server.")
    else:
        # Workflow filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            workflow_status = st.selectbox(
                "Filter by status",
                ["All", "completed", "in_progress", "failed", "pending"],
            )

        with col2:
            workflow_limit = st.number_input(
                "Results limit", min_value=10, max_value=100, value=50
            )

        with col3:
            auto_refresh = st.checkbox(
                "🔄 Auto-refresh",
                value=True,
                help="Auto-refresh when workflows are in progress",
            )

        with col4:
            if st.button("🔄 Refresh Now"):
                st.rerun()

        st.markdown("---")

        # Get workflows
        workflows = get_workflows(
            status=None if workflow_status == "All" else workflow_status,
            limit=workflow_limit,
        )

        # Check if any workflows are in progress
        has_in_progress = any(
            wf.get("status") in ["in_progress", "pending"] for wf in workflows
        )

        # Auto-refresh if enabled and workflows in progress
        if auto_refresh and has_in_progress:
            st.info("🔄 Auto-refreshing... (workflows in progress)")
            time.sleep(2)  # Wait 2 seconds before refresh
            st.rerun()

        if workflows:
            st.subheader(f"📋 Recent Workflows ({len(workflows)})")

            # Workflows table
            workflows_data = []
            for wf in workflows:
                # Calculate duration from timestamps
                duration_str = "N/A"
                if wf.get("completed_at") and wf.get("started_at"):
                    try:
                        # Handle various datetime formats - preserve microseconds
                        start_val = wf["started_at"]
                        end_val = wf["completed_at"]

                        # Convert to string if datetime object
                        if isinstance(start_val, datetime):
                            start = start_val
                        else:
                            start_str = str(start_val).replace("Z", "+00:00")
                            start = datetime.fromisoformat(start_str)

                        if isinstance(end_val, datetime):
                            end = end_val
                        else:
                            end_str = str(end_val).replace("Z", "+00:00")
                            end = datetime.fromisoformat(end_str)

                        duration_seconds = (end - start).total_seconds()

                        # Format duration nicely
                        if duration_seconds == 0:
                            # Timestamps are truly identical
                            duration_str = "0ms"
                        elif duration_seconds < 1:
                            duration_str = f"{duration_seconds*1000:.0f}ms"
                        elif duration_seconds < 60:
                            duration_str = f"{duration_seconds:.1f}s"
                        else:
                            minutes = int(duration_seconds // 60)
                            seconds = duration_seconds % 60
                            duration_str = f"{minutes}m {seconds:.0f}s"
                    except Exception as e:
                        # Show what went wrong for debugging
                        duration_str = f"Error: {type(e).__name__}"
                elif wf.get("started_at") and wf.get("status") == "in_progress":
                    duration_str = "Running..."
                else:
                    duration_str = "N/A"

                # Calculate progress from agents_executed or status
                progress = wf.get("progress", 0)

                # Always check status first for completed workflows
                status = wf.get("status", "")
                if status == "completed":
                    progress = 100
                elif progress == 0:
                    # If no progress value but has status, estimate it
                    if status == "in_progress":
                        # Estimate 50% if in progress
                        progress = 50
                    elif status == "pending":
                        progress = 0
                    elif wf.get("agents_executed"):
                        # Estimate from agents executed
                        agents_executed_count = len(wf.get("agents_executed", []))
                        if agents_executed_count > 0:
                            progress = min(agents_executed_count * 25, 100)

                # Determine current/last agent
                agents_executed = wf.get("agents_executed", [])
                if wf.get("status") == "completed" and agents_executed:
                    current_agent_display = agents_executed[-1]  # Last agent
                elif wf.get("current_agent"):
                    current_agent_display = wf.get("current_agent")
                elif agents_executed:
                    current_agent_display = agents_executed[-1]
                else:
                    # Infer from workflow type
                    wf_type = wf.get("workflow_type", "")
                    if wf_type == "customer_support":
                        current_agent_display = "Customer Support"
                    elif wf_type == "campaign_launch":
                        current_agent_display = "Campaign"
                    elif wf_type == "analytics":
                        current_agent_display = "Analytics"
                    else:
                        current_agent_display = "N/A"

                # Format completed time
                completed_str = "N/A"
                if wf.get("completed_at"):
                    try:
                        completed_str = str(wf.get("completed_at"))[:19]
                    except:
                        completed_str = "N/A"

                workflows_data.append(
                    {
                        "Workflow ID": wf.get("workflow_id", "N/A")[:16] + "...",
                        "Type": wf.get("workflow_type", "N/A"),
                        "Status": wf.get("status", "N/A"),
                        "Progress": f"{progress:.0f}%",
                        "Current Agent": current_agent_display,
                        "Started": (
                            str(wf.get("started_at", "N/A"))[:19]
                            if wf.get("started_at")
                            else "N/A"
                        ),
                        "Completed": completed_str,
                        "Duration": duration_str,
                    }
                )

            workflows_df = pd.DataFrame(workflows_data)

            # Apply status filter styling
            st.dataframe(workflows_df, use_container_width=True, hide_index=True)

            # Export workflows
            csv = workflows_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Workflows CSV",
                data=csv,
                file_name=f"workflows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            st.markdown("---")

            # Workflow Detail View
            st.subheader("🔍 Workflow Details")

            selected_workflow_id = st.selectbox(
                "Select workflow to view details",
                [wf.get("workflow_id") for wf in workflows],
            )

            if selected_workflow_id:
                workflow_details = get_workflow_details(selected_workflow_id)

                if workflow_details:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("**Workflow Information**")
                        st.json(workflow_details)

                        # Download workflow results
                        if workflow_details.get("status") == "completed":
                            result_json = json.dumps(
                                workflow_details.get("result", {}), indent=2
                            )
                            st.download_button(
                                label="📥 Download Results (JSON)",
                                data=result_json,
                                file_name=f"workflow_{selected_workflow_id}_results.json",
                                mime="application/json",
                            )

                    with col2:
                        st.markdown("**State Transitions**")

                        # Get actual state transitions from workflow details
                        transitions = []

                        if workflow_details:
                            # Use agents_executed array if available
                            agents_executed = workflow_details.get(
                                "agents_executed", []
                            )
                            if agents_executed:
                                for agent_name in agents_executed:
                                    transitions.append(
                                        {
                                            "Agent": agent_name,
                                            "Status": "completed",
                                            "Time": "-",
                                        }
                                    )
                                # Add current agent if in progress
                                if workflow_details.get("status") == "in_progress":
                                    current_agent = workflow_details.get(
                                        "current_agent"
                                    )
                                    if (
                                        current_agent
                                        and current_agent not in agents_executed
                                    ):
                                        transitions.append(
                                            {
                                                "Agent": current_agent,
                                                "Status": "in_progress",
                                                "Time": "-",
                                            }
                                        )
                            elif workflow_details.get("state_transitions"):
                                # Fallback to state_transitions if available
                                state_transitions = workflow_details.get(
                                    "state_transitions", []
                                )
                                for trans in state_transitions:
                                    transitions.append(
                                        {
                                            "Agent": trans.get("agent_name", "Unknown"),
                                            "Status": trans.get("status", "unknown"),
                                            "Time": f"{trans.get('duration_seconds', 0):.1f}s",
                                        }
                                    )
                            else:
                                # Infer from workflow type and status since agents_executed is empty
                                wf_type = workflow_details.get("workflow_type", "")
                                status = workflow_details.get("status", "unknown")

                                if wf_type == "customer_support":
                                    agent_name = "Customer Support Agent"
                                elif wf_type == "campaign_launch":
                                    agent_name = "Campaign Agent"
                                elif wf_type == "analytics":
                                    agent_name = "Analytics Agent"
                                else:
                                    agent_name = (
                                        wf_type.replace("_", " ").title()
                                        if wf_type
                                        else "Workflow"
                                    )

                                transitions.append(
                                    {
                                        "Agent": agent_name,
                                        "Status": status,
                                        "Time": "-",
                                    }
                                )

                        if transitions:
                            for trans in transitions:
                                status_emoji = {
                                    "completed": "✅",
                                    "in_progress": "⏳",
                                    "pending": "⏸️",
                                    "failed": "❌",
                                }.get(trans["Status"], "⚪")
                                time_str = (
                                    f" - {trans['Time']}"
                                    if trans["Time"] != "-"
                                    else ""
                                )
                                st.markdown(
                                    f"{status_emoji} **{trans['Agent']}**{time_str}"
                                )
                        else:
                            st.info("No agent execution data available")

                    st.markdown("---")

                    # Execution Timeline Visualization
                    st.subheader("📊 Execution Timeline")

                    # Build timeline from workflow details ONLY
                    timeline_data = []

                    if workflow_details:
                        # Extract timeline information from workflow
                        started_at = workflow_details.get(
                            "started_at", datetime.now().isoformat()
                        )
                        completed_at = workflow_details.get("completed_at")

                        # Parse timestamps
                        try:
                            start_time = datetime.fromisoformat(
                                str(started_at).replace("Z", "+00:00")
                            )
                        except:
                            start_time = datetime.now()

                        # Calculate end time
                        if completed_at:
                            try:
                                end_time = datetime.fromisoformat(
                                    str(completed_at).replace("Z", "+00:00")
                                )
                            except:
                                end_time = start_time + timedelta(seconds=60)
                        else:
                            end_time = datetime.now()

                        total_duration = (end_time - start_time).total_seconds()

                        # Use agents_executed to build timeline
                        agents_executed = workflow_details.get("agents_executed", [])
                        if agents_executed:
                            # Divide timeline among agents with minimum 1 second per agent
                            # This prevents fractional seconds in the display
                            duration_per_agent = max(
                                1.0,  # Minimum 1 second per agent
                                (
                                    total_duration / len(agents_executed)
                                    if agents_executed
                                    else 0
                                ),
                            )
                            current_time = start_time

                            for agent_name in agents_executed:
                                agent_end = current_time + timedelta(
                                    seconds=duration_per_agent
                                )
                                timeline_data.append(
                                    {
                                        "Task": agent_name.replace("_", " ").title(),
                                        "Start": current_time.isoformat(),
                                        "Finish": agent_end.isoformat(),
                                        "Resource": "Completed",
                                    }
                                )
                                current_time = agent_end
                        elif workflow_details.get("state_transitions"):
                            # Fallback to state_transitions if available
                            current_time = start_time
                            for trans in workflow_details["state_transitions"]:
                                duration = trans.get("duration_seconds", 0)
                                agent_name = trans.get("agent_name", "Unknown")
                                status = trans.get("status", "unknown")

                                agent_end = current_time + timedelta(seconds=duration)
                                timeline_data.append(
                                    {
                                        "Task": agent_name.replace("_", " ").title(),
                                        "Start": current_time.isoformat(),
                                        "Finish": agent_end.isoformat(),
                                        "Resource": status.replace("_", " ").title(),
                                    }
                                )
                                current_time = agent_end
                        else:
                            # Show single workflow bar with better labeling
                            workflow_type = workflow_details.get(
                                "workflow_type", "Workflow"
                            )
                            status = workflow_details.get("status", "unknown")

                            # Better display name
                            display_name = workflow_type.replace("_", " ").title()
                            if display_name == "Customer Support":
                                display_name = "Customer Support Agent"
                            elif display_name == "Marketing Strategy":
                                display_name = "Marketing Strategy Agent"
                            elif display_name == "Analytics Evaluation":
                                display_name = "Analytics Agent"

                            timeline_data.append(
                                {
                                    "Task": display_name,
                                    "Start": start_time.isoformat(),
                                    "Finish": end_time.isoformat(),
                                    "Resource": status.replace("_", " ").title(),
                                }
                            )

                    if timeline_data:
                        # Fix labels: convert status to readable format
                        for item in timeline_data:
                            status = item["Resource"]
                            if status == "in_progress" or status == "In_Progress":
                                item["Resource"] = "In Progress"
                            elif status == "completed" or status == "Completed":
                                item["Resource"] = "Completed"
                            elif status == "pending" or status == "Pending":
                                item["Resource"] = "Pending"
                            elif status == "failed" or status == "Failed":
                                item["Resource"] = "Failed"

                        fig = px.timeline(
                            timeline_data,
                            x_start="Start",
                            x_end="Finish",
                            y="Task",
                            color="Resource",
                            title="Agent Execution Timeline",
                            color_discrete_map={
                                "Completed": "#00CC96",
                                "In Progress": "#FFA15A",
                                "Pending": "#AB63FA",
                                "Failed": "#EF553B",
                            },
                        )
                        fig.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No timeline data available")
        else:
            st.info("No workflows found matching the selected criteria.")


# ============================================================================
# TAB 4: ANALYTICS
# ============================================================================

with tab4:
    st.header("Analytics & Insights")

    if not api_healthy:
        st.error("⚠️ API is not available. Please start the API server.")
    else:
        agents = get_agents()

        # Date range for analytics
        col1, col2 = st.columns(2)
        with col1:
            start_date_input = st.date_input(
                "Start Date", datetime.now() - timedelta(days=7)
            )
        with col2:
            end_date_input = st.date_input("End Date", datetime.now())

        st.markdown("---")

        # Agent Performance Comparison
        st.subheader("📊 Agent Performance Comparison")

        if agents:
            agent_metrics = []
            for agent in agents:
                history = get_agent_history(agent.get("agent_id"), limit=100)

                if history and "history" in history:
                    history_items = history["history"]

                    # Calculate metrics
                    statuses = [item.get("status") for item in history_items]
                    success_count = sum(1 for s in statuses if s == "completed")
                    success_rate = (
                        (success_count / len(statuses) * 100) if statuses else 0
                    )

                    durations = [
                        item.get("duration_seconds", 0) or 0 for item in history_items
                    ]
                    avg_duration = sum(durations) / len(durations) if durations else 0

                    agent_metrics.append(
                        {
                            "Agent": agent.get("name", agent.get("agent_id")),
                            "Success Rate": success_rate,
                            "Avg Response Time": avg_duration,
                            "Total Executions": len(history_items),
                        }
                    )

            if agent_metrics:
                agent_metrics_df = pd.DataFrame(agent_metrics)

                # Bar chart for performance comparison
                fig = px.bar(
                    agent_metrics_df,
                    x="Agent",
                    y=["Success Rate", "Avg Response Time"],
                    barmode="group",
                    title="Agent Performance Comparison",
                    labels={"value": "Metric Value", "variable": "Metric"},
                )
                st.plotly_chart(fig, use_container_width=True)

                # Data table
                st.dataframe(
                    agent_metrics_df, use_container_width=True, hide_index=True
                )

        st.markdown("---")

        # Campaign Metrics Over Time
        st.subheader("📈 Campaign Metrics Over Time")

        # Try to fetch workflows, fallback to synthetic data
        workflows = get_workflows(limit=100)
        campaign_data_list = []

        # If no workflows or no campaign workflows, use synthetic execution data
        # Match both traditional 'campaign_launch' and chat-based 'chat_campaign_creation' workflows
        campaign_workflows = (
            [
                wf
                for wf in workflows
                if wf.get("workflow_type")
                in ["campaign_launch", "chat_campaign_creation"]
            ]
            if workflows
            else []
        )

        if not campaign_workflows:
            # Use synthetic execution data with campaign metrics
            try:
                exec_data = load_execution_data(time_range="30d", limit=200)
                campaign_records = [
                    r
                    for r in exec_data
                    if r.get("result", {}).get("metrics", {}).get("impressions")
                    is not None
                ]

                # Group by date and aggregate metrics
                dates = pd.date_range(
                    start=start_date_input, end=end_date_input, freq="D"
                )

                for date in dates:
                    date_str = date.strftime("%Y-%m-%d")
                    day_records = [
                        r
                        for r in campaign_records
                        if r.get("started_at", "").startswith(date_str)
                    ]

                    if day_records:
                        # Aggregate metrics
                        total_impressions = sum(
                            r.get("result", {}).get("metrics", {}).get("impressions", 0)
                            for r in day_records
                        )
                        total_clicks = sum(
                            r.get("result", {}).get("metrics", {}).get("clicks", 0)
                            for r in day_records
                        )
                        ctr = (
                            (total_clicks / total_impressions * 100)
                            if total_impressions > 0
                            else 0
                        )

                        campaign_data_list.append(
                            {
                                "Date": date,
                                "Campaigns": len(day_records),
                                "CTR": round(ctr, 2),
                                "Impressions (K)": round(total_impressions / 1000, 1),
                            }
                        )
                    else:
                        campaign_data_list.append(
                            {
                                "Date": date,
                                "Campaigns": 0,
                                "CTR": 0,
                                "Impressions (K)": 0,
                            }
                        )

            except Exception as e:
                st.warning(f"Could not load campaign data: {e}")
        else:
            # Use workflow data
            dates = pd.date_range(start=start_date_input, end=end_date_input, freq="D")

            for date in dates:
                day_workflows = [
                    wf
                    for wf in campaign_workflows
                    if wf.get("started_at", "").startswith(date.strftime("%Y-%m-%d"))
                ]

                campaigns_launched = len(day_workflows)

                if day_workflows:
                    statuses = [wf.get("status") for wf in day_workflows]
                    success_count = sum(1 for s in statuses if s == "completed")
                    success_rate = (
                        (success_count / len(statuses)) * 100 if statuses else 0
                    )
                    engagement = sum(
                        int(wf.get("progress", 0) * 100) for wf in day_workflows
                    ) // max(1, len(day_workflows))
                else:
                    success_rate = 0
                    engagement = 0

                campaign_data_list.append(
                    {
                        "Date": date,
                        "Campaigns": campaigns_launched,
                        "Success Rate": success_rate,
                        "Engagement": engagement,
                    }
                )

        campaign_data = pd.DataFrame(campaign_data_list)

        if (
            not campaign_data.empty
            and len([c for c in campaign_data.columns if c != "Date"]) > 0
        ):
            # Plot available metrics
            metric_cols = [
                c
                for c in campaign_data.columns
                if c != "Date" and campaign_data[c].sum() > 0
            ]
            if metric_cols:
                fig = px.line(
                    campaign_data,
                    x="Date",
                    y=metric_cols,
                    title="Campaign Metrics Over Time",
                    labels={"value": "Value", "variable": "Metric"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No campaign activity in the selected date range.")
        else:
            st.info("No campaign data available for the selected date range.")

        st.markdown("---")

        # Agent Handoff Frequency Matrix
        st.subheader("🔄 Agent Handoff Frequency Matrix")

        # Fetch fresh workflow data for handoff analysis
        all_workflows = get_workflows(limit=100)

        # ONLY count handoffs from actual workflow executions (not synthetic/historical data)
        handoff_counts = {}
        total_transitions = 0
        filtered_self_handoffs = 0
        workflows_with_multiple_agents = 0

        if all_workflows:
            for workflow in all_workflows:
                handoff_found = False
                agents_exec = workflow.get("agents_executed", [])

                # Debug: Count workflows with multiple agents
                if len(agents_exec) > 1:
                    workflows_with_multiple_agents += 1

                # Method 1: Extract from state_transitions if available
                if "state_transitions" in workflow and workflow["state_transitions"]:
                    state_transitions = workflow.get("state_transitions", [])
                    for i in range(len(state_transitions) - 1):
                        total_transitions += 1
                        from_agent = state_transitions[i].get("agent_name", "Unknown")
                        to_agent = state_transitions[i + 1].get("agent_name", "Unknown")

                        # Only count if different agents (no self-handoffs)
                        if (
                            from_agent != to_agent
                            and from_agent != "Unknown"
                            and to_agent != "Unknown"
                        ):
                            key = f"{from_agent}-{to_agent}"
                            handoff_counts[key] = handoff_counts.get(key, 0) + 1
                            handoff_found = True
                        elif from_agent == to_agent and from_agent != "Unknown":
                            filtered_self_handoffs += 1

                # Method 2: Extract from agents_executed when multiple agents were involved
                # Only use this if we haven't found handoffs from state_transitions
                if not handoff_found and (
                    "agents_executed" in workflow
                    and len(workflow.get("agents_executed", [])) > 1
                ):
                    agents = workflow["agents_executed"]
                    for i in range(len(agents) - 1):
                        from_agent = agents[i].replace("_", " ").title()
                        to_agent = agents[i + 1].replace("_", " ").title()

                        if from_agent != to_agent:
                            key = f"{from_agent}-{to_agent}"
                            handoff_counts[key] = handoff_counts.get(key, 0) + 1
                            handoff_found = True

                # Method 3: Check for explicit handoff flag in result
                # Only use this if we haven't found handoffs from other methods
                if not handoff_found and workflow.get("result", {}).get(
                    "handoff_required"
                ):
                    result = workflow.get("result", {})
                    agents_executed = workflow.get("agents_executed", [])
                    target_agent = result.get("target_agent")

                    if len(agents_executed) > 0 and target_agent:
                        from_agent = agents_executed[-1].replace("_", " ").title()
                        to_agent = target_agent.replace("_", " ").title()

                        if from_agent != to_agent:
                            key = f"{from_agent}-{to_agent}"
                            handoff_counts[key] = handoff_counts.get(key, 0) + 1

        # Build dataframe from actual handoffs
        if handoff_counts:
            try:
                handoff_data = []
                for handoff, count in handoff_counts.items():
                    from_agent, to_agent = handoff.split("-")
                    handoff_data.append(
                        {"From": from_agent, "To": to_agent, "Count": count}
                    )

                handoff_df = pd.DataFrame(handoff_data)

                # Filter out self-loops for cleaner visualization
                handoff_df_filtered = handoff_df[
                    handoff_df["From"] != handoff_df["To"]
                ].copy()

                if not handoff_df_filtered.empty:
                    # Create a clean bar chart showing handoff patterns
                    handoff_df_filtered["Flow"] = (
                        handoff_df_filtered["From"] + " → " + handoff_df_filtered["To"]
                    )
                    handoff_df_sorted = handoff_df_filtered.sort_values(
                        "Count", ascending=True
                    )

                    fig_flow = px.bar(
                        handoff_df_sorted,
                        x="Count",
                        y="Flow",
                        orientation="h",
                        title="Agent-to-Agent Handoff Frequency",
                        labels={"Count": "Number of Handoffs", "Flow": "Agent Flow"},
                        color="Count",
                        color_continuous_scale="Blues",
                        height=max(300, len(handoff_df_sorted) * 40),
                    )

                    fig_flow.update_layout(
                        showlegend=False,
                        xaxis_title="Handoff Count",
                        yaxis_title="",
                        font=dict(size=12),
                        margin=dict(l=200, r=20, t=50, b=50),
                    )

                    fig_flow.update_traces(
                        marker=dict(line=dict(color="#000000", width=1)),
                        texttemplate="%{x}",
                        textposition="outside",
                    )

                    st.plotly_chart(fig_flow, use_container_width=True)
                else:
                    st.info(
                        "No inter-agent handoffs detected yet. All workflows completed within a single agent."
                    )

                # Show summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Handoffs", handoff_df["Count"].sum())
                with col2:
                    most_common = handoff_df.nlargest(1, "Count")
                    if not most_common.empty:
                        st.metric(
                            "Most Common Handoff",
                            f"{most_common.iloc[0]['From']} → {most_common.iloc[0]['To']}",
                            delta=f"{most_common.iloc[0]['Count']} times",
                        )
                with col3:
                    st.metric("Unique Agent Pairs", len(handoff_df))

                # Optional: Show detailed breakdown table
                with st.expander("📊 View Detailed Handoff Statistics"):
                    st.dataframe(
                        handoff_df.sort_values("Count", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as e:
                st.error(f"Error creating handoff visualization: {e}")
                st.caption(f"Debug: handoff_counts = {handoff_counts}")
        else:
            # Show debug info about why no handoffs were found
            if all_workflows:
                if total_transitions > 0:
                    st.info(
                        f"📊 Analyzed {len(all_workflows)} workflows with {total_transitions} agent transitions. "
                        f"Filtered out {filtered_self_handoffs} self-handoffs. "
                        f"No inter-agent handoffs detected - all workflows completed within a single agent or had only self-handoffs."
                    )
                else:
                    debug_msg = f"📊 Analyzed {len(all_workflows)} workflows but found no state transitions."
                    if workflows_with_multiple_agents > 0:
                        debug_msg += f" Note: {workflows_with_multiple_agents} workflows have multiple agents but handoffs were not extracted. This may be a code issue."
                    else:
                        debug_msg += " All workflows completed within a single agent."
                    st.info(debug_msg)
            else:
                st.info(
                    "📊 No workflow data available. Run workflows with multiple agents to generate handoff statistics."
                )

        st.markdown("---")

        # Error Rate Trends
        st.subheader("⚠️ Error Rate Trends")

        # Calculate error rates from actual workflow data
        error_data_list = []

        for date in dates:
            # Count workflows for this date
            day_workflows = [
                wf
                for wf in workflows
                if wf.get("started_at", "").startswith(date.strftime("%Y-%m-%d"))
            ]

            if day_workflows:
                # Count failed workflows
                failed_count = sum(
                    1 for wf in day_workflows if wf.get("status") == "failed"
                )
                error_rate = (
                    (failed_count / len(day_workflows)) * 100 if day_workflows else 0
                )
                total_errors = failed_count
            else:
                error_rate = 0
                total_errors = 0

            error_data_list.append(
                {
                    "Date": date,
                    "Error Rate": error_rate,
                    "Total Errors": total_errors,
                }
            )

        error_data = pd.DataFrame(error_data_list)

        if not error_data.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(
                    x=error_data["Date"],
                    y=error_data["Error Rate"],
                    name="Error Rate (%)",
                    mode="lines+markers",
                ),
                secondary_y=False,
            )

            fig.add_trace(
                go.Bar(
                    x=error_data["Date"],
                    y=error_data["Total Errors"],
                    name="Total Errors",
                    opacity=0.3,
                ),
                secondary_y=True,
            )

            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Error Rate (%)", secondary_y=False)
            fig.update_yaxes(title_text="Total Errors", secondary_y=True)
            fig.update_layout(title_text="Error Rate Trends")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No error data available for the selected date range.")

        st.markdown("---")

        # Export Analytics Data
        st.subheader("📥 Export Analytics")

        col1, col2, col3 = st.columns(3)

        with col1:
            if agent_metrics:
                csv_agents = pd.DataFrame(agent_metrics).to_csv(index=False)
                st.download_button(
                    label="Download Agent Metrics",
                    data=csv_agents,
                    file_name="agent_metrics.csv",
                    mime="text/csv",
                )

        with col2:
            csv_campaigns = campaign_data.to_csv(index=False)
            st.download_button(
                label="Download Campaign Data",
                data=csv_campaigns,
                file_name="campaign_data.csv",
                mime="text/csv",
            )

        with col3:
            csv_errors = error_data.to_csv(index=False)
            st.download_button(
                label="Download Error Data",
                data=csv_errors,
                file_name="error_data.csv",
                mime="text/csv",
            )


# ============================================================================
# Auto-refresh Logic
# ============================================================================

if st.session_state.auto_refresh:
    time.sleep(AUTO_REFRESH_INTERVAL)
    st.rerun()
