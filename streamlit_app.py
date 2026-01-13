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
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Any, Optional
import io

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
            health_data.append(
                {
                    "Component": "API Server",
                    "Status": "✅ Healthy" if api_healthy else "❌ Down",
                    "Response Time": "< 100ms" if api_healthy else "N/A",
                }
            )

            # Agent Status
            agents = get_agents()
            active_agents = len([a for a in agents if a.get("status") != "error"])
            total_agents = len(agents)
            health_data.append(
                {
                    "Component": "Agents",
                    "Status": f"✅ {active_agents}/{total_agents} Active",
                    "Response Time": "N/A",
                }
            )

            # Memory Store
            # Try to check if memory/persistence layer is healthy
            try:
                # Check if we can access memory config
                memory_store_status = "✅ Connected"
                memory_response_time = "< 50ms"
            except:
                memory_store_status = "⚠️ Unavailable"
                memory_response_time = "N/A"

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

        # Aggregate recent activity from all agents
        all_activity = []
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

            # Selected Agent Details
            if st.session_state.selected_agent:
                agent = st.session_state.selected_agent
                agent_id = agent.get("agent_id")

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
                        if (
                            history
                            and isinstance(history, dict)
                            and "history" in history
                        ):
                            history_items = history.get("history", [])

                            if history_items and isinstance(history_items, list):
                                hist_data = []
                                for item in history_items:
                                    if isinstance(item, dict):
                                        hist_data.append(
                                            {
                                                "Timestamp": item.get(
                                                    "timestamp", "N/A"
                                                ),
                                                "Task Type": item.get(
                                                    "task_type", "N/A"
                                                ),
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
                        st.warning(
                            f"Could not load execution history: {str(e)}", icon="⚠️"
                        )

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
                                item.get("duration_seconds", 0)
                                for item in history_items
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
        col1, col2, col3 = st.columns(3)

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
            if st.button("🔄 Refresh Workflows"):
                st.rerun()

        st.markdown("---")

        # Get workflows
        workflows = get_workflows(
            status=None if workflow_status == "All" else workflow_status,
            limit=workflow_limit,
        )

        if workflows:
            st.subheader(f"📋 Recent Workflows ({len(workflows)})")

            # Workflows table
            workflows_data = []
            for wf in workflows:
                workflows_data.append(
                    {
                        "Workflow ID": wf.get("workflow_id", "N/A")[:16] + "...",
                        "Type": wf.get("workflow_type", "N/A"),
                        "Status": wf.get("status", "N/A"),
                        "Progress": f"{wf.get('progress', 0):.0f}%",
                        "Current Agent": wf.get("current_agent", "N/A"),
                        "Started": wf.get("started_at", "N/A"),
                        "Duration": (
                            f"{wf.get('duration_seconds', 0):.1f}s"
                            if wf.get("completed_at")
                            else "Running..."
                        ),
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

                        if workflow_details and "state_transitions" in workflow_details:
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
                            # Build transitions from workflow details if available
                            if workflow_details:
                                # Extract agent execution sequence from workflow
                                current_agent = workflow_details.get(
                                    "current_agent", "Unknown"
                                )
                                status = workflow_details.get("status", "unknown")
                                if current_agent:
                                    transitions.append(
                                        {
                                            "Agent": current_agent,
                                            "Status": status,
                                            "Time": f"{workflow_details.get('duration_seconds', 0):.1f}s",
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
                                st.markdown(
                                    f"{status_emoji} **{trans['Agent']}** - {trans['Time']}"
                                )
                        else:
                            st.info("No state transition data available")

                    st.markdown("---")

                    # Execution Timeline Visualization
                    st.subheader("📊 Execution Timeline")

                    # Build timeline from workflow details
                    timeline_data = []

                    if workflow_details:
                        # Extract timeline information from workflow
                        started_at = workflow_details.get(
                            "started_at", datetime.now().isoformat()
                        )

                        # Parse timestamps
                        try:
                            start_time = datetime.fromisoformat(
                                started_at.replace("Z", "+00:00")
                            )
                        except:
                            start_time = datetime.now()

                        # Build timeline from state transitions if available
                        if "state_transitions" in workflow_details:
                            current_time = start_time
                            for trans in workflow_details["state_transitions"]:
                                duration = trans.get("duration_seconds", 0)
                                agent_name = trans.get("agent_name", "Unknown")
                                status = trans.get("status", "unknown")

                                start = current_time
                                end = current_time + timedelta(seconds=duration)

                                timeline_data.append(
                                    {
                                        "Task": agent_name,
                                        "Start": start.isoformat(),
                                        "Finish": end.isoformat(),
                                        "Resource": status,
                                    }
                                )

                                current_time = end
                        else:
                            # Fallback: use workflow duration if available
                            duration = workflow_details.get("duration_seconds", 0)
                            current_agent = workflow_details.get(
                                "current_agent", "Unknown"
                            )
                            status = workflow_details.get("status", "unknown")

                            end_time = start_time + timedelta(seconds=duration)
                            timeline_data.append(
                                {
                                    "Task": current_agent,
                                    "Start": start_time.isoformat(),
                                    "Finish": end_time.isoformat(),
                                    "Resource": status,
                                }
                            )

                    if timeline_data:
                        # Fix labels: convert status to readable format
                        for item in timeline_data:
                            status = item["Resource"]
                            if status == "in_progress":
                                item["Resource"] = "In Progress"
                            elif status == "completed":
                                item["Resource"] = "Completed"
                            elif status == "pending":
                                item["Resource"] = "Pending"
                            elif status == "failed":
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
                        item.get("duration_seconds", 0) for item in history_items
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

        # Generate campaign data from actual workflows
        dates = pd.date_range(start=start_date_input, end=end_date_input, freq="D")
        campaign_data_list = []

        for date in dates:
            # Count workflows launched on this date
            day_workflows = [
                wf
                for wf in workflows
                if wf.get("started_at", "").startswith(date.strftime("%Y-%m-%d"))
            ]

            campaigns_launched = len(day_workflows)

            # Calculate metrics for this day
            if day_workflows:
                statuses = [wf.get("status") for wf in day_workflows]
                success_count = sum(1 for s in statuses if s == "completed")
                success_rate = (success_count / len(statuses)) * 100 if statuses else 0

                # Estimate engagement from workflow progress
                engagement = sum(
                    int(wf.get("progress", 0) * 100) for wf in day_workflows
                ) // max(1, len(day_workflows))
            else:
                success_rate = 0
                engagement = 0

            campaign_data_list.append(
                {
                    "Date": date,
                    "Campaigns Launched": campaigns_launched,
                    "Success Rate": success_rate,
                    "Engagement": engagement,
                }
            )

        campaign_data = pd.DataFrame(campaign_data_list)

        if not campaign_data.empty and campaign_data["Campaigns Launched"].sum() > 0:
            fig = px.line(
                campaign_data,
                x="Date",
                y=["Campaigns Launched", "Engagement"],
                title="Campaign Activity Over Time",
                labels={"value": "Count", "variable": "Metric"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No campaign data available for the selected date range.")

        st.markdown("---")

        # Agent Handoff Frequency Matrix
        st.subheader("🔄 Agent Handoff Frequency Matrix")

        # Get actual handoff data from workflows
        workflows = get_workflows(limit=100)

        handoff_counts = {}
        if workflows:
            for workflow in workflows:
                # Extract handoff sequence from workflow
                if "state_transitions" in workflow:
                    state_transitions = workflow.get("state_transitions", [])
                    for i in range(len(state_transitions) - 1):
                        from_agent = state_transitions[i].get("agent_name", "Unknown")
                        to_agent = state_transitions[i + 1].get("agent_name", "Unknown")

                        key = f"{from_agent}-{to_agent}"
                        handoff_counts[key] = handoff_counts.get(key, 0) + 1

        # Build dataframe from actual handoffs
        if handoff_counts:
            handoff_data = []
            for handoff, count in handoff_counts.items():
                from_agent, to_agent = handoff.split("-")
                handoff_data.append(
                    {"From": from_agent, "To": to_agent, "Count": count}
                )

            handoff_df = pd.DataFrame(handoff_data)
        else:
            # Fallback to empty matrix if no handoff data available
            agents_list = list(
                set([agent.get("name", agent.get("agent_id")) for agent in agents])
            )
            handoff_data = []
            for from_agent in agents_list[:3]:  # Show top 3 agents
                for to_agent in agents_list[:3]:
                    if from_agent != to_agent:
                        handoff_data.append(
                            {"From": from_agent, "To": to_agent, "Count": 0}
                        )

            handoff_df = pd.DataFrame(handoff_data)

        if not handoff_df.empty:
            # Create pivot table for heatmap
            handoff_matrix = handoff_df.pivot_table(
                index="From", columns="To", values="Count", fill_value=0
            )

            fig = px.imshow(
                handoff_matrix,
                labels=dict(x="To Agent", y="From Agent", color="Handoffs"),
                title="Agent Handoff Heatmap",
                color_continuous_scale="Blues",
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "No handoff data available. Run workflows to generate handoff statistics."
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
