# UI Components - Enterprise Marketing AI Agents

This directory contains two main UI components for interacting with the multi-agent system:

1. **Streamlit Dashboard** (`streamlit_app.py`) - System monitoring and analytics
2. **Gradio Conversational UI** (`gradio_app.py`) - Chat-based agent interaction

---

## Streamlit Dashboard - System Monitoring

Interactive web dashboard for monitoring and managing the Enterprise Marketing AI Agents system.

**Technology Stack:** Streamlit 1.30+, Plotly, Pandas, Requests
**Port:** 8501 (default)
**Launch Script:** `./start_dashboard.sh`

## Features

### 📈 Overview Tab
- **System Health Indicators**: Real-time API and agent status
- **Key Metrics Cards**: Total executions, success rate, average response time, active agents
- **Real-time Activity Log**: Last 20 events across all agents
- **System Resources**: CPU, memory, and API request monitoring

### 🤖 Agents Tab
- **Agent Cards**: Visual display of all agents with status indicators
- **Execution History**: Detailed history table for each agent
- **Performance Metrics**: Success rates, average durations, execution trends
- **Tool Usage Statistics**: Track which tools agents use most
- **Interactive Testing**: Execute agents directly from the dashboard

### 🔄 Workflows Tab
- **Workflow List**: Recent workflows with filtering and search
- **Detail View**: Comprehensive workflow information and state
- **Execution Timeline**: Visual Gantt-style timeline of agent handoffs
- **State Transitions**: Track workflow progress through agents
- **Download Results**: Export workflow results as JSON

### 📊 Analytics Tab
- **Agent Performance Comparison**: Bar charts comparing all agents
- **Campaign Metrics**: Time series of campaign activity
- **Handoff Frequency Matrix**: Heatmap showing agent collaboration patterns
- **Error Rate Trends**: Track and visualize system errors over time
- **Data Export**: Download analytics as CSV for further analysis

## Installation

The dashboard is included with the project. Ensure you have Streamlit installed:

```bash
pip install -r requirements.txt
```

## Usage

### Start the Dashboard

**Method 1: Using the launch script (Recommended)**
```bash
./start_dashboard.sh
```

**Method 2: Direct Streamlit command**
```bash
cd ui
streamlit run streamlit_app.py
```

The dashboard will be available at: http://localhost:8501

### Prerequisites

**The API server must be running** for the dashboard to function:
```bash
./start_all.sh  # Starts API, Streamlit, and Gradio
# OR just API:
python api/main.py
```

API should be available at: http://localhost:8000

## Configuration

### Streamlit Settings

Configuration is available in `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "localhost"

[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
```

### Dashboard Settings

Configuration is defined in `ui/streamlit_app.py`:

```python
API_BASE_URL = "http://127.0.0.1:8000/api/v1"
AUTO_REFRESH_INTERVAL = 5  # seconds
MAX_RETRIES = 3
REQUEST_TIMEOUT = 10
```

For production, configure via environment variables or settings.py.

## Dashboard Features

### Auto-Refresh
- Toggle auto-refresh in the sidebar
- Refreshes every 5 seconds when enabled
- Manual refresh button always available

### Filtering & Search
- **Agent Filter**: Focus on specific agents
- **Time Range**: Last hour, 6 hours, 24 hours, 7 days, 30 days
- **Status Filter**: Filter by workflow/execution status
- **Search**: Find specific workflows or executions

### Data Export
- **CSV Export**: Download tables as CSV files
- **JSON Export**: Export workflow results and configurations
- **Analytics Data**: Export charts and metrics for reporting

### Interactive Testing
Execute agents directly from the dashboard:
1. Navigate to Agents tab
2. Select an agent
3. Scroll to "Test Agent Execution"
4. Enter task data in JSON format
5. Click "Execute Agent"
6. View results in real-time

## API Integration

The dashboard connects to these API endpoints:

- `GET /health` - System health check
- `GET /metrics` - System metrics
- `GET /api/v1/agents` - List all agents
- `GET /api/v1/agents/{agent_id}` - Agent details
- `GET /api/v1/agents/{agent_id}/history` - Agent execution history
- `POST /api/v1/workflows/*` - Execute workflows (campaign-launch, customer-support, analytics, feedback-learning)
- `GET /api/v1/workflows/{workflow_id}/status` - Workflow status
- `GET /api/v1/workflows/{workflow_id}/results` - Workflow results

## Troubleshooting

### Dashboard won't start
```bash
# Check if Streamlit is installed
streamlit --version

# Reinstall if needed
pip install streamlit --upgrade
```

### API Connection Failed
```bash
# Verify API is running
curl http://localhost:8000/health

# Start all services if not running
./start_all.sh

# OR start API only:
python api/main.py
```

### Port Already in Use
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Or use a different port
streamlit run streamlit_app.py --server.port=8502
```

### Missing Data in Dashboard
- Ensure API is running and healthy
- Check API logs for errors
- Verify agents have execution history
- Try manual refresh or restart dashboard

## Screenshots

### Overview Tab
Real-time system monitoring with health indicators and activity logs.

### Agents Tab
Manage and test individual agents with detailed performance metrics.

### Workflows Tab
Track multi-agent workflows from start to finish with visual timelines.

### Analytics Tab
Comprehensive analytics with interactive charts and data export.

## Development

### Adding New Visualizations

Add new charts to the Analytics tab:

```python
import plotly.express as px

# Your data
df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

# Create chart
fig = px.line(df, x="x", y="y", title="My Chart")

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
```

### Adding New Metrics

Add custom metrics to the Overview tab:

```python
# In the Overview tab section
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("My Metric", "123", delta="5")
```

### Custom API Endpoints

If you add new API endpoints, update the `api_request` function:

```python
def get_custom_data():
    """Get data from custom endpoint."""
    return api_request("/custom/endpoint")
```

---

## Gradio Conversational UI

### Overview

Interactive chat interface for natural language interaction with AI agents.

**Technology Stack:** Gradio 4.0+, httpx (async), Python 3.10+
**Port:** 7860 (default)
**Launch Script:** `./start_gradio.sh`
**URL:** http://localhost:7860
**Features:** Async workflow execution, intelligent intent detection, conversation export

### Features

#### 🎯 Core Functionality
- **Real-time Chat**: Natural language interaction with AI agents
- **Auto-routing**: Intelligent routing to appropriate agent based on query intent
- **Agent Handoffs**: Visualize transitions between specialized agents
- **Streaming Responses**: Token-by-token response streaming (when supported)
- **Intermediate Results**: Display KB search results, metrics, and agent status
- **Conversation History**: Maintain context across multiple interactions
- **Export Conversations**: Save conversation history as JSON

#### 🤖 Supported Agents
1. **Marketing Strategy Agent** - Campaign planning, content strategy
2. **Customer Support Agent** - Knowledge base queries, documentation search
3. **Analytics Agent** - Performance analysis, metrics, insights
4. **Feedback Learning Agent** - Improvements, optimizations, learning

#### 💬 Example Queries
```
✓ "Plan a campaign for our new payment API product"
✓ "How does Stripe handle subscription billing?"
✓ "Analyze our campaign performance from last month"
✓ "What improvements can we make to agent performance?"
✓ "Create a content strategy for developer outreach"
```

### Quick Start

```bash
# 1. Start all services (recommended)
./start_all.sh

# OR start Gradio separately (requires API running):
./start_gradio.sh

# 2. Open browser to http://localhost:7860
```

### Interface Components

**Main Chat Area**:
- Chatbot display with conversation history
- Multi-line message input
- Send, Clear, and Export buttons

**Sidebar**:
- Agent selector (Auto/Manual routing)
- Real-time status display
- Workflow metrics panel
- Example queries

### Configuration

Configuration is loaded from `config/settings.py` and defined in `ui/gradio_app.py`:

```python
API_BASE_URL = f"http://localhost:8000/api/v1"
TIMEOUT = 300.0  # seconds

# Conversation export
EXPORT_DIR = "data/conversations"
```

The Gradio app auto-detects the API port from settings.

### Troubleshooting

**FastAPI Not Running**:
```bash
# Start all services
./start_all.sh

# OR start API only:
python api/main.py
```

**Port 7860 In Use**:
```bash
# Kill existing process
lsof -ti:7860 | xargs kill -9
```

**Timeout Issues**:
- Increase `max_poll_attempts` in config
- Check FastAPI workflow timeouts
- Verify agent initialization

### API Integration

Gradio uses these endpoints:
```
POST /api/v1/workflows/campaign-launch
POST /api/v1/workflows/customer-support
POST /api/v1/workflows/analytics
POST /api/v1/workflows/feedback-learning
GET  /api/v1/workflows/{id}/status
GET  /api/v1/workflows/{id}/results
```

### Export Format

Conversations exported to `data/conversations/`:
```json
{
  "conversation_id": "uuid",
  "timestamp": "2026-01-12T10:30:00",
  "messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ]
}
```

---

## Comparison: Streamlit vs Gradio

| Feature | Streamlit | Gradio |
|---------|-----------|--------|
| **Purpose** | System monitoring & analytics | Conversational AI interaction |
| **Port** | 8501 | 7860 |
| **Use Case** | Admin/DevOps dashboard | End-user chat interface |
| **Interactivity** | Dashboards, charts, tables | Chat, conversational flow |
| **Best For** | Monitoring, metrics, debugging | Natural language queries |

### When to Use Each

**Use Streamlit** when you need to:
- Monitor system health and performance
- View agent execution history
- Analyze workflow patterns
- Debug agent behaviors
- View detailed logs and metrics

**Use Gradio** when you need to:
- Have natural conversations with agents
- Test agent responses interactively
- Demonstrate system capabilities
- Provide end-user interface
- Export conversation histories

---

## Performance Tips

1. **Cache API Calls**: Use `st.cache_data` (Streamlit) or state management (Gradio)
2. **Limit Data**: Use pagination and limits for large datasets
3. **Optimize Queries**: Filter data on the API side when possible
4. **Debounce Auto-Refresh**: Adjust refresh interval based on load

## Security

- Both dashboards run locally by default
- No authentication required for local development
- For production deployment, enable authentication in config
- Use HTTPS for production deployments
- Configure CORS appropriately in API config

## Contributing

When adding UI features:

1. Follow framework best practices (Streamlit/Gradio)
2. Use session state for caching
3. Add error handling for API calls
4. Include loading indicators
5. Update this documentation

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Project API Reference](../docs/api_reference.md)
