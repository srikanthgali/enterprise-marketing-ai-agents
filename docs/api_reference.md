# Enterprise Marketing AI Agents - API Reference

## 📋 Overview

**Base URL:** `http://localhost:8000`
**API Version:** 1.0.0
**Protocol:** REST (FastAPI)

Interactive API documentation available at:
- **Swagger UI:** `http://localhost:8000/api/docs`
- **ReDoc:** `http://localhost:8000/api/redoc`
- **OpenAPI Spec:** `http://localhost:8000/api/openapi.json`

---

## 🔐 Authentication

Currently, the API does not require authentication. In production, implement:
- API key authentication
- JWT tokens for user sessions
- Rate limiting per client

---

## 📊 Health & Monitoring

### GET `/health`

Check system health status.

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "timestamp": "2026-01-15T10:30:00",
  "version": "1.0.0",
  "components": {
    "agents": "healthy",
    "memory": "healthy",
    "redis": "disabled"
  },
  "uptime_seconds": 3600.5
}
```

**Status Values:** `healthy`, `degraded`, `unhealthy`

---

### GET `/metrics`

Get system performance metrics.

**Response:** `200 OK`
```json
{
  "timestamp": "2026-01-15T10:30:00",
  "system": {
    "cpu_percent": 25.5,
    "memory_percent": 45.2,
    "memory_used_mb": 2048.5,
    "memory_available_mb": 2500.0
  },
  "agents": {
    "marketing_strategy": {
      "total_executions": 150,
      "avg_duration": 3.2,
      "success_rate": 0.95
    },
    "customer_support": {...},
    "analytics_evaluation": {...},
    "feedback_learning": {...}
  },
  "workflows": {
    "campaign_launch": {
      "total": 25,
      "completed": 23,
      "failed": 2,
      "avg_duration": 18.5
    }
  }
}
```

---

## 💬 Chat & Natural Language Interface

### POST `/api/v1/chat`

**NEW: Unified natural language endpoint with LLM-driven intent classification.**

Send raw natural language messages and receive intelligent, context-aware responses. The system automatically:
- Classifies user intent using LLM (GPT-4o-mini)
- Extracts relevant entities (campaign names, budgets, dates, etc.)
- Routes to appropriate agent
- Handles multi-agent handoffs
- Returns structured responses with metadata

**Request Body:**
```json
{
  "message": "Create a Q2 marketing campaign for our payment API with a $50k budget",
  "session_id": "user_123_session",
  "metadata": {
    "user_id": "user_123",
    "channel": "web"
  }
}
```

**Response:** `200 OK`
```json
{
  "response": "I've created a comprehensive Q2 marketing campaign strategy...",
  "intent": "campaign_creation",
  "confidence": 0.92,
  "entities": {
    "campaign_name": "Q2 Payment API Campaign",
    "budget": 50000,
    "quarter": "Q2"
  },
  "agents_executed": ["marketing_strategy"],
  "handoffs": [],
  "session_id": "user_123_session",
  "timestamp": "2026-01-17T10:30:00"
}
```

**Example Use Cases:**

1. **Customer Support Query:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I handle refunds for subscription payments?",
    "session_id": "support_session_1"
  }'
```

2. **Analytics Request:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me conversion rates for last month",
    "session_id": "analytics_session_1"
  }'
```

3. **Complex Multi-Agent Query:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Customer satisfaction dropped 15%. Analyze why and suggest improvements.",
    "session_id": "complex_session_1"
  }'
```

**Benefits:**
- ✅ 90%+ intent classification accuracy (vs ~60% with keywords)
- ✅ Handles typos, synonyms, complex phrasing
- ✅ Automatic entity extraction
- ✅ Confidence scoring for routing decisions
- ✅ Session management for conversation context
- ✅ Single endpoint for all agent interactions

---

### POST `/api/v1/chat/stream`

**Streaming variant of the chat endpoint for real-time responses.**

**Request Body:** Same as `/api/v1/chat`

**Response:** Server-Sent Events (SSE) stream

```
data: {"type": "classification", "intent": "campaign_creation", "confidence": 0.92}

data: {"type": "agent_start", "agent": "marketing_strategy"}

data: {"type": "message", "content": "I've analyzed your requirements..."}

data: {"type": "agent_complete", "agent": "marketing_strategy"}

data: {"type": "done", "session_id": "user_123_session"}
```

---

## 🤖 Agent Management

### GET `/api/v1/agents`

List all available agents.

**Response:** `200 OK`
```json
[
  {
    "agent_id": "marketing_strategy",
    "name": "Marketing Strategy Agent",
    "status": "idle",
    "capabilities": ["market_research", "campaign_planning", "audience_segmentation"],
    "active_tasks": 0,
    "total_executions": 150,
    "avg_execution_time": 3.2
  },
  {...}
]
```

---

### GET `/api/v1/agents/{agent_id}`

Get detailed information about a specific agent.

**Path Parameters:**
- `agent_id` (string, required): Agent identifier

**Response:** `200 OK`
```json
{
  "agent_id": "marketing_strategy",
  "name": "Marketing Strategy Agent",
  "status": "idle",
  "capabilities": ["market_research", "campaign_planning", "budget_allocation"],
  "active_tasks": 0,
  "total_executions": 150,
  "avg_execution_time": 3.2
}
```

**Error Response:** `404 Not Found`
```json
{
  "detail": "Agent 'invalid_agent' not found"
}
```

---

### GET `/api/v1/agents/{agent_id}/history`

Get execution history for an agent.

**Path Parameters:**
- `agent_id` (string, required): Agent identifier

**Query Parameters:**
- `limit` (integer, optional, default: 50): Max history entries

**Response:** `200 OK`
```json
{
  "agent_id": "marketing_strategy",
  "total_executions": 150,
  "history": [
    {
      "execution_id": "exec_abc123",
      "timestamp": "2026-01-15T10:25:00",
      "task_type": "campaign_planning",
      "status": "completed",
      "duration_seconds": 3.5,
      "summary": "Created Q2 campaign strategy"
    },
    {...}
  ]
}
```

---

## 🔄 Workflow Execution

### POST `/api/v1/workflows/campaign-launch`

Launch a campaign creation workflow.

**Request Body:**
```json
{
  "campaign_name": "Q2 Product Launch",
  "objectives": ["awareness", "leads"],
  "target_audience": "B2B SaaS companies",
  "budget": 75000,
  "duration_weeks": 12,
  "channels": ["email", "social", "content"],
  "additional_context": {
    "product_type": "API Platform",
    "key_features": ["Security", "Scalability"]
  }
}
```

**Valid Objectives:** `awareness`, `leads`, `conversions`, `retention`, `engagement`

**Response:** `202 Accepted`
```json
{
  "workflow_id": "wf_abc123def456",
  "status": "pending",
  "message": "Workflow initiated successfully",
  "workflow_type": "campaign_launch",
  "estimated_duration": 600
}
```

---

### POST `/api/v1/workflows/customer-support`

Execute customer support workflow.

**Request Body:**
```json
{
  "query": "How does Stripe handle subscription refunds?",
  "customer_id": "cust_123",
  "priority": "high",
  "context": {
    "subscription_id": "sub_456",
    "issue_type": "billing"
  }
}
```

**Valid Priorities:** `low`, `medium`, `high`, `critical`

**Response:** `202 Accepted`
```json
{
  "workflow_id": "wf_support789",
  "status": "pending",
  "message": "Support workflow initiated",
  "workflow_type": "customer_support",
  "estimated_duration": 300
}
```

---

### POST `/api/v1/workflows/analytics`

Execute analytics evaluation workflow.

**Request Body:**
```json
{
  "analysis_type": "campaign_performance",
  "time_range": "last_30_days",
  "metrics": ["ctr", "conversions", "roi", "cac"],
  "filters": {
    "campaign_id": "camp_123",
    "channel": "email"
  },
  "include_forecast": true
}
```

**Valid Analysis Types:** `campaign_performance`, `agent_performance`, `trend_analysis`, `ab_test`

**Response:** `202 Accepted`
```json
{
  "workflow_id": "wf_analytics123",
  "status": "pending",
  "message": "Analytics workflow initiated",
  "workflow_type": "analytics",
  "estimated_duration": 300
}
```

---

### POST `/api/v1/workflows/feedback-learning`

Execute feedback and learning workflow.

**Request Body:**
```json
{
  "feedback_type": "agent_optimization",
  "agent_id": "marketing_strategy",
  "time_range": "last_7_days",
  "analysis_scope": ["success_patterns", "failure_modes", "prompt_optimization"]
}
```

**Valid Feedback Types:** `agent_optimization`, `workflow_optimization`, `system_wide_learning`

**Response:** `202 Accepted`
```json
{
  "workflow_id": "wf_learning456",
  "status": "pending",
  "message": "Learning workflow initiated",
  "workflow_type": "feedback_learning",
  "estimated_duration": 300
}
```

---

### GET `/api/v1/workflows/{workflow_id}/status`

Get workflow execution status.

**Path Parameters:**
- `workflow_id` (string, required): Workflow identifier

**Response:** `200 OK`
```json
{
  "workflow_id": "wf_abc123def456",
  "status": "in_progress",
  "workflow_type": "campaign_launch",
  "started_at": "2026-01-15T10:20:00",
  "progress": 0.6,
  "current_agent": "analytics_evaluation",
  "agents_executed": ["orchestrator", "marketing_strategy"],
  "state_transitions": [
    {
      "agent_name": "orchestrator",
      "timestamp": "2026-01-15T10:20:00",
      "status": "completed"
    },
    {
      "agent_name": "marketing_strategy",
      "timestamp": "2026-01-15T10:20:15",
      "status": "completed"
    }
  ],
  "error": null
}
```

**Status Values:** `pending`, `in_progress`, `completed`, `failed`

---

### GET `/api/v1/workflows/{workflow_id}/results`

Get workflow execution results (blocking until complete).

**Path Parameters:**
- `workflow_id` (string, required): Workflow identifier

**Query Parameters:**
- `timeout` (integer, optional, default: 300): Max wait time in seconds

**Response:** `200 OK` (for campaign launch)
```json
{
  "workflow_id": "wf_abc123def456",
  "workflow_type": "campaign_launch",
  "status": "completed",
  "duration_seconds": 22.5,
  "results": {
    "strategy": {
      "campaign_plan": {...},
      "target_audience": {...},
      "budget_allocation": {...},
      "content_strategy": {...},
      "success_metrics": {...}
    },
    "validation": {
      "feasibility_score": 0.85,
      "risk_assessment": "low",
      "recommendations": [...]
    },
    "learning_insights": {
      "applied_best_practices": [...],
      "optimization_suggestions": [...]
    }
  },
  "agents_executed": ["orchestrator", "marketing_strategy", "analytics_evaluation", "feedback_learning"],
  "execution_summary": {
    "total_steps": 4,
    "successful_steps": 4,
    "failed_steps": 0,
    "handoffs": 3
  }
}
```

**Error Response:** `408 Request Timeout`
```json
{
  "detail": "Workflow execution timed out after 300 seconds"
}
```

---

## 📝 Prompt Management

### GET `/api/v1/prompts/{agent_id}`

Get current system prompt for an agent.

**Path Parameters:**
- `agent_id` (string, required): Agent identifier

**Response:** `200 OK`
```json
{
  "agent_id": "marketing_strategy",
  "current_prompt": "You are an expert Marketing Strategy Agent...",
  "version": "v1.2",
  "updated_at": "2026-01-10T08:00:00"
}
```

---

### PUT `/api/v1/prompts/{agent_id}`

Update system prompt for an agent.

**Path Parameters:**
- `agent_id` (string, required): Agent identifier

**Request Body:**
```json
{
  "prompt": "You are an expert Marketing Strategy Agent specialized in...",
  "reason": "Enhanced budget allocation guidance"
}
```

**Response:** `200 OK`
```json
{
  "message": "Prompt updated successfully",
  "agent_id": "marketing_strategy",
  "version": "v1.3",
  "previous_version": "v1.2"
}
```

---

### GET `/api/v1/prompts/{agent_id}/versions`

List all prompt versions for an agent.

**Response:** `200 OK`
```json
{
  "agent_id": "marketing_strategy",
  "versions": [
    {
      "version": "v1.3",
      "updated_at": "2026-01-15T10:00:00",
      "reason": "Enhanced budget allocation guidance"
    },
    {
      "version": "v1.2",
      "updated_at": "2026-01-10T08:00:00",
      "reason": "Added KPI measurement framework"
    }
  ]
}
```

---

### POST `/api/v1/prompts/{agent_id}/rollback`

Rollback to a previous prompt version.

**Request Body:**
```json
{
  "version_id": "v1.2"
}
```

**Response:** `200 OK`
```json
{
  "message": "Prompt rolled back successfully",
  "agent_id": "marketing_strategy",
  "current_version": "v1.2"
}
```

---

## 🔍 Search & Query

### POST `/api/v1/knowledge-base/search`

Search the knowledge base (Stripe documentation).

**Request Body:**
```json
{
  "query": "subscription billing refunds",
  "top_k": 5,
  "filters": {
    "doc_type": "api_reference"
  }
}
```

**Response:** `200 OK`
```json
{
  "query": "subscription billing refunds",
  "results": [
    {
      "content": "Stripe provides flexible options for subscription refunds...",
      "source": "https://stripe.com/docs/billing/subscriptions/cancel",
      "relevance_score": 0.92,
      "metadata": {
        "doc_type": "guide",
        "category": "billing"
      }
    },
    {...}
  ],
  "total_results": 5
}
```

---

## 📊 Data Models

### AgentStatusResponse
```typescript
{
  agent_id: string;
  name: string;
  status: "idle" | "processing" | "error";
  capabilities: string[];
  active_tasks: number;
  total_executions: number;
  avg_execution_time: number | null;
}
```

### WorkflowStatusResponse
```typescript
{
  workflow_id: string;
  status: "pending" | "in_progress" | "completed" | "failed";
  workflow_type: string;
  started_at: string; // ISO 8601
  completed_at?: string; // ISO 8601
  progress: number; // 0.0 to 1.0
  current_agent?: string;
  agents_executed: string[];
  state_transitions: Array<{
    agent_name: string;
    timestamp: string;
    status: string;
  }>;
  error?: string;
}
```

### HealthCheckResponse
```typescript
{
  status: "healthy" | "degraded" | "unhealthy";
  timestamp: string;
  version: string;
  components: Record<string, string>;
  uptime_seconds: number;
}
```

---

## 🚨 Error Responses

### Standard Error Format
```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| `200` | OK | Request successful |
| `202` | Accepted | Async workflow initiated |
| `400` | Bad Request | Invalid request parameters |
| `404` | Not Found | Resource not found |
| `408` | Request Timeout | Workflow execution timeout |
| `422` | Unprocessable Entity | Validation error |
| `500` | Internal Server Error | Server error |

### Example Error Response
```json
{
  "detail": [
    {
      "loc": ["body", "budget"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ]
}
```

---

## 🔄 Async Workflow Pattern

Most workflows follow this pattern:

1. **Initiate:** POST to workflow endpoint → Receive `workflow_id`
2. **Poll:** GET `/workflows/{workflow_id}/status` → Check progress
3. **Retrieve:** GET `/workflows/{workflow_id}/results` → Get final results (blocking)

### Example Flow
```bash
# 1. Initiate workflow
curl -X POST http://localhost:8000/api/v1/workflows/campaign-launch \
  -H "Content-Type: application/json" \
  -d '{"campaign_name": "Q2 Launch", "objectives": ["leads"], ...}'

# Response: {"workflow_id": "wf_abc123", "status": "pending"}

# 2. Poll status (optional)
curl http://localhost:8000/api/v1/workflows/wf_abc123/status

# 3. Get results (blocks until complete)
curl http://localhost:8000/api/v1/workflows/wf_abc123/results?timeout=600
```

---

## 📦 SDK Examples

### Python (httpx)
```python
import httpx
import asyncio

async def launch_campaign():
    async with httpx.AsyncClient() as client:
        # Initiate workflow
        response = await client.post(
            "http://localhost:8000/api/v1/workflows/campaign-launch",
            json={
                "campaign_name": "Q2 Product Launch",
                "objectives": ["awareness", "leads"],
                "target_audience": "B2B SaaS",
                "budget": 75000,
                "duration_weeks": 12
            }
        )
        workflow_id = response.json()["workflow_id"]

        # Get results (blocking)
        results = await client.get(
            f"http://localhost:8000/api/v1/workflows/{workflow_id}/results",
            params={"timeout": 600}
        )
        return results.json()

# Run
results = asyncio.run(launch_campaign())
print(results)
```

### JavaScript (fetch)
```javascript
async function launchCampaign() {
  // Initiate workflow
  const initResponse = await fetch(
    'http://localhost:8000/api/v1/workflows/campaign-launch',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        campaign_name: 'Q2 Product Launch',
        objectives: ['awareness', 'leads'],
        target_audience: 'B2B SaaS',
        budget: 75000,
        duration_weeks: 12
      })
    }
  );
  const { workflow_id } = await initResponse.json();

  // Get results
  const resultsResponse = await fetch(
    `http://localhost:8000/api/v1/workflows/${workflow_id}/results?timeout=600`
  );
  return await resultsResponse.json();
}
```

### cURL
```bash
# Campaign launch
curl -X POST http://localhost:8000/api/v1/workflows/campaign-launch \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_name": "Q2 Product Launch",
    "objectives": ["awareness", "leads"],
    "target_audience": "B2B SaaS companies",
    "budget": 75000,
    "duration_weeks": 12
  }'

# Customer support
curl -X POST http://localhost:8000/api/v1/workflows/customer-support \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does subscription billing work?",
    "customer_id": "cust_123",
    "priority": "high"
  }'

# Check health
curl http://localhost:8000/health

# List agents
curl http://localhost:8000/api/v1/agents
```

---

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# CORS
ALLOWED_ORIGINS=localhost:3000,localhost:8501

# Timeouts
DEFAULT_WORKFLOW_TIMEOUT=300
MAX_WORKFLOW_TIMEOUT=900

# Rate Limiting (if enabled)
RATE_LIMIT_PER_MINUTE=60
```

---

## 📈 Rate Limiting

**Current Status:** Not implemented (development)

**Production Implementation:**
- Rate limit: 60 requests/minute per IP
- Burst allowance: 10 requests
- Headers returned:
  - `X-RateLimit-Limit`: Requests allowed per window
  - `X-RateLimit-Remaining`: Requests remaining
  - `X-RateLimit-Reset`: Unix timestamp when limit resets

---

## 🔐 Security Best Practices

1. **Use HTTPS** in production
2. **Implement authentication** (API keys or JWT)
3. **Enable rate limiting** to prevent abuse
4. **Validate all inputs** (handled by Pydantic)
5. **Set CORS** appropriately for your domains
6. **Monitor API usage** via `/metrics` endpoint
7. **Use timeouts** on all workflow requests

---

## 📚 Additional Resources

- **Interactive Docs:** http://localhost:8000/api/docs
- **Streamlit Dashboard:** http://localhost:8501
- **Gradio Chat:** http://localhost:7860
- **Source Code:** `/api` directory
- **Schemas:** `/api/schemas`
- **Routes:** `/api/routes`

---

**Last Updated:** January 15, 2026
**API Version:** 1.0.0
**Contact:** srikanthgali137@gmail.com
