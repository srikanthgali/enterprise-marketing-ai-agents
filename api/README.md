# FastAPI Application - Quick Start Guide

## Quick Start (Recommended)

```bash
# Start all services (API + Streamlit + Gradio)
./start_all.sh

# Or start API only:
python api/main.py

# Verify it's running:
curl http://localhost:8000/health
```

API will be available at:
- **Base URL:** http://localhost:8000
- **Docs:** http://localhost:8000/api/docs
- **Health:** http://localhost:8000/health

## Running the API

### Development Mode

```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install fastapi uvicorn[standard] psutil

# Run the API
python api/main.py

# Or with uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Install production server
pip install gunicorn

# Run with multiple workers
gunicorn api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300
```

## API Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI JSON**: http://localhost:8000/api/openapi.json

## API Endpoints

### Health & Metrics
- `GET /health` - System health check
- `GET /metrics` - System metrics and statistics

### Agents
- `GET /api/v1/agents` - List all agents
- `GET /api/v1/agents/{agent_id}` - Get agent details
- `GET /api/v1/agents/{agent_id}/history` - Get execution history

### Workflows
- `POST /api/v1/workflows/campaign-launch` - Launch campaign workflow
- `POST /api/v1/workflows/customer-support` - Handle support inquiry
- `POST /api/v1/workflows/analytics` - Generate analytics report
- `POST /api/v1/workflows/feedback-learning` - Run feedback/learning analysis
- `GET /api/v1/workflows/{workflow_id}/status` - Get workflow status
- `GET /api/v1/workflows/{workflow_id}/results` - Get workflow results (blocking)

### Prompts (Advanced)
- `GET /api/v1/prompts/{agent_id}` - Get agent's current prompt
- `PUT /api/v1/prompts/{agent_id}` - Update agent prompt
- `GET /api/v1/prompts/{agent_id}/versions` - List prompt versions
- `POST /api/v1/prompts/{agent_id}/rollback` - Rollback to previous version

## Example API Calls

### 1. Check System Health

```bash
curl http://localhost:8000/health
```

### 2. List All Agents

```bash
curl http://localhost:8000/api/v1/agents
```

### 3. Launch Campaign Workflow

```bash
curl -X POST http://localhost:8000/api/v1/workflows/campaign-launch \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_name": "Q2 Product Launch",
    "objectives": ["awareness", "leads"],
    "target_audience": "B2B SaaS companies",
    "budget": 75000,
    "duration_weeks": 12,
    "channels": ["email", "social", "content"]
  }'
```

### 4. Check Workflow Status

```bash
# Replace {workflow_id} with the ID from the launch response
curl http://localhost:8000/api/v1/workflows/{workflow_id}/status
```

### 5. Get Workflow Results

```bash
curl http://localhost:8000/api/v1/workflows/{workflow_id}/results
```

### 6. Customer Support Inquiry

```bash
curl -X POST http://localhost:8000/api/v1/workflows/customer-support \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does Stripe handle subscription refunds?",
    "customer_id": "cust_abc123",
    "priority": "high"
  }'
```

### 7. Generate Analytics Report

```bash
curl -X POST http://localhost:8000/api/v1/workflows/analytics \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_type": "campaign_performance",
    "time_range": "last_30_days",
    "metrics": ["ctr", "conversions", "roi", "cac"],
    "include_forecast": true
  }'
```

### 8. Feedback & Learning Analysis

```bash
curl -X POST http://localhost:8000/api/v1/workflows/feedback-learning \
  -H "Content-Type: application/json" \
  -d '{
    "feedback_type": "agent_optimization",
    "agent_id": "marketing_strategy",
    "time_range": "last_7_days"
  }'
```

## Authentication

**Current Status:** Not implemented (development mode)

For production deployment, implement:
- API key authentication via headers
- JWT tokens for session management
- Rate limiting per client

See `docs/api_reference.md` for security best practices.

## Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI API (Required)
OPENAI_API_KEY=your-openai-api-key
OPENAI_ORG_ID=your-org-id

# System Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO

# API Settings (Optional - defaults provided)
API_HOST=0.0.0.0
API_PORT=8000

# CORS (Optional)
ALLOWED_ORIGINS=localhost,127.0.0.1

# Vector Store
EMBEDDINGS_PATH=data/embeddings
KB_INDEX_NAME=stripe_knowledge_base
```

## Error Handling

All errors return FastAPI's standard format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Validation errors return detailed information:
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

Common HTTP status codes:
- `200` - Success
- `202` - Accepted (async workflow started)
- `400` - Bad Request
- `404` - Not Found
- `408` - Request Timeout (workflow timeout)
- `422` - Validation Error
- `500` - Internal Server Error

## Rate Limiting

**Current Status:** Not implemented (development mode)

For production, implement rate limiting with:
- Redis-based token bucket algorithm
- Per-IP or per-API-key limits
- Configurable limits per endpoint type

## Monitoring

Get real-time metrics:

```bash
curl http://localhost:8000/metrics
```

Returns:
- Agent execution statistics
- Workflow statistics
- System resource usage
- Performance metrics

## Troubleshooting

### Port Already in Use

```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn api.main:app --port 8001
```

### Import Errors

```bash
# Ensure you're in the project root
cd /path/to/enterprise-marketing-ai-agents

# Install in editable mode
pip install -e .
```

### Agent Not Found

Ensure agents are properly initialized by checking logs:
```bash
tail -f logs/api.log

# Or check agent status
curl http://localhost:8000/api/v1/agents
```

## Next Steps

1. **Integrate with Frontend**: Use the OpenAPI spec to generate client SDKs
2. **Add Webhooks**: Implement callback URLs for async workflow completion
3. **Stream Progress**: Add SSE or WebSocket for real-time workflow updates
4. **Deploy**: Use Docker and Kubernetes for production deployment

## Support

For issues or questions:
- Check logs in `logs/api/`
- Review OpenAPI docs at `/api/docs`
- See main project README for architecture details
