"""
Health check and metrics API routes.

Endpoints for system health monitoring and performance metrics.
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
from datetime import datetime
import time
import psutil
import os

from api.schemas.response import HealthCheckResponse, MetricsResponse
from api.dependencies import get_orchestrator, get_memory_manager
from config.settings import get_settings

router = APIRouter(tags=["health"])

# Track application start time
START_TIME = time.time()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    orchestrator=Depends(get_orchestrator),
):
    """
    Check the health status of the system and its components.

    Returns:
        Health status of all system components
    """
    settings = get_settings()
    components = {}

    # Check orchestrator and agents
    try:
        if orchestrator and len(orchestrator.agents) > 0:
            components["agents"] = "healthy"
        else:
            components["agents"] = "degraded"
    except Exception:
        components["agents"] = "unhealthy"

    # Check Redis connection if enabled
    if settings.redis.enabled:
        try:
            # Try to import redis and check connection
            import redis

            r = redis.from_url(settings.redis.url)
            r.ping()
            components["redis"] = "healthy"
        except Exception:
            components["redis"] = "unhealthy"
    else:
        components["redis"] = "disabled"

    # Check memory manager
    try:
        memory_manager = get_memory_manager()
        if memory_manager:
            components["memory"] = "healthy"
        else:
            components["memory"] = "degraded"
    except Exception:
        components["memory"] = "unhealthy"

    # Determine overall status
    unhealthy_count = sum(1 for status in components.values() if status == "unhealthy")
    if unhealthy_count == 0:
        overall_status = "healthy"
    elif unhealthy_count < len(components) / 2:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version=settings.version,
        components=components,
        uptime_seconds=time.time() - START_TIME,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    orchestrator=Depends(get_orchestrator),
):
    """
    Get system performance metrics.

    Returns:
        Detailed system metrics including agent statistics,
        workflow statistics, and system resources
    """
    # Collect agent metrics
    agent_metrics = {}
    for agent_id, agent in orchestrator.agents.items():
        history = getattr(agent, "execution_history", [])
        successful = sum(1 for h in history if h.get("status") == "completed")
        failed = sum(1 for h in history if h.get("status") == "failed")
        durations = [
            h.get("duration_seconds", 0) for h in history if h.get("duration_seconds")
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        agent_metrics[agent_id] = {
            "total_executions": len(history),
            "successful": successful,
            "failed": failed,
            "avg_duration": round(avg_duration, 2),
            "status": agent.status.value,
        }

    # Collect workflow metrics
    from api.routes.workflows import workflow_states

    workflow_metrics = {
        "total": len(workflow_states),
        "completed": sum(
            1 for w in workflow_states.values() if w["status"] == "completed"
        ),
        "in_progress": sum(
            1 for w in workflow_states.values() if w["status"] == "in_progress"
        ),
        "failed": sum(1 for w in workflow_states.values() if w["status"] == "failed"),
        "pending": sum(1 for w in workflow_states.values() if w["status"] == "pending"),
    }

    # Collect system metrics
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)

        system_metrics = {
            "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
            "cpu_usage_percent": round(cpu_percent, 2),
            "uptime_seconds": round(time.time() - START_TIME, 2),
            "active_threads": process.num_threads(),
        }
    except Exception:
        system_metrics = {
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0,
            "uptime_seconds": round(time.time() - START_TIME, 2),
            "active_threads": 0,
        }

    # Performance metrics from orchestrator
    performance_metrics = {
        "total_requests": orchestrator.health_metrics.get("total_requests", 0),
        "successful_requests": orchestrator.health_metrics.get(
            "successful_requests", 0
        ),
        "failed_requests": orchestrator.health_metrics.get("failed_requests", 0),
        "workflows_running": orchestrator.health_metrics.get("workflows_running", 0),
    }

    # Calculate success rate
    total = performance_metrics["total_requests"]
    if total > 0:
        performance_metrics["success_rate"] = round(
            performance_metrics["successful_requests"] / total * 100, 2
        )
    else:
        performance_metrics["success_rate"] = 100.0

    return MetricsResponse(
        timestamp=datetime.now(),
        agents=agent_metrics,
        workflows=workflow_metrics,
        system=system_metrics,
        performance=performance_metrics,
    )
