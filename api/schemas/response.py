"""Response models for API endpoints."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class PromptResponse(BaseModel):
    """Response model for prompt content."""

    agent_id: str
    version: str
    prompt: str
    size: int


class VersionInfo(BaseModel):
    """Information about a prompt version."""

    version_id: str
    file_path: str
    size: int
    modified: str
    is_current: bool


class ComparisonResponse(BaseModel):
    """Response model for prompt comparison."""

    agent_id: str
    version1: str
    version2: str
    diff_unified: str
    changes_summary: Dict[str, int]


class ChangeLogEntry(BaseModel):
    """Log entry for a prompt change."""

    timestamp: str
    agent_id: str
    version_id: str
    reason: str


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""

    agent_id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Current agent status")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    active_tasks: int = Field(..., description="Number of active tasks")
    total_executions: int = Field(..., description="Total executions count")
    avg_execution_time: Optional[float] = Field(
        None, description="Average execution time in seconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "marketing_strategy",
                "name": "Marketing Strategy Agent",
                "status": "idle",
                "capabilities": ["strategy", "planning", "analysis"],
                "active_tasks": 0,
                "total_executions": 42,
                "avg_execution_time": 12.5,
            }
        }


class AgentExecutionResponse(BaseModel):
    """Response model for agent execution."""

    execution_id: str = Field(..., description="Unique execution ID")
    agent_id: str = Field(..., description="Agent identifier")
    status: str = Field(..., description="Execution status")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result")
    error: Optional[str] = Field(None, description="Error message if failed")
    started_at: datetime = Field(..., description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution end time")
    duration_seconds: Optional[float] = Field(None, description="Execution duration")

    class Config:
        json_schema_extra = {
            "example": {
                "execution_id": "exec_abc123",
                "agent_id": "marketing_strategy",
                "status": "completed",
                "result": {
                    "campaign_plan": {"channels": ["email", "social"], "budget": 50000}
                },
                "started_at": "2026-01-12T10:00:00Z",
                "completed_at": "2026-01-12T10:00:15Z",
                "duration_seconds": 15.3,
            }
        }


class ExecutionHistoryItem(BaseModel):
    """Single execution history entry."""

    execution_id: str
    timestamp: datetime
    task_type: str
    status: str
    duration_seconds: Optional[float]
    summary: Optional[str]


class AgentHistoryResponse(BaseModel):
    """Response model for agent execution history."""

    agent_id: str
    total_executions: int
    history: List[ExecutionHistoryItem]

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "marketing_strategy",
                "total_executions": 42,
                "history": [
                    {
                        "execution_id": "exec_abc123",
                        "timestamp": "2026-01-12T10:00:00Z",
                        "task_type": "campaign_planning",
                        "status": "completed",
                        "duration_seconds": 15.3,
                        "summary": "Campaign plan created",
                    }
                ],
            }
        }


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status."""

    workflow_id: str = Field(..., description="Workflow identifier")
    workflow_type: str = Field(..., description="Type of workflow")
    status: str = Field(..., description="Current workflow status")
    current_agent: Optional[str] = Field(None, description="Currently executing agent")
    progress: float = Field(..., description="Progress percentage (0-100)")
    started_at: datetime = Field(..., description="Workflow start time")
    completed_at: Optional[datetime] = Field(
        None, description="Workflow completion time"
    )
    estimated_completion: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )
    agents_executed: List[str] = Field(..., description="List of executed agents")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "wf_xyz789",
                "workflow_type": "campaign_launch",
                "status": "in_progress",
                "current_agent": "marketing_strategy",
                "progress": 45.0,
                "started_at": "2026-01-12T10:00:00Z",
                "agents_executed": ["orchestrator"],
            }
        }


class WorkflowResultResponse(BaseModel):
    """Response model for workflow results."""

    workflow_id: str = Field(..., description="Workflow identifier")
    workflow_type: str = Field(..., description="Type of workflow")
    status: str = Field(..., description="Final workflow status")
    results: Dict[str, Any] = Field(..., description="Workflow results by agent")
    summary: str = Field(..., description="Human-readable summary")
    execution_trace: List[Dict[str, Any]] = Field(
        ..., description="Detailed execution trace"
    )
    started_at: datetime = Field(..., description="Workflow start time")
    completed_at: datetime = Field(..., description="Workflow completion time")
    total_duration_seconds: float = Field(..., description="Total execution time")

    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "wf_xyz789",
                "workflow_type": "campaign_launch",
                "status": "completed",
                "results": {
                    "marketing_strategy": {"campaign_plan": "..."},
                    "analytics_evaluation": {"metrics": "..."},
                },
                "summary": "Campaign launch workflow completed successfully",
                "execution_trace": [],
                "started_at": "2026-01-12T10:00:00Z",
                "completed_at": "2026-01-12T10:05:00Z",
                "total_duration_seconds": 300.0,
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Overall system health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    components: Dict[str, str] = Field(..., description="Component health status")
    uptime_seconds: float = Field(..., description="System uptime in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2026-01-12T10:00:00Z",
                "version": "0.1.0",
                "components": {
                    "database": "healthy",
                    "redis": "healthy",
                    "agents": "healthy",
                },
                "uptime_seconds": 86400.0,
            }
        }


class MetricsResponse(BaseModel):
    """Response model for system metrics."""

    timestamp: datetime
    agents: Dict[str, Dict[str, Any]] = Field(..., description="Agent-specific metrics")
    workflows: Dict[str, int] = Field(..., description="Workflow statistics")
    system: Dict[str, Any] = Field(..., description="System-level metrics")
    performance: Dict[str, float] = Field(..., description="Performance metrics")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-01-12T10:00:00Z",
                "agents": {
                    "marketing_strategy": {
                        "total_executions": 42,
                        "successful": 40,
                        "failed": 2,
                        "avg_duration": 12.5,
                    }
                },
                "workflows": {
                    "total": 100,
                    "completed": 95,
                    "in_progress": 3,
                    "failed": 2,
                },
                "system": {
                    "memory_usage_mb": 512,
                    "cpu_usage_percent": 25.5,
                },
                "performance": {
                    "avg_response_time_ms": 250.0,
                    "requests_per_minute": 45.0,
                },
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(..., description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "details": {"field": "budget", "issue": "must be positive"},
                "timestamp": "2026-01-12T10:00:00Z",
            }
        }
