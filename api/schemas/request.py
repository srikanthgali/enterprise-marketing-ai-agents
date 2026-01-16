"""Request models for API endpoints."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime


class PromptUpdateRequest(BaseModel):
    """Request model for updating a prompt."""

    prompt: str = Field(..., description="New prompt content")
    reason: str = Field(..., description="Reason for the update")


class RollbackRequest(BaseModel):
    """Request model for rolling back a prompt."""

    version_id: str = Field(..., description="Version ID to rollback to")


class AgentExecuteRequest(BaseModel):
    """Request model for agent execution."""

    task_data: Dict[str, Any] = Field(
        ...,
        description="Task data to pass to the agent",
        example={"query": "Analyze campaign performance"},
    )
    session_id: Optional[str] = Field(
        None, description="Optional session ID for conversation context"
    )
    priority: Optional[str] = Field(
        "medium", description="Task priority", example="high"
    )

    @validator("priority")
    def validate_priority(cls, v):
        """Validate priority level."""
        valid = ["low", "medium", "high", "critical"]
        if v not in valid:
            raise ValueError(f"Priority must be one of {valid}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "task_data": {
                    "query": "Analyze Q1 campaign performance",
                    "metrics": ["ctr", "conversions", "roi"],
                },
                "session_id": "sess_123abc",
                "priority": "high",
            }
        }


class CampaignLaunchRequest(BaseModel):
    """Request model for campaign launch workflow."""

    campaign_name: str = Field(..., description="Name of the campaign", min_length=1)
    objectives: List[str] = Field(
        ..., description="Campaign objectives", example=["awareness", "leads"]
    )
    target_audience: str = Field(
        ..., description="Target audience description", min_length=1
    )
    budget: float = Field(..., description="Campaign budget in dollars", gt=0)
    duration_weeks: int = Field(
        ..., description="Campaign duration in weeks", gt=0, le=52
    )
    channels: Optional[List[str]] = Field(None, description="Marketing channels to use")
    additional_context: Optional[Dict[str, Any]] = Field(
        None, description="Additional campaign context"
    )

    @validator("objectives")
    def validate_objectives(cls, v):
        """Validate objectives."""
        valid = ["awareness", "leads", "conversions", "retention", "engagement"]
        for obj in v:
            if obj not in valid:
                raise ValueError(f"Each objective must be one of {valid}, got: {obj}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "campaign_name": "Q2 Product Launch",
                "objectives": ["awareness", "leads"],
                "target_audience": "B2B SaaS companies",
                "budget": 75000,
                "duration_weeks": 12,
                "channels": ["email", "social", "content"],
            }
        }


class CustomerSupportRequest(BaseModel):
    """Request model for customer support workflow."""

    inquiry: str = Field(..., description="Customer inquiry or question", min_length=1)
    customer_id: Optional[str] = Field(None, description="Customer ID if available")
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context about the inquiry"
    )
    urgency: Optional[str] = Field("normal", description="Urgency level")

    @validator("urgency")
    def validate_urgency(cls, v):
        """Validate urgency level."""
        valid = ["low", "normal", "high", "critical"]
        if v not in valid:
            raise ValueError(f"Urgency must be one of {valid}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "inquiry": "How do I integrate the payment API?",
                "customer_id": "cust_abc123",
                "context": {"product": "Payment API", "plan": "Enterprise"},
                "urgency": "high",
            }
        }


class AnalyticsRequest(BaseModel):
    """Request model for analytics report generation."""

    report_type: str = Field(..., description="Type of analytics report to generate")
    date_range: Dict[str, str] = Field(
        ...,
        description="Date range for the report",
        example={"start": "2026-01-01", "end": "2026-01-31"},
    )
    message: Optional[str] = Field(
        None, description="Natural language query about the data"
    )
    metrics: Optional[List[str]] = Field(
        None, description="Specific metrics to include"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional filters (can include user_query for natural language queries)",
    )
    format: Optional[str] = Field("json", description="Output format")

    @validator("report_type")
    def validate_report_type(cls, v):
        """Validate report type."""
        valid = ["campaign_performance", "customer_engagement", "revenue", "trends"]
        if v not in valid:
            raise ValueError(f"Report type must be one of {valid}")
        return v

    @validator("format")
    def validate_format(cls, v):
        """Validate output format."""
        valid = ["json", "pdf", "csv"]
        if v not in valid:
            raise ValueError(f"Format must be one of {valid}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "report_type": "campaign_performance",
                "date_range": {"start": "2026-01-01", "end": "2026-01-31"},
                "metrics": ["ctr", "conversions", "roi"],
                "format": "json",
            }
        }


class FeedbackLearningRequest(BaseModel):
    """Request model for feedback learning workflow."""

    message: str = Field(
        ..., description="Feedback or learning request message", min_length=1
    )
    request_type: Optional[str] = Field(
        "analyze_feedback", description="Type of learning request"
    )
    time_range: Optional[str] = Field(
        "last_7_days", description="Time range for feedback analysis"
    )
    agent_id: Optional[str] = Field(
        None, description="Specific agent to analyze (if applicable)"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context for the learning request"
    )

    @validator("request_type")
    def validate_request_type(cls, v):
        """Validate request type."""
        valid = [
            "analyze_feedback",
            "optimize_agent",
            "track_experiment",
            "detect_patterns",
            "prediction_improvement",
        ]
        if v not in valid:
            raise ValueError(f"Request type must be one of {valid}")
        return v

    @validator("time_range")
    def validate_time_range(cls, v):
        """Validate time range."""
        valid = ["last_24h", "last_7_days", "last_30_days", "all"]
        if v not in valid:
            raise ValueError(f"Time range must be one of {valid}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Rate the quality of the campaign strategy: 2/5 stars. It was too generic.",
                "request_type": "analyze_feedback",
                "time_range": "last_7_days",
                "context": {"source": "user_rating", "rating": 2},
            }
        }
