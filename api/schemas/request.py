"""Request models for API endpoints."""

from pydantic import BaseModel, Field


class PromptUpdateRequest(BaseModel):
    """Request model for updating a prompt."""

    prompt: str = Field(..., description="New prompt content")
    reason: str = Field(..., description="Reason for the update")


class RollbackRequest(BaseModel):
    """Request model for rolling back a prompt."""

    version_id: str = Field(..., description="Version ID to rollback to")
