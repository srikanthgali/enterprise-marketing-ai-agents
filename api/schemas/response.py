"""Response models for API endpoints."""

from typing import Dict
from pydantic import BaseModel, Field


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
