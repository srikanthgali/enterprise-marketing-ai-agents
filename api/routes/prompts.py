"""
API endpoints for prompt management.

Provides REST API for:
- Viewing current prompts
- Updating prompts
- Listing versions
- Comparing versions
- Rolling back prompts
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from src.marketing_agents.core.prompt_manager import PromptManager
from api.dependencies import get_prompt_manager
from api.schemas.request import PromptUpdateRequest, RollbackRequest
from api.schemas.response import (
    PromptResponse,
    VersionInfo,
    ComparisonResponse,
    ChangeLogEntry,
)


router = APIRouter(prefix="/prompts", tags=["prompts"])


# Endpoints
@router.get("/{agent_id}", response_model=PromptResponse)
async def get_prompt(
    agent_id: str,
    version: str = "latest",
    prompt_manager: PromptManager = Depends(get_prompt_manager),
):
    """
    Get the current or specific version of a prompt.

    Args:
        agent_id: Identifier for the agent
        version: Version to retrieve (default: latest)

    Returns:
        Prompt content and metadata
    """
    try:
        prompt = prompt_manager.load_prompt(agent_id, version)
        return PromptResponse(
            agent_id=agent_id, version=version, prompt=prompt, size=len(prompt)
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Prompt not found for agent '{agent_id}'"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{agent_id}")
async def update_prompt(
    agent_id: str,
    request: PromptUpdateRequest,
    prompt_manager: PromptManager = Depends(get_prompt_manager),
):
    """
    Update a prompt and create a new version.

    Args:
        agent_id: Identifier for the agent
        request: Update request with new prompt and reason

    Returns:
        Success status and version information
    """
    try:
        success = prompt_manager.update_prompt(agent_id, request.prompt, request.reason)

        if success:
            versions = prompt_manager.list_versions(agent_id)
            latest_version = versions[0] if versions else None

            return {
                "success": True,
                "agent_id": agent_id,
                "message": "Prompt updated successfully",
                "version": latest_version,
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update prompt")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}/versions", response_model=List[VersionInfo])
async def list_versions(
    agent_id: str, prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """
    List all available versions for a prompt.

    Args:
        agent_id: Identifier for the agent

    Returns:
        List of version metadata
    """
    try:
        versions = prompt_manager.list_versions(agent_id)
        return [VersionInfo(**v) for v in versions]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/rollback")
async def rollback_prompt(
    agent_id: str,
    request: RollbackRequest,
    prompt_manager: PromptManager = Depends(get_prompt_manager),
):
    """
    Rollback a prompt to a previous version.

    Args:
        agent_id: Identifier for the agent
        request: Rollback request with version ID

    Returns:
        Success status and rollback information
    """
    try:
        success = prompt_manager.rollback_prompt(agent_id, request.version_id)

        if success:
            return {
                "success": True,
                "agent_id": agent_id,
                "message": f"Rolled back to version {request.version_id}",
                "version_id": request.version_id,
            }
        else:
            raise HTTPException(status_code=500, detail="Rollback failed")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}/compare")
async def compare_prompts(
    agent_id: str,
    version1: str,
    version2: str = "latest",
    prompt_manager: PromptManager = Depends(get_prompt_manager),
):
    """
    Compare two versions of a prompt.

    Args:
        agent_id: Identifier for the agent
        version1: First version to compare
        version2: Second version to compare (default: latest)

    Returns:
        Comparison results with diff and summary
    """
    try:
        comparison = prompt_manager.compare_prompts(agent_id, version1, version2)

        if "error" in comparison:
            raise HTTPException(status_code=400, detail=comparison["error"])

        return {
            "agent_id": agent_id,
            "version1": comparison["version1"],
            "version2": comparison["version2"],
            "diff_unified": comparison["diff_unified"],
            "diff_html": comparison["diff_html"],
            "changes_summary": comparison["changes_summary"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}/history", response_model=List[ChangeLogEntry])
async def get_change_history(
    agent_id: str,
    limit: int = 20,
    prompt_manager: PromptManager = Depends(get_prompt_manager),
):
    """
    Get the change history for a prompt.

    Args:
        agent_id: Identifier for the agent
        limit: Maximum number of entries to return

    Returns:
        List of change log entries
    """
    try:
        history = prompt_manager.get_change_history(agent_id, limit)
        return [ChangeLogEntry(**entry) for entry in history]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/reload")
async def reload_prompt(
    agent_id: str, prompt_manager: PromptManager = Depends(get_prompt_manager)
):
    """
    Force reload a prompt from disk, clearing the cache.

    Args:
        agent_id: Identifier for the agent

    Returns:
        Reloaded prompt information
    """
    try:
        prompt = prompt_manager.reload_prompt(agent_id)
        return {
            "success": True,
            "agent_id": agent_id,
            "message": "Prompt reloaded from disk",
            "size": len(prompt),
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Prompt not found for agent '{agent_id}'"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
