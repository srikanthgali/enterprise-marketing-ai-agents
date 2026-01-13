"""
Agent management API routes.

Endpoints for executing agents and retrieving agent information.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, List
import uuid
from datetime import datetime

from api.schemas.request import AgentExecuteRequest
from api.schemas.response import (
    AgentStatusResponse,
    AgentExecutionResponse,
    AgentHistoryResponse,
    ExecutionHistoryItem,
    ErrorResponse,
)
from api.dependencies import get_orchestrator, get_memory_manager

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=List[AgentStatusResponse])
async def list_agents(orchestrator=Depends(get_orchestrator)):
    """
    List all available agents with their current status.

    Returns:
        List of agent status information
    """
    agents_list = []

    for agent_id, agent in orchestrator.agents.items():
        # Get execution history for the agent
        history = getattr(agent, "execution_history", [])

        # Calculate stats
        total_executions = len(history)
        avg_time = None
        if history:
            durations = [
                h.get("duration_seconds", 0)
                for h in history
                if h.get("duration_seconds")
            ]
            if durations:
                avg_time = sum(durations) / len(durations)

        agents_list.append(
            AgentStatusResponse(
                agent_id=agent.agent_id,
                name=agent.name,
                status=agent.status.value,
                capabilities=agent.capabilities,
                active_tasks=0,  # Could be tracked via workflow state
                total_executions=total_executions,
                avg_execution_time=avg_time,
            )
        )

    return agents_list


@router.get("/{agent_id}", response_model=AgentStatusResponse)
async def get_agent_details(agent_id: str, orchestrator=Depends(get_orchestrator)):
    """
    Get detailed information about a specific agent.

    Args:
        agent_id: The agent identifier

    Returns:
        Agent status and details
    """
    if agent_id not in orchestrator.agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found",
        )

    agent = orchestrator.agents[agent_id]
    history = getattr(agent, "execution_history", [])

    # Calculate stats
    total_executions = len(history)
    avg_time = None
    if history:
        durations = [
            h.get("duration_seconds", 0) for h in history if h.get("duration_seconds")
        ]
        if durations:
            avg_time = sum(durations) / len(durations)

    return AgentStatusResponse(
        agent_id=agent.agent_id,
        name=agent.name,
        status=agent.status.value,
        capabilities=agent.capabilities,
        active_tasks=0,
        total_executions=total_executions,
        avg_execution_time=avg_time,
    )


@router.get("/{agent_id}/history", response_model=AgentHistoryResponse)
async def get_agent_history(
    agent_id: str,
    limit: int = 50,
    orchestrator=Depends(get_orchestrator),
):
    """
    Get execution history for a specific agent.

    Args:
        agent_id: The agent identifier
        limit: Maximum number of history entries to return

    Returns:
        Agent execution history
    """
    if agent_id not in orchestrator.agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found",
        )

    agent = orchestrator.agents[agent_id]
    history = getattr(agent, "execution_history", [])

    # Convert to response format
    history_items = []
    for entry in history[-limit:]:
        history_items.append(
            ExecutionHistoryItem(
                execution_id=entry.get("execution_id", str(uuid.uuid4())),
                timestamp=entry.get("timestamp", datetime.now()),
                task_type=entry.get("task_type", "unknown"),
                status=entry.get("status", "unknown"),
                duration_seconds=entry.get("duration_seconds"),
                summary=entry.get("summary"),
            )
        )

    return AgentHistoryResponse(
        agent_id=agent_id,
        total_executions=len(history),
        history=history_items,
    )


@router.post("/{agent_id}/execute", response_model=AgentExecutionResponse)
async def execute_agent(
    agent_id: str,
    request: AgentExecuteRequest,
    orchestrator=Depends(get_orchestrator),
    memory_manager=Depends(get_memory_manager),
):
    """
    Execute a specific agent with the provided task data.

    Args:
        agent_id: The agent identifier
        request: Execution request with task data

    Returns:
        Execution result
    """
    if agent_id not in orchestrator.agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found",
        )

    agent = orchestrator.agents[agent_id]
    execution_id = f"exec_{uuid.uuid4().hex[:12]}"
    started_at = datetime.now()

    try:
        # Execute the agent
        result = await agent.process(request.task_data)

        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()

        # Store execution record if session_id provided
        if request.session_id and memory_manager:
            memory_manager.save_short_term(
                key=f"execution:{execution_id}",
                value={
                    "agent_id": agent_id,
                    "task_data": request.task_data,
                    "result": result,
                    "duration": duration,
                },
            )

        return AgentExecutionResponse(
            execution_id=execution_id,
            agent_id=agent_id,
            status="completed",
            result=result,
            error=None,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )

    except Exception as e:
        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()

        return AgentExecutionResponse(
            execution_id=execution_id,
            agent_id=agent_id,
            status="failed",
            result=None,
            error=str(e),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )
