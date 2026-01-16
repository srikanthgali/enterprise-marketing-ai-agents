"""
Workflow execution API routes.

Endpoints for executing multi-agent workflows.
"""

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from typing import Dict, Any
import uuid
from datetime import datetime, timedelta

from api.schemas.request import (
    CampaignLaunchRequest,
    CustomerSupportRequest,
    AnalyticsRequest,
    FeedbackLearningRequest,
)
from api.schemas.response import (
    WorkflowStatusResponse,
    WorkflowResultResponse,
    ErrorResponse,
)
from api.dependencies import get_orchestrator, get_memory_manager
from config.settings import get_settings

router = APIRouter(prefix="/workflows", tags=["workflows"])

# In-memory workflow state storage (in production, use Redis or database)
workflow_states: Dict[str, Dict[str, Any]] = {}


def get_workflow_timeout(workflow_type: str) -> int:
    """Get timeout for workflow from config."""
    settings = get_settings()
    workflow_config = settings.get_workflow_config(workflow_type)
    if workflow_config:
        # Calculate total timeout from steps
        timeout = sum(
            step.get("timeout", 60) for step in workflow_config.get("steps", [])
        )
        return timeout
    # Default timeouts
    defaults = {
        "campaign_launch": 600,
        "customer_support": 300,
        "analytics": 300,
    }
    return defaults.get(workflow_type, 300)


async def execute_workflow_async(
    workflow_id: str,
    workflow_type: str,
    task_data: Dict[str, Any],
    orchestrator,
    memory_manager,
):
    """
    Execute workflow asynchronously in background.

    Args:
        workflow_id: Unique workflow identifier
        workflow_type: Type of workflow
        task_data: Data for the workflow
        orchestrator: Orchestrator agent instance
        memory_manager: Memory manager instance
    """
    workflow_states[workflow_id]["status"] = "in_progress"
    workflow_states[workflow_id]["started_at"] = datetime.now()

    try:
        # Execute the workflow
        result = await orchestrator.process(
            {
                "task_type": workflow_type,
                "data": task_data,
                "workflow_id": workflow_id,
            }
        )

        # Extract agents_executed from execution_summary
        agents_executed = []
        state_transitions = []
        if result.get("execution_summary"):
            execution_summary = result["execution_summary"]
            # Get agents_executed directly from execution_summary
            # (get_execution_summary already extracts them)
            agents_executed = execution_summary.get("agents_executed", [])

            # Build state_transitions from execution_history for analytics
            execution_history = execution_summary.get("execution_history", [])
            for record in execution_history:
                state_transitions.append(
                    {
                        "agent_name": record.get("agent_id", "unknown"),
                        "timestamp": record.get("started_at"),
                        "status": record.get("status", "unknown"),
                    }
                )

        # Calculate progress (100% when completed)
        progress = 1.0

        # Update workflow state
        workflow_states[workflow_id].update(
            {
                "status": "completed",
                "completed_at": datetime.now(),
                "result": result,
                "agents_executed": agents_executed,
                "state_transitions": state_transitions,
                "progress": progress,
                "current_agent": None,
                "error": None,
            }
        )

    except Exception as e:
        workflow_states[workflow_id].update(
            {
                "status": "failed",
                "completed_at": datetime.now(),
                "error": str(e),
            }
        )


@router.post("/campaign-launch", response_model=WorkflowStatusResponse)
async def launch_campaign(
    request: CampaignLaunchRequest,
    background_tasks: BackgroundTasks,
    orchestrator=Depends(get_orchestrator),
    memory_manager=Depends(get_memory_manager),
):
    """
    Launch a campaign creation workflow.

    This workflow involves:
    1. Marketing Strategy Agent - Creates campaign plan
    2. Analytics Evaluation Agent - Validates feasibility
    3. Feedback Learning Agent - Applies learned best practices

    Args:
        request: Campaign launch request

    Returns:
        Workflow status with workflow_id for tracking
    """
    workflow_id = f"wf_{uuid.uuid4().hex[:12]}"

    # Get workflow configuration
    settings = get_settings()
    workflow_config = settings.get_workflow_config("campaign_launch")
    timeout_minutes = get_workflow_timeout("campaign_launch") // 60

    # Initialize workflow state
    workflow_states[workflow_id] = {
        "workflow_id": workflow_id,
        "workflow_type": "campaign_launch",
        "status": "pending",
        "current_agent": None,
        "progress": 0.0,
        "started_at": None,
        "completed_at": None,
        "estimated_completion": datetime.now() + timedelta(minutes=timeout_minutes),
        "agents_executed": [],
        "state_transitions": [],
        "error": None,
        "request_data": request.dict(),
        "config": workflow_config,
    }

    # Start workflow in background
    background_tasks.add_task(
        execute_workflow_async,
        workflow_id,
        "campaign_launch",
        request.dict(),
        orchestrator,
        memory_manager,
    )

    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        workflow_type="campaign_launch",
        status="pending",
        current_agent=None,
        progress=0.0,
        started_at=datetime.now(),
        completed_at=None,
        estimated_completion=datetime.now() + timedelta(minutes=timeout_minutes),
        agents_executed=[],
        error=None,
    )


@router.post("/customer-support", response_model=WorkflowStatusResponse)
async def handle_support_inquiry(
    request: CustomerSupportRequest,
    background_tasks: BackgroundTasks,
    orchestrator=Depends(get_orchestrator),
    memory_manager=Depends(get_memory_manager),
):
    """
    Handle a customer support inquiry workflow.

    This workflow routes to the Customer Support Agent which can:
    - Search knowledge base for answers
    - Escalate to human support if needed
    - Learn from resolution patterns

    Args:
        request: Customer support request

    Returns:
        Workflow status with workflow_id for tracking
    """
    workflow_id = f"wf_{uuid.uuid4().hex[:12]}"

    # Initialize workflow state
    workflow_states[workflow_id] = {
        "workflow_id": workflow_id,
        "workflow_type": "customer_support",
        "status": "pending",
        "current_agent": None,
        "progress": 0.0,
        "started_at": None,
        "completed_at": None,
        "estimated_completion": datetime.now() + timedelta(minutes=2),
        "agents_executed": [],
        "state_transitions": [],
        "error": None,
        "request_data": request.dict(),
    }

    # Start workflow in background
    background_tasks.add_task(
        execute_workflow_async,
        workflow_id,
        "customer_support",
        request.dict(),
        orchestrator,
        memory_manager,
    )

    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        workflow_type="customer_support",
        status="pending",
        current_agent=None,
        progress=0.0,
        started_at=datetime.now(),
        completed_at=None,
        estimated_completion=datetime.now() + timedelta(minutes=2),
        agents_executed=[],
        error=None,
    )


@router.post("/analytics", response_model=WorkflowStatusResponse)
async def generate_analytics_report(
    request: AnalyticsRequest,
    background_tasks: BackgroundTasks,
    orchestrator=Depends(get_orchestrator),
    memory_manager=Depends(get_memory_manager),
):
    """
    Generate an analytics report workflow.

    This workflow uses the Analytics Evaluation Agent to:
    - Analyze campaign performance data
    - Generate insights and recommendations
    - Create visualizations and reports

    Args:
        request: Analytics report request

    Returns:
        Workflow status with workflow_id for tracking
    """
    workflow_id = f"wf_{uuid.uuid4().hex[:12]}"

    # Initialize workflow state
    workflow_states[workflow_id] = {
        "workflow_id": workflow_id,
        "workflow_type": "analytics",
        "status": "pending",
        "current_agent": None,
        "progress": 0.0,
        "started_at": None,
        "completed_at": None,
        "estimated_completion": datetime.now() + timedelta(minutes=3),
        "agents_executed": [],
        "state_transitions": [],
        "error": None,
        "request_data": request.dict(),
    }

    # Start workflow in background
    background_tasks.add_task(
        execute_workflow_async,
        workflow_id,
        "analytics",
        request.dict(),
        orchestrator,
        memory_manager,
    )

    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        workflow_type="analytics",
        status="pending",
        current_agent=None,
        progress=0.0,
        started_at=datetime.now(),
        completed_at=None,
        estimated_completion=datetime.now() + timedelta(minutes=3),
        agents_executed=[],
        error=None,
    )


@router.post("/feedback-learning", response_model=WorkflowStatusResponse)
async def process_feedback_learning(
    request: FeedbackLearningRequest,
    background_tasks: BackgroundTasks,
    orchestrator=Depends(get_orchestrator),
    memory_manager=Depends(get_memory_manager),
):
    """
    Process feedback and learning workflow.

    This workflow uses the Feedback Learning Agent to:
    - Collect and analyze feedback from users and agents
    - Identify patterns and improvement opportunities
    - Generate recommendations for system optimization
    - Investigate recurring issues

    Args:
        request: Feedback learning request

    Returns:
        Workflow status with workflow_id for tracking
    """
    workflow_id = f"wf_{uuid.uuid4().hex[:12]}"

    # Initialize workflow state
    workflow_states[workflow_id] = {
        "workflow_id": workflow_id,
        "workflow_type": "feedback_learning",
        "status": "pending",
        "current_agent": None,
        "progress": 0.0,
        "started_at": None,
        "completed_at": None,
        "estimated_completion": datetime.now() + timedelta(minutes=2),
        "agents_executed": [],
        "state_transitions": [],
        "error": None,
        "request_data": request.dict(),
    }

    # Start workflow in background
    background_tasks.add_task(
        execute_workflow_async,
        workflow_id,
        "feedback_learning",
        request.dict(),
        orchestrator,
        memory_manager,
    )

    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        workflow_type="feedback_learning",
        status="pending",
        current_agent=None,
        progress=0.0,
        started_at=datetime.now(),
        completed_at=None,
        estimated_completion=datetime.now() + timedelta(minutes=2),
        agents_executed=[],
        error=None,
    )


@router.get("", response_model=Dict[str, Any])
async def list_workflows(
    status: str = None,
    workflow_type: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    List all workflows with optional filtering.

    Args:
        status: Filter by workflow status (pending, in_progress, completed, failed)
        workflow_type: Filter by workflow type (campaign_launch, customer_support, analytics)
        limit: Maximum number of workflows to return
        offset: Number of workflows to skip

    Returns:
        List of workflows with metadata
    """
    # Filter workflows
    filtered_workflows = []

    for workflow_id, state in workflow_states.items():
        # Apply status filter
        if status and state.get("status") != status:
            continue

        # Apply workflow_type filter
        if workflow_type and state.get("workflow_type") != workflow_type:
            continue

        # Build workflow summary
        workflow_summary = {
            "workflow_id": workflow_id,
            "workflow_type": state.get("workflow_type"),
            "status": state.get("status"),
            "current_agent": state.get("current_agent"),
            "progress": state.get("progress", 0.0),
            "started_at": state.get("started_at"),
            "completed_at": state.get("completed_at"),
            "estimated_completion": state.get("estimated_completion"),
            "agents_executed": state.get("agents_executed", []),
            "state_transitions": state.get("state_transitions", []),
            "error": state.get("error"),
        }

        filtered_workflows.append(workflow_summary)

    # Sort by started_at (most recent first)
    filtered_workflows.sort(
        key=lambda x: x.get("started_at") or datetime.min, reverse=True
    )

    # Apply pagination
    total_count = len(filtered_workflows)
    paginated_workflows = filtered_workflows[offset : offset + limit]

    return {
        "workflows": paginated_workflows,
        "total": total_count,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """
    Get the current status of a workflow.

    Args:
        workflow_id: The workflow identifier

    Returns:
        Current workflow status
    """
    if workflow_id not in workflow_states:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found",
        )

    state = workflow_states[workflow_id]

    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        workflow_type=state["workflow_type"],
        status=state["status"],
        current_agent=state.get("current_agent"),
        progress=state.get("progress", 0.0),
        started_at=state.get("started_at", datetime.now()),
        completed_at=state.get("completed_at"),
        estimated_completion=state.get("estimated_completion"),
        agents_executed=state.get("agents_executed", []),
        error=state.get("error"),
    )


@router.get("/{workflow_id}/results", response_model=WorkflowResultResponse)
async def get_workflow_results(workflow_id: str):
    """
    Get the final results of a completed workflow.

    Args:
        workflow_id: The workflow identifier

    Returns:
        Workflow results
    """
    if workflow_id not in workflow_states:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found",
        )

    state = workflow_states[workflow_id]

    if state["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow is still {state['status']}. Results not available yet.",
        )

    result = state.get("result", {})
    started_at = state.get("started_at", datetime.now())
    completed_at = state.get("completed_at", datetime.now())
    duration = (completed_at - started_at).total_seconds()

    # Build execution trace
    execution_trace = []
    if result and "execution_summary" in result:
        execution_trace = result.get("execution_summary", {}).get("execution_trace", [])

    # Extract results by agent
    results_by_agent = {}
    if result:
        if "final_result" in result:
            results_by_agent = result["final_result"]
        elif "result" in result:
            results_by_agent = result["result"]

    # Ensure results is a dictionary (never None)
    if results_by_agent is None or not isinstance(results_by_agent, dict):
        results_by_agent = {}

    # For failed workflows, include error information
    summary = "Workflow execution completed"
    if state["status"] == "failed":
        error_msg = state.get("error", "Unknown error")
        summary = f"Workflow failed: {error_msg}"
        if not results_by_agent:
            results_by_agent = {"error": error_msg}
    elif result:
        summary = result.get("summary", summary)

    return WorkflowResultResponse(
        workflow_id=workflow_id,
        workflow_type=state["workflow_type"],
        status=state["status"],
        results=results_by_agent,
        summary=summary,
        execution_trace=execution_trace,
        started_at=started_at,
        completed_at=completed_at,
        total_duration_seconds=duration,
    )
