"""
LangGraph State Definition - Multi-Agent System State Management.

Defines the shared state structure for the multi-agent workflow system,
including agent coordination, message history, and workflow tracking.
"""

from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4
import operator

from langchain_core.messages import BaseMessage


class AgentExecutionRecord(TypedDict):
    """Record of a single agent execution within a workflow."""

    agent_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    status: str  # 'running', 'completed', 'failed'
    result: Optional[Dict[str, Any]]
    error: Optional[str]


class AgentState(TypedDict):
    """
    Shared state for multi-agent workflow execution.

    This state is passed between agents in the LangGraph workflow and tracks
    the complete execution context, message history, and results.

    Fields:
        messages: Message history for LLM context (accumulated with operator.add)
        current_agent: ID of the currently active agent
        task_type: Type of task being executed (e.g., 'campaign_planning', 'content_creation')
        task_data: Input data and parameters for the task
        next_action: Next action to take in the workflow ('continue', 'handoff', 'complete', 'error')
        handoff_required: Whether a handoff to another agent is needed
        target_agent: ID of the agent to hand off to (if handoff_required is True)
        workflow_id: Unique identifier for this workflow execution
        started_at: Timestamp when workflow started
        execution_history: List of agent execution records in chronological order
        intermediate_results: Dictionary mapping agent_id to their results
        final_result: Final result of the workflow (None until complete)
        error: Error message if workflow failed
        retry_count: Number of times the current step has been retried
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    task_type: str
    task_data: Dict[str, Any]
    next_action: str
    handoff_required: bool
    target_agent: Optional[str]
    workflow_id: str
    started_at: datetime
    execution_history: List[AgentExecutionRecord]
    intermediate_results: Dict[str, Any]
    final_result: Optional[Dict[str, Any]]
    error: Optional[str]
    retry_count: int


def create_initial_state(task_type: str, task_data: Dict[str, Any]) -> AgentState:
    """
    Create an initial state for a new workflow execution.

    Args:
        task_type: Type of task to execute (e.g., 'campaign_planning', 'content_creation',
                  'seo_optimization', 'performance_analysis')
        task_data: Input data and parameters for the task

    Returns:
        AgentState: Initialized state ready for workflow execution

    Example:
        >>> state = create_initial_state(
        ...     task_type='campaign_planning',
        ...     task_data={
        ...         'product': 'Enterprise CDP',
        ...         'target_audience': 'B2B SaaS companies',
        ...         'campaign_goal': 'increase_awareness'
        ...     }
        ... )
    """
    return AgentState(
        messages=[],
        current_agent="",
        task_type=task_type,
        task_data=task_data,
        next_action="continue",
        handoff_required=False,
        target_agent=None,
        workflow_id=str(uuid4()),
        started_at=datetime.utcnow(),
        execution_history=[],
        intermediate_results={},
        final_result=None,
        error=None,
        retry_count=0,
    )


def update_state_with_result(
    state: AgentState, agent_id: str, result: Dict[str, Any]
) -> AgentState:
    """
    Update the workflow state with an agent's execution result.

    This function:
    1. Stores the result in intermediate_results
    2. Updates the execution history with completion status
    3. Resets retry count on successful execution

    Args:
        state: Current workflow state
        agent_id: ID of the agent that produced the result
        result: Result dictionary from the agent's execution

    Returns:
        AgentState: Updated state with new results

    Example:
        >>> result = {
        ...     'status': 'success',
        ...     'campaign_plan': {...},
        ...     'confidence': 0.95
        ... }
        >>> state = update_state_with_result(state, 'campaign_manager', result)
    """
    # Store the result
    updated_results = state["intermediate_results"].copy()
    updated_results[agent_id] = result

    # Update execution history
    updated_history = state["execution_history"].copy()
    for record in reversed(updated_history):
        if record["agent_id"] == agent_id and record["status"] == "running":
            record["status"] = "completed"
            record["completed_at"] = datetime.utcnow()
            record["result"] = result
            break

    return AgentState(
        messages=state["messages"],
        current_agent=state["current_agent"],
        task_type=state["task_type"],
        task_data=state["task_data"],
        next_action=state["next_action"],
        handoff_required=state["handoff_required"],
        target_agent=state["target_agent"],
        workflow_id=state["workflow_id"],
        started_at=state["started_at"],
        execution_history=updated_history,
        intermediate_results=updated_results,
        final_result=state["final_result"],
        error=state["error"],
        retry_count=0,  # Reset retry count on successful execution
    )


def record_agent_start(state: AgentState, agent_id: str) -> AgentState:
    """
    Record the start of an agent's execution in the workflow.

    Args:
        state: Current workflow state
        agent_id: ID of the agent starting execution

    Returns:
        AgentState: Updated state with new execution record
    """
    execution_record: AgentExecutionRecord = {
        "agent_id": agent_id,
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "status": "running",
        "result": None,
        "error": None,
    }

    updated_history = state["execution_history"].copy()
    updated_history.append(execution_record)

    return AgentState(
        messages=state["messages"],
        current_agent=agent_id,
        task_type=state["task_type"],
        task_data=state["task_data"],
        next_action=state["next_action"],
        handoff_required=state["handoff_required"],
        target_agent=state["target_agent"],
        workflow_id=state["workflow_id"],
        started_at=state["started_at"],
        execution_history=updated_history,
        intermediate_results=state["intermediate_results"],
        final_result=state["final_result"],
        error=state["error"],
        retry_count=state["retry_count"],
    )


def record_agent_error(
    state: AgentState, agent_id: str, error_message: str
) -> AgentState:
    """
    Record an error during agent execution.

    Args:
        state: Current workflow state
        agent_id: ID of the agent that encountered an error
        error_message: Description of the error

    Returns:
        AgentState: Updated state with error information
    """
    # Update execution history
    updated_history = state["execution_history"].copy()
    for record in reversed(updated_history):
        if record["agent_id"] == agent_id and record["status"] == "running":
            record["status"] = "failed"
            record["completed_at"] = datetime.utcnow()
            record["error"] = error_message
            break

    return AgentState(
        messages=state["messages"],
        current_agent=state["current_agent"],
        task_type=state["task_type"],
        task_data=state["task_data"],
        next_action="error",
        handoff_required=state["handoff_required"],
        target_agent=state["target_agent"],
        workflow_id=state["workflow_id"],
        started_at=state["started_at"],
        execution_history=updated_history,
        intermediate_results=state["intermediate_results"],
        final_result=state["final_result"],
        error=error_message,
        retry_count=state["retry_count"] + 1,
    )


def check_workflow_complete(state: AgentState) -> bool:
    """
    Check if the workflow has completed successfully.

    A workflow is considered complete when:
    1. next_action is 'complete'
    2. final_result is set
    3. There are no errors

    Args:
        state: Current workflow state

    Returns:
        bool: True if workflow is complete, False otherwise

    Example:
        >>> if check_workflow_complete(state):
        ...     print(f"Workflow {state['workflow_id']} completed!")
        ...     print(f"Result: {state['final_result']}")
    """
    return (
        state["next_action"] == "complete"
        and state["final_result"] is not None
        and state["error"] is None
    )


def set_handoff(state: AgentState, target_agent: str, reason: str = "") -> AgentState:
    """
    Configure the state for a handoff to another agent.

    Args:
        state: Current workflow state
        target_agent: ID of the agent to hand off to
        reason: Optional reason for the handoff

    Returns:
        AgentState: Updated state configured for handoff
    """
    return AgentState(
        messages=state["messages"],
        current_agent=state["current_agent"],
        task_type=state["task_type"],
        task_data=state["task_data"],
        next_action="handoff",
        handoff_required=True,
        target_agent=target_agent,
        workflow_id=state["workflow_id"],
        started_at=state["started_at"],
        execution_history=state["execution_history"],
        intermediate_results=state["intermediate_results"],
        final_result=state["final_result"],
        error=state["error"],
        retry_count=state["retry_count"],
    )


def set_final_result(state: AgentState, result: Dict[str, Any]) -> AgentState:
    """
    Set the final result and mark the workflow as complete.

    Args:
        state: Current workflow state
        result: Final result dictionary

    Returns:
        AgentState: Updated state with final result and complete status
    """
    return AgentState(
        messages=state["messages"],
        current_agent=state["current_agent"],
        task_type=state["task_type"],
        task_data=state["task_data"],
        next_action="complete",
        handoff_required=False,
        target_agent=None,
        workflow_id=state["workflow_id"],
        started_at=state["started_at"],
        execution_history=state["execution_history"],
        intermediate_results=state["intermediate_results"],
        final_result=result,
        error=None,
        retry_count=state["retry_count"],
    )


def get_agent_results(state: AgentState, agent_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the results from a specific agent's execution.

    Args:
        state: Current workflow state
        agent_id: ID of the agent whose results to retrieve

    Returns:
        Optional[Dict[str, Any]]: Agent's results or None if not found
    """
    return state["intermediate_results"].get(agent_id)


def get_workflow_duration(state: AgentState) -> float:
    """
    Calculate the total duration of the workflow in seconds.

    Args:
        state: Current workflow state

    Returns:
        float: Duration in seconds
    """
    return (datetime.utcnow() - state["started_at"]).total_seconds()


def get_execution_summary(state: AgentState) -> Dict[str, Any]:
    """
    Generate a summary of the workflow execution.

    Args:
        state: Current workflow state

    Returns:
        Dict[str, Any]: Summary including agents executed, duration, and status
    """
    agents_executed = [
        record["agent_id"]
        for record in state["execution_history"]
        if record["status"] == "completed"
    ]

    failed_agents = [
        record["agent_id"]
        for record in state["execution_history"]
        if record["status"] == "failed"
    ]

    return {
        "workflow_id": state["workflow_id"],
        "task_type": state["task_type"],
        "status": state["next_action"],
        "duration_seconds": get_workflow_duration(state),
        "agents_executed": agents_executed,
        "failed_agents": failed_agents,
        "total_retries": state["retry_count"],
        "has_error": state["error"] is not None,
        "is_complete": check_workflow_complete(state),
    }
