"""
Core components for the multi-agent system.

Provides base classes, orchestration, handoffs, and communication infrastructure.
"""

from .base_agent import BaseAgent, AgentStatus, HandoffRequest
from .orchestrator import OrchestratorAgent
from .handoff_manager import HandoffManager, HandoffStatus
from .message_bus import MessageBus
from .prompt_manager import PromptManager
from .state import (
    AgentState,
    AgentExecutionRecord,
    create_initial_state,
    update_state_with_result,
    check_workflow_complete,
    record_agent_start,
    record_agent_error,
    set_handoff,
    set_final_result,
    get_agent_results,
    get_workflow_duration,
    get_execution_summary,
)
from .graph_builder import WorkflowGraphBuilder, create_workflow_graph

__all__ = [
    "BaseAgent",
    "AgentStatus",
    "HandoffRequest",
    "OrchestratorAgent",
    "HandoffManager",
    "HandoffStatus",
    "MessageBus",
    "PromptManager",
    "AgentState",
    "AgentExecutionRecord",
    "create_initial_state",
    "update_state_with_result",
    "check_workflow_complete",
    "record_agent_start",
    "record_agent_error",
    "set_handoff",
    "set_final_result",
    "get_agent_results",
    "get_workflow_duration",
    "get_execution_summary",
    "WorkflowGraphBuilder",
    "create_workflow_graph",
]
