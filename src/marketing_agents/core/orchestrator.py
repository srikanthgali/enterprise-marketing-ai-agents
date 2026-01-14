"""
Orchestrator Agent - Central coordinator for the multi-agent system.

Manages agent lifecycle, routes tasks, coordinates handoffs,
and monitors overall system health and performance.
"""

from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage

from .base_agent import BaseAgent, AgentStatus
from .handoff_manager import HandoffManager
from .message_bus import MessageBus
from .state import (
    AgentState,
    create_initial_state,
    update_state_with_result,
    set_handoff,
    set_final_result,
    get_execution_summary,
)
from .graph_builder import WorkflowGraphBuilder


class OrchestratorAgent(BaseAgent):
    """
    Central orchestrator that coordinates all agents in the system.

    Responsibilities:
    - Route incoming requests to appropriate agents
    - Manage multi-agent workflows
    - Handle agent failures and fallbacks
    - Monitor system performance
    - Coordinate inter-agent handoffs
    """

    def __init__(
        self,
        config: Dict[str, Any],
        memory_manager=None,
        message_bus: Optional[MessageBus] = None,
    ):
        super().__init__(
            agent_id="orchestrator",
            name="Orchestrator Agent",
            description="Central coordinator managing agent lifecycle and workflows",
            config=config,
            memory_manager=memory_manager,
            message_bus=message_bus,
        )

        self.handoff_manager = HandoffManager(
            timeout_seconds=config.get("handoff_timeout", 300),
            max_retries=config.get("max_retries", 3),
            message_bus=message_bus,
        )

        # Registry of available agents
        self.agents: Dict[str, BaseAgent] = {}

        # Active workflows
        self.active_workflows: Dict[str, Dict] = {}

        # System health metrics
        self.health_metrics = {
            "agents_active": 0,
            "workflows_running": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
        }

        # LangGraph workflow - will be initialized after agents are registered
        self.workflow_graph = None

    def _initialize_workflow_graph(self) -> None:
        """Initialize the LangGraph workflow after agents are registered."""
        if len(self.agents) == 0:
            self.logger.warning(
                "No agents registered yet. Workflow graph will be initialized after agent registration."
            )
            return

        # Create agent node executors
        agent_nodes = {
            "orchestrator": lambda state: asyncio.run(self.orchestrator_node(state)),
            "marketing_strategy": lambda state: asyncio.run(
                self.marketing_strategy_node(state)
            ),
            "customer_support": lambda state: asyncio.run(
                self.customer_support_node(state)
            ),
            "analytics_evaluation": lambda state: asyncio.run(
                self.analytics_evaluation_node(state)
            ),
            "feedback_learning": lambda state: asyncio.run(
                self.feedback_learning_node(state)
            ),
        }

        try:
            builder = WorkflowGraphBuilder(agent_nodes)
            self.workflow_graph = builder.build()
            self.logger.info("LangGraph workflow initialized successfully")
        except Exception as e:
            self.logger.error(
                f"Failed to initialize workflow graph: {e}", exc_info=True
            )
            raise

    def _register_tools(self) -> None:
        """Register orchestrator-specific tools."""
        self.register_tool("route_task", self._route_task)
        self.register_tool("get_agent_status", self._get_agent_status)
        self.register_tool("execute_workflow", self._execute_workflow)
        self.register_tool("monitor_health", self._monitor_health)

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the orchestrator.

        Args:
            agent: Agent instance to register
        """
        self.agents[agent.agent_id] = agent
        self.handoff_manager.register_agent(agent.agent_id, agent)
        self.health_metrics["agents_active"] = len(self.agents)

        # Log registration with prompt info
        prompt_status = (
            "with system prompt" if agent.system_prompt else "without system prompt"
        )
        self.logger.info(f"Registered agent: {agent.agent_id} {prompt_status}")

        # Re-initialize workflow graph if we have all required agents
        required_agents = {
            "marketing_strategy",
            "customer_support",
            "analytics_evaluation",
            "feedback_learning",
        }
        registered_agents = set(self.agents.keys())
        if required_agents.issubset(registered_agents) and self.workflow_graph is None:
            self._initialize_workflow_graph()

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming request using LangGraph workflow.

        Args:
            input_data: Request data containing task type and parameters

        Returns:
            Processing result
        """
        self.status = AgentStatus.PROCESSING
        self.health_metrics["total_requests"] += 1
        start_time = datetime.utcnow()

        try:
            task_type = input_data.get("task_type")
            request_data = input_data.get("data", {})

            self.logger.info(f"Processing request via LangGraph workflow: {task_type}")

            # Execute via LangGraph workflow
            result = await self.execute_workflow(task_type, request_data)

            self.health_metrics["successful_requests"] += 1
            self.status = AgentStatus.IDLE

            duration = (datetime.utcnow() - start_time).total_seconds()

            return {
                "success": True,
                "workflow_id": result.get("workflow_id"),
                "result": result.get("final_result"),
                "execution_summary": result.get("execution_summary"),
                "duration": duration,
            }

        except Exception as e:
            self.logger.error(f"Request processing failed: {e}", exc_info=True)
            self.health_metrics["failed_requests"] += 1
            self.status = AgentStatus.ERROR

            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def _determine_target_agent(self, task_type: str, data: Dict[str, Any]) -> str:
        """
        Determine which agent should handle the task.

        Args:
            task_type: Type of task
            data: Task data

        Returns:
            Target agent ID
        """
        # Routing logic based on task type
        routing_map = {
            "campaign_planning": "marketing_strategy",
            "content_strategy": "marketing_strategy",
            "market_analysis": "marketing_strategy",
            "customer_inquiry": "customer_support",
            "support_ticket": "customer_support",
            "performance_analysis": "analytics_evaluation",
            "generate_report": "analytics_evaluation",
            "system_optimization": "feedback_learning",
            "model_tuning": "feedback_learning",
        }

        target = routing_map.get(task_type, "marketing_strategy")
        self.logger.info(f"Routing {task_type} -> {target}")

        return target

    async def _route_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route task to appropriate agent."""
        return await self.process(task)

    async def execute_workflow(
        self, task_type: str, task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a multi-agent workflow using LangGraph.

        Args:
            task_type: Type of task to execute
            task_data: Input data for the task

        Returns:
            Dict containing workflow results and execution summary
        """
        if self.workflow_graph is None:
            raise RuntimeError(
                "Workflow graph not initialized. Ensure all required agents are registered."
            )

        # Create initial state
        initial_state = create_initial_state(task_type, task_data)
        workflow_id = initial_state["workflow_id"]

        self.logger.info(f"Starting workflow {workflow_id} for task: {task_type}")

        # Track active workflow
        self.active_workflows[workflow_id] = {
            "task_type": task_type,
            "status": "running",
            "started_at": initial_state["started_at"],
            "steps_completed": 0,
        }
        self.health_metrics["workflows_running"] = len(self.active_workflows)

        try:
            # Execute workflow through LangGraph with increased recursion limit
            # to handle complex multi-agent handoff scenarios
            final_state = await self.workflow_graph.ainvoke(
                initial_state, config={"recursion_limit": 50}
            )

            # Update workflow tracking
            self.active_workflows[workflow_id]["status"] = "completed"
            self.active_workflows[workflow_id]["steps_completed"] = len(
                final_state["execution_history"]
            )

            # Generate execution summary
            execution_summary = get_execution_summary(final_state)

            self.logger.info(
                f"Workflow {workflow_id} completed. Status: {execution_summary['status']}"
            )

            return {
                "workflow_id": workflow_id,
                "final_result": final_state["final_result"],
                "execution_summary": execution_summary,
                "error": final_state.get("error"),
            }

        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}", exc_info=True)
            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["error"] = str(e)

            # Return error information instead of raising
            return {
                "workflow_id": workflow_id,
                "final_result": None,
                "execution_summary": {
                    "status": "failed",
                    "error": str(e),
                    "agents_executed": [],
                    "total_steps": 0,
                },
                "error": str(e),
            }

        finally:
            # Clean up completed workflow
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            self.health_metrics["workflows_running"] = len(self.active_workflows)

    async def orchestrator_node(self, state: AgentState) -> AgentState:
        """
        Orchestrator node - entry point for workflow routing.

        Args:
            state: Current workflow state

        Returns:
            Updated state with routing decision
        """
        self.logger.info(
            f"Orchestrator node processing workflow {state['workflow_id']}"
        )

        # Orchestrator just routes - actual processing happens in target agent nodes
        # State is already set up with task_type for routing
        return state

    async def marketing_strategy_node(self, state: AgentState) -> AgentState:
        """Execute marketing strategy agent and update state."""
        return await self._execute_agent_node(state, "marketing_strategy")

    async def customer_support_node(self, state: AgentState) -> AgentState:
        """Execute customer support agent and update state."""
        return await self._execute_agent_node(state, "customer_support")

    async def analytics_evaluation_node(self, state: AgentState) -> AgentState:
        """Execute analytics evaluation agent and update state."""
        return await self._execute_agent_node(state, "analytics_evaluation")

    async def feedback_learning_node(self, state: AgentState) -> AgentState:
        """Execute feedback learning agent and update state."""
        return await self._execute_agent_node(state, "feedback_learning")

    async def _execute_agent_node(self, state: AgentState, agent_id: str) -> AgentState:
        """
        Common execution logic for agent nodes.

        Args:
            state: Current workflow state
            agent_id: ID of the agent to execute

        Returns:
            Updated state with agent results
        """
        self.logger.info(f"Executing agent node: {agent_id}")

        # Get agent from registry
        if agent_id not in self.agents:
            error_msg = f"Agent '{agent_id}' not found in registry"
            self.logger.error(error_msg)
            state["error"] = error_msg
            state["next_action"] = "error"
            return state

        # Record agent start in workflow state
        from .state import record_agent_start

        state = record_agent_start(state, agent_id)

        agent = self.agents[agent_id]
        execution_start = datetime.utcnow()

        # Check if this agent was the target of a handoff
        was_handoff_target = (
            state.get("handoff_required") and state.get("target_agent") == agent_id
        )

        try:
            # Execute agent with task data
            result = await agent.process(state["task_data"])
            execution_end = datetime.utcnow()
            duration = (execution_end - execution_start).total_seconds()

            # Record execution in agent's history
            execution_record = {
                "execution_id": str(id(result)),
                "timestamp": execution_start,
                "task_type": state.get("task_type", "unknown"),
                "status": "completed" if result.get("success", True) else "failed",
                "duration_seconds": duration,
                "summary": result.get("summary", "Task completed"),
            }
            agent.execution_history.append(execution_record)

            # Add result message to conversation
            if state["messages"]:
                messages = list(state["messages"])
            else:
                messages = []

            messages.append(
                AIMessage(
                    content=f"Agent {agent_id} completed: {result.get('summary', 'Task completed')}",
                    name=agent_id,
                )
            )

            # Update state with results
            updated_state = update_state_with_result(state, agent_id, result)
            updated_state["messages"] = messages

            # Prefer explicit handoff signals returned by the agent itself.
            # Several agents (e.g., customer_support) attach `handoff_required` and
            # `target_agent` directly to their response payload.
            if (
                isinstance(result, dict)
                and result.get("handoff_required")
                and result.get("target_agent")
            ):
                to_agent = str(result["target_agent"])
                reason = str(result.get("handoff_reason") or "handoff_requested")
                context = result.get("context")
                if not isinstance(context, dict):
                    context = result

                self.logger.info(
                    f"Handoff requested from {agent_id} to {to_agent}: {reason} (explicit)"
                )

                await self.handoff_manager.request_handoff(
                    from_agent=agent_id,
                    to_agent=to_agent,
                    reason=reason,
                    context=context,
                )

                handoff_state = set_handoff(updated_state, to_agent, reason)

                # Update task_data for the target agent using context from the source agent
                # This ensures the target agent receives the handoff context, not the original task data
                if context and isinstance(context, dict):
                    # We create a new task_data dictionary merging original data with context
                    # Context takes precedence
                    new_task_data = updated_state.get("task_data", {}).copy()
                    new_task_data.update(context)
                    handoff_state["task_data"] = new_task_data

                    # Update task_type if provided in context
                    if "type" in context:
                        handoff_state["task_type"] = context["type"]

                self.logger.info(
                    f"Created handoff state: next_action={handoff_state.get('next_action')}, "
                    f"handoff_required={handoff_state.get('handoff_required')}, "
                    f"target_agent={handoff_state.get('target_agent')}"
                )
                return handoff_state

            # If this agent was the target of a handoff, clear the handoff flags NOW
            # (after state update, not before)
            if was_handoff_target:
                self.logger.info(
                    f"Clearing handoff flags - {agent_id} processed handed-off task"
                )
                updated_state["handoff_required"] = False
                updated_state["target_agent"] = None

            # Check for handoffs
            handoff = agent.should_handoff(result)
            if handoff:
                self.logger.info(
                    f"Handoff requested from {agent_id} to {handoff.to_agent}: {handoff.reason}"
                )

                # Notify handoff manager
                await self.handoff_manager.request_handoff(
                    from_agent=agent_id,
                    to_agent=handoff.to_agent,
                    reason=handoff.reason,
                    context=handoff.context,
                )

                # Update state for handoff
                updated_state = set_handoff(
                    updated_state, handoff.to_agent, handoff.reason
                )
            else:
                # No handoff - check if this completes the workflow
                if result.get("is_final", False):
                    updated_state = set_final_result(updated_state, result)
                    self.logger.info(
                        f"Workflow {state['workflow_id']} marked complete by {agent_id}"
                    )
                else:
                    # Return to orchestrator for next routing decision
                    updated_state["next_action"] = "continue"

            return updated_state

        except Exception as e:
            error_msg = f"Error executing {agent_id}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # Create a proper error state update
            error_state = dict(state)
            error_state["error"] = error_msg
            error_state["next_action"] = "error"
            error_state["current_agent"] = agent_id
            return error_state

    async def _get_agent_status(self, agent_id: Optional[str] = None) -> Dict:
        """Get status of specific agent or all agents."""
        if agent_id:
            if agent_id in self.agents:
                return self.agents[agent_id].get_status()
            return {"error": f"Agent {agent_id} not found"}

        return {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}

    async def _execute_workflow(
        self, workflow_name: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Backward compatibility wrapper for execute_workflow.

        Args:
            workflow_name: Name/type of workflow (used as task_type)
            input_data: Input data for the workflow

        Returns:
            Workflow execution results
        """
        return await self.execute_workflow(workflow_name, input_data)

    async def _monitor_health(self) -> Dict[str, Any]:
        """Monitor system health and return metrics."""
        health_data = self.health_metrics.copy()
        health_data["handoff_metrics"] = self.handoff_manager.get_metrics()
        health_data["timestamp"] = datetime.utcnow().isoformat()

        return health_data

    def get_workflow_history(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the execution history for a specific workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Workflow history or None if not found
        """
        return self.active_workflows.get(workflow_id)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "orchestrator_status": self.get_status(),
            "agents": {
                agent_id: agent.get_status() for agent_id, agent in self.agents.items()
            },
            "health_metrics": self.health_metrics,
            "active_workflows": len(self.active_workflows),
            "handoff_metrics": self.handoff_manager.get_metrics(),
            "workflow_graph_initialized": self.workflow_graph is not None,
        }
