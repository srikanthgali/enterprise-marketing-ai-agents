"""
LangGraph Workflow Graph Builder - Multi-Agent System Orchestration.

Constructs the LangGraph StateGraph that orchestrates agent interactions,
manages workflow routing, and handles agent-to-agent handoffs.
"""

from typing import Dict, Any, Callable, Literal
import logging

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from .state import AgentState, record_agent_start, record_agent_error
from src.marketing_agents.utils import get_logger


class WorkflowGraphBuilder:
    """
    Builds and configures the LangGraph workflow for multi-agent coordination.

    This class constructs a StateGraph that:
    - Registers all agent nodes (orchestrator, marketing_strategy, customer_support,
      analytics_evaluation, feedback_learning)
    - Defines conditional routing between agents
    - Handles handoffs based on agent decisions
    - Manages workflow completion and error states
    """

    def __init__(self, agent_nodes: Dict[str, Callable[[AgentState], AgentState]]):
        """
        Initialize the workflow graph builder.

        Args:
            agent_nodes: Dictionary mapping agent_id to agent execution functions.
                        Each function takes AgentState and returns updated AgentState.
                        Example: {
                            'orchestrator': orchestrator_node,
                            'marketing_strategy': marketing_strategy_node,
                            ...
                        }
        """
        self.logger = get_logger(__name__)
        self.agent_nodes = agent_nodes
        self.graph = StateGraph(AgentState)

        # Validate required agents
        self._validate_agents()

        self.logger.info(
            "WorkflowGraphBuilder initialized with agents: %s", list(agent_nodes.keys())
        )

    def _validate_agents(self) -> None:
        """Validate that all required agents are provided."""
        required_agents = [
            "orchestrator",
            "marketing_strategy",
            "customer_support",
            "analytics_evaluation",
            "feedback_learning",
        ]

        missing_agents = [
            agent for agent in required_agents if agent not in self.agent_nodes
        ]

        if missing_agents:
            raise ValueError(
                f"Missing required agent nodes: {missing_agents}. "
                f"Provided: {list(self.agent_nodes.keys())}"
            )

    def _add_agent_nodes(self) -> None:
        """Register all agent nodes with the graph."""
        for agent_id, agent_func in self.agent_nodes.items():
            # Wrap agent function with error handling
            wrapped_func = self._wrap_agent_node(agent_id, agent_func)
            self.graph.add_node(agent_id, wrapped_func)
            self.logger.debug(f"Added node: {agent_id}")

    def _wrap_agent_node(
        self, agent_id: str, agent_func: Callable[[AgentState], AgentState]
    ) -> Callable[[AgentState], AgentState]:
        """
        Wrap agent function with error handling and logging.

        Args:
            agent_id: ID of the agent
            agent_func: Original agent function

        Returns:
            Wrapped function with error handling
        """

        def wrapped(state: AgentState) -> AgentState:
            try:
                self.logger.info(
                    f"Executing agent: {agent_id} (workflow: {state['workflow_id']})"
                )

                # Record agent start
                state = record_agent_start(state, agent_id)

                # Execute agent
                result_state = agent_func(state)

                self.logger.info(
                    f"Agent {agent_id} completed. Next action: {result_state['next_action']}"
                )

                return result_state

            except Exception as e:
                self.logger.error(
                    f"Error in agent {agent_id}: {str(e)}",
                    exc_info=True,
                )
                # Record error
                error_state = record_agent_error(state, agent_id, str(e))
                return error_state

        return wrapped

    def _add_conditional_edges(self) -> None:
        """Add conditional routing edges between agents."""
        # Orchestrator routes to all agents based on task_type
        self.graph.add_conditional_edges(
            "orchestrator",
            self.route_from_orchestrator,
            {
                "marketing_strategy": "marketing_strategy",
                "customer_support": "customer_support",
                "analytics_evaluation": "analytics_evaluation",
                "feedback_learning": "feedback_learning",
                "complete": END,
                "error": END,
            },
        )

        # Marketing Strategy can route to analytics, customer_support, or complete
        self.graph.add_conditional_edges(
            "marketing_strategy",
            self.route_from_marketing_strategy,
            {
                "analytics_evaluation": "analytics_evaluation",
                "customer_support": "customer_support",
                "feedback_learning": "feedback_learning",
                "orchestrator": "orchestrator",
                "complete": END,
                "error": END,
            },
        )

        # Customer Support can route to marketing_strategy, analytics, or complete
        self.graph.add_conditional_edges(
            "customer_support",
            self.route_from_customer_support,
            {
                "marketing_strategy": "marketing_strategy",
                "analytics_evaluation": "analytics_evaluation",
                "feedback_learning": "feedback_learning",
                "orchestrator": "orchestrator",
                "complete": END,
                "error": END,
            },
        )

        # Analytics can route to marketing_strategy, feedback_learning, or complete
        self.graph.add_conditional_edges(
            "analytics_evaluation",
            self.route_from_analytics,
            {
                "marketing_strategy": "marketing_strategy",
                "feedback_learning": "feedback_learning",
                "customer_support": "customer_support",
                "orchestrator": "orchestrator",
                "complete": END,
                "error": END,
            },
        )

        # Feedback Learning typically returns to orchestrator or completes
        self.graph.add_conditional_edges(
            "feedback_learning",
            self.route_from_feedback_learning,
            {
                "orchestrator": "orchestrator",
                "marketing_strategy": "marketing_strategy",
                "analytics_evaluation": "analytics_evaluation",
                "complete": END,
                "error": END,
            },
        )

        self.logger.debug("Conditional edges added successfully")

    def route_from_orchestrator(self, state: AgentState) -> Literal[
        "marketing_strategy",
        "customer_support",
        "analytics_evaluation",
        "feedback_learning",
        "complete",
        "error",
    ]:
        """
        Route from orchestrator based on task_type.

        Routing rules:
        - campaign_planning, content_strategy, brand_positioning → marketing_strategy
        - customer_inquiry, support_ticket, issue_resolution → customer_support
        - performance_analysis, metrics_evaluation, reporting → analytics_evaluation
        - system_improvement, model_tuning, optimization → feedback_learning

        Args:
            state: Current workflow state

        Returns:
            Next agent or workflow action
        """
        if state["error"]:
            self.logger.warning(f"Routing to error from orchestrator: {state['error']}")
            return "error"

        if state["next_action"] == "complete":
            self.logger.info("Workflow complete from orchestrator")
            return "complete"

        task_type = state["task_type"]

        # Marketing strategy tasks (including campaign launch)
        if task_type in [
            "campaign_planning",
            "campaign_launch",
            "content_strategy",
            "brand_positioning",
            "marketing_strategy",
        ]:
            self.logger.info(f"Routing to marketing_strategy for task: {task_type}")
            return "marketing_strategy"

        # Customer support tasks
        elif task_type in [
            "customer_inquiry",
            "customer_support_request",
            "support_ticket",
            "issue_resolution",
            "customer_support",
        ]:
            self.logger.info(f"Routing to customer_support for task: {task_type}")
            return "customer_support"

        # Analytics tasks
        elif task_type in [
            "performance_analysis",
            "metrics_evaluation",
            "reporting",
            "analytics",
        ]:
            self.logger.info(f"Routing to analytics_evaluation for task: {task_type}")
            return "analytics_evaluation"

        # Learning tasks
        elif task_type in [
            "system_improvement",
            "model_tuning",
            "optimization",
            "learning",
        ]:
            self.logger.info(f"Routing to feedback_learning for task: {task_type}")
            return "feedback_learning"

        else:
            self.logger.warning(f"Unknown task_type: {task_type}, routing to error")
            return "error"

    def route_from_marketing_strategy(self, state: AgentState) -> Literal[
        "analytics_evaluation",
        "customer_support",
        "feedback_learning",
        "orchestrator",
        "complete",
        "error",
    ]:
        """
        Route from marketing_strategy agent.

        Handoff rules (from agents_config.yaml):
        - strategy_validation_needed → analytics_evaluation
        - customer_insights_needed → customer_support
        - optimization_needed → feedback_learning
        - Otherwise returns to orchestrator or completes

        Args:
            state: Current workflow state

        Returns:
            Next agent or workflow action
        """
        if state["error"]:
            self.logger.warning(
                f"Routing to error from marketing_strategy: {state['error']}"
            )
            return "error"

        if state["next_action"] == "complete":
            self.logger.info("Workflow complete from marketing_strategy")
            return "complete"

        if state["handoff_required"] and state["target_agent"]:
            target = state["target_agent"]
            self.logger.info(f"Handoff from marketing_strategy to {target}")

            # Validate target
            valid_targets = [
                "analytics_evaluation",
                "customer_support",
                "feedback_learning",
            ]
            if target in valid_targets:
                return target  # type: ignore

            self.logger.warning(
                f"Invalid target agent: {target}, returning to orchestrator"
            )

        # Default: return to orchestrator
        self.logger.info("Marketing_strategy returning to orchestrator")
        return "orchestrator"

    def route_from_customer_support(self, state: AgentState) -> Literal[
        "marketing_strategy",
        "analytics_evaluation",
        "feedback_learning",
        "orchestrator",
        "complete",
        "error",
    ]:
        """
        Route from customer_support agent.

        Handoff rules (from agents_config.yaml):
        - strategic_insight_found → marketing_strategy
        - analytics_required → analytics_evaluation
        - system_improvement_identified → feedback_learning
        - Otherwise returns to orchestrator or completes

        Args:
            state: Current workflow state

        Returns:
            Next agent or workflow action
        """
        if state["error"]:
            self.logger.warning(
                f"Routing to error from customer_support: {state['error']}"
            )
            return "error"

        if state["next_action"] == "complete":
            self.logger.info("Workflow complete from customer_support")
            return "complete"

        if state["handoff_required"] and state["target_agent"]:
            target = state["target_agent"]
            self.logger.info(f"Handoff from customer_support to {target}")

            # Validate target
            valid_targets = [
                "marketing_strategy",
                "analytics_evaluation",
                "feedback_learning",
            ]
            if target in valid_targets:
                return target  # type: ignore

            self.logger.warning(
                f"Invalid target agent: {target}, returning to orchestrator"
            )

        # Default: return to orchestrator
        self.logger.info("Customer_support returning to orchestrator")
        return "orchestrator"

    def route_from_analytics(self, state: AgentState) -> Literal[
        "marketing_strategy",
        "feedback_learning",
        "customer_support",
        "orchestrator",
        "complete",
        "error",
    ]:
        """
        Route from analytics_evaluation agent.

        Handoff rules (from agents_config.yaml):
        - strategic_pivot_needed → marketing_strategy
        - learning_opportunity → feedback_learning
        - customer_issue_detected → customer_support
        - Otherwise returns to orchestrator or completes

        Args:
            state: Current workflow state

        Returns:
            Next agent or workflow action
        """
        if state["error"]:
            self.logger.warning(
                f"Routing to error from analytics_evaluation: {state['error']}"
            )
            return "error"

        if state["next_action"] == "complete":
            self.logger.info("Workflow complete from analytics_evaluation")
            return "complete"

        if state["handoff_required"] and state["target_agent"]:
            target = state["target_agent"]
            self.logger.info(f"Handoff from analytics_evaluation to {target}")

            # Validate target
            valid_targets = [
                "marketing_strategy",
                "feedback_learning",
                "customer_support",
            ]
            if target in valid_targets:
                return target  # type: ignore

            self.logger.warning(
                f"Invalid target agent: {target}, returning to orchestrator"
            )

        # Default: return to orchestrator
        self.logger.info("Analytics_evaluation returning to orchestrator")
        return "orchestrator"

    def route_from_feedback_learning(self, state: AgentState) -> Literal[
        "orchestrator",
        "marketing_strategy",
        "analytics_evaluation",
        "complete",
        "error",
    ]:
        """
        Route from feedback_learning agent.

        Handoff rules (from agents_config.yaml):
        - system_update_ready → orchestrator
        - strategic_learning → marketing_strategy
        - analysis_needed → analytics_evaluation
        - Otherwise completes

        Args:
            state: Current workflow state

        Returns:
            Next agent or workflow action
        """
        if state["error"]:
            self.logger.warning(
                f"Routing to error from feedback_learning: {state['error']}"
            )
            return "error"

        if state["next_action"] == "complete":
            self.logger.info("Workflow complete from feedback_learning")
            return "complete"

        if state["handoff_required"] and state["target_agent"]:
            target = state["target_agent"]
            self.logger.info(f"Handoff from feedback_learning to {target}")

            # Validate target
            valid_targets = [
                "orchestrator",
                "marketing_strategy",
                "analytics_evaluation",
            ]
            if target in valid_targets:
                return target  # type: ignore

            self.logger.warning(
                f"Invalid target agent: {target}, returning to orchestrator"
            )

        # Default: return to orchestrator
        self.logger.info("Feedback_learning returning to orchestrator")
        return "orchestrator"

    def build(self):
        """
        Build and compile the workflow graph.

        Returns:
            Compiled LangGraph that can be invoked with initial state

        Example:
            >>> builder = WorkflowGraphBuilder(agent_nodes)
            >>> graph = builder.build()
            >>> result = graph.invoke(initial_state)
        """
        try:
            # Add all agent nodes
            self._add_agent_nodes()

            # Set orchestrator as entry point
            self.graph.set_entry_point("orchestrator")
            self.logger.info("Entry point set to orchestrator")

            # Add conditional edges for routing
            self._add_conditional_edges()

            # Compile the graph
            compiled_graph = self.graph.compile()

            self.logger.info("Workflow graph compiled successfully")
            return compiled_graph

        except Exception as e:
            self.logger.error(
                f"Failed to build workflow graph: {str(e)}", exc_info=True
            )
            raise


def create_workflow_graph(agent_nodes: Dict[str, Callable[[AgentState], AgentState]]):
    """
    Convenience function to create and build a workflow graph.

    Args:
        agent_nodes: Dictionary mapping agent_id to agent execution functions

    Returns:
        Compiled LangGraph ready for execution

    Example:
        >>> graph = create_workflow_graph({
        ...     'orchestrator': orchestrator_node,
        ...     'marketing_strategy': marketing_strategy_node,
        ...     'customer_support': customer_support_node,
        ...     'analytics_evaluation': analytics_node,
        ...     'feedback_learning': feedback_node,
        ... })
        >>> result = graph.invoke(initial_state)
    """
    builder = WorkflowGraphBuilder(agent_nodes)
    return builder.build()
