"""
Base Agent Class - Foundation for all marketing agents.

Provides common functionality for agent lifecycle, memory management,
tool execution, and inter-agent communication.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import uuid

from langchain_openai import ChatOpenAI

from config.settings import get_settings
from src.marketing_agents.core.prompt_manager import PromptManager
from src.marketing_agents.utils import get_logger, log_execution
from src.marketing_agents.utils.exceptions import (
    AgentError,
    ValidationError,
    ToolExecutionError,
)


class AgentStatus(Enum):
    """Agent lifecycle states."""

    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_HANDOFF = "waiting_handoff"
    ERROR = "error"
    STOPPED = "stopped"


class HandoffRequest:
    """Represents a request to hand off to another agent."""

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        reason: str,
        context: Dict[str, Any],
        priority: str = "medium",
    ):
        self.id = str(uuid.uuid4())
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.reason = reason
        self.context = context
        self.priority = priority
        self.timestamp = datetime.utcnow()
        self.status = "pending"

    def to_dict(self) -> Dict[str, Any]:
        """Convert handoff request to dictionary."""
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "reason": self.reason,
            "context": self.context,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all marketing agents.

    Provides core functionality including:
    - Agent lifecycle management
    - Memory operations
    - Tool execution
    - Inter-agent communication
    - Event publishing
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        config: Dict[str, Any],
        memory_manager=None,
        message_bus=None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        """
        Initialize base agent.

        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Agent's purpose and capabilities
            config: Agent configuration dictionary
            memory_manager: Shared memory manager instance
            message_bus: Event bus for inter-agent communication
            prompt_manager: PromptManager instance (auto-created if None)
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.config = config
        self.memory_manager = memory_manager
        self.message_bus = message_bus

        self.status = AgentStatus.IDLE
        self.tools: Dict[str, Callable] = {}
        self.capabilities: List[str] = config.get("capabilities", [])
        self.handoff_rules: List[Dict] = config.get("handoff_rules", [])

        # Use centralized logger with agent context
        self.logger = get_logger(__name__, agent_id=agent_id)

        # Load settings
        self.settings = get_settings()

        self.execution_history: List[Dict] = []

        # Initialize PromptManager (create if not provided)
        self.prompt_manager = prompt_manager if prompt_manager else PromptManager()

        # Load system prompt
        self.system_prompt: Optional[str] = None
        try:
            self.system_prompt = self.prompt_manager.load_prompt(agent_id)
            self.logger.info(f"Loaded system prompt for agent: {agent_id}")

            # Validate prompt structure
            self._validate_prompt()

        except FileNotFoundError:
            self.logger.warning(
                f"No prompt file found for agent: {agent_id}. "
                "Agent will operate without a system prompt."
            )
        except Exception as e:
            self.logger.error(f"Failed to load prompt: {e}")

        # Initialize LLM with system prompt
        self.llm: Optional[ChatOpenAI] = None
        self._initialize_llm()

        self._register_tools()

        self.logger.info(f"Initialized agent: {agent_id}")

    @abstractmethod
    def _register_tools(self) -> None:
        """Register agent-specific tools. Must be implemented by subclasses."""
        pass

    def _initialize_llm(self) -> None:
        """
        Initialize the LLM with current configuration and system prompt.
        """
        try:
            model_config = self.config.get("model", {})
            model_name = model_config.get("name", "gpt-4o-mini")
            temperature = model_config.get("temperature", 0.7)

            # Create LLM (system prompt will be used in messages)
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
            )

            if self.system_prompt:
                self.logger.info(f"Initialized LLM ({model_name}) with system prompt")
            else:
                self.logger.info(
                    f"Initialized LLM ({model_name}) without system prompt"
                )

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None

    def _validate_prompt(self) -> None:
        """
        Validate that the system prompt contains required sections.
        Logs warnings if sections are missing.
        """
        if not self.system_prompt:
            return

        required_sections = [
            "ROLE DEFINITION",
            "CAPABILITIES",
            "OUTPUT FORMAT",
        ]

        recommended_sections = [
            "CONSTRAINTS",
            "EXAMPLES",
            "HANDOFF",
        ]

        prompt_upper = self.system_prompt.upper()

        # Check required sections
        missing_required = []
        for section in required_sections:
            if section not in prompt_upper:
                missing_required.append(section)

        if missing_required:
            self.logger.warning(
                f"Prompt for {self.agent_id} missing REQUIRED sections: {missing_required}"
            )

        # Check recommended sections
        missing_recommended = []
        for section in recommended_sections:
            if section not in prompt_upper:
                missing_recommended.append(section)

        if missing_recommended:
            self.logger.info(
                f"Prompt for {self.agent_id} missing recommended sections: {missing_recommended}"
            )

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and generate output. Must be implemented by subclasses.

        Args:
            input_data: Input data to process

        Returns:
            Processing results
        """
        pass

    def register_tool(self, tool_name: str, tool_func: Callable) -> None:
        """Register a tool that the agent can use."""
        self.tools[tool_name] = tool_func
        self.logger.info(f"Registered tool: {tool_name}")

    @log_execution()
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a registered tool.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If tool execution fails
        """
        if tool_name not in self.tools:
            raise ToolExecutionError(
                f"Tool '{tool_name}' not found in agent {self.agent_id}",
                agent_id=self.agent_id,
                tool_name=tool_name,
            )

        self.logger.info(f"Executing tool: {tool_name}")
        try:
            result = await self.tools[tool_name](**kwargs)
            self.logger.info(f"Tool {tool_name} executed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Tool {tool_name} execution failed: {e}", exc_info=True)
            raise ToolExecutionError(
                f"Tool execution failed: {str(e)}",
                agent_id=self.agent_id,
                tool_name=tool_name,
            )

    def save_to_memory(
        self, key: str, value: Any, memory_type: str = "short_term"
    ) -> None:
        """Save data to agent's memory."""
        if self.memory_manager:
            self.memory_manager.save(
                agent_id=self.agent_id, key=key, value=value, memory_type=memory_type
            )
            self.logger.debug(f"Saved to {memory_type} memory: {key}")

    def retrieve_from_memory(
        self, key: str, memory_type: str = "short_term"
    ) -> Optional[Any]:
        """Retrieve data from agent's memory."""
        if self.memory_manager:
            value = self.memory_manager.retrieve(
                agent_id=self.agent_id, key=key, memory_type=memory_type
            )
            self.logger.debug(f"Retrieved from {memory_type} memory: {key}")
            return value
        return None

    def should_handoff(self, context: Dict[str, Any]) -> Optional[HandoffRequest]:
        """
        Determine if handoff to another agent is needed.

        Args:
            context: Current execution context

        Returns:
            HandoffRequest if handoff needed, None otherwise
        """
        for rule in self.handoff_rules:
            trigger = rule.get("trigger")
            target = rule.get("target")
            conditions = rule.get("conditions", [])

            # Check if any condition matches
            if any(cond in str(context) for cond in conditions):
                self.logger.info(f"Handoff triggered: {trigger} -> {target}")
                return HandoffRequest(
                    from_agent=self.agent_id,
                    to_agent=target,
                    reason=trigger,
                    context=context,
                )

        return None

    async def request_handoff(self, handoff: HandoffRequest) -> None:
        """Request handoff to another agent."""
        self.status = AgentStatus.WAITING_HANDOFF

        if self.message_bus:
            await self.message_bus.publish(
                channel="handoff.requests", message=handoff.to_dict()
            )

        self.logger.info(f"Handoff requested: {self.agent_id} -> {handoff.to_agent}")

    async def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish event to message bus."""
        if self.message_bus:
            event = {
                "event_type": event_type,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
            }
            await self.message_bus.publish(channel="agent.events", message=event)
            self.logger.debug(f"Published event: {event_type}")

    def log_execution(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        duration: float,
        success: bool,
    ) -> None:
        """Log execution details for analysis."""
        execution_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "input": input_data,
            "output": output_data,
            "duration": duration,
            "success": success,
        }
        self.execution_history.append(execution_record)

        # Keep only recent history
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "tools_count": len(self.tools),
            "executions_count": len(self.execution_history),
        }

    def reset(self) -> None:
        """Reset agent state."""
        self.status = AgentStatus.IDLE
        self.execution_history = []
        self.logger.info(f"Agent {self.agent_id} reset")

    def reload_prompt(self, version: str = "latest") -> bool:
        """
        Reload the system prompt from disk without restarting the agent.

        Args:
            version: Version to load (default: 'latest')

        Returns:
            True if prompt was reloaded successfully, False otherwise
        """
        try:
            if version == "latest":
                self.system_prompt = self.prompt_manager.reload_prompt(self.agent_id)
            else:
                self.system_prompt = self.prompt_manager.load_prompt(
                    self.agent_id, version
                )

            self.logger.info(
                f"Reloaded system prompt for agent: {self.agent_id} (version: {version})"
            )

            # Validate reloaded prompt
            self._validate_prompt()

            # Reinitialize LLM with new prompt
            self._initialize_llm()

            return True
        except Exception as e:
            self.logger.error(f"Failed to reload prompt: {e}")
            return False

    def get_system_prompt(self) -> Optional[str]:
        """
        Get the current system prompt.

        Returns:
            The system prompt string, or None if not loaded
        """
        return self.system_prompt

    async def start(self) -> None:
        """Start the agent."""
        self.status = AgentStatus.IDLE
        await self.publish_event("agent.started", {"agent_id": self.agent_id})
        self.logger.info(f"Agent {self.agent_id} started")

    async def stop(self) -> None:
        """Stop the agent."""
        self.status = AgentStatus.STOPPED
        await self.publish_event("agent.stopped", {"agent_id": self.agent_id})
        self.logger.info(f"Agent {self.agent_id} stopped")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.agent_id}, status={self.status.value})>"
