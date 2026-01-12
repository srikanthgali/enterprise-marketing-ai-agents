"""
Session Context Management for Agent Workflows.

Provides workflow-scoped memory isolation and automatic cleanup.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SessionContext:
    """
    Session context for workflow-scoped memory management.

    Features:
    - Isolated memory per workflow_id
    - Automatic cleanup on session end
    - Agent activity tracking
    - Session state management
    """

    def __init__(
        self,
        workflow_id: str,
        memory_manager,
        metadata: Optional[Dict[str, Any]] = None,
        auto_cleanup: bool = True,
    ):
        """
        Initialize session context.

        Args:
            workflow_id: Unique workflow identifier
            memory_manager: MemoryManager instance
            metadata: Additional session metadata
            auto_cleanup: Auto-cleanup on session end
        """
        self.workflow_id = workflow_id
        self.memory_manager = memory_manager
        self.metadata = metadata or {}
        self.auto_cleanup = auto_cleanup

        # Session state
        self.started_at = datetime.utcnow()
        self.ended_at: Optional[datetime] = None
        self.is_active = True

        # Track agent activities
        self.agent_interactions: List[Dict[str, Any]] = []
        self.active_agents: List[str] = []

        # Create session in memory manager
        self.memory_manager.create_session(workflow_id, metadata)

        logger.info(f"SessionContext created for workflow {workflow_id}")

    def __enter__(self):
        """Context manager entry."""
        logger.debug(f"Entering SessionContext {self.workflow_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        logger.debug(f"Exiting SessionContext {self.workflow_id}")
        self.end_session()

        # Don't suppress exceptions
        return False

    def save_agent_memory(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: str = "short_term",
    ) -> bool:
        """
        Save data to agent's memory within this session.

        Args:
            agent_id: Agent identifier
            key: Memory key
            value: Value to store
            memory_type: "short_term" or "long_term"

        Returns:
            True if successful
        """
        if not self.is_active:
            logger.warning(
                f"Attempted to save memory in inactive session {self.workflow_id}"
            )
            return False

        # Add workflow_id prefix to key for isolation
        scoped_key = f"workflow_{self.workflow_id}_{key}"

        # Track agent activity
        if agent_id not in self.active_agents:
            self.active_agents.append(agent_id)

        self.agent_interactions.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "action": "save",
                "key": key,
                "memory_type": memory_type,
            }
        )

        return self.memory_manager.save(
            agent_id=agent_id,
            key=scoped_key,
            value=value,
            memory_type=memory_type,
        )

    def retrieve_agent_memory(
        self,
        agent_id: str,
        key: str,
        memory_type: str = "short_term",
    ) -> Optional[Any]:
        """
        Retrieve data from agent's memory within this session.

        Args:
            agent_id: Agent identifier
            key: Memory key
            memory_type: "short_term" or "long_term"

        Returns:
            Stored value or None
        """
        if not self.is_active:
            logger.warning(
                f"Attempted to retrieve memory from inactive session {self.workflow_id}"
            )
            return None

        # Add workflow_id prefix to key for isolation
        scoped_key = f"workflow_{self.workflow_id}_{key}"

        # Track agent activity
        if agent_id not in self.active_agents:
            self.active_agents.append(agent_id)

        self.agent_interactions.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "action": "retrieve",
                "key": key,
                "memory_type": memory_type,
            }
        )

        return self.memory_manager.retrieve(
            agent_id=agent_id,
            key=scoped_key,
            memory_type=memory_type,
        )

    def add_conversation_message(
        self,
        agent_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Add a message to this session's conversation history.

        Args:
            agent_id: Agent identifier
            role: Message role ("user", "assistant", "system")
            content: Message content
            metadata: Additional metadata

        Returns:
            True if successful
        """
        if not self.is_active:
            logger.warning(
                f"Attempted to add message to inactive session {self.workflow_id}"
            )
            return False

        # Track agent activity
        if agent_id not in self.active_agents:
            self.active_agents.append(agent_id)

        self.agent_interactions.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "action": "message",
                "role": role,
            }
        )

        return self.memory_manager.add_conversation_message(
            workflow_id=self.workflow_id,
            agent_id=agent_id,
            role=role,
            content=content,
            metadata=metadata,
        )

    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get conversation history for this session.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of conversation messages
        """
        return self.memory_manager.get_conversation_history(
            workflow_id=self.workflow_id,
            limit=limit,
        )

    def save_execution_record(
        self,
        agent_id: str,
        record: Dict[str, Any],
    ) -> str:
        """
        Save an execution record for this session.

        Args:
            agent_id: Agent identifier
            record: Execution details

        Returns:
            Record ID
        """
        # Add workflow_id to record
        record["workflow_id"] = self.workflow_id

        return self.memory_manager.save_execution_record(
            agent_id=agent_id,
            record=record,
        )

    def get_session_duration(self) -> timedelta:
        """Get session duration."""
        end_time = self.ended_at or datetime.utcnow()
        return end_time - self.started_at

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of session activity.

        Returns:
            Summary dictionary
        """
        duration = self.get_session_duration()

        return {
            "workflow_id": self.workflow_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": duration.total_seconds(),
            "is_active": self.is_active,
            "active_agents": self.active_agents,
            "total_interactions": len(self.agent_interactions),
            "conversation_messages": len(self.get_conversation_history()),
            "metadata": self.metadata,
        }

    def end_session(self) -> bool:
        """
        End the session and perform cleanup.

        Returns:
            True if successful
        """
        if not self.is_active:
            logger.warning(f"Session {self.workflow_id} already ended")
            return False

        try:
            self.ended_at = datetime.utcnow()
            self.is_active = False

            # Log session summary
            summary = self.get_session_summary()
            logger.info(
                f"Session {self.workflow_id} ended - "
                f"Duration: {summary['duration_seconds']:.1f}s, "
                f"Agents: {len(self.active_agents)}, "
                f"Interactions: {summary['total_interactions']}"
            )

            # Auto-cleanup if enabled
            if self.auto_cleanup:
                logger.debug(f"Auto-cleanup enabled for session {self.workflow_id}")
                return self.cleanup()

            return True

        except Exception as e:
            logger.error(f"Failed to end session {self.workflow_id}: {e}")
            return False

    def cleanup(self) -> bool:
        """
        Clean up session data (short-term memory, conversation history).

        Note: Long-term memory is preserved for learning.

        Returns:
            True if successful
        """
        try:
            success = self.memory_manager.clear_session(self.workflow_id)
            if success:
                logger.info(f"Cleaned up session {self.workflow_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to cleanup session {self.workflow_id}: {e}")
            return False


@contextmanager
def create_session(
    workflow_id: str,
    memory_manager,
    metadata: Optional[Dict[str, Any]] = None,
    auto_cleanup: bool = True,
):
    """
    Context manager for creating a session.

    Usage:
        with create_session(workflow_id, memory_manager) as session:
            session.save_agent_memory("agent1", "key", "value")
            session.add_conversation_message("agent1", "user", "Hello")

    Args:
        workflow_id: Unique workflow identifier
        memory_manager: MemoryManager instance
        metadata: Additional session metadata
        auto_cleanup: Auto-cleanup on session end

    Yields:
        SessionContext instance
    """
    session = SessionContext(
        workflow_id=workflow_id,
        memory_manager=memory_manager,
        metadata=metadata,
        auto_cleanup=auto_cleanup,
    )

    try:
        yield session
    finally:
        if session.is_active:
            session.end_session()
