"""
Handoff Manager - Coordinates agent-to-agent handoffs.

Manages the lifecycle of handoff requests, validates handoffs,
and ensures smooth transitions between agents.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio


class HandoffStatus(Enum):
    """Handoff request states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class HandoffManager:
    """
    Manages handoffs between agents in the multi-agent system.

    Responsibilities:
    - Validate handoff requests
    - Route requests to target agents
    - Track handoff lifecycle
    - Handle timeouts and failures
    - Maintain handoff history
    """

    def __init__(
        self, timeout_seconds: int = 300, max_retries: int = 3, message_bus=None
    ):
        """
        Initialize handoff manager.

        Args:
            timeout_seconds: Maximum time for handoff completion
            max_retries: Maximum retry attempts for failed handoffs
            message_bus: Event bus for communication
        """
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.message_bus = message_bus

        self.logger = logging.getLogger("handoff_manager")

        # Active handoffs
        self.active_handoffs: Dict[str, Dict] = {}

        # Handoff history
        self.history: List[Dict] = []

        # Agent registry (maps agent_id -> agent instance)
        self.agent_registry: Dict[str, Any] = {}

        # Handoff callbacks
        self.on_handoff_completed: Optional[Callable] = None
        self.on_handoff_failed: Optional[Callable] = None

    def register_agent(self, agent_id: str, agent_instance: Any) -> None:
        """
        Register an agent for handoff routing.

        Args:
            agent_id: Unique agent identifier
            agent_instance: Agent instance
        """
        self.agent_registry[agent_id] = agent_instance
        self.logger.info(f"Registered agent: {agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self.agent_registry:
            del self.agent_registry[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")

    async def request_handoff(
        self,
        from_agent: str,
        to_agent: str,
        reason: str,
        context: Dict[str, Any],
        priority: str = "medium",
    ) -> str:
        """
        Create and process a handoff request.

        Args:
            from_agent: Source agent ID
            to_agent: Target agent ID
            reason: Reason for handoff
            context: Context data to pass
            priority: Priority level (low, medium, high)

        Returns:
            Handoff request ID
        """
        # Validate agents exist
        if to_agent not in self.agent_registry:
            raise ValueError(f"Target agent '{to_agent}' not registered")

        # Create handoff record
        handoff_id = f"handoff_{datetime.utcnow().timestamp()}"
        handoff_record = {
            "id": handoff_id,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "reason": reason,
            "context": context,
            "priority": priority,
            "status": HandoffStatus.PENDING.value,
            "created_at": datetime.utcnow(),
            "attempts": 0,
            "errors": [],
        }

        self.active_handoffs[handoff_id] = handoff_record

        self.logger.info(
            f"Handoff requested: {from_agent} -> {to_agent} (ID: {handoff_id})"
        )

        # Publish event
        if self.message_bus:
            await self.message_bus.publish(
                channel="handoff.requests",
                message={
                    "event": "handoff_requested",
                    "handoff_id": handoff_id,
                    **handoff_record,
                },
            )

        # Execute handoff asynchronously
        asyncio.create_task(self._execute_handoff(handoff_id))

        return handoff_id

    async def _execute_handoff(self, handoff_id: str) -> None:
        """
        Execute a handoff request.

        Args:
            handoff_id: Handoff request ID
        """
        if handoff_id not in self.active_handoffs:
            self.logger.error(f"Handoff {handoff_id} not found")
            return

        handoff = self.active_handoffs[handoff_id]
        handoff["status"] = HandoffStatus.IN_PROGRESS.value
        handoff["started_at"] = datetime.utcnow()

        target_agent_id = handoff["to_agent"]
        target_agent = self.agent_registry.get(target_agent_id)

        if not target_agent:
            await self._handle_handoff_failure(
                handoff_id, f"Target agent {target_agent_id} not available"
            )
            return

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                target_agent.process(handoff["context"]), timeout=self.timeout_seconds
            )

            await self._handle_handoff_success(handoff_id, result)

        except asyncio.TimeoutError:
            await self._handle_handoff_timeout(handoff_id)
        except Exception as e:
            await self._handle_handoff_failure(handoff_id, str(e))

    async def _handle_handoff_success(
        self, handoff_id: str, result: Dict[str, Any]
    ) -> None:
        """Handle successful handoff completion."""
        if handoff_id not in self.active_handoffs:
            return

        handoff = self.active_handoffs[handoff_id]
        handoff["status"] = HandoffStatus.COMPLETED.value
        handoff["completed_at"] = datetime.utcnow()
        handoff["result"] = result
        handoff["duration"] = (
            handoff["completed_at"] - handoff["started_at"]
        ).total_seconds()

        self.logger.info(f"Handoff {handoff_id} completed successfully")

        # Move to history
        self.history.append(handoff)
        del self.active_handoffs[handoff_id]

        # Publish event
        if self.message_bus:
            await self.message_bus.publish(
                channel="handoff.completed",
                message={
                    "event": "handoff_completed",
                    "handoff_id": handoff_id,
                    "result": result,
                },
            )

        # Call callback if registered
        if self.on_handoff_completed:
            await self.on_handoff_completed(handoff)

    async def _handle_handoff_failure(self, handoff_id: str, error: str) -> None:
        """Handle handoff failure with retry logic."""
        if handoff_id not in self.active_handoffs:
            return

        handoff = self.active_handoffs[handoff_id]
        handoff["attempts"] += 1
        handoff["errors"].append(
            {"timestamp": datetime.utcnow().isoformat(), "error": error}
        )

        self.logger.error(
            f"Handoff {handoff_id} failed (attempt {handoff['attempts']}): {error}"
        )

        # Retry if attempts remain
        if handoff["attempts"] < self.max_retries:
            self.logger.info(f"Retrying handoff {handoff_id}")
            handoff["status"] = HandoffStatus.PENDING.value
            await asyncio.sleep(2 ** handoff["attempts"])  # Exponential backoff
            await self._execute_handoff(handoff_id)
        else:
            # Max retries exceeded
            handoff["status"] = HandoffStatus.FAILED.value
            handoff["failed_at"] = datetime.utcnow()

            self.logger.error(f"Handoff {handoff_id} failed permanently")

            # Move to history
            self.history.append(handoff)
            del self.active_handoffs[handoff_id]

            # Publish event
            if self.message_bus:
                await self.message_bus.publish(
                    channel="handoff.failed",
                    message={
                        "event": "handoff_failed",
                        "handoff_id": handoff_id,
                        "errors": handoff["errors"],
                    },
                )

            # Call callback if registered
            if self.on_handoff_failed:
                await self.on_handoff_failed(handoff)

    async def _handle_handoff_timeout(self, handoff_id: str) -> None:
        """Handle handoff timeout."""
        await self._handle_handoff_failure(
            handoff_id, f"Handoff timed out after {self.timeout_seconds} seconds"
        )

    def get_handoff_status(self, handoff_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a handoff request.

        Args:
            handoff_id: Handoff request ID

        Returns:
            Handoff status dictionary or None if not found
        """
        if handoff_id in self.active_handoffs:
            return self.active_handoffs[handoff_id].copy()

        # Check history
        for handoff in reversed(self.history):
            if handoff["id"] == handoff_id:
                return handoff.copy()

        return None

    def get_active_handoffs(self) -> List[Dict[str, Any]]:
        """Get all active handoff requests."""
        return list(self.active_handoffs.values())

    def get_handoff_history(
        self, limit: int = 100, agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get handoff history.

        Args:
            limit: Maximum number of records to return
            agent_id: Filter by agent ID (from or to)

        Returns:
            List of handoff records
        """
        history = self.history

        if agent_id:
            history = [
                h
                for h in history
                if h["from_agent"] == agent_id or h["to_agent"] == agent_id
            ]

        return history[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get handoff performance metrics.

        Returns:
            Dictionary of metrics
        """
        total_handoffs = len(self.history)

        if total_handoffs == 0:
            return {
                "total_handoffs": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "active_count": len(self.active_handoffs),
            }

        completed = [
            h for h in self.history if h["status"] == HandoffStatus.COMPLETED.value
        ]
        failed = [h for h in self.history if h["status"] == HandoffStatus.FAILED.value]

        avg_duration = 0.0
        if completed:
            avg_duration = sum(h.get("duration", 0) for h in completed) / len(completed)

        return {
            "total_handoffs": total_handoffs,
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": (
                len(completed) / total_handoffs if total_handoffs > 0 else 0.0
            ),
            "avg_duration": avg_duration,
            "active_count": len(self.active_handoffs),
        }

    def clear_history(self, days_to_keep: int = 7) -> int:
        """
        Clear old handoff history.

        Args:
            days_to_keep: Number of days of history to retain

        Returns:
            Number of records cleared
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        old_count = len(self.history)
        self.history = [h for h in self.history if h["created_at"] > cutoff_date]

        cleared = old_count - len(self.history)
        self.logger.info(f"Cleared {cleared} old handoff records")

        return cleared
