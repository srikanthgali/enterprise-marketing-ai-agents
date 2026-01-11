"""
Message Bus - Event-driven communication between agents.

Provides pub/sub messaging for inter-agent communication,
event broadcasting, and system-wide notifications.
"""

from typing import Dict, List, Callable, Any, Optional
import asyncio
import logging
from datetime import datetime


class MessageBus:
    """
    Event-driven message bus for agent communication.

    Implements publish-subscribe pattern for:
    - Agent events
    - Handoff requests
    - Workflow notifications
    - System alerts
    """

    def __init__(self):
        """Initialize message bus."""
        self.logger = logging.getLogger("message_bus")

        # Channel subscriptions: {channel: [callbacks]}
        self.subscriptions: Dict[str, List[Callable]] = {}

        # Message history for debugging
        self.message_history: List[Dict] = []

        # Statistics
        self.stats = {
            "messages_published": 0,
            "messages_delivered": 0,
            "active_channels": 0,
        }

    def subscribe(
        self, channel: str, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to a channel.

        Args:
            channel: Channel name
            callback: Async function to call when message received
        """
        if channel not in self.subscriptions:
            self.subscriptions[channel] = []
            self.stats["active_channels"] = len(self.subscriptions)

        self.subscriptions[channel].append(callback)
        self.logger.info(f"New subscription to channel: {channel}")

    def unsubscribe(self, channel: str, callback: Callable) -> None:
        """
        Unsubscribe from a channel.

        Args:
            channel: Channel name
            callback: Callback to remove
        """
        if channel in self.subscriptions:
            if callback in self.subscriptions[channel]:
                self.subscriptions[channel].remove(callback)
                self.logger.info(f"Unsubscribed from channel: {channel}")

    async def publish(self, channel: str, message: Dict[str, Any]) -> None:
        """
        Publish message to a channel.

        Args:
            channel: Channel name
            message: Message data
        """
        self.stats["messages_published"] += 1

        # Add metadata
        enriched_message = {
            **message,
            "_channel": channel,
            "_timestamp": datetime.utcnow().isoformat(),
        }

        # Store in history
        self.message_history.append(enriched_message)
        if len(self.message_history) > 1000:
            self.message_history = self.message_history[-1000:]

        # Deliver to subscribers
        if channel in self.subscriptions:
            callbacks = self.subscriptions[channel]
            self.logger.debug(f"Publishing to {channel}: {len(callbacks)} subscribers")

            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(enriched_message)
                    else:
                        callback(enriched_message)

                    self.stats["messages_delivered"] += 1

                except Exception as e:
                    self.logger.error(f"Error delivering message to subscriber: {e}")

    def get_channels(self) -> List[str]:
        """Get list of active channels."""
        return list(self.subscriptions.keys())

    def get_subscriber_count(self, channel: str) -> int:
        """Get number of subscribers for a channel."""
        return len(self.subscriptions.get(channel, []))

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return self.stats.copy()

    def get_recent_messages(
        self, channel: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """
        Get recent messages.

        Args:
            channel: Filter by channel (None for all)
            limit: Maximum number of messages

        Returns:
            List of messages
        """
        messages = self.message_history

        if channel:
            messages = [m for m in messages if m.get("_channel") == channel]

        return messages[-limit:]


from typing import Optional  # Add to imports at top
