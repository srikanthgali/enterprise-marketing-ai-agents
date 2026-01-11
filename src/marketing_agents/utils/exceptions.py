"""
Custom exceptions for the marketing agents system.

Provides domain-specific exceptions for better error handling
and debugging throughout the application.
"""


class AgentError(Exception):
    """Base exception for agent-related errors."""

    def __init__(self, message: str, agent_id: str = None, **kwargs):
        self.agent_id = agent_id
        self.details = kwargs
        super().__init__(message)


class HandoffError(AgentError):
    """Exception raised during agent handoff failures."""

    def __init__(
        self, message: str, from_agent: str = None, to_agent: str = None, **kwargs
    ):
        self.from_agent = from_agent
        self.to_agent = to_agent
        super().__init__(message, **kwargs)


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""

    def __init__(self, message: str, config_key: str = None, **kwargs):
        self.config_key = config_key
        self.details = kwargs
        super().__init__(message)


class ValidationError(Exception):
    """Exception raised for validation failures."""

    def __init__(self, message: str, field: str = None, value: any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class MemoryError(AgentError):
    """Exception raised for memory-related errors."""

    pass


class ToolExecutionError(AgentError):
    """Exception raised during tool execution failures."""

    def __init__(self, message: str, tool_name: str = None, **kwargs):
        self.tool_name = tool_name
        super().__init__(message, **kwargs)


class TimeoutError(AgentError):
    """Exception raised when operations timeout."""

    def __init__(self, message: str, timeout_seconds: int = None, **kwargs):
        self.timeout_seconds = timeout_seconds
        super().__init__(message, **kwargs)


class MessageBusError(Exception):
    """Exception raised for message bus errors."""

    def __init__(self, message: str, channel: str = None, **kwargs):
        self.channel = channel
        self.details = kwargs
        super().__init__(message)
