"""
Common utilities for the marketing agents system.

Provides helper functions, validators, formatters, and other
shared functionality used across the application.
"""

from .logging import get_logger, log_execution
from .validators import (
    validate_email,
    validate_url,
    validate_json,
    validate_agent_config,
)
from .formatters import (
    format_timestamp,
    format_duration,
    format_number,
    truncate_text,
)
from .exceptions import (
    AgentError,
    HandoffError,
    ConfigurationError,
    ValidationError,
)

__all__ = [
    # Logging
    "get_logger",
    "log_execution",
    # Validators
    "validate_email",
    "validate_url",
    "validate_json",
    "validate_agent_config",
    # Formatters
    "format_timestamp",
    "format_duration",
    "format_number",
    "truncate_text",
    # Exceptions
    "AgentError",
    "HandoffError",
    "ConfigurationError",
    "ValidationError",
    "MemoryError",
    "ToolExecutionError",
    "TimeoutError",
    "MessageBusError",
]
