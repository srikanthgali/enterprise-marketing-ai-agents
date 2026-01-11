"""
Validation utilities for the marketing agents system.

Provides validation functions for various data types and formats
used throughout the application.
"""

import re
import json
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from .exceptions import ValidationError


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If email is invalid
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email):
        raise ValidationError(f"Invalid email format: {email}", field="email")
    return True


def validate_url(url: str, require_https: bool = False) -> bool:
    """
    Validate URL format.

    Args:
        url: URL to validate
        require_https: Whether to require HTTPS protocol

    Returns:
        True if valid

    Raises:
        ValidationError: If URL is invalid
    """
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValidationError(f"Invalid URL format: {url}", field="url")

        if require_https and result.scheme != "https":
            raise ValidationError(f"URL must use HTTPS protocol: {url}", field="url")

        return True
    except Exception as e:
        raise ValidationError(f"Invalid URL: {url} - {str(e)}", field="url")


def validate_json(data: str) -> Dict[str, Any]:
    """
    Validate and parse JSON string.

    Args:
        data: JSON string to validate

    Returns:
        Parsed JSON data

    Raises:
        ValidationError: If JSON is invalid
    """
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {str(e)}", field="json_data")


def validate_agent_config(config: Dict[str, Any]) -> bool:
    """
    Validate agent configuration structure.

    Args:
        config: Agent configuration dictionary

    Returns:
        True if valid

    Raises:
        ValidationError: If configuration is invalid
    """
    required_fields = ["id", "name", "description", "capabilities"]

    for field in required_fields:
        if field not in config:
            raise ValidationError(
                f"Missing required field in agent config: {field}", field=field
            )

    # Validate capabilities is a list
    if not isinstance(config["capabilities"], list):
        raise ValidationError("Agent capabilities must be a list", field="capabilities")

    return True


def validate_handoff_request(handoff: Dict[str, Any]) -> bool:
    """
    Validate handoff request structure.

    Args:
        handoff: Handoff request dictionary

    Returns:
        True if valid

    Raises:
        ValidationError: If handoff request is invalid
    """
    required_fields = ["from_agent", "to_agent", "reason", "context"]

    for field in required_fields:
        if field not in handoff:
            raise ValidationError(
                f"Missing required field in handoff request: {field}", field=field
            )

    return True


def validate_priority(priority: str) -> bool:
    """
    Validate priority level.

    Args:
        priority: Priority level string

    Returns:
        True if valid

    Raises:
        ValidationError: If priority is invalid
    """
    valid_priorities = ["low", "medium", "high", "urgent"]
    if priority.lower() not in valid_priorities:
        raise ValidationError(
            f"Invalid priority level: {priority}. Must be one of {valid_priorities}",
            field="priority",
            value=priority,
        )
    return True
