"""
Formatting utilities for the marketing agents system.

Provides formatting functions for timestamps, numbers, text,
and other data types used in the application.
"""

from datetime import datetime, timedelta
from typing import Optional, Any
import json


def format_timestamp(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """
    Format datetime object as string.

    Args:
        dt: Datetime object to format
        format_str: Format string

    Returns:
        Formatted timestamp string
    """
    return dt.strftime(format_str)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds as human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "2h 30m 15s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"

    duration = timedelta(seconds=seconds)
    parts = []

    hours = duration.seconds // 3600
    if hours > 0:
        parts.append(f"{hours}h")

    minutes = (duration.seconds % 3600) // 60
    if minutes > 0:
        parts.append(f"{minutes}m")

    secs = duration.seconds % 60
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_number(number: float, precision: int = 2, use_comma: bool = True) -> str:
    """
    Format number with specified precision and comma separator.

    Args:
        number: Number to format
        precision: Decimal places
        use_comma: Whether to use comma as thousands separator

    Returns:
        Formatted number string
    """
    if use_comma:
        return f"{number:,.{precision}f}"
    return f"{number:.{precision}f}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_dict_as_json(data: dict, indent: int = 2, sort_keys: bool = True) -> str:
    """
    Format dictionary as pretty-printed JSON.

    Args:
        data: Dictionary to format
        indent: Indentation spaces
        sort_keys: Whether to sort keys

    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent, sort_keys=sort_keys)


def format_list_as_string(
    items: list, separator: str = ", ", conjunction: str = "and"
) -> str:
    """
    Format list as human-readable string.

    Args:
        items: List of items to format
        separator: Separator between items
        conjunction: Conjunction word for last item

    Returns:
        Formatted string (e.g., "item1, item2, and item3")
    """
    if not items:
        return ""
    if len(items) == 1:
        return str(items[0])
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"

    return f"{separator.join(str(i) for i in items[:-1])}{separator}{conjunction} {items[-1]}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename
    """
    import re

    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")
