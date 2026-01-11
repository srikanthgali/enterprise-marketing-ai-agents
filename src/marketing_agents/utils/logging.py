"""
Centralized Logging Utility.

Provides structured logging with support for multiple handlers,
log rotation, and context-aware logging for the agent system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json
from functools import wraps
import traceback

from config.settings import get_settings


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "agent_id"):
            log_data["agent_id"] = record.agent_id
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data)


class AgentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds agent context to log records."""

    def process(self, msg, kwargs):
        """Add agent context to log message."""
        # Add extra context from adapter
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"].update(self.extra)
        return msg, kwargs


class LoggerManager:
    """Centralized logger manager for the application."""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern for logger manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger manager."""
        if not self._initialized:
            self.settings = get_settings()
            self._setup_root_logger()
            self._initialized = True

    def _setup_root_logger(self) -> None:
        """Set up root logger with handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.settings.logging.level)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Console handler
        if self.settings.logging.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.settings.logging.level)
            console_formatter = logging.Formatter(self.settings.logging.format)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # File handler with rotation
        if self.settings.logging.file_enabled:
            log_file = Path(self.settings.logging.file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=self.settings.logging.file_max_bytes,
                backupCount=self.settings.logging.file_backup_count,
            )
            file_handler.setLevel(self.settings.logging.level)

            # Use structured formatter for file logs
            if self.settings.system.environment == "production":
                file_formatter = StructuredFormatter()
            else:
                file_formatter = logging.Formatter(self.settings.logging.format)

            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

    def get_logger(
        self,
        name: str,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> logging.Logger:
        """
        Get a logger instance with optional context.

        Args:
            name: Logger name
            agent_id: Optional agent identifier
            task_id: Optional task identifier

        Returns:
            Logger instance
        """
        logger = logging.getLogger(name)

        # Add context if provided
        if agent_id or task_id:
            extra = {}
            if agent_id:
                extra["agent_id"] = agent_id
            if task_id:
                extra["task_id"] = task_id
            logger = AgentLoggerAdapter(logger, extra)

        return logger

    def set_level(self, level: str) -> None:
        """
        Set logging level for all handlers.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = getattr(logging, level.upper())
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        for handler in root_logger.handlers:
            handler.setLevel(log_level)


def get_logger(
    name: str,
    agent_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        agent_id: Optional agent identifier
        task_id: Optional task identifier

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__, agent_id="marketing_strategy")
        >>> logger.info("Processing task", extra={"task_type": "campaign"})
    """
    manager = LoggerManager()
    return manager.get_logger(name, agent_id, task_id)


def log_execution(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution with timing and error handling.

    Args:
        logger: Optional logger instance

    Example:
        >>> @log_execution()
        >>> async def process_task(data):
        >>>     return result
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            start_time = datetime.utcnow()

            func_logger.info(
                f"Starting execution: {func.__name__}",
                extra={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                },
            )

            try:
                result = await func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()

                func_logger.info(
                    f"Completed execution: {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "duration_seconds": duration,
                        "success": True,
                    },
                )

                return result

            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()

                func_logger.error(
                    f"Failed execution: {func.__name__}",
                    exc_info=True,
                    extra={
                        "function": func.__name__,
                        "duration_seconds": duration,
                        "success": False,
                        "error_type": type(e).__name__,
                    },
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            start_time = datetime.utcnow()

            func_logger.info(
                f"Starting execution: {func.__name__}",
                extra={"function": func.__name__},
            )

            try:
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()

                func_logger.info(
                    f"Completed execution: {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "duration_seconds": duration,
                        "success": True,
                    },
                )

                return result

            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()

                func_logger.error(
                    f"Failed execution: {func.__name__}",
                    exc_info=True,
                    extra={
                        "function": func.__name__,
                        "duration_seconds": duration,
                        "success": False,
                        "error_type": type(e).__name__,
                    },
                )
                raise

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
