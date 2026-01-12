"""
Memory Management for Marketing Agents.

Provides comprehensive memory management including:
- Short-term memory (session-scoped)
- Long-term memory (persistent)
- Shared knowledge base
- Session context management
- Conversation history tracking
- Execution record persistence
"""

from .memory_manager import MemoryManager, InMemoryBackend, RedisBackend, MemoryBackend
from .session import SessionContext, create_session
from .utils import create_memory_manager, get_redis_client, get_memory_config

__all__ = [
    "MemoryManager",
    "MemoryBackend",
    "InMemoryBackend",
    "RedisBackend",
    "SessionContext",
    "create_session",
    "create_memory_manager",
    "get_redis_client",
    "get_memory_config",
]
