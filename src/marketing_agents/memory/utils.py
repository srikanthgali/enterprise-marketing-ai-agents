"""
Helper utilities for memory management initialization and configuration.
"""

import logging
from typing import Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


def get_redis_client():
    """
    Get Redis client if Redis is enabled in settings.

    Returns:
        Redis client instance or None
    """
    settings = get_settings()

    if not settings.memory.use_redis:
        logger.info("Redis disabled in memory settings")
        return None

    try:
        import redis

        redis_config = settings.redis

        # Create Redis client with connection pool
        client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password=(
                redis_config.password.get_secret_value()
                if redis_config.password
                else None
            ),
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )

        # Test connection
        client.ping()
        logger.info(
            f"Redis client connected to {redis_config.host}:{redis_config.port}"
        )
        return client

    except ImportError:
        logger.error(
            "Redis backend requested but 'redis' package not installed. "
            "Install with: pip install redis"
        )
        logger.info("Falling back to in-memory backend")
        return None

    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.info("Falling back to in-memory backend")
        return None


def create_memory_manager(redis_client=None):
    """
    Create a MemoryManager instance with proper configuration.

    Args:
        redis_client: Optional Redis client. If None, will be created based on settings.

    Returns:
        MemoryManager instance
    """
    from pathlib import Path
    from src.marketing_agents.memory import MemoryManager

    settings = get_settings()

    # Get Redis client if needed and not provided
    if redis_client is None and settings.memory.use_redis:
        redis_client = get_redis_client()

    # Create memory manager
    memory_manager = MemoryManager(
        storage_dir=Path(settings.memory.storage_dir),
        use_redis=settings.memory.use_redis and redis_client is not None,
        redis_client=redis_client,
        embeddings_dir=Path(settings.memory.vector_store_dir),
    )

    logger.info(
        f"MemoryManager created - Backend: "
        f"{'Redis' if (settings.memory.use_redis and redis_client) else 'In-Memory'}"
    )

    return memory_manager


def get_memory_config() -> dict:
    """
    Get memory configuration summary.

    Returns:
        Dictionary with memory configuration details
    """
    settings = get_settings()
    mem_settings = settings.memory

    return {
        "backend": {
            "type": mem_settings.backend_type,
            "use_redis": mem_settings.use_redis,
        },
        "storage": {
            "storage_dir": mem_settings.storage_dir,
            "execution_records_dir": mem_settings.execution_records_dir,
            "vector_store_dir": mem_settings.vector_store_dir,
            "vector_store_name": mem_settings.vector_store_name,
        },
        "ttl": {
            "short_term": mem_settings.short_term_ttl,
            "session_timeout": mem_settings.session_timeout,
            "search_cache": mem_settings.search_cache_ttl,
        },
        "session": {
            "auto_cleanup": mem_settings.auto_cleanup,
            "max_conversation_history": mem_settings.max_conversation_history,
            "max_duration": mem_settings.max_session_duration,
            "persist_sessions": mem_settings.persist_sessions,
        },
        "vector_search": {
            "enabled": mem_settings.enable_vector_search,
            "embedding_model": mem_settings.embedding_model,
            "top_k": mem_settings.vector_search_top_k,
            "similarity_threshold": mem_settings.similarity_threshold,
            "reranking": mem_settings.enable_reranking,
        },
        "performance": {
            "cache_enabled": mem_settings.cache_enabled,
            "max_cache_size": mem_settings.max_cache_size,
            "cache_policy": mem_settings.cache_policy,
            "lazy_load_vector_store": mem_settings.lazy_load_vector_store,
        },
        "execution_records": {
            "enabled": mem_settings.execution_records_enabled,
            "format": mem_settings.execution_records_format,
            "max_per_file": mem_settings.max_records_per_file,
            "retention_days": mem_settings.retention_days,
        },
        "monitoring": {
            "enabled": mem_settings.monitoring_enabled,
            "stats_interval": mem_settings.stats_interval,
            "alerts": {
                "short_term_high": mem_settings.alert_short_term_high,
                "active_sessions_high": mem_settings.alert_active_sessions_high,
            },
        },
    }
