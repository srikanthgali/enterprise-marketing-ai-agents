"""
Example: Using Memory Manager with YAML Configuration

Demonstrates how the memory manager uses settings from memory_config.yaml
and automatically configures Redis backend based on environment.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from src.marketing_agents.memory import (
    create_memory_manager,
    create_session,
    get_memory_config,
    get_redis_client,
)


def example_1_config_loading():
    """Example 1: View loaded configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Configuration Loading from YAML")
    print("=" * 70 + "\n")

    settings = get_settings()

    print(f"Environment: {settings.system.environment}")
    print(f"Backend Type: {settings.memory.backend_type}")
    print(f"Use Redis: {settings.memory.use_redis}")
    print(f"Storage Dir: {settings.memory.storage_dir}")
    print(f"Vector Search Enabled: {settings.memory.enable_vector_search}")
    print(f"Embedding Model: {settings.memory.embedding_model}")
    print(f"Short-term TTL: {settings.memory.short_term_ttl}s")
    print(f"Session Timeout: {settings.memory.session_timeout}s")
    print(f"Auto Cleanup: {settings.memory.auto_cleanup}")
    print(f"Max Conversation History: {settings.memory.max_conversation_history}")

    print("\n📋 Full Memory Configuration:")
    config = get_memory_config()
    import json

    print(json.dumps(config, indent=2))


def example_2_automatic_initialization():
    """Example 2: Automatic initialization with config."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Automatic Initialization")
    print("=" * 70 + "\n")

    # Create memory manager - automatically uses config
    print("1. Creating memory manager from config...")
    memory_manager = create_memory_manager()

    stats = memory_manager.get_stats()
    print(f"\n✓ Memory manager initialized")
    print(
        f"   Backend: {'Redis' if isinstance(memory_manager.short_term_backend.__class__.__name__, 'RedisBackend') else 'In-Memory'}"
    )
    print(f"   Storage dir: {stats['storage_dir']}")
    print(f"   Embeddings dir: {stats['embeddings_dir']}")

    # Test basic operations
    print("\n2. Testing basic operations...")
    memory_manager.save(
        agent_id="test_agent",
        key="test_key",
        value={"message": "Hello from YAML config!"},
        memory_type="short_term",
    )
    print("✓ Saved to short-term memory")

    value = memory_manager.retrieve(
        agent_id="test_agent",
        key="test_key",
        memory_type="short_term",
    )
    print(f"✓ Retrieved: {value}")


async def example_3_session_with_config():
    """Example 3: Session management using config settings."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Session Management with Config")
    print("=" * 70 + "\n")

    settings = get_settings()
    memory_manager = create_memory_manager()

    print(f"Session settings from config:")
    print(f"  Auto cleanup: {settings.memory.auto_cleanup}")
    print(f"  Max conversation history: {settings.memory.max_conversation_history}")
    print(f"  Session timeout: {settings.memory.session_timeout}s")
    print()

    # Create session - automatically uses config settings
    with create_session(
        workflow_id="config_demo_workflow",
        memory_manager=memory_manager,
        metadata={"source": "yaml_config"},
        auto_cleanup=settings.memory.auto_cleanup,  # From config
    ) as session:
        print("✓ Session created with config settings")

        # Add messages
        for i in range(3):
            session.add_conversation_message(
                agent_id="agent_001",
                role="assistant",
                content=f"Message {i+1}",
            )

        history = session.get_conversation_history(
            limit=settings.memory.max_conversation_history  # From config
        )
        print(f"✓ Added {len(history)} messages (respecting config limit)")

        summary = session.get_session_summary()
        print(f"✓ Session summary:")
        print(f"   Duration: {summary['duration_seconds']:.2f}s")
        print(f"   Active agents: {summary['active_agents']}")

    print("✓ Session auto-cleaned (as per config)")


def example_4_environment_specific():
    """Example 4: Environment-specific configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Environment-Specific Settings")
    print("=" * 70 + "\n")

    settings = get_settings()

    print("Configuration varies by environment:")
    print("\nCurrent Environment:", settings.system.environment)
    print("\nDevelopment typically uses:")
    print("  - Backend: In-Memory (faster for dev)")
    print("  - Auto cleanup: Enabled")
    print("  - Monitoring: Disabled")

    print("\nProduction typically uses:")
    print("  - Backend: Redis (distributed, persistent)")
    print("  - Auto cleanup: Enabled")
    print("  - Monitoring: Enabled")
    print("  - Session persistence: Enabled")
    print("  - Replication: Enabled")

    print("\nTo change environment, set:")
    print("  export ENVIRONMENT=production")


def example_5_redis_connection():
    """Example 5: Redis connection testing."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Redis Connection (if enabled)")
    print("=" * 70 + "\n")

    settings = get_settings()

    if settings.memory.use_redis:
        print(f"Redis is ENABLED in config")
        print(f"  Host: {settings.redis.host}")
        print(f"  Port: {settings.redis.port}")
        print(f"  DB: {settings.redis.db}")
        print()

        print("Attempting to connect to Redis...")
        redis_client = get_redis_client()

        if redis_client:
            print("✓ Redis connection successful")

            # Test Redis operations
            try:
                redis_client.set("test_key", "test_value")
                value = redis_client.get("test_key")
                print(f"✓ Redis test: set/get successful (value: {value})")
                redis_client.delete("test_key")
            except Exception as e:
                print(f"✗ Redis operation failed: {e}")
        else:
            print("✗ Redis connection failed")
            print("  Memory manager will fall back to in-memory backend")
    else:
        print("Redis is DISABLED in config")
        print("Using in-memory backend")


def example_6_monitoring_config():
    """Example 6: Monitoring configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Monitoring Configuration")
    print("=" * 70 + "\n")

    settings = get_settings()

    print("Monitoring settings from config:")
    print(f"  Enabled: {settings.memory.monitoring_enabled}")
    print(f"  Stats interval: {settings.memory.stats_interval}s")
    print(f"  Alert thresholds:")
    print(f"    - Short-term entries: {settings.memory.alert_short_term_high}")
    print(f"    - Active sessions: {settings.memory.alert_active_sessions_high}")

    if settings.memory.monitoring_enabled:
        memory_manager = create_memory_manager()
        stats = memory_manager.get_stats()

        print(f"\nCurrent stats:")
        print(f"  Short-term entries: {stats['short_term_entries']}")
        print(f"  Long-term entries: {stats['long_term_entries']}")
        print(f"  Active sessions: {stats['active_sessions']}")

        # Check against thresholds
        if stats["short_term_entries"] > settings.memory.alert_short_term_high:
            print(f"\n⚠️  Alert: Short-term entries above threshold!")
        else:
            print(f"\n✓ All metrics within thresholds")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MEMORY CONFIGURATION EXAMPLES")
    print("=" * 70)

    try:
        example_1_config_loading()
        example_2_automatic_initialization()
        await example_3_session_with_config()
        example_4_environment_specific()
        example_5_redis_connection()
        example_6_monitoring_config()

        print("\n" + "=" * 70)
        print("✓ All examples completed!")
        print("=" * 70)
        print("\nNote: To use Redis backend, ensure:")
        print("  1. Redis server is running: redis-server")
        print("  2. Config has backend.type: redis")
        print("  3. Redis connection settings are correct")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
