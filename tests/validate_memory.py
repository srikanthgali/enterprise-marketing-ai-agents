"""Quick validation test for memory management implementation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.marketing_agents.memory import (
            MemoryManager,
            MemoryBackend,
            InMemoryBackend,
            RedisBackend,
            SessionContext,
            create_session,
        )

        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_memory_manager():
    """Test basic MemoryManager functionality."""
    print("\nTesting MemoryManager...")

    try:
        from src.marketing_agents.memory import MemoryManager

        # Initialize
        memory_manager = MemoryManager()
        print("✓ MemoryManager initialized")

        # Save short-term memory
        success = memory_manager.save(
            agent_id="test_agent",
            key="test_key",
            value="test_value",
            memory_type="short_term",
        )
        assert success, "Failed to save short-term memory"
        print("✓ Short-term memory saved")

        # Retrieve short-term memory
        value = memory_manager.retrieve(
            agent_id="test_agent",
            key="test_key",
            memory_type="short_term",
        )
        assert value == "test_value", f"Expected 'test_value', got {value}"
        print("✓ Short-term memory retrieved")

        # Save long-term memory
        success = memory_manager.save(
            agent_id="test_agent",
            key="long_term_key",
            value={"data": "persistent"},
            memory_type="long_term",
        )
        assert success, "Failed to save long-term memory"
        print("✓ Long-term memory saved")

        # Retrieve long-term memory
        value = memory_manager.retrieve(
            agent_id="test_agent",
            key="long_term_key",
            memory_type="long_term",
        )
        assert value == {"data": "persistent"}, f"Unexpected value: {value}"
        print("✓ Long-term memory retrieved")

        # Get stats
        stats = memory_manager.get_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        print(f"✓ Stats retrieved: {stats}")

        return True

    except Exception as e:
        print(f"✗ MemoryManager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_session_context():
    """Test SessionContext functionality."""
    print("\nTesting SessionContext...")

    try:
        from src.marketing_agents.memory import MemoryManager, create_session

        memory_manager = MemoryManager()

        # Create session
        with create_session(
            workflow_id="test_workflow",
            memory_manager=memory_manager,
            metadata={"test": True},
            auto_cleanup=False,
        ) as session:
            print("✓ Session created")

            # Save in session
            success = session.save_agent_memory(
                agent_id="test_agent",
                key="session_key",
                value="session_value",
            )
            assert success, "Failed to save in session"
            print("✓ Session memory saved")

            # Retrieve from session
            value = session.retrieve_agent_memory(
                agent_id="test_agent",
                key="session_key",
            )
            assert value == "session_value", f"Expected 'session_value', got {value}"
            print("✓ Session memory retrieved")

            # Add conversation message
            success = session.add_conversation_message(
                agent_id="test_agent",
                role="user",
                content="Test message",
            )
            assert success, "Failed to add conversation message"
            print("✓ Conversation message added")

            # Get conversation history
            history = session.get_conversation_history()
            assert len(history) == 1, f"Expected 1 message, got {len(history)}"
            print("✓ Conversation history retrieved")

            # Get session summary
            summary = session.get_session_summary()
            assert summary["workflow_id"] == "test_workflow"
            print(f"✓ Session summary: {summary['workflow_id']}")

        print("✓ Session context closed")
        return True

    except Exception as e:
        print(f"✗ SessionContext test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config():
    """Test memory configuration."""
    print("\nTesting configuration...")

    try:
        from config.settings import get_settings

        settings = get_settings()
        memory_settings = settings.memory

        assert hasattr(memory_settings, "storage_dir")
        assert hasattr(memory_settings, "use_redis")
        assert hasattr(memory_settings, "short_term_ttl")

        print(f"✓ Storage dir: {memory_settings.storage_dir}")
        print(f"✓ Use Redis: {memory_settings.use_redis}")
        print(f"✓ TTL: {memory_settings.short_term_ttl}")

        return True

    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("MEMORY MANAGEMENT VALIDATION")
    print("=" * 70)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("MemoryManager", test_memory_manager()))
    results.append(("SessionContext", test_session_context()))
    results.append(("Configuration", test_config()))

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✓ All validation tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
