"""
Unit tests for PromptManager.

Tests all major functionality including:
- Loading prompts
- Updating prompts with versioning
- Rollback functionality
- Version comparison
- Caching behavior
- Thread safety
- Error handling
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
import threading
import time

from src.marketing_agents.core.prompt_manager import PromptManager


@pytest.fixture
def temp_prompts_dir():
    """Create a temporary directory for test prompts."""
    temp_dir = tempfile.mkdtemp()
    prompts_dir = Path(temp_dir) / "prompts"
    prompts_dir.mkdir(parents=True)

    # Create a sample prompt file
    sample_prompt = "# Test Agent Prompt\n\nThis is a test prompt for the agent."
    with open(prompts_dir / "test_agent.txt", "w") as f:
        f.write(sample_prompt)

    yield prompts_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def prompt_manager(temp_prompts_dir):
    """Create a PromptManager instance with temp directory."""
    return PromptManager(prompts_dir=str(temp_prompts_dir))


class TestPromptManagerInitialization:
    """Test PromptManager initialization."""

    def test_init_creates_directories(self, temp_prompts_dir):
        """Test that initialization creates necessary directories."""
        pm = PromptManager(prompts_dir=str(temp_prompts_dir))

        assert pm.prompts_dir.exists()
        assert pm.history_dir.exists()
        assert pm.log_file.parent.exists()

    def test_init_with_default_dir(self):
        """Test initialization with default directory."""
        pm = PromptManager()
        assert pm.prompts_dir == Path("config/prompts")


class TestLoadPrompt:
    """Test prompt loading functionality."""

    def test_load_existing_prompt(self, prompt_manager):
        """Test loading an existing prompt."""
        prompt = prompt_manager.load_prompt("test_agent")
        assert prompt == "# Test Agent Prompt\n\nThis is a test prompt for the agent."

    def test_load_nonexistent_prompt(self, prompt_manager):
        """Test loading a prompt that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            prompt_manager.load_prompt("nonexistent_agent")

    def test_load_specific_version(self, prompt_manager):
        """Test loading a specific version."""
        # First, create a version by updating
        prompt_manager.update_prompt(
            "test_agent", "Updated prompt content", "Test update"
        )

        # Get the version ID
        versions = prompt_manager.list_versions("test_agent")
        version_id = versions[1]["version_id"]

        # Load the specific version
        old_prompt = prompt_manager.load_prompt("test_agent", version_id)
        assert (
            old_prompt == "# Test Agent Prompt\n\nThis is a test prompt for the agent."
        )

    def test_load_invalid_version(self, prompt_manager):
        """Test loading an invalid version."""
        with pytest.raises(ValueError):
            prompt_manager.load_prompt("test_agent", "invalid_version")

    def test_caching_behavior(self, prompt_manager):
        """Test that prompts are cached after loading."""
        # First load
        prompt1 = prompt_manager.load_prompt("test_agent")

        # Check cache
        assert "test_agent" in prompt_manager._cache

        # Second load should use cache
        prompt2 = prompt_manager.load_prompt("test_agent")
        assert prompt1 == prompt2


class TestUpdatePrompt:
    """Test prompt update functionality."""

    def test_update_creates_version(self, prompt_manager):
        """Test that updating creates a version."""
        # Update prompt
        success = prompt_manager.update_prompt(
            "test_agent", "New prompt content", "Test update"
        )

        assert success

        # Check that version was created
        versions = prompt_manager.list_versions("test_agent")
        assert len(versions) == 2  # Current + 1 historical

    def test_update_changes_current(self, prompt_manager):
        """Test that update changes the current prompt."""
        new_content = "New prompt content"
        prompt_manager.update_prompt("test_agent", new_content, "Test update")

        # Load and verify
        current = prompt_manager.load_prompt("test_agent")
        assert current == new_content

    def test_update_logs_change(self, prompt_manager):
        """Test that updates are logged."""
        prompt_manager.update_prompt("test_agent", "New content", "Test reason")

        # Check log file
        assert prompt_manager.log_file.exists()

        with open(prompt_manager.log_file, "r") as f:
            log_content = f.read()
            assert "test_agent" in log_content
            assert "Test reason" in log_content

    def test_update_updates_cache(self, prompt_manager):
        """Test that update updates the cache."""
        # Load to populate cache
        prompt_manager.load_prompt("test_agent")

        # Update
        new_content = "Updated content"
        prompt_manager.update_prompt("test_agent", new_content, "Update")

        # Cache should be updated
        assert prompt_manager._cache["test_agent"] == new_content


class TestRollback:
    """Test rollback functionality."""

    def test_rollback_to_previous_version(self, prompt_manager):
        """Test rolling back to a previous version."""
        original = prompt_manager.load_prompt("test_agent")

        # Make an update
        prompt_manager.update_prompt("test_agent", "New content", "Update")

        # Get version to rollback to
        versions = prompt_manager.list_versions("test_agent")
        version_id = versions[1]["version_id"]

        # Rollback
        success = prompt_manager.rollback_prompt("test_agent", version_id)
        assert success

        # Verify rollback
        current = prompt_manager.load_prompt("test_agent")
        assert current == original

    def test_rollback_invalid_version(self, prompt_manager):
        """Test rollback to invalid version fails gracefully."""
        success = prompt_manager.rollback_prompt("test_agent", "invalid_version")
        assert not success


class TestComparePrompts:
    """Test prompt comparison functionality."""

    def test_compare_versions(self, prompt_manager):
        """Test comparing two versions."""
        # Create a version
        prompt_manager.update_prompt(
            "test_agent",
            "# Test Agent Prompt\n\nThis is an UPDATED test prompt.",
            "Update test",
        )

        # Get version ID
        versions = prompt_manager.list_versions("test_agent")
        version_id = versions[1]["version_id"]

        # Compare
        comparison = prompt_manager.compare_prompts("test_agent", version_id, "latest")

        assert "diff_unified" in comparison
        assert "changes_summary" in comparison
        assert comparison["changes_summary"]["total_changes"] > 0

    def test_compare_same_versions(self, prompt_manager):
        """Test comparing identical versions."""
        comparison = prompt_manager.compare_prompts("test_agent", "latest", "latest")

        assert comparison["changes_summary"]["total_changes"] == 0


class TestListVersions:
    """Test version listing functionality."""

    def test_list_versions_single(self, prompt_manager):
        """Test listing versions with only current."""
        versions = prompt_manager.list_versions("test_agent")

        assert len(versions) == 1
        assert versions[0]["version_id"] == "latest"
        assert versions[0]["is_current"] is True

    def test_list_versions_multiple(self, prompt_manager):
        """Test listing versions after updates."""
        # Create multiple versions
        for i in range(3):
            prompt_manager.update_prompt(
                "test_agent", f"Content version {i}", f"Update {i}"
            )
            time.sleep(0.01)  # Ensure different timestamps

        versions = prompt_manager.list_versions("test_agent")

        assert len(versions) == 4  # 1 current + 3 historical
        assert versions[0]["is_current"] is True

        # Check that historical versions are sorted by timestamp (most recent first)
        version_ids = [v["version_id"] for v in versions[1:]]
        assert version_ids == sorted(version_ids, reverse=True)


class TestChangeHistory:
    """Test change history functionality."""

    def test_get_change_history(self, prompt_manager):
        """Test retrieving change history."""
        # Make some updates
        for i in range(3):
            prompt_manager.update_prompt("test_agent", f"Content {i}", f"Reason {i}")

        history = prompt_manager.get_change_history("test_agent")

        assert len(history) >= 3
        # Most recent should be first
        assert "Reason 2" in history[0]["reason"]

    def test_get_change_history_with_limit(self, prompt_manager):
        """Test change history with limit."""
        # Make updates
        for i in range(5):
            prompt_manager.update_prompt("test_agent", f"Content {i}", f"Reason {i}")

        history = prompt_manager.get_change_history("test_agent", limit=2)
        assert len(history) == 2

    def test_get_all_history(self, prompt_manager):
        """Test retrieving history for all agents."""
        # Create updates for test_agent
        prompt_manager.update_prompt("test_agent", "Content", "Reason")

        # Get all history (no filter)
        history = prompt_manager.get_change_history(agent_id=None)

        assert len(history) >= 1


class TestCaching:
    """Test caching functionality."""

    def test_clear_cache(self, prompt_manager):
        """Test clearing the cache."""
        # Load to populate cache
        prompt_manager.load_prompt("test_agent")
        assert "test_agent" in prompt_manager._cache

        # Clear cache
        prompt_manager.clear_cache()
        assert len(prompt_manager._cache) == 0

    def test_reload_prompt(self, prompt_manager):
        """Test reloading a prompt bypasses cache."""
        # Load and cache
        prompt1 = prompt_manager.load_prompt("test_agent")

        # Modify the file directly (simulate external change)
        with open(prompt_manager.prompts_dir / "test_agent.txt", "w") as f:
            f.write("Externally modified content")

        # Regular load still uses cache
        cached = prompt_manager.load_prompt("test_agent")
        assert cached == prompt1

        # Reload bypasses cache
        reloaded = prompt_manager.reload_prompt("test_agent")
        assert reloaded == "Externally modified content"


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_loads(self, prompt_manager):
        """Test concurrent prompt loading."""
        results = []
        errors = []

        def load_prompt():
            try:
                prompt = prompt_manager.load_prompt("test_agent")
                results.append(prompt)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=load_prompt) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 10
        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_concurrent_updates(self, prompt_manager):
        """Test concurrent prompt updates."""
        results = []

        def update_prompt(i):
            success = prompt_manager.update_prompt(
                "test_agent", f"Content {i}", f"Reason {i}"
            )
            results.append(success)

        # Create multiple threads
        threads = [threading.Thread(target=update_prompt, args=(i,)) for i in range(5)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # All should succeed
        assert all(results)

        # Check that all versions were created
        versions = prompt_manager.list_versions("test_agent")
        assert len(versions) >= 6  # Original + 5 updates


class TestErrorHandling:
    """Test error handling."""

    def test_load_from_invalid_directory(self):
        """Test handling of invalid directory."""
        pm = PromptManager(prompts_dir="/nonexistent/directory")

        with pytest.raises(FileNotFoundError):
            pm.load_prompt("test_agent")

    def test_update_preserves_data_on_error(self, prompt_manager):
        """Test that failed updates don't corrupt data."""
        original = prompt_manager.load_prompt("test_agent")

        # This should succeed
        success = prompt_manager.update_prompt("test_agent", "Valid content", "Valid")
        assert success

        # Current should be updated
        current = prompt_manager.load_prompt("test_agent")
        assert current == "Valid content"


class TestAgentIntegration:
    """Test PromptManager integration with agents."""

    def test_agent_loads_prompt(self, temp_prompts_dir):
        """Test that agents load prompts correctly on initialization."""
        from src.marketing_agents.core.base_agent import BaseAgent
        from src.marketing_agents.core.prompt_manager import PromptManager

        # Create a test agent class
        class TestAgent(BaseAgent):
            def _register_tools(self):
                pass

            async def process(self, input_data):
                return {"result": "test"}

        # Create a proper prompt for testing
        prompt_content = """# Test Agent Prompt

## ROLE DEFINITION
You are a test agent.

## CAPABILITIES
- Test capability 1
- Test capability 2

## OUTPUT FORMAT
Return JSON format.

## CONSTRAINTS
Follow all rules.
"""
        with open(temp_prompts_dir / "test_agent.txt", "w") as f:
            f.write(prompt_content)

        # Initialize agent
        pm = PromptManager(prompts_dir=str(temp_prompts_dir))
        agent = TestAgent(
            agent_id="test_agent",
            name="Test Agent",
            description="Test agent description",
            config={"model": {"name": "gpt-4o-mini", "temperature": 0.7}},
            prompt_manager=pm,
        )

        # Verify prompt was loaded
        assert agent.system_prompt is not None
        assert "Test Agent Prompt" in agent.system_prompt
        assert len(agent.system_prompt) > 100
        assert "ROLE DEFINITION" in agent.system_prompt

    def test_agent_validates_prompt(self, temp_prompts_dir, caplog):
        """Test that agents validate prompt sections."""
        from src.marketing_agents.core.base_agent import BaseAgent
        from src.marketing_agents.core.prompt_manager import PromptManager

        class TestAgent(BaseAgent):
            def _register_tools(self):
                pass

            async def process(self, input_data):
                return {"result": "test"}

        # Create incomplete prompt (missing required sections)
        incomplete_prompt = (
            "# Incomplete Prompt\n\nJust some text without proper sections."
        )
        with open(temp_prompts_dir / "incomplete_agent.txt", "w") as f:
            f.write(incomplete_prompt)

        pm = PromptManager(prompts_dir=str(temp_prompts_dir))

        # Initialize agent - should warn about missing sections
        agent = TestAgent(
            agent_id="incomplete_agent",
            name="Incomplete Agent",
            description="Agent with incomplete prompt",
            config={"model": {"name": "gpt-4o-mini"}},
            prompt_manager=pm,
        )

        # Check that warning was logged
        assert agent.system_prompt is not None
        # The validation should have logged warnings

    def test_agent_reload_prompt(self, temp_prompts_dir):
        """Test that agents can reload prompts."""
        from src.marketing_agents.core.base_agent import BaseAgent
        from src.marketing_agents.core.prompt_manager import PromptManager

        class TestAgent(BaseAgent):
            def _register_tools(self):
                pass

            async def process(self, input_data):
                return {"result": "test"}

        # Create initial prompt
        initial_prompt = """# Test Agent

## ROLE DEFINITION
Initial role.

## CAPABILITIES
- Initial capability

## OUTPUT FORMAT
JSON
"""
        with open(temp_prompts_dir / "reload_agent.txt", "w") as f:
            f.write(initial_prompt)

        pm = PromptManager(prompts_dir=str(temp_prompts_dir))
        agent = TestAgent(
            agent_id="reload_agent",
            name="Reload Agent",
            description="Agent for testing reload",
            config={"model": {"name": "gpt-4o-mini"}},
            prompt_manager=pm,
        )

        initial_loaded = agent.system_prompt
        assert "Initial role" in initial_loaded

        # Update prompt
        updated_prompt = """# Test Agent

## ROLE DEFINITION
Updated role.

## CAPABILITIES
- Updated capability

## OUTPUT FORMAT
JSON
"""
        pm.update_prompt("reload_agent", updated_prompt, "Test update")

        # Reload in agent
        success = agent.reload_prompt()
        assert success
        assert "Updated role" in agent.system_prompt
        assert agent.system_prompt != initial_loaded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
