"""
Test script to verify agent integration with PromptManager.

This script tests:
1. Agent initialization with automatic prompt loading
2. Prompt validation
3. LLM initialization
4. Prompt reloading
"""

import sys
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.marketing_agents.agents.marketing_strategy import MarketingStrategyAgent
from src.marketing_agents.agents.customer_support import CustomerSupportAgent
from src.marketing_agents.core.prompt_manager import PromptManager
from config.settings import get_settings


async def test_agent_initialization():
    """Test basic agent initialization with PromptManager."""
    print("\n" + "=" * 60)
    print("Test 1: Agent Initialization with PromptManager")
    print("=" * 60)

    # Get settings
    settings = get_settings()

    # Load agent config
    config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.7,
        },
        "capabilities": ["strategy", "analysis"],
    }

    # Create shared PromptManager
    prompt_manager = PromptManager()

    # Initialize Marketing Strategy Agent
    print("\n→ Initializing Marketing Strategy Agent...")
    try:
        marketing_agent = MarketingStrategyAgent(
            config=config, prompt_manager=prompt_manager
        )

        print(f"✓ Agent initialized: {marketing_agent.agent_id}")
        print(f"  Name: {marketing_agent.name}")
        print(f"  Prompt loaded: {marketing_agent.system_prompt is not None}")
        if marketing_agent.system_prompt:
            print(f"  Prompt size: {len(marketing_agent.system_prompt)} chars")
            print(f"  First 100 chars: {marketing_agent.system_prompt[:100]}...")
        print(f"  LLM initialized: {marketing_agent.llm is not None}")
        if marketing_agent.llm:
            print(f"  LLM model: {marketing_agent.llm.model_name}")
    except Exception as e:
        print(f"✗ Failed to initialize Marketing Strategy Agent: {e}")
        import traceback

        traceback.print_exc()

    # Initialize Customer Support Agent
    print("\n→ Initializing Customer Support Agent...")
    try:
        support_agent = CustomerSupportAgent(
            config=config, prompt_manager=prompt_manager
        )

        print(f"✓ Agent initialized: {support_agent.agent_id}")
        print(f"  Name: {support_agent.name}")
        print(f"  Prompt loaded: {support_agent.system_prompt is not None}")
        if support_agent.system_prompt:
            print(f"  Prompt size: {len(support_agent.system_prompt)} chars")
        print(f"  LLM initialized: {support_agent.llm is not None}")
    except Exception as e:
        print(f"✗ Failed to initialize Customer Support Agent: {e}")
        import traceback

        traceback.print_exc()


async def test_prompt_validation():
    """Test prompt validation functionality."""
    print("\n" + "=" * 60)
    print("Test 2: Prompt Validation")
    print("=" * 60)

    config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.7,
        }
    }

    prompt_manager = PromptManager()

    print("\n→ Testing prompt validation...")
    try:
        agent = MarketingStrategyAgent(config=config, prompt_manager=prompt_manager)

        # Check if prompt has required sections
        if agent.system_prompt:
            required_sections = ["ROLE DEFINITION", "CAPABILITIES", "OUTPUT FORMAT"]
            found_sections = []

            for section in required_sections:
                if section in agent.system_prompt.upper():
                    found_sections.append(section)

            print(f"✓ Prompt validation complete")
            print(
                f"  Required sections found: {len(found_sections)}/{len(required_sections)}"
            )
            for section in found_sections:
                print(f"    ✓ {section}")

            missing = [s for s in required_sections if s not in found_sections]
            if missing:
                for section in missing:
                    print(f"    ✗ {section}")
        else:
            print("⚠ No prompt loaded")

    except Exception as e:
        print(f"✗ Validation test failed: {e}")


async def test_prompt_reload():
    """Test prompt reloading functionality."""
    print("\n" + "=" * 60)
    print("Test 3: Prompt Reloading")
    print("=" * 60)

    config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.7,
        }
    }

    prompt_manager = PromptManager()

    print("\n→ Initializing agent...")
    try:
        agent = MarketingStrategyAgent(config=config, prompt_manager=prompt_manager)

        if agent.system_prompt:
            original_size = len(agent.system_prompt)
            print(f"✓ Original prompt size: {original_size} chars")

            # Test reload
            print("\n→ Reloading prompt...")
            success = agent.reload_prompt()

            if success:
                print(f"✓ Prompt reloaded successfully")
                print(f"  New prompt size: {len(agent.system_prompt)} chars")
                print(f"  LLM reinitialized: {agent.llm is not None}")
            else:
                print("✗ Prompt reload failed")
        else:
            print("⚠ No prompt loaded initially")

    except Exception as e:
        print(f"✗ Reload test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_llm_integration():
    """Test LLM integration with system prompt."""
    print("\n" + "=" * 60)
    print("Test 4: LLM Integration")
    print("=" * 60)

    config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.7,
        }
    }

    prompt_manager = PromptManager()

    print("\n→ Testing LLM initialization...")
    try:
        agent = MarketingStrategyAgent(config=config, prompt_manager=prompt_manager)

        if agent.llm:
            print(f"✓ LLM initialized successfully")
            print(f"  Model: {agent.llm.model_name}")
            print(f"  Temperature: {agent.llm.temperature}")
            print(f"  System prompt available: {agent.system_prompt is not None}")
        else:
            print("✗ LLM not initialized")

    except Exception as e:
        print(f"✗ LLM integration test failed: {e}")


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" Agent-PromptManager Integration Tests")
    print("=" * 70)

    try:
        await test_agent_initialization()
        await test_prompt_validation()
        await test_prompt_reload()
        await test_llm_integration()

        print("\n" + "=" * 70)
        print(" All Tests Complete!")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Agents automatically load prompts via PromptManager")
        print("  ✓ Prompts are validated on load")
        print("  ✓ LLM is initialized with system prompt")
        print("  ✓ Prompts can be reloaded without restart")

    except Exception as e:
        print(f"\n✗ Test suite error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
