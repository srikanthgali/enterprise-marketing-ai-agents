"""
Example: Integrating PromptManager with Marketing Agents

This example demonstrates how to:
1. Initialize agents with PromptManager
2. Use prompts in agent processing
3. Update prompts dynamically
4. Reload prompts without restarting agents
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from src.marketing_agents.core import PromptManager
from src.marketing_agents.core.base_agent import BaseAgent


class ExampleMarketingAgent(BaseAgent):
    """Example marketing agent using PromptManager."""

    def _register_tools(self):
        """Register agent-specific tools."""
        self.register_tool("analyze_market", self._analyze_market)
        self.register_tool("generate_strategy", self._generate_strategy)

    async def process(self, input_data: dict) -> dict:
        """
        Process input using the system prompt.

        In a real implementation, this would:
        1. Use the system_prompt to guide LLM behavior
        2. Apply the prompt as context for analysis
        3. Generate responses based on prompt instructions
        """
        # Simulate using the prompt
        if self.system_prompt:
            print(f"\n[Agent {self.agent_id}] Processing with prompt...")
            print(f"Prompt length: {len(self.system_prompt)} characters")
            print(f"First 100 chars: {self.system_prompt[:100]}...")

        # Process the input (simplified)
        result = {
            "agent_id": self.agent_id,
            "input": input_data,
            "output": f"Processed by {self.name}",
            "prompt_used": self.system_prompt is not None,
            "prompt_size": len(self.system_prompt) if self.system_prompt else 0,
        }

        return result

    async def _analyze_market(self, **kwargs):
        """Analyze market trends."""
        return {"analysis": "Market analysis based on prompt guidance"}

    async def _generate_strategy(self, **kwargs):
        """Generate marketing strategy."""
        return {"strategy": "Strategy generated using prompt framework"}


async def example_basic_usage():
    """Example 1: Basic usage with PromptManager."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Initialize PromptManager
    prompt_manager = PromptManager()

    # Create agent with prompt manager
    agent = ExampleMarketingAgent(
        agent_id="marketing_strategy",
        name="Marketing Strategy Agent",
        description="Develops marketing strategies",
        config={"capabilities": ["strategy", "analysis"]},
        prompt_manager=prompt_manager,
    )

    # The prompt is automatically loaded during initialization
    print(f"\n✓ Agent initialized with prompt")
    print(f"  System prompt loaded: {agent.system_prompt is not None}")
    print(
        f"  Prompt size: {len(agent.system_prompt) if agent.system_prompt else 0} chars"
    )

    # Process some input
    result = await agent.process({"task": "Analyze Q1 marketing performance"})
    print(f"\n✓ Processing complete:")
    print(f"  Result: {result['output']}")
    print(f"  Used prompt: {result['prompt_used']}")


async def example_dynamic_update():
    """Example 2: Updating prompts dynamically."""
    print("\n" + "=" * 60)
    print("Example 2: Dynamic Prompt Updates")
    print("=" * 60)

    # Initialize
    prompt_manager = PromptManager()
    agent = ExampleMarketingAgent(
        agent_id="marketing_strategy",
        name="Marketing Strategy Agent",
        description="Develops marketing strategies",
        config={},
        prompt_manager=prompt_manager,
    )

    original_prompt = agent.system_prompt
    print(f"\n✓ Original prompt loaded ({len(original_prompt)} chars)")

    # Process with original prompt
    result1 = await agent.process({"task": "Create campaign strategy"})
    print(f"\n✓ Processed with original prompt")

    # Update the prompt
    print(f"\n→ Updating prompt...")
    new_prompt = (
        original_prompt
        + "\n\n## NEW REQUIREMENT\nAll strategies must include ROI projections."
    )

    success = prompt_manager.update_prompt(
        "marketing_strategy", new_prompt, "Added ROI projection requirement"
    )

    if success:
        print(f"✓ Prompt updated successfully")

        # Reload the prompt in the agent
        if agent.reload_prompt():
            print(f"✓ Agent prompt reloaded without restart")
            print(f"  New prompt size: {len(agent.system_prompt)} chars")

            # Process with new prompt
            result2 = await agent.process({"task": "Create campaign strategy"})
            print(f"\n✓ Processed with updated prompt")

    # Rollback to original
    print(f"\n→ Rolling back to original prompt...")
    versions = prompt_manager.list_versions("marketing_strategy")
    if len(versions) >= 2:
        version_id = versions[1]["version_id"]
        if prompt_manager.rollback_prompt("marketing_strategy", version_id):
            agent.reload_prompt()
            print(f"✓ Rolled back to version {version_id}")
            print(f"  Prompt size: {len(agent.system_prompt)} chars")


async def example_multi_agent():
    """Example 3: Multiple agents with different prompts."""
    print("\n" + "=" * 60)
    print("Example 3: Multiple Agents with Different Prompts")
    print("=" * 60)

    # Shared prompt manager
    prompt_manager = PromptManager()

    # Create multiple agents
    agents = []
    agent_ids = ["marketing_strategy", "customer_support", "orchestrator"]

    for agent_id in agent_ids:
        try:
            agent = ExampleMarketingAgent(
                agent_id=agent_id,
                name=f"{agent_id.replace('_', ' ').title()} Agent",
                description=f"Handles {agent_id} tasks",
                config={},
                prompt_manager=prompt_manager,
            )
            agents.append(agent)
            print(f"\n✓ Created {agent.name}")
            print(f"  Prompt loaded: {agent.system_prompt is not None}")
            if agent.system_prompt:
                print(f"  Prompt size: {len(agent.system_prompt)} chars")
        except FileNotFoundError:
            print(f"\n⚠ No prompt file found for {agent_id}")

    # Process tasks with each agent
    print(f"\n→ Processing tasks with {len(agents)} agents...")
    for agent in agents:
        result = await agent.process({"task": f"Handle {agent.agent_id} task"})
        print(f"  {agent.name}: {result['output']}")


async def example_prompt_comparison():
    """Example 4: Comparing prompt versions."""
    print("\n" + "=" * 60)
    print("Example 4: Comparing Prompt Versions")
    print("=" * 60)

    prompt_manager = PromptManager()

    # Create some versions
    print("\n→ Creating prompt versions...")
    original = prompt_manager.load_prompt("marketing_strategy")

    # Version 1: Add budget section
    v1_prompt = original + "\n\n## BUDGET CONSIDERATIONS\nConsider budget constraints."
    prompt_manager.update_prompt(
        "marketing_strategy", v1_prompt, "Added budget section"
    )
    print("✓ Created version 1: Added budget section")

    # Version 2: Add timeline section
    v2_prompt = v1_prompt + "\n\n## TIMELINE REQUIREMENTS\nInclude realistic timelines."
    prompt_manager.update_prompt(
        "marketing_strategy", v2_prompt, "Added timeline section"
    )
    print("✓ Created version 2: Added timeline section")

    # List all versions
    versions = prompt_manager.list_versions("marketing_strategy")
    print(f"\n→ Available versions: {len(versions)}")
    for v in versions[:3]:  # Show first 3
        print(f"  - {v['version_id']}: {v['size']} bytes")

    # Compare versions
    if len(versions) >= 3:
        print(f"\n→ Comparing versions...")
        comparison = prompt_manager.compare_prompts(
            "marketing_strategy", versions[2]["version_id"], "latest"  # Original
        )

        summary = comparison["changes_summary"]
        print(f"✓ Comparison complete:")
        print(f"  Added lines: {summary['added_lines']}")
        print(f"  Removed lines: {summary['removed_lines']}")
        print(f"  Total changes: {summary['total_changes']}")


async def example_change_tracking():
    """Example 5: Tracking prompt changes."""
    print("\n" + "=" * 60)
    print("Example 5: Change History Tracking")
    print("=" * 60)

    prompt_manager = PromptManager()

    # View change history
    history = prompt_manager.get_change_history("marketing_strategy", limit=5)

    print(f"\n→ Recent changes ({len(history)} entries):")
    for i, entry in enumerate(history, 1):
        print(f"\n  {i}. Version: {entry['version_id']}")
        print(f"     Time: {entry['timestamp']}")
        print(f"     Reason: {entry['reason']}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" PromptManager Integration Examples")
    print("=" * 70)

    try:
        await example_basic_usage()
        await example_dynamic_update()
        await example_multi_agent()
        await example_prompt_comparison()
        await example_change_tracking()

        print("\n" + "=" * 70)
        print(" All Examples Complete!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  1. PromptManager provides centralized prompt storage")
        print("  2. Agents automatically load prompts on initialization")
        print("  3. Prompts can be updated without restarting agents")
        print("  4. All changes are versioned and trackable")
        print("  5. Multiple agents can share the same PromptManager")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
