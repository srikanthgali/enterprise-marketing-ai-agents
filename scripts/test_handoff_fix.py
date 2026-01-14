#!/usr/bin/env python3
"""
Test script to verify the handoff routing fix.
This simulates a customer support request that should trigger a handoff to marketing.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from marketing_agents.core.orchestrator import OrchestratorAgent
from marketing_agents.agents.customer_support import CustomerSupportAgent
from marketing_agents.agents.marketing_strategy import MarketingStrategyAgent
from marketing_agents.agents.analytics_evaluation import AnalyticsEvaluationAgent
from marketing_agents.agents.feedback_learning import FeedbackLearningAgent
from config.settings import Settings

settings = Settings()


async def test_handoff():
    """Test the customer support to marketing handoff."""

    print("=" * 80)
    print("Testing Customer Support → Marketing Strategy Handoff")
    print("=" * 80)

    # Load agent configs
    agents_config = settings.get_agents_config()

    # Initialize orchestrator
    orchestrator = OrchestratorAgent(
        config=agents_config.get("orchestrator", {}),
        memory_manager=None,
        message_bus=None,
    )

    # Initialize and register all agents
    agents = {
        "customer_support": CustomerSupportAgent(
            config=agents_config.get("customer_support", {}),
            memory_manager=None,
            message_bus=None,
        ),
        "marketing_strategy": MarketingStrategyAgent(
            config=agents_config.get("marketing_strategy", {}),
            memory_manager=None,
            message_bus=None,
        ),
        "analytics_evaluation": AnalyticsEvaluationAgent(
            config=agents_config.get("analytics_evaluation", {}),
            memory_manager=None,
            message_bus=None,
        ),
        "feedback_learning": FeedbackLearningAgent(
            config=agents_config.get("feedback_learning", {}),
            memory_manager=None,
            message_bus=None,
        ),
    }

    for agent_id, agent in agents.items():
        orchestrator.register_agent(agent_id, agent)

    # Test request that should trigger handoff
    test_request = {
        "message": "Multiple clients are asking if you support cryptocurrency payments. Any plans for that?",
        "customer_id": "test_customer_001",
        "ticket_id": "test_ticket_001",
        "request_type": "inquiry",
    }

    print("\n📝 Test Request:")
    print(f"   Message: {test_request['message']}")
    print(f"   Request Type: {test_request['request_type']}")

    try:
        print("\n🚀 Starting workflow...")
        result = await orchestrator.execute_workflow(
            task_type="customer_support", task_data=test_request
        )

        print("\n✅ Workflow completed successfully!")
        print(f"\n📊 Execution Summary:")

        exec_summary = result.get("execution_summary", {})
        print(f"   Status: {exec_summary.get('status', 'unknown')}")
        print(f"   Total Steps: {exec_summary.get('total_steps', 0)}")
        print(
            f"   Agents Executed: {', '.join(exec_summary.get('agents_executed', []))}"
        )

        if exec_summary.get("handoffs"):
            print(f"\n🔄 Handoffs:")
            for handoff in exec_summary.get("handoffs", []):
                print(
                    f"   {handoff['from']} → {handoff['to']}: {handoff.get('reason', 'N/A')}"
                )

        if result.get("error"):
            print(f"\n❌ Error: {result['error']}")
            return False

        # Verify the expected handoff occurred
        agents_executed = exec_summary.get("agents_executed", [])
        if (
            "customer_support" in agents_executed
            and "marketing_strategy" in agents_executed
        ):
            print(
                "\n✅ SUCCESS: Handoff from customer_support to marketing_strategy worked!"
            )
            return True
        else:
            print(
                f"\n⚠️  WARNING: Expected both customer_support and marketing_strategy"
            )
            print(f"   but got: {agents_executed}")
            return False

    except Exception as e:
        print(f"\n❌ Workflow failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_handoff())
    sys.exit(0 if success else 1)
