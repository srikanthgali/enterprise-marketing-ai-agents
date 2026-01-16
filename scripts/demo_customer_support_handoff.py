#!/usr/bin/env python3
"""
Quick demo of Customer Support Agent handoff detection.
This demonstrates the fix in action with a simple example.
"""
import asyncio
from src.marketing_agents.agents.customer_support import CustomerSupportAgent
from src.marketing_agents.memory.memory_manager import MemoryManager


async def demo_handoff():
    """Demonstrate handoff detection with a campaign issue."""

    print("=" * 80)
    print("CUSTOMER SUPPORT AGENT HANDOFF DEMO")
    print("=" * 80)

    # Initialize agent
    agent_config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.5,
        },
    }

    memory_manager = MemoryManager()
    agent = CustomerSupportAgent(config=agent_config, memory_manager=memory_manager)

    # Test campaign issue that should trigger handoff
    test_query = "I signed up through your Black Friday promotion but didn't receive the discount code."

    print(f"\n📥 Customer Query:")
    print(f"   {test_query}")

    print("\n⚙️  Processing...")

    result = await agent.process(
        {
            "type": "inquiry",
            "message": test_query,
        }
    )

    print("\n📤 Response:")
    print(f"   Success: {result.get('success')}")
    print(f"   Is Final: {result.get('is_final')}")

    if result.get("handoff_required"):
        print("\n🔀 HANDOFF DETECTED!")
        print(f"   Target Agent: {result.get('target_agent')}")
        print(f"   Reason: {result.get('handoff_reason')}")
        context = result.get("context", {})
        print(f"   Issue Type: {context.get('issue_type')}")
        print(f"   Sentiment: {context.get('sentiment')}")
        print(f"   Recommendation: {context.get('recommendation')}")
    else:
        print("\n✓ No handoff needed - standard support query")

    print(f"\n💬 Agent Response:")
    response_text = result.get("response", "")
    # Print first 200 chars
    print(f"   {response_text[:200]}...")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(demo_handoff())
