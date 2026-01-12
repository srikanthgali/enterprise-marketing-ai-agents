"""
Memory Management Usage Examples.

Demonstrates various features of the comprehensive memory management system:
1. Basic memory operations (save/retrieve)
2. Session context management
3. Conversation history tracking
4. Semantic search in memory
5. Execution record tracking
6. Multiple agents in a workflow
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.marketing_agents.memory import MemoryManager, SessionContext, create_session


async def example_1_basic_operations():
    """Example 1: Basic memory operations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Memory Operations")
    print("=" * 70 + "\n")

    # Initialize memory manager
    memory_manager = MemoryManager()

    # Save short-term memory
    print("1. Saving short-term memory...")
    memory_manager.save(
        agent_id="marketing_strategy_agent",
        key="current_campaign",
        value={
            "name": "Q1 Product Launch",
            "budget": 50000,
            "channels": ["email", "social"],
        },
        memory_type="short_term",
    )
    print("✓ Saved short-term memory\n")

    # Retrieve short-term memory
    print("2. Retrieving short-term memory...")
    campaign = memory_manager.retrieve(
        agent_id="marketing_strategy_agent",
        key="current_campaign",
        memory_type="short_term",
    )
    print(f"✓ Retrieved: {campaign}\n")

    # Save long-term memory
    print("3. Saving long-term memory...")
    memory_manager.save(
        agent_id="marketing_strategy_agent",
        key="successful_campaigns",
        value=[
            {"name": "Q4 Holiday", "roi": 3.2, "lessons": "Start earlier next time"},
            {"name": "Q3 Back to School", "roi": 2.8, "lessons": "Focus on parents"},
        ],
        memory_type="long_term",
    )
    print("✓ Saved long-term memory\n")

    # Retrieve long-term memory
    print("4. Retrieving long-term memory...")
    campaigns = memory_manager.retrieve(
        agent_id="marketing_strategy_agent",
        key="successful_campaigns",
        memory_type="long_term",
    )
    print(f"✓ Retrieved: {campaigns}\n")

    # Get stats
    stats = memory_manager.get_stats()
    print(f"Memory Stats: {stats}\n")


async def example_2_session_context():
    """Example 2: Session context management."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Session Context Management")
    print("=" * 70 + "\n")

    memory_manager = MemoryManager()

    # Create session with context manager
    print("1. Creating session with context manager...")
    with create_session(
        workflow_id="workflow_123",
        memory_manager=memory_manager,
        metadata={"user": "john@example.com", "campaign": "Q1 Launch"},
        auto_cleanup=False,  # Don't auto-cleanup for demo
    ) as session:
        print(f"✓ Session created: {session.workflow_id}\n")

        # Save memory within session
        print("2. Saving data in session scope...")
        session.save_agent_memory(
            agent_id="content_agent",
            key="draft_headlines",
            value=["Launch Your Success", "Innovate Today", "Transform Tomorrow"],
        )
        print("✓ Saved session-scoped memory\n")

        # Retrieve memory
        print("3. Retrieving data from session...")
        headlines = session.retrieve_agent_memory(
            agent_id="content_agent",
            key="draft_headlines",
        )
        print(f"✓ Retrieved: {headlines}\n")

        # Add conversation messages
        print("4. Adding conversation messages...")
        session.add_conversation_message(
            agent_id="strategy_agent",
            role="user",
            content="Create a marketing campaign for Q1 product launch",
        )
        session.add_conversation_message(
            agent_id="strategy_agent",
            role="assistant",
            content="I'll create a comprehensive campaign focusing on email and social media.",
        )
        session.add_conversation_message(
            agent_id="content_agent",
            role="assistant",
            content="I've generated three headline options for the campaign.",
        )
        print("✓ Added 3 messages\n")

        # Get conversation history
        print("5. Retrieving conversation history...")
        history = session.get_conversation_history()
        print(f"✓ Retrieved {len(history)} messages:")
        for msg in history:
            print(f"   [{msg['role']}] {msg['agent_id']}: {msg['content'][:50]}...")
        print()

        # Get session summary
        print("6. Session summary...")
        summary = session.get_session_summary()
        print(f"✓ Workflow: {summary['workflow_id']}")
        print(f"   Active agents: {summary['active_agents']}")
        print(f"   Interactions: {summary['total_interactions']}")
        print(f"   Messages: {summary['conversation_messages']}")
        print()

    print("✓ Session ended and cleaned up\n")


async def example_3_execution_records():
    """Example 3: Execution record tracking."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Execution Record Tracking")
    print("=" * 70 + "\n")

    memory_manager = MemoryManager()

    # Save execution records
    print("1. Saving execution records...")

    record_id_1 = memory_manager.save_execution_record(
        agent_id="analytics_agent",
        record={
            "status": "success",
            "input": {"query": "engagement metrics", "timeframe": "last_30_days"},
            "output": {"avg_engagement": 0.45, "total_users": 12500},
            "duration_ms": 234,
        },
    )
    print(f"✓ Saved record 1: {record_id_1}")

    record_id_2 = memory_manager.save_execution_record(
        agent_id="analytics_agent",
        record={
            "status": "success",
            "input": {"query": "conversion rates", "timeframe": "last_7_days"},
            "output": {"conversion_rate": 0.032, "total_conversions": 156},
            "duration_ms": 189,
        },
    )
    print(f"✓ Saved record 2: {record_id_2}\n")

    # Retrieve execution records
    print("2. Retrieving execution records...")
    records = memory_manager.get_execution_records(
        agent_id="analytics_agent",
        limit=5,
    )
    print(f"✓ Retrieved {len(records)} records:")
    for record in records:
        print(
            f"   [{record['status']}] {record['input']['query']} - {record['duration_ms']}ms"
        )
    print()

    # Filter by status
    print("3. Filtering by status...")
    success_records = memory_manager.get_execution_records(
        agent_id="analytics_agent",
        status_filter="success",
        limit=10,
    )
    print(f"✓ Found {len(success_records)} successful executions\n")


async def example_4_semantic_search():
    """Example 4: Semantic search in memory."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Semantic Search in Memory")
    print("=" * 70 + "\n")

    memory_manager = MemoryManager()

    # Save some marketing insights in long-term memory
    print("1. Saving marketing insights...")

    insights = [
        "Email campaigns with personalized subject lines increase open rates by 26%",
        "Social media posts with video content get 48% more engagement",
        "Customer segmentation based on behavior improves conversion by 15%",
        "A/B testing headlines can improve click-through rates by up to 20%",
        "Retargeting abandoned carts increases recovery rate by 35%",
    ]

    for i, insight in enumerate(insights):
        memory_manager.save(
            agent_id="strategy_agent",
            key=f"insight_{i}",
            value=insight,
            memory_type="long_term",
        )
    print(f"✓ Saved {len(insights)} insights\n")

    # Wait for vector store to process
    await asyncio.sleep(1)

    # Search for similar memories
    print("2. Searching for similar insights...")
    results = memory_manager.search_similar(
        agent_id="strategy_agent",
        query="How can I improve email marketing performance?",
        top_k=3,
    )

    print(f"✓ Found {len(results)} relevant insights:")
    for result in results:
        print(f"   Score: {result['similarity_score']:.3f}")
        print(f"   Content: {result['content']}")
        print()


async def example_5_multi_agent_workflow():
    """Example 5: Multiple agents in a workflow."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Multi-Agent Workflow")
    print("=" * 70 + "\n")

    memory_manager = MemoryManager()

    with create_session(
        workflow_id="campaign_creation_001",
        memory_manager=memory_manager,
        metadata={"type": "campaign_creation", "priority": "high"},
        auto_cleanup=False,
    ) as session:
        print("1. Strategy Agent - Analyzing target audience...")
        session.save_agent_memory(
            agent_id="strategy_agent",
            key="target_audience",
            value={
                "segments": ["tech_enthusiasts", "early_adopters"],
                "size": 50000,
                "demographics": {
                    "age_range": "25-45",
                    "interests": ["technology", "innovation"],
                },
            },
        )
        session.add_conversation_message(
            agent_id="strategy_agent",
            role="assistant",
            content="Identified 2 key segments with 50,000 potential customers",
        )
        session.save_execution_record(
            agent_id="strategy_agent",
            record={
                "status": "success",
                "action": "audience_analysis",
                "duration_ms": 450,
            },
        )
        print("✓ Strategy agent completed\n")

        print("2. Content Agent - Creating campaign content...")
        audience = session.retrieve_agent_memory("strategy_agent", "target_audience")
        session.save_agent_memory(
            agent_id="content_agent",
            key="campaign_content",
            value={
                "headline": "Innovate Your Future",
                "tagline": "Technology that transforms",
                "cta": "Get Early Access",
            },
        )
        session.add_conversation_message(
            agent_id="content_agent",
            role="assistant",
            content="Created campaign content tailored to tech enthusiasts",
        )
        session.save_execution_record(
            agent_id="content_agent",
            record={
                "status": "success",
                "action": "content_creation",
                "duration_ms": 890,
            },
        )
        print("✓ Content agent completed\n")

        print("3. Analytics Agent - Setting up tracking...")
        session.save_agent_memory(
            agent_id="analytics_agent",
            key="tracking_config",
            value={
                "metrics": ["open_rate", "click_rate", "conversion_rate"],
                "goals": {
                    "open_rate": 0.25,
                    "click_rate": 0.05,
                    "conversion_rate": 0.02,
                },
            },
        )
        session.add_conversation_message(
            agent_id="analytics_agent",
            role="assistant",
            content="Tracking configured with 3 key metrics and goals",
        )
        session.save_execution_record(
            agent_id="analytics_agent",
            record={
                "status": "success",
                "action": "tracking_setup",
                "duration_ms": 120,
            },
        )
        print("✓ Analytics agent completed\n")

        # Get session summary
        print("4. Workflow Summary...")
        summary = session.get_session_summary()
        print(f"✓ Workflow: {summary['workflow_id']}")
        print(f"   Agents involved: {', '.join(summary['active_agents'])}")
        print(f"   Total interactions: {summary['total_interactions']}")
        print(f"   Duration: {summary['duration_seconds']:.1f}s")
        print()

        # Get conversation history
        print("5. Conversation Flow...")
        history = session.get_conversation_history()
        for msg in history:
            print(f"   {msg['agent_id']}: {msg['content']}")
        print()

    print("✓ Multi-agent workflow completed\n")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MEMORY MANAGEMENT EXAMPLES")
    print("=" * 70)

    try:
        await example_1_basic_operations()
        await example_2_session_context()
        await example_3_execution_records()
        await example_4_semantic_search()
        await example_5_multi_agent_workflow()

        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
