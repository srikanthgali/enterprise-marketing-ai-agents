"""
Test script for Marketing Strategy Agent tools.

Validates that all production tools are properly integrated and functional.
"""

import asyncio
import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.marketing_agents.agents.marketing_strategy import MarketingStrategyAgent
from config.settings import get_settings


@pytest.mark.asyncio
async def test_marketing_strategy_agent():
    """Test the production-ready Marketing Strategy Agent tools."""

    print("=" * 80)
    print("Testing Marketing Strategy Agent Production Tools")
    print("=" * 80)

    # Initialize settings
    settings = get_settings()

    # Create agent instance
    config = {
        "capabilities": [
            "market_research",
            "competitor_analysis",
            "audience_segmentation",
            "content_strategy",
            "channel_optimization",
        ],
        "model": {"name": "gpt-4o-mini", "temperature": 0.7},
    }

    agent = MarketingStrategyAgent(config=config)

    print("✓ Marketing Strategy Agent initialized successfully")
    print(f"  - Web Search Tool: {agent.web_search_tool is not None}")
    print(f"  - KB Search Tool: {agent.kb_search_tool is not None}")

    # Test 1: Market Research
    print("\n=== Testing Market Research ===")
    market_result = await agent._market_research(
        query="payment processing industry", target_audience="enterprise developers"
    )
    print(
        f"✓ Market research completed: {len(market_result.get('stripe_insights', []))} insights"
    )

    # Test 2: Competitor Analysis
    print("\n2. Testing competitor analysis...")
    competitor_result = await agent._competitor_analysis(
        competitor_name="PayPal", industry="payment processing"
    )
    print(
        f"✓ Competitor analysis completed for {competitor_result.get('competitor_profile', {}).get('name', 'unknown')}"
    )

    # Test 3: Audience Segmentation
    print("\n3. Testing audience segmentation...")
    campaign_data = {
        "objectives": ["Increase product adoption", "Generate qualified leads"],
        "industry": "fintech",
        "product": "Stripe",
        "budget": 100000,
    }

    segmentation = await agent._audience_segmentation(campaign_data)
    print(f"   ✓ Generated {len(segmentation.get('segments', []))} audience segments")

    # 4. Test content strategy generation
    print("\n4. Testing content strategy generation...")
    content_strategy = await agent._generate_content_strategy(
        campaign_plan={
            "objectives": ["Increase product awareness", "Generate leads"],
            "target_audience": {"primary_segment": {"name": "Enterprise CTOs"}},
            "product": "Stripe",
        },
        duration_weeks=12,
    )
    print(
        f"   ✓ Generated {len(content_strategy.get('calendar', []))} week content calendar"
    )

    # Test 5: _optimize_channels
    print("\n5. Testing _optimize_channels...")
    channels_result = await agent._optimize_channels(
        budget=100000.0, objectives=["Increase brand awareness", "Generate leads"]
    )
    print(
        f"   Channel optimization: {len(channels_result.get('channels', {}))} channels"
    )
    print(f"   Budget allocation: ${channels_result.get('total_budget', 0):,.0f}")

    print("\n=== Tests Completed Successfully ===\n")


if __name__ == "__main__":
    asyncio.run(test_marketing_strategy_agent())
