#!/usr/bin/env python
"""
Quick manual test for intent routing - Test specific use cases interactively.

Usage:
    python scripts/quick_test_intent.py
"""

import asyncio
import httpx
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

API_BASE_URL = "http://localhost:8000/api/v1"


async def test_single_message(message: str):
    """Test a single message and print results."""
    print(f"\n{'='*80}")
    print(f"Testing: {message}")
    print(f"{'='*80}")

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{API_BASE_URL}/chat", json={"message": message}
            )

            if response.status_code == 200:
                result = response.json()

                print(f"\n✓ Success!")
                print(f"\nIntent: {result.get('intent')}")
                print(f"Confidence: {result.get('confidence'):.2%}")
                print(
                    f"Initial Agent: {result.get('agents_executed', [])[0] if result.get('agents_executed') else 'N/A'}"
                )
                print(f"All Agents: {' → '.join(result.get('agents_executed', []))}")

                if result.get("handoffs"):
                    print(f"\nHandoffs:")
                    for handoff in result["handoffs"]:
                        print(
                            f"  • {handoff['from']} → {handoff['to']}: {handoff.get('reason', 'N/A')}"
                        )
                else:
                    print(f"\nHandoffs: None")

                print(f"\nResponse Preview:")
                response_msg = result.get("message", "")
                preview = (
                    response_msg[:300] + "..."
                    if len(response_msg) > 300
                    else response_msg
                )
                print(f"{preview}")

            else:
                print(f"\n✗ Error: {response.status_code}")
                print(response.text)

        except Exception as e:
            print(f"\n✗ Exception: {e}")


async def main():
    """Run interactive tests."""

    # Check API
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get(f"{API_BASE_URL}/health")
            if health.status_code != 200:
                print("❌ API is not healthy. Please start the API server.")
                return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Start with: python scripts/run_api.py")
        return

    print("✓ API is running\n")

    # Test cases from user requirements
    test_cases = [
        # Handoff scenarios
        "Customer satisfaction scores dropped 15% this month. What's causing this?",
        "Recommend improvements for our campaign performance",
        "Are our conversion rate predictions accurate? How can we improve them?",
        "What positioning strategy should we use to stand out in the payment processing market?",
        "Our conversion rates are falling. Need help optimizing our approach.",
        # Direct routing
        "Our checkout is throwing a 400 error when customers try to pay with saved cards",
        "How do I implement webhooks for payment confirmations?",
        "I need to create a marketing campaign to promote our new payment processing feature for small businesses.",
        "Generate a monthly performance report for December.",
        "Show me the conversion funnel for our checkout process.",
        "Rate the quality of the campaign strategy provided by the Marketing Agent: 2/5 stars. It was too generic.",
        "Multiple agents are reporting that mobile checkout has issues. Can you investigate and recommend improvements?",
    ]

    print("Testing all use cases...")
    print("=" * 80)

    for i, message in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}]")
        await test_single_message(message)
        await asyncio.sleep(1)  # Brief pause between tests

    print(f"\n{'='*80}")
    print("All tests complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
