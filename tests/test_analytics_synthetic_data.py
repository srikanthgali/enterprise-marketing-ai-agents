"""
Test script to verify analytics agent can load and process synthetic data.
"""

import asyncio
import json
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.marketing_agents.agents.analytics_evaluation import AnalyticsEvaluationAgent
from src.marketing_agents.tools.synthetic_data_loader import load_execution_data


async def test_synthetic_data_loading():
    """Test loading synthetic data."""
    print("\n" + "=" * 80)
    print("TEST: Synthetic Data Loading")
    print("=" * 80)

    # Load data directly
    execution_data = load_execution_data(time_range="365d", limit=100)

    print(f"\n✅ Loaded {len(execution_data)} execution records")

    if execution_data:
        # Show sample record
        sample = execution_data[0]
        print("\n📄 Sample Record:")
        print(json.dumps(sample, indent=2))

        # Show data distribution
        agents = {}
        statuses = {}
        for record in execution_data:
            agent = record.get("agent_id", "unknown")
            status = record.get("status", "unknown")
            agents[agent] = agents.get(agent, 0) + 1
            statuses[status] = statuses.get(status, 0) + 1

        print("\n📊 Data Distribution:")
        print(f"   By Agent: {agents}")
        print(f"   By Status: {statuses}")

    return len(execution_data) > 0


async def test_analytics_with_synthetic_data():
    """Test analytics agent with synthetic data."""
    print("\n" + "=" * 80)
    print("TEST: Analytics Agent with Synthetic Data")
    print("=" * 80)

    # Create analytics agent
    agent = AnalyticsEvaluationAgent(config={"test_mode": True})

    # Request analytics report for last 30 days
    result = await agent.process(
        {
            "type": "calculate_metrics",
            "time_range": "30d",
            "metric_types": ["campaign", "agent", "system"],
        }
    )

    if result["success"]:
        metrics = result["analytics"]["metrics"]
        print("\n✅ Metrics calculated successfully!")

        # Show campaign metrics
        campaign = metrics.get("campaign_metrics", {})
        print(f"\n📊 Campaign Metrics:")
        print(f"   CTR: {campaign.get('ctr', 0):.2f}%")
        print(f"   Conversion Rate: {campaign.get('conversion_rate', 0):.2f}%")
        print(f"   ROI: {campaign.get('roi', 0):.2f}%")
        print(f"   Total Impressions: {campaign.get('total_impressions', 0):,}")
        print(f"   Total Clicks: {campaign.get('total_clicks', 0):,}")
        print(f"   Total Conversions: {campaign.get('total_conversions', 0):,}")
        print(f"   Total Revenue: ${campaign.get('total_revenue', 0):,.2f}")

        # Show agent metrics
        agent_metrics = metrics.get("agent_metrics", {})
        print(f"\n🤖 Agent Metrics:")
        print(f"   Success Rate: {agent_metrics.get('success_rate', 0):.2f}%")
        print(f"   Avg Response Time: {agent_metrics.get('avg_response_time', 0):.2f}s")
        print(f"   Total Executions: {agent_metrics.get('total_executions', 0):,}")
        print(f"   Error Rate: {agent_metrics.get('error_rate', 0):.2f}%")

        # Show system metrics
        system = metrics.get("system_metrics", {})
        print(f"\n⚙️  System Metrics:")
        print(f"   Throughput: {system.get('throughput', 0):.2f} exec/hour")
        print(f"   Error Rate: {system.get('error_rate', 0):.2f}%")
        print(f"   P99 Latency: {system.get('latency_p99', 0):.3f}s")

        print(f"\n📈 Data Points Analyzed: {metrics.get('data_points', 0)}")

        # Verify non-zero values
        has_real_data = (
            campaign.get("total_impressions", 0) > 0
            and campaign.get("ctr", 0) > 0
            and agent_metrics.get("total_executions", 0) > 0
        )

        if has_real_data:
            print("\n✅ SUCCESS: Analytics showing real data from synthetic files!")
            return True
        else:
            print("\n❌ WARNING: Analytics still showing zero values")
            return False

    else:
        print(f"\n❌ Failed: {result.get('error')}")
        return False


async def test_report_generation():
    """Test full report generation."""
    print("\n" + "=" * 80)
    print("TEST: Report Generation")
    print("=" * 80)

    agent = AnalyticsEvaluationAgent(config={"test_mode": True})

    # Generate full report
    result = await agent.process(
        {
            "type": "generate_report",
            "time_range": "30d",
        }
    )

    if result["success"]:
        report = result["analytics"]["report"]
        report_content = report.get("report_content", "")

        print("\n✅ Report generated successfully!")
        print("\n" + "=" * 80)
        print("GENERATED REPORT:")
        print("=" * 80)
        print(report_content)

        # Verify report has real data
        has_data = (
            "0.00%" not in report_content
            or report_content.count("0.00%") < 5  # Allow some zeros but not all
        )

        return has_data

    else:
        print(f"\n❌ Failed: {result.get('error')}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🧪 ANALYTICS SYNTHETIC DATA INTEGRATION TEST")
    print("=" * 80)

    success_count = 0
    total_tests = 3

    try:
        # Test 1: Load synthetic data
        if await test_synthetic_data_loading():
            success_count += 1

        # Test 2: Analytics with synthetic data
        if await test_analytics_with_synthetic_data():
            success_count += 1

        # Test 3: Report generation
        if await test_report_generation():
            success_count += 1

        print("\n" + "=" * 80)
        print(f"📊 TEST RESULTS: {success_count}/{total_tests} tests passed")
        print("=" * 80)

        if success_count == total_tests:
            print("\n✅ ALL TESTS PASSED! Analytics now using synthetic data.")
        else:
            print(f"\n⚠️  {total_tests - success_count} test(s) failed.")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
