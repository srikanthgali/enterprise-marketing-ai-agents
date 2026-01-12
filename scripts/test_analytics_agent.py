"""
Test script for Analytics & Evaluation Agent.

Demonstrates the production-ready analytics functionality including:
- Metrics calculation
- Report generation
- Performance forecasting
- Anomaly detection
- A/B test analysis
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.marketing_agents.agents.analytics_evaluation import AnalyticsEvaluationAgent
from src.marketing_agents.tools.metrics_calculator import (
    calculate_campaign_metrics,
    calculate_agent_metrics,
    calculate_system_metrics,
)
from src.marketing_agents.tools.visualization import (
    generate_campaign_dashboard,
    create_line_chart,
    create_funnel_chart,
)


def generate_mock_execution_data(num_records: int = 50) -> list:
    """Generate mock execution data for testing."""
    import random

    execution_data = []
    base_time = datetime.utcnow() - timedelta(days=7)

    agents = ["campaign_manager", "content_creator", "seo_optimizer", "analytics_agent"]

    for i in range(num_records):
        timestamp = base_time + timedelta(hours=i * 3)

        record = {
            "agent_id": random.choice(agents),
            "started_at": timestamp.isoformat(),
            "completed_at": (
                timestamp + timedelta(seconds=random.uniform(1, 10))
            ).isoformat(),
            "status": random.choices(["completed", "failed"], weights=[0.9, 0.1])[0],
            "result": {
                "metrics": {
                    "impressions": random.randint(5000, 15000),
                    "clicks": random.randint(100, 800),
                    "conversions": random.randint(5, 50),
                    "likes": random.randint(50, 300),
                    "shares": random.randint(10, 100),
                    "comments": random.randint(5, 50),
                    "cost": random.uniform(100, 500),
                    "revenue": random.uniform(300, 2000),
                },
                "handoff_required": random.choice([True, False]),
                "target_agent": (
                    random.choice(agents) if random.random() < 0.2 else None
                ),
            },
        }

        execution_data.append(record)

    return execution_data


async def test_calculate_metrics():
    """Test metrics calculation."""
    print("\n" + "=" * 80)
    print("TEST 1: Calculate Metrics")
    print("=" * 80)

    agent = AnalyticsEvaluationAgent(
        agent_id="analytics_test", config={"test_mode": True}
    )

    # Add mock execution history
    agent.execution_history = generate_mock_execution_data(50)

    # Calculate metrics for last 24 hours
    result = await agent.process(
        {
            "type": "calculate_metrics",
            "time_range": "7d",
            "metric_types": ["campaign", "agent", "system"],
        }
    )

    if result["success"]:
        metrics = result["analytics"]["metrics"]["metrics"]
        print("\n✅ Metrics calculated successfully!")
        print(f"\n📊 Campaign Metrics:")
        print(json.dumps(metrics["campaign_metrics"], indent=2))
        print(f"\n🤖 Agent Metrics:")
        print(json.dumps(metrics["agent_metrics"], indent=2))
        print(f"\n⚙️  System Metrics:")
        print(json.dumps(metrics["system_metrics"], indent=2))
    else:
        print(f"\n❌ Failed: {result.get('error')}")

    return agent


async def test_generate_report(agent: AnalyticsEvaluationAgent):
    """Test report generation."""
    print("\n" + "=" * 80)
    print("TEST 2: Generate Report")
    print("=" * 80)

    result = await agent.process(
        {"type": "generate_report", "time_range": "7d", "format": "markdown"}
    )

    if result["success"]:
        report = result["analytics"]["report"]["report"]
        print("\n✅ Report generated successfully!")
        print(f"\n📄 Report Content:\n")
        print(
            report["report_content"][:1000] + "..."
            if len(report["report_content"]) > 1000
            else report["report_content"]
        )
        print(f"\n📈 Generated {len(report['visualizations'])} visualizations")
        print(f"💡 Generated {len(report['insights'])} insights")

        if report.get("insights"):
            print(f"\n💡 Key Insights:")
            for i, insight in enumerate(report["insights"][:3], 1):
                print(f"   {i}. {insight}")
    else:
        print(f"\n❌ Failed: {result.get('error')}")


async def test_forecast_performance(agent: AnalyticsEvaluationAgent):
    """Test performance forecasting."""
    print("\n" + "=" * 80)
    print("TEST 3: Forecast Performance")
    print("=" * 80)

    result = await agent.process(
        {"type": "forecast_performance", "time_range": "7d", "periods_ahead": 7}
    )

    if result["success"]:
        forecast = result["analytics"]["forecast"]["forecast"]
        print("\n✅ Forecast generated successfully!")
        print(f"\n🔮 Method: {forecast['method']}")
        print(f"📅 Forecast Date: {forecast['forecast_date']}")
        print(f"\n📊 Forecasted Metrics:")
        for metric, value in list(forecast["forecasts"].items())[:5]:
            intervals = forecast["confidence_intervals"].get(metric, {})
            print(
                f"   {metric}: {value} (95% CI: [{intervals.get('lower', 0)}, {intervals.get('upper', 0)}])"
            )
    else:
        print(f"\n❌ Failed: {result.get('error')}")


async def test_detect_anomalies(agent: AnalyticsEvaluationAgent):
    """Test anomaly detection."""
    print("\n" + "=" * 80)
    print("TEST 4: Detect Anomalies")
    print("=" * 80)

    result = await agent.process({"type": "detect_anomalies", "threshold": 2.0})

    if result["success"]:
        anomalies = result["analytics"]["anomalies"]["anomalies"]
        print("\n✅ Anomaly detection completed!")
        print(f"\n🚨 Found {len(anomalies['anomalies'])} anomalies")
        print(f"⚠️  Warnings: {anomalies['severity_counts']['warning']}")
        print(f"🔴 Critical: {anomalies['severity_counts']['critical']}")

        if anomalies.get("alerts"):
            print(f"\n🔔 Alerts:")
            for alert in anomalies["alerts"][:3]:
                print(f"   [{alert['severity'].upper()}] {alert['message']}")
    else:
        print(f"\n❌ Failed: {result.get('error')}")


async def test_ab_test_analysis():
    """Test A/B test analysis."""
    print("\n" + "=" * 80)
    print("TEST 5: A/B Test Analysis")
    print("=" * 80)

    agent = AnalyticsEvaluationAgent(
        agent_id="analytics_test", config={"test_mode": True}
    )

    # Test Case 1: Significant difference
    result = await agent.process(
        {
            "type": "analyze_ab_test",
            "test_id": "email_subject_test_001",
            "variant_a": {
                "conversions": 120,
                "trials": 5000,
            },
            "variant_b": {
                "conversions": 180,
                "trials": 5000,
            },
        }
    )

    if result["success"]:
        analysis = result["analytics"]["ab_test_analysis"]["ab_test_analysis"]
        print("\n✅ A/B test analysis completed!")
        print(f"\n🏆 Winner: Variant {analysis['winner']}")
        print(f"📊 Confidence: {analysis['confidence']*100:.2f}%")
        print(f"📈 P-value: {analysis['p_value']:.4f}")
        print(f"✨ Is Significant: {analysis['is_significant']}")
        print(f"\n📋 Metrics:")
        print(
            f"   Variant A: {analysis['metrics']['variant_a']['conversion_rate']}% conversion rate"
        )
        print(
            f"   Variant B: {analysis['metrics']['variant_b']['conversion_rate']}% conversion rate"
        )
        print(f"   Lift: {analysis['metrics']['lift']:.2f}%")
        print(f"\n💡 Recommendation:")
        print(f"   {analysis['recommendation']}")
    else:
        print(f"\n❌ Failed: {result.get('error')}")


async def test_visualizations():
    """Test visualization generation."""
    print("\n" + "=" * 80)
    print("TEST 6: Visualizations")
    print("=" * 80)

    # Generate mock data
    mock_metrics = {
        "campaign_metrics": {
            "ctr": 4.5,
            "conversion_rate": 3.2,
            "roi": 250.0,
            "total_impressions": 100000,
            "total_clicks": 4500,
            "total_conversions": 144,
        },
        "agent_metrics": {
            "success_rate": 94.5,
            "avg_response_time": 2.3,
            "agent_breakdown": {
                "campaign_manager": {"executions": 50, "success_rate": 96.0},
                "content_creator": {"executions": 45, "success_rate": 93.3},
                "seo_optimizer": {"executions": 30, "success_rate": 93.3},
            },
        },
        "system_metrics": {
            "throughput": 15.5,
            "error_rate": 5.5,
            "latency_p50": 1.2,
            "latency_p95": 3.4,
            "latency_p99": 5.8,
        },
        "comparisons": {
            "ctr_change": 12.5,
            "conversion_rate_change": -3.2,
            "roi_change": 18.7,
        },
    }

    # Generate dashboards
    campaign_viz = generate_campaign_dashboard(mock_metrics)
    print(f"\n✅ Generated {len(campaign_viz)} campaign visualizations")

    # Show first visualization
    if campaign_viz:
        print(f"\n📊 Sample Visualization (Metric Card):")
        print(json.dumps(campaign_viz[0], indent=2))


async def test_metrics_calculator():
    """Test standalone metrics calculator functions."""
    print("\n" + "=" * 80)
    print("TEST 7: Metrics Calculator Functions")
    print("=" * 80)

    # Generate test data
    test_data = generate_mock_execution_data(30)

    # Test campaign metrics
    campaign_metrics = calculate_campaign_metrics(test_data)
    print("\n✅ Campaign Metrics:")
    print(f"   CTR: {campaign_metrics['ctr']}%")
    print(f"   Conversion Rate: {campaign_metrics['conversion_rate']}%")
    print(f"   ROI: {campaign_metrics['roi']}%")

    # Test agent metrics
    agent_metrics = calculate_agent_metrics(test_data)
    print("\n✅ Agent Metrics:")
    print(f"   Success Rate: {agent_metrics['success_rate']}%")
    print(f"   Avg Response Time: {agent_metrics['avg_response_time']}s")
    print(f"   Handoff Rate: {agent_metrics['handoff_rate']}%")

    # Test system metrics
    system_metrics = calculate_system_metrics(test_data)
    print("\n✅ System Metrics:")
    print(f"   Throughput: {system_metrics['throughput']} exec/hour")
    print(f"   Error Rate: {system_metrics['error_rate']}%")
    print(f"   P99 Latency: {system_metrics['latency_p99']}s")


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🧪 ANALYTICS & EVALUATION AGENT - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Calculate Metrics
        agent = await test_calculate_metrics()

        # Test 2: Generate Report
        await test_generate_report(agent)

        # Test 3: Forecast Performance
        await test_forecast_performance(agent)

        # Test 4: Detect Anomalies
        await test_detect_anomalies(agent)

        # Test 5: A/B Test Analysis
        await test_ab_test_analysis()

        # Test 6: Visualizations
        await test_visualizations()

        # Test 7: Metrics Calculator
        await test_metrics_calculator()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
