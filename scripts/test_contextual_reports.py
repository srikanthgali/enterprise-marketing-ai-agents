"""Test script for enhanced contextual reports."""

import sys

sys.path.insert(
    0, "/Users/srikanthgali/Documents/repos/code/enterprise-marketing-ai-agents"
)

from src.marketing_agents.agents.analytics_evaluation import AnalyticsEvaluationAgent

# Initialize agent
agent = AnalyticsEvaluationAgent(config={}, memory_manager=None, message_bus=None)

# Test all four problematic queries
test_cases = [
    {
        "query": "Create a report analyzing mobile vs. desktop conversion rates across all channels for Q4.",
        "expected_keywords": ["Mobile vs Desktop", "Device Performance"],
    },
    {
        "query": "Show me how our website traffic has trended over the past 6 months.",
        "expected_keywords": ["Traffic Trend Analysis", "Trend Analysis"],
    },
    {
        "query": "Compare email, social, and paid search performance side by side.",
        "expected_keywords": [
            "Channel Performance Comparison",
            "Email Marketing",
            "Social Media",
            "Paid Search",
        ],
    },
    {
        "query": "Show me the conversion funnel for our checkout process.",
        "expected_keywords": ["Conversion Funnel", "Funnel Overview", "Stage-by-Stage"],
    },
]

print("\n" + "=" * 70)
print("TESTING ENHANCED CONTEXTUAL REPORTS")
print("=" * 70)

for i, test in enumerate(test_cases, 1):
    query = test["query"]
    expected = test["expected_keywords"]

    print(f"\n{i}. Testing: {query[:60]}...")

    # Calculate metrics and generate report
    metrics = agent._calculate_metrics(time_range="365d")
    report = agent._generate_contextual_report(metrics, query)

    # Check if we got specific response (not generic)
    is_specific = any(keyword in report for keyword in expected)
    is_generic = "# Analytics Report" in report and "## Key Metrics" in report

    if is_specific and not is_generic:
        print(f"   ✅ PASS - Got specific contextual response")
        print(f"   Found: {[k for k in expected if k in report]}")
    elif is_generic:
        print(f"   ❌ FAIL - Got generic response")
    else:
        print(f"   ⚠️  PARTIAL - Response generated but keywords not found")

    # Show first line of report
    first_line = report.split("\n")[0]
    print(f"   Report Title: {first_line}")

    # Show a snippet of the report
    print(f"\n   Report Preview (first 300 chars):")
    print(f"   {report[:300]}...")

print("\n" + "=" * 70)
print("TESTING COMPLETE")
print("=" * 70)
