"""
Analytics Agent Test Script

Tests Analytics & Evaluation Agent with realistic queries from the test guide.
Based on ANALYTICS_AGENT_TEST_GUIDE.md
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, Any


# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
TIMEOUT = 120.0


class AnalyticsAgentTester:
    """Test harness for Analytics & Evaluation Agent."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        self.test_results = []

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    async def test_analytics_workflow(
        self, test_name: str, query: str, report_type: str = "campaign_performance"
    ) -> Dict[str, Any]:
        """
        Test analytics workflow with a query.

        Args:
            test_name: Name of the test
            query: User query to analyze
            report_type: Type of report to generate

        Returns:
            Test result with status and details
        """
        print(f"\n{'='*80}")
        print(f"TEST: {test_name}")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Report Type: {report_type}")
        print("-" * 80)

        try:
            # Create analytics request payload
            payload = {
                "report_type": report_type,
                "date_range": {"start": "2026-01-01", "end": "2026-01-31"},
                "metrics": ["conversion_rate", "roi", "engagement"],
                "filters": {"user_query": query},
            }

            # Submit workflow
            print("📤 Submitting analytics workflow...")
            response = await self.client.post(
                f"{API_BASE_URL}/workflows/analytics", json=payload
            )

            if response.status_code != 200:
                error_msg = f"Failed to submit workflow: {response.status_code}"
                print(f"❌ {error_msg}")
                return {
                    "test_name": test_name,
                    "status": "failed",
                    "error": error_msg,
                    "query": query,
                }

            workflow_data = response.json()
            workflow_id = workflow_data.get("workflow_id")
            print(f"✅ Workflow submitted: {workflow_id}")

            # Poll for completion
            print("⏳ Waiting for workflow completion...")
            max_polls = 60  # 60 polls * 2 seconds = 2 minutes
            poll_count = 0

            while poll_count < max_polls:
                poll_count += 1
                await asyncio.sleep(2)

                # Get workflow status
                status_response = await self.client.get(
                    f"{API_BASE_URL}/workflows/{workflow_id}"
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    current_status = status_data.get("status")
                    progress = status_data.get("progress", 0)

                    print(
                        f"   Status: {current_status} (progress: {progress:.0%})",
                        end="\r",
                    )

                    if current_status == "completed":
                        print("\n✅ Workflow completed!")

                        # Get results
                        result_response = await self.client.get(
                            f"{API_BASE_URL}/workflows/{workflow_id}/results"
                        )

                        if result_response.status_code == 200:
                            result_data = result_response.json()

                            # Extract agent execution info
                            agents_executed = status_data.get("agents_executed", [])
                            analytics_triggered = (
                                "analytics_evaluation" in agents_executed
                            )

                            print(f"\n📊 Agents Executed: {agents_executed}")
                            print(
                                f"🎯 Analytics Agent Triggered: {'YES ✅' if analytics_triggered else 'NO ❌'}"
                            )

                            # Extract result
                            results = result_data.get("results", {})
                            if isinstance(results, dict):
                                if "final_result" in results:
                                    results = results["final_result"]
                                elif "result" in results:
                                    results = results["result"]

                            # Show summary
                            if isinstance(results, dict):
                                summary = results.get("summary", "No summary available")
                                print(f"\n📝 Summary: {summary}")

                                if "analytics" in results:
                                    analytics_data = results["analytics"]
                                    print(f"\n📈 Analytics Data:")
                                    print(json.dumps(analytics_data, indent=2)[:500])

                            return {
                                "test_name": test_name,
                                "status": "passed" if analytics_triggered else "failed",
                                "query": query,
                                "workflow_id": workflow_id,
                                "agents_executed": agents_executed,
                                "analytics_triggered": analytics_triggered,
                                "result": results,
                            }

                    elif current_status == "failed":
                        error = status_data.get("error", "Unknown error")
                        print(f"\n❌ Workflow failed: {error}")
                        return {
                            "test_name": test_name,
                            "status": "failed",
                            "query": query,
                            "workflow_id": workflow_id,
                            "error": error,
                        }

            # Timeout
            print(f"\n⏰ Timeout after {max_polls * 2} seconds")
            return {
                "test_name": test_name,
                "status": "timeout",
                "query": query,
                "workflow_id": workflow_id,
            }

        except Exception as e:
            print(f"\n❌ Exception: {str(e)}")
            return {
                "test_name": test_name,
                "status": "error",
                "query": query,
                "error": str(e),
            }

    async def run_all_tests(self):
        """Run all test cases from the test guide."""

        # Test cases based on ANALYTICS_AGENT_TEST_GUIDE.md
        # Note: report_type must be one of: campaign_performance, customer_engagement, revenue, trends
        test_cases = [
            # 1. Metrics Calculation & KPI Tracking
            {
                "name": "Test #1: Basic Metrics - Conversion Rate",
                "query": "What's the conversion rate for our Q4 campaign?",
                "report_type": "campaign_performance",
            },
            {
                "name": "Test #2: Multiple KPI Dashboard",
                "query": "Show me all key metrics for last month's email campaigns.",
                "report_type": "campaign_performance",
            },
            {
                "name": "Test #3: Real-Time Monitoring",
                "query": "How is our Black Friday campaign performing right now?",
                "report_type": "campaign_performance",
            },
            # 2. Report Generation
            {
                "name": "Test #4: Standard Performance Report",
                "query": "Generate a monthly performance report for December.",
                "report_type": "campaign_performance",
            },
            {
                "name": "Test #5: Custom Report - Device Analysis",
                "query": "Create a report analyzing mobile vs. desktop conversion rates across all channels for Q4.",
                "report_type": "campaign_performance",
            },
            {
                "name": "Test #6: Executive Summary",
                "query": "I need a one-page executive summary for the board meeting tomorrow.",
                "report_type": "revenue",
            },
            # 3. Data Visualization
            {
                "name": "Test #7: Trend Visualization",
                "query": "Show me how our website traffic has trended over the past 6 months.",
                "report_type": "trends",
            },
            {
                "name": "Test #8: Comparative Visualization",
                "query": "Compare email, social, and paid search performance side by side.",
                "report_type": "campaign_performance",
            },
            {
                "name": "Test #9: Funnel Visualization",
                "query": "Show me the conversion funnel for our checkout process.",
                "report_type": "campaign_performance",
            },
            # 4. Statistical Analysis
            {
                "name": "Test #10: Correlation Analysis",
                "query": "Is there a correlation between our email send frequency and unsubscribe rate?",
                "report_type": "customer_engagement",
            },
            {
                "name": "Test #11: Segment Performance",
                "query": "Do enterprise customers convert better than SMB customers?",
                "report_type": "customer_engagement",
            },
            {
                "name": "Test #12: Regression Analysis",
                "query": "What factors most influence our conversion rate?",
                "report_type": "campaign_performance",
            },
            # 5. A/B Test Analysis
            {
                "name": "Test #13: A/B Test Results",
                "query": "Analyze the results of our landing page A/B test that ran last week.",
                "report_type": "campaign_performance",
            },
            {
                "name": "Test #14: A/B Test Design",
                "query": "I want to test two different email subject lines. How should I set up the experiment?",
                "report_type": "campaign_performance",
            },
            # 6. Attribution Modeling
            {
                "name": "Test #16: Multi-Touch Attribution",
                "query": "How do different marketing channels contribute to conversions?",
                "report_type": "campaign_performance",
            },
            {
                "name": "Test #17: Customer Journey",
                "query": "What's the typical path customers take before purchasing?",
                "report_type": "customer_engagement",
            },
            # 7. Forecasting & Prediction
            {
                "name": "Test #18: Performance Forecast",
                "query": "What conversion rate can we expect next quarter?",
                "report_type": "trends",
            },
            {
                "name": "Test #19: Budget Impact Forecast",
                "query": "If we increase our paid search budget by 30%, what impact will that have?",
                "report_type": "revenue",
            },
            # 8. Anomaly Detection
            {
                "name": "Test #21: Performance Drop Alert",
                "query": "Our email open rates suddenly dropped. What's going on?",
                "report_type": "campaign_performance",
            },
            {
                "name": "Test #22: Unusual Pattern Detection",
                "query": "Are there any unusual patterns in our campaign data I should know about?",
                "report_type": "trends",
            },
        ]

        print("\n" + "=" * 80)
        print("ANALYTICS AGENT TEST SUITE")
        print("=" * 80)
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total tests: {len(test_cases)}")
        print("=" * 80)

        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n\nRunning test {i}/{len(test_cases)}...")
            result = await self.test_analytics_workflow(
                test_name=test_case["name"],
                query=test_case["query"],
                report_type=test_case["report_type"],
            )
            results.append(result)

            # Brief pause between tests
            if i < len(test_cases):
                await asyncio.sleep(2)

        # Summary
        print("\n\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in results if r.get("status") == "passed")
        failed = sum(1 for r in results if r.get("status") == "failed")
        errors = sum(1 for r in results if r.get("status") == "error")
        timeouts = sum(1 for r in results if r.get("status") == "timeout")

        print(f"✅ Passed: {passed}/{len(results)}")
        print(f"❌ Failed: {failed}/{len(results)}")
        print(f"⚠️  Errors: {errors}/{len(results)}")
        print(f"⏰ Timeouts: {timeouts}/{len(results)}")

        # Show failed tests
        if failed > 0 or errors > 0:
            print("\n" + "-" * 80)
            print("FAILED/ERROR TESTS:")
            print("-" * 80)
            for result in results:
                if result.get("status") in ["failed", "error"]:
                    print(f"\n❌ {result['test_name']}")
                    print(f"   Query: {result['query']}")
                    if "error" in result:
                        print(f"   Error: {result['error']}")
                    if "agents_executed" in result:
                        print(f"   Agents Executed: {result['agents_executed']}")
                    print(
                        f"   Analytics Triggered: {result.get('analytics_triggered', False)}"
                    )

        # Show analytics agent trigger rate
        analytics_triggered_count = sum(
            1 for r in results if r.get("analytics_triggered", False)
        )
        print("\n" + "-" * 80)
        print(
            f"📊 Analytics Agent Trigger Rate: {analytics_triggered_count}/{len(results)} ({analytics_triggered_count/len(results)*100:.1f}%)"
        )
        print("-" * 80)

        return results


async def main():
    """Main test runner."""
    tester = AnalyticsAgentTester()

    try:
        # Check API health
        print("🏥 Checking API health...")
        try:
            health_response = await tester.client.get(f"{API_BASE_URL}/health")
            if health_response.status_code == 200:
                print("✅ API is healthy and responding")
            else:
                print(f"⚠️  API returned status: {health_response.status_code}")
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            print(f"   Make sure the API is running at {API_BASE_URL}")
            return

        # Run all tests
        await tester.run_all_tests()

    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
