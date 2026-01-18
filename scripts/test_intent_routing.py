#!/usr/bin/env python
"""
Test Script for LLM-Based Intent Routing System.

Tests all use cases including handoffs between agents:
- Customer Support → Analytics
- Analytics → Feedback Learning
- Analytics → Marketing Strategy
- Direct agent routing
"""

import asyncio
import httpx
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
TIMEOUT = 300.0

console = Console()


class TestCase:
    """Represents a test case with expected behavior."""

    def __init__(
        self,
        name: str,
        message: str,
        expected_intent: str,
        expected_initial_agent: str,
        expected_handoffs: List[Dict[str, str]] = None,
        description: str = "",
    ):
        self.name = name
        self.message = message
        self.expected_intent = expected_intent
        self.expected_initial_agent = expected_initial_agent
        self.expected_handoffs = expected_handoffs or []
        self.description = description


# Define all test cases
TEST_CASES = [
    # ========== HANDOFF TEST CASES ==========
    TestCase(
        name="Handoff: Customer Support → Analytics",
        message="Customer satisfaction scores dropped 15% this month. What's causing this?",
        expected_intent="customer_support",
        expected_initial_agent="customer_support",
        expected_handoffs=[{"from": "customer_support", "to": "analytics_evaluation"}],
        description="Customer satisfaction issue should start with support, then handoff to analytics for investigation",
    ),
    TestCase(
        name="Handoff: Analytics → Feedback Learning (Campaign Improvements)",
        message="Recommend improvements for our campaign performance",
        expected_intent="feedback_analysis",
        expected_initial_agent="feedback_learning",
        expected_handoffs=[],
        description="Campaign improvement recommendations should go directly to feedback learning",
    ),
    TestCase(
        name="Handoff: Analytics → Feedback Learning (Prediction Accuracy)",
        message="Are our conversion rate predictions accurate? How can we improve them?",
        expected_intent="feedback_analysis",
        expected_initial_agent="feedback_learning",
        expected_handoffs=[],
        description="Questions about prediction accuracy should go to feedback learning",
    ),
    TestCase(
        name="Handoff: Analytics → Marketing Strategy (Positioning)",
        message="What positioning strategy should we use to stand out in the payment processing market?",
        expected_intent="market_analysis",
        expected_initial_agent="marketing_strategy",
        expected_handoffs=[],
        description="Market positioning questions should go to marketing strategy",
    ),
    TestCase(
        name="Handoff: Analytics → Marketing Strategy (Conversion Optimization)",
        message="Our conversion rates are falling. Need help optimizing our approach.",
        expected_intent="performance_analytics",
        expected_initial_agent="analytics_evaluation",
        expected_handoffs=[
            {"from": "analytics_evaluation", "to": "marketing_strategy"}
        ],
        description="Falling conversion rates should trigger analytics analysis, then handoff to marketing strategy",
    ),
    # ========== CUSTOMER SUPPORT TEST CASES ==========
    TestCase(
        name="Customer Support: Technical Error",
        message="Our checkout is throwing a 400 error when customers try to pay with saved cards",
        expected_intent="customer_support",
        expected_initial_agent="customer_support",
        expected_handoffs=[],
        description="Technical errors should route to customer support",
    ),
    TestCase(
        name="Customer Support: Implementation Question",
        message="How do I implement webhooks for payment confirmations?",
        expected_intent="customer_support",
        expected_initial_agent="customer_support",
        expected_handoffs=[],
        description="Implementation questions should route to customer support",
    ),
    # ========== MARKETING STRATEGY TEST CASES ==========
    TestCase(
        name="Marketing Strategy: Campaign Creation",
        message="I need to create a marketing campaign to promote our new payment processing feature for small businesses.",
        expected_intent="campaign_creation",
        expected_initial_agent="marketing_strategy",
        expected_handoffs=[],
        description="Campaign creation requests should route to marketing strategy",
    ),
    # ========== ANALYTICS TEST CASES ==========
    TestCase(
        name="Analytics: Monthly Report",
        message="Generate a monthly performance report for December.",
        expected_intent="performance_analytics",
        expected_initial_agent="analytics_evaluation",
        expected_handoffs=[],
        description="Report generation should route to analytics",
    ),
    TestCase(
        name="Analytics: Conversion Funnel",
        message="Show me the conversion funnel for our checkout process.",
        expected_intent="performance_analytics",
        expected_initial_agent="analytics_evaluation",
        expected_handoffs=[],
        description="Funnel analysis should route to analytics",
    ),
    # ========== FEEDBACK LEARNING TEST CASES ==========
    TestCase(
        name="Feedback Learning: Quality Rating",
        message="Rate the quality of the campaign strategy provided by the Marketing Agent: 2/5 stars. It was too generic.",
        expected_intent="feedback_analysis",
        expected_initial_agent="feedback_learning",
        expected_handoffs=[],
        description="Quality ratings should route to feedback learning",
    ),
    TestCase(
        name="Feedback Learning: Pattern Detection",
        message="Multiple agents are reporting that mobile checkout has issues. Can you investigate and recommend improvements?",
        expected_intent="feedback_analysis",
        expected_initial_agent="feedback_learning",
        expected_handoffs=[],
        description="Pattern detection and issue investigation should route to feedback learning",
    ),
]


class TestResult:
    """Stores test execution results."""

    def __init__(self, test_case: TestCase):
        self.test_case = test_case
        self.passed = False
        self.actual_intent = None
        self.actual_confidence = None
        self.actual_agents = []
        self.actual_handoffs = []
        self.error = None
        self.response_time = None
        self.response_message = None


async def run_test_case(client: httpx.AsyncClient, test_case: TestCase) -> TestResult:
    """
    Execute a single test case.

    Args:
        client: HTTP client
        test_case: Test case to execute

    Returns:
        Test result with pass/fail status
    """
    result = TestResult(test_case)
    start_time = datetime.now()

    try:
        # Call unified chat endpoint
        response = await client.post(
            f"{API_BASE_URL}/chat", json={"message": test_case.message}, timeout=TIMEOUT
        )

        result.response_time = (datetime.now() - start_time).total_seconds()

        if response.status_code == 200:
            data = response.json()

            # Extract results
            result.actual_intent = data.get("intent")
            result.actual_confidence = data.get("confidence")
            result.actual_agents = data.get("agents_executed", [])
            result.actual_handoffs = data.get("handoffs", [])
            result.response_message = data.get("message", "")

            # Validate results
            intent_match = result.actual_intent == test_case.expected_intent
            initial_agent_match = (
                len(result.actual_agents) > 0
                and result.actual_agents[0] == test_case.expected_initial_agent
            )

            # Check handoffs (allow for actual handoffs to have more detail)
            handoff_match = True
            if test_case.expected_handoffs:
                if len(result.actual_handoffs) != len(test_case.expected_handoffs):
                    handoff_match = False
                else:
                    for expected, actual in zip(
                        test_case.expected_handoffs, result.actual_handoffs
                    ):
                        if expected["from"] != actual.get("from") or expected[
                            "to"
                        ] != actual.get("to"):
                            handoff_match = False
                            break

            result.passed = intent_match and initial_agent_match and handoff_match

        else:
            result.error = f"API Error: {response.status_code} - {response.text}"
            result.passed = False

    except Exception as e:
        result.error = str(e)
        result.passed = False
        result.response_time = (datetime.now() - start_time).total_seconds()

    return result


async def run_all_tests():
    """Run all test cases and display results."""

    console.print(
        "\n[bold cyan]🧪 Testing LLM-Based Intent Routing System[/bold cyan]\n"
    )
    console.print(f"API Endpoint: {API_BASE_URL}")
    console.print(f"Test Cases: {len(TEST_CASES)}\n")

    # Check if API is running
    try:
        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"{API_BASE_URL}/health")
            if health_response.status_code != 200:
                console.print(
                    "[bold red]❌ API is not healthy. Please start the API server.[/bold red]"
                )
                return
    except Exception as e:
        console.print(f"[bold red]❌ Cannot connect to API: {e}[/bold red]")
        console.print("[yellow]Start the API with: python scripts/run_api.py[/yellow]")
        return

    console.print("[green]✓ API is running[/green]\n")

    # Run tests
    results = []
    async with httpx.AsyncClient() as client:
        for i, test_case in enumerate(TEST_CASES, 1):
            console.print(
                f"[cyan]Running test {i}/{len(TEST_CASES)}:[/cyan] {test_case.name}"
            )

            result = await run_test_case(client, test_case)
            results.append(result)

            # Show immediate result
            if result.passed:
                console.print(
                    f"  [green]✓ PASSED[/green] ({result.response_time:.2f}s)"
                )
            else:
                console.print(f"  [red]✗ FAILED[/red] ({result.response_time:.2f}s)")
                if result.error:
                    console.print(f"    [red]Error: {result.error}[/red]")

            console.print()

    # Display summary
    display_results_summary(results)


def display_results_summary(results: List[TestResult]):
    """Display comprehensive test results summary."""

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    success_rate = (passed / len(results)) * 100 if results else 0

    # Summary panel
    console.print("\n" + "=" * 80)
    console.print(
        Panel.fit(
            f"[bold]Test Results Summary[/bold]\n\n"
            f"Total Tests: {len(results)}\n"
            f"[green]Passed: {passed}[/green]\n"
            f"[red]Failed: {failed}[/red]\n"
            f"Success Rate: {success_rate:.1f}%",
            title="📊 Results",
            border_style="cyan",
        )
    )

    # Detailed results table
    table = Table(
        title="\nDetailed Test Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Test Case", style="cyan", width=40)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Intent", width=20)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Agents", width=30)
    table.add_column("Handoffs", width=15)
    table.add_column("Time", justify="right", width=8)

    for result in results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        status_style = "green" if result.passed else "red"

        intent = result.actual_intent or "N/A"
        confidence = (
            f"{result.actual_confidence:.2f}" if result.actual_confidence else "N/A"
        )
        agents = " → ".join(result.actual_agents) if result.actual_agents else "N/A"
        handoffs_count = len(result.actual_handoffs)
        handoffs_str = f"{handoffs_count} handoff(s)" if handoffs_count > 0 else "None"
        time_str = f"{result.response_time:.2f}s" if result.response_time else "N/A"

        table.add_row(
            result.test_case.name,
            f"[{status_style}]{status}[/{status_style}]",
            intent,
            confidence,
            agents,
            handoffs_str,
            time_str,
        )

    console.print(table)

    # Failed tests details
    if failed > 0:
        console.print("\n[bold red]Failed Tests Details:[/bold red]\n")
        for result in results:
            if not result.passed:
                console.print(
                    Panel(
                        f"[bold]Test:[/bold] {result.test_case.name}\n"
                        f"[bold]Message:[/bold] {result.test_case.message}\n\n"
                        f"[yellow]Expected:[/yellow]\n"
                        f"  Intent: {result.test_case.expected_intent}\n"
                        f"  Initial Agent: {result.test_case.expected_initial_agent}\n"
                        f"  Handoffs: {result.test_case.expected_handoffs}\n\n"
                        f"[red]Actual:[/red]\n"
                        f"  Intent: {result.actual_intent}\n"
                        f"  Agents: {result.actual_agents}\n"
                        f"  Handoffs: {result.actual_handoffs}\n"
                        f"  Error: {result.error}",
                        title=f"❌ {result.test_case.name}",
                        border_style="red",
                    )
                )

    # Handoff analysis
    console.print("\n[bold cyan]Handoff Analysis:[/bold cyan]\n")
    handoff_tests = [r for r in results if r.test_case.expected_handoffs]
    if handoff_tests:
        for result in handoff_tests:
            handoff_status = "✓" if result.passed else "✗"
            color = "green" if result.passed else "red"
            console.print(
                f"[{color}]{handoff_status}[/{color}] {result.test_case.name}"
            )
            console.print(f"   Expected: {result.test_case.expected_handoffs}")
            console.print(f"   Actual:   {result.actual_handoffs}\n")
    else:
        console.print("No handoff test cases found.\n")


def main():
    """Main entry point."""
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Test execution failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()
