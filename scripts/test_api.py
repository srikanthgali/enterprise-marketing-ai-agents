"""
Test script for the FastAPI application.

Run this to verify the API is working correctly.
"""

import requests
import json
import time
import sys


BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_health_check():
    """Test the health check endpoint."""
    print_section("Testing Health Check")

    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_list_agents():
    """Test listing all agents."""
    print_section("Testing List Agents")

    try:
        response = requests.get(f"{BASE_URL}/api/v1/agents")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Found {len(data)} agents:")
        for agent in data:
            print(f"  - {agent['agent_id']}: {agent['name']} ({agent['status']})")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_campaign_launch():
    """Test campaign launch workflow."""
    print_section("Testing Campaign Launch Workflow")

    payload = {
        "campaign_name": "Test Campaign Q1 2026",
        "objectives": ["awareness", "leads"],
        "target_audience": "B2B SaaS companies",
        "budget": 50000,
        "duration_weeks": 8,
        "channels": ["email", "social"],
    }

    try:
        print("Launching campaign...")
        response = requests.post(
            f"{BASE_URL}/api/v1/workflows/campaign-launch",
            json=payload,
        )
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Workflow ID: {data['workflow_id']}")
        print(f"Status: {data['status']}")

        workflow_id = data["workflow_id"]

        # Wait a bit for processing
        print("\nWaiting 3 seconds for workflow to process...")
        time.sleep(3)

        # Check status
        print("\nChecking workflow status...")
        status_response = requests.get(f"{BASE_URL}/api/v1/workflows/{workflow_id}")
        print(f"Status Code: {status_response.status_code}")
        status_data = status_response.json()
        print(f"Current Status: {status_data['status']}")
        print(f"Progress: {status_data['progress']}%")

        return response.status_code == 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_customer_support():
    """Test customer support workflow."""
    print_section("Testing Customer Support Workflow")

    payload = {
        "inquiry": "How do I reset my API key?",
        "customer_id": "test_customer_123",
        "urgency": "high",
    }

    try:
        print("Submitting support inquiry...")
        response = requests.post(
            f"{BASE_URL}/api/v1/workflows/customer-support",
            json=payload,
        )
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Workflow ID: {data['workflow_id']}")
        print(f"Status: {data['status']}")

        return response.status_code == 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_metrics():
    """Test metrics endpoint."""
    print_section("Testing Metrics")

    try:
        response = requests.get(f"{BASE_URL}/api/v1/metrics")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"\nAgent Metrics:")
        for agent_id, metrics in data["agents"].items():
            print(f"  {agent_id}: {metrics['total_executions']} executions")

        print(f"\nWorkflow Metrics:")
        print(f"  Total: {data['workflows']['total']}")
        print(f"  Completed: {data['workflows']['completed']}")
        print(f"  In Progress: {data['workflows']['in_progress']}")

        return response.status_code == 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  FastAPI Application Test Suite")
    print("  Make sure the API is running on http://localhost:8000")
    print("=" * 70)

    # Check if server is reachable
    try:
        response = requests.get(BASE_URL, timeout=2)
        print("✓ Server is reachable")
    except Exception:
        print("❌ Server is not reachable. Please start the API first:")
        print("   python scripts/run_api.py")
        sys.exit(1)

    # Run tests
    results = []
    results.append(("Health Check", test_health_check()))
    results.append(("List Agents", test_list_agents()))
    results.append(("Campaign Launch", test_campaign_launch()))
    results.append(("Customer Support", test_customer_support()))
    results.append(("Metrics", test_metrics()))

    # Print summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
