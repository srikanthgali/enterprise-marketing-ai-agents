"""
Test script for Streamlit Dashboard integration with API.

This script verifies that the dashboard can connect to the API
and retrieve data correctly.
"""

import requests
import sys
from typing import Dict, Optional

API_BASE_URL = "http://127.0.0.1:8000/api/v1"


def test_api_connection() -> bool:
    """Test if API is accessible."""
    print("Testing API connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API connection successful")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_get_agents() -> bool:
    """Test retrieving agents list."""
    print("\nTesting agents endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/agents", timeout=10)
        response.raise_for_status()
        agents = response.json()

        if isinstance(agents, list):
            print(f"✅ Retrieved {len(agents)} agents")
            if agents:
                print(f"   Sample agent: {agents[0].get('name', 'Unknown')}")
            return True
        else:
            print("❌ Unexpected response format")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_get_workflows() -> bool:
    """Test retrieving workflows list."""
    print("\nTesting workflows endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/workflows?limit=10", timeout=10)
        response.raise_for_status()
        workflows = response.json()

        if isinstance(workflows, list):
            print(f"✅ Retrieved {len(workflows)} workflows")
            return True
        else:
            print("❌ Unexpected response format")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_agent_details() -> bool:
    """Test retrieving agent details."""
    print("\nTesting agent details endpoint...")
    try:
        # First get list of agents
        response = requests.get(f"{API_BASE_URL}/agents", timeout=10)
        response.raise_for_status()
        agents = response.json()

        if not agents:
            print("⚠️  No agents available to test")
            return True

        # Test first agent
        agent_id = agents[0].get("agent_id")
        response = requests.get(f"{API_BASE_URL}/agents/{agent_id}", timeout=10)
        response.raise_for_status()
        agent_details = response.json()

        print(f"✅ Retrieved details for agent: {agent_details.get('name', 'Unknown')}")
        print(f"   Status: {agent_details.get('status', 'N/A')}")
        print(f"   Capabilities: {len(agent_details.get('capabilities', []))}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_agent_history() -> bool:
    """Test retrieving agent execution history."""
    print("\nTesting agent history endpoint...")
    try:
        # Get first agent
        response = requests.get(f"{API_BASE_URL}/agents", timeout=10)
        response.raise_for_status()
        agents = response.json()

        if not agents:
            print("⚠️  No agents available to test")
            return True

        # Get history for first agent
        agent_id = agents[0].get("agent_id")
        response = requests.get(
            f"{API_BASE_URL}/agents/{agent_id}/history?limit=10", timeout=10
        )
        response.raise_for_status()
        history = response.json()

        total = history.get("total_executions", 0)
        items = len(history.get("history", []))
        print(f"✅ Retrieved execution history: {items} items (total: {total})")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all dashboard integration tests."""
    print("=" * 80)
    print("Streamlit Dashboard - API Integration Tests")
    print("=" * 80)
    print()

    tests = [
        ("API Connection", test_api_connection),
        ("Get Agents", test_get_agents),
        ("Get Workflows", test_get_workflows),
        ("Agent Details", test_agent_details),
        ("Agent History", test_agent_history),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)

    if passed == total:
        print("\n✅ All tests passed! Dashboard is ready to use.")
        print("\nStart the dashboard with:")
        print("  python scripts/run_dashboard.py")
        print("  or")
        print("  ./start_dashboard.sh")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the API server and configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
