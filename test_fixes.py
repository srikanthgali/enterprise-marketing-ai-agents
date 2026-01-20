#!/usr/bin/env python3
"""
Test script to verify all fixes are working correctly.
"""
import requests
import json
import time

API_URL = "http://127.0.0.1:8000/api/v1"


def test_conversion_rate_query():
    """Test conversion rate improvement query."""
    print("\n" + "=" * 80)
    print("TEST 1: Conversion Rate Improvement Query")
    print("=" * 80)

    response = requests.post(
        f"{API_URL}/chat",
        json={
            "message": "Our conversion rate is 2.5%. Now recommend how to improve it."
        },
    )

    data = response.json()

    print(f"✓ Intent: {data.get('intent')}")
    print(f"✓ Confidence: {data.get('confidence')}")
    print(f"✓ Agent: {data.get('agent')}")
    print(f"✓ Agents executed: {data.get('agents_executed')}")

    message = data.get("message", "")

    # Check if user-provided metric (2.5%) is in the response
    if "2.50%" in message or "2.5%" in message:
        print("✅ PASS: User-provided conversion rate (2.5%) is used in analysis")
    else:
        print("❌ FAIL: User-provided conversion rate not found in response")
        print(f"   Response preview: {message[:200]}")

    # Check if detailed recommendations are present
    if (
        "Recommendations to Improve Conversion Rate" in message
        or "Quick Wins" in message
    ):
        print("✅ PASS: Detailed recommendations are provided")
    else:
        print("❌ FAIL: Detailed recommendations not found")

    # Check that it's not a generic learning response
    if "Learning analysis completed" in message:
        print("❌ FAIL: Generic learning response detected")
    else:
        print("✅ PASS: Not a generic response")

    return data


def test_prediction_accuracy_query():
    """Test prediction accuracy query."""
    print("\n" + "=" * 80)
    print("TEST 2: Prediction Accuracy Query")
    print("=" * 80)

    response = requests.post(
        f"{API_URL}/chat",
        json={
            "message": "Are our conversion rate predictions accurate? How can we improve them?"
        },
    )

    data = response.json()

    print(f"✓ Intent: {data.get('intent')}")
    print(f"✓ Agent: {data.get('agent')}")
    print(f"✓ Agents executed: {data.get('agents_executed')}")

    message = data.get("message", "")

    # Check if prediction-specific content is present
    if "Prediction" in message and "Accuracy" in message:
        print("✅ PASS: Prediction accuracy analysis present")
    else:
        print("❌ FAIL: Prediction accuracy analysis not found")

    # Check for recommendations
    if "Recommendations" in message or "recommendations" in message:
        print("✅ PASS: Recommendations are provided")
    else:
        print("❌ FAIL: Recommendations not found")

    return data


def test_handoff_workflows():
    """Test workflows with handoffs."""
    print("\n" + "=" * 80)
    print("TEST 3: Handoff Matrix Data")
    print("=" * 80)

    # Create some workflows with multiple agents
    print("Creating test workflows...")

    queries = [
        "Create a holiday campaign and validate with data",
        "Design a retention strategy",
        "Analyze campaign performance and suggest improvements",
    ]

    for query in queries:
        requests.post(f"{API_URL}/chat", json={"message": query})
        time.sleep(1)

    # Check workflows
    response = requests.get(f"{API_URL}/workflows?limit=10")
    data = response.json()

    workflows = data.get("workflows", [])
    multi_agent_workflows = [
        w for w in workflows if len(w.get("agents_executed", [])) > 1
    ]

    print(f"✓ Total workflows: {len(workflows)}")
    print(f"✓ Multi-agent workflows: {len(multi_agent_workflows)}")

    if multi_agent_workflows:
        print("✅ PASS: Workflows with multiple agents found")
        print("\nHandoff patterns:")
        for w in multi_agent_workflows[:5]:
            agents = w.get("agents_executed", [])
            print(f"  • {' → '.join(agents)}")
    else:
        print("❌ FAIL: No multi-agent workflows found")

    return workflows


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" TESTING ALL FIXES")
    print("=" * 80)

    try:
        # Test API health
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ API is healthy")
        else:
            print("❌ API health check failed")
            return

        # Run tests
        test_conversion_rate_query()
        test_prediction_accuracy_query()
        test_handoff_workflows()

        print("\n" + "=" * 80)
        print(" TEST SUMMARY")
        print("=" * 80)
        print("All tests completed. Check results above.")
        print("\nGradio UI: http://localhost:7860")
        print("Streamlit Dashboard: http://localhost:8501")
        print("\nPlease test in the UIs to verify the fixes work there too.")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
