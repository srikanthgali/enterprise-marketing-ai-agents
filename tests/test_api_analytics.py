#!/usr/bin/env python3
"""Quick test of analytics API endpoint."""

import requests
import json

API_URL = "http://localhost:8000/api/v1"


def test_analytics():
    """Test analytics workflow execution."""
    print("\n" + "=" * 80)
    print("Testing Analytics API Endpoint")
    print("=" * 80)

    # Test analytics workflow
    response = requests.post(
        f"{API_URL}/workflows/analytics",
        json={
            "report_type": "campaign_performance",
            "date_range": {"start": "2025-12-01", "end": "2026-01-13"},
            "format": "markdown",
        },
        timeout=30,
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print("\n✅ Response received!")

        # Extract analytics report
        if "analytics" in data:
            analytics = data["analytics"]
            if "report" in analytics:
                report = analytics["report"]
                report_content = report.get("report_content", "")

                print("\n" + "=" * 80)
                print("ANALYTICS REPORT FROM API:")
                print("=" * 80)
                print(report_content[:1000])  # First 1000 chars

                # Check for zeros
                if "0.00%" in report_content:
                    zero_count = report_content.count("0.00%")
                    print(f"\n⚠️  Found {zero_count} instances of '0.00%'")
                    if zero_count > 5:
                        print("❌ Still showing mostly zeros!")
                        return False
                    else:
                        print("✅ Some real data present!")
                        return True
                else:
                    print("\n✅ No zeros found - all real data!")
                    return True
            else:
                print("❌ No 'report' in analytics")
                print(f"Analytics keys: {analytics.keys()}")
        else:
            print("❌ No 'analytics' in response")
            print(f"Response keys: {data.keys()}")

    else:
        print(f"❌ Error: {response.text}")

    return False


if __name__ == "__main__":
    success = test_analytics()
    exit(0 if success else 1)
