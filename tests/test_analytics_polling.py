#!/usr/bin/env python3
"""Test analytics with polling."""

import requests
import json
import time

# Submit workflow
response = requests.post(
    "http://localhost:8000/api/v1/workflows/analytics",
    json={
        "report_type": "campaign_performance",
        "date_range": {"start": "2025-12-01", "end": "2026-01-13"},
        "format": "json",
    },
    timeout=60,
)

if response.status_code != 200:
    print(f"Failed to start: {response.text}")
    exit(1)

workflow_id = response.json()["workflow_id"]
print(f"Workflow ID: {workflow_id}")
print("Waiting for completion...")

# Poll for results
for i in range(30):  # 30 seconds max
    time.sleep(1)
    status_response = requests.get(
        f"http://localhost:8000/api/v1/workflows/{workflow_id}"
    )

    if status_response.status_code == 200:
        status_data = status_response.json()
        current_status = status_data.get("status")
        print(
            f"[{i+1}s] Status: {current_status}, Progress: {status_data.get('progress', 0):.0%}"
        )

        if current_status == "completed":
            print("\n✅ Workflow completed!")

            # Check the result
            result = status_data.get("result", {})
            if "analytics" in result:
                analytics = result["analytics"]
                if "metrics" in analytics:
                    metrics = analytics["metrics"]
                    campaign = metrics.get("campaign_metrics", {})
                    print(f"\n📊 Campaign Metrics:")
                    print(f"  CTR: {campaign.get('ctr', 0):.2f}%")
                    print(
                        f"  Conversion Rate: {campaign.get('conversion_rate', 0):.2f}%"
                    )
                    print(f"  ROI: {campaign.get('roi', 0):.2f}%")
                    print(
                        f"  Total Impressions: {campaign.get('total_impressions', 0):,}"
                    )
                    print(
                        f"  Total Conversions: {campaign.get('total_conversions', 0):,}"
                    )
                    print(f"  Total Revenue: ${campaign.get('total_revenue', 0):,.2f}")

                    if campaign.get("total_impressions", 0) > 0:
                        print("\n✅ SUCCESS: Real data from synthetic files!")
                    else:
                        print("\n❌ FAIL: Still showing zeros!")
            break

        elif current_status == "failed":
            print(f"\n❌ Workflow failed: {status_data.get('error')}")
            break
    else:
        print(f"Status check failed: {status_response.status_code}")

print("\nDone.")
