#!/usr/bin/env python3
"""Test script to see the exact API response structure"""
import httpx
import json
import time

# 1. Create workflow
print("Creating analytics workflow...")
response = httpx.post(
    "http://localhost:8000/api/v1/workflows/analytics",
    json={
        "report_type": "campaign_performance",
        "date_range": {"start": "2026-01-01", "end": "2026-01-31"},
        "metrics": ["conversion_rate", "roi", "engagement"],
        "filters": {"user_query": "Show me the conversion funnel"},
    },
    timeout=30.0,
)
print(f"Create response: {response.status_code}")
workflow_data = response.json()
workflow_id = workflow_data["workflow_id"]
print(f"Workflow ID: {workflow_id}")

# 2. Wait for completion
print("\nWaiting for workflow to complete...")
for i in range(20):
    time.sleep(1)
    status_response = httpx.get(
        f"http://localhost:8000/api/v1/workflows/{workflow_id}",
        timeout=10.0,
    )
    status_data = status_response.json()
    print(f"Status: {status_data['status']}", end="\r")
    if status_data["status"] in ["completed", "failed"]:
        break

# 3. Get results
print("\n\nFetching results...")
results_response = httpx.get(
    f"http://localhost:8000/api/v1/workflows/{workflow_id}/results",
    timeout=10.0,
)
results_data = results_response.json()

print("\n=== FULL RESPONSE ===")
print(json.dumps(results_data, indent=2))

# 4. Check specific fields
print("\n=== CHECKING ANALYTICS FIELD ===")
if "results" in results_data:
    print(f"results type: {type(results_data['results'])}")
    print(
        f"results keys: {results_data['results'].keys() if isinstance(results_data['results'], dict) else 'not a dict'}"
    )

    if "analytics" in results_data["results"]:
        analytics = results_data["results"]["analytics"]
        print(f"\nanalytics type: {type(analytics)}")
        print(
            f"analytics keys: {analytics.keys() if isinstance(analytics, dict) else 'not a dict'}"
        )

        if isinstance(analytics, dict) and "report" in analytics:
            report = analytics["report"]
            print(f"\nreport type: {type(report)}")
            print(f"report length: {len(report) if isinstance(report, str) else 'N/A'}")
            if isinstance(report, str):
                print(f"\nFirst 300 chars of report:\n{report[:300]}")
