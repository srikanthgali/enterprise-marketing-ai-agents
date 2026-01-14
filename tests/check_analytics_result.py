#!/usr/bin/env python3
"""Check analytics result."""

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

workflow_id = response.json()["workflow_id"]
print(f"Workflow ID: {workflow_id}")

# Wait a bit
time.sleep(3)

# Get full result
status_response = requests.get(f"http://localhost:8000/api/v1/workflows/{workflow_id}")

if status_response.status_code == 200:
    data = status_response.json()
    print("\nFull result:")
    print(json.dumps(data, indent=2))
