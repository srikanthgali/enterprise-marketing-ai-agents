#!/usr/bin/env python3
"""Quick test of analytics API endpoint."""

import requests
import json

# Test via Gradio chat interface which should use analytics agent
response = requests.post(
    "http://localhost:8000/api/v1/workflows/analytics",
    json={
        "report_type": "campaign_performance",
        "date_range": {"start": "2025-12-01", "end": "2026-01-13"},
        "format": "json",
    },
    timeout=60,
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=2)[:2000])
else:
    print(f"Error: {response.text}")
