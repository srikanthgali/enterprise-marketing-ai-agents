#!/bin/bash
# Quick start script for the Streamlit Dashboard

set -e

echo "=============================================================================="
echo "Enterprise Marketing AI Agents - Dashboard Quick Start"
echo "=============================================================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated. Activating..."
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Virtual environment not found. Run: python -m venv .venv"
        exit 1
    fi
fi

# Check if API is running
echo ""
echo "Checking API status..."
if curl -s --connect-timeout 2 --max-time 3 http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "✅ API is running at http://localhost:8000"
else
    echo "⚠️  API is not running. Starting API server..."
    echo ""

    # Create logs directory if it doesn't exist
    mkdir -p logs

    # Start API in background
    python scripts/run_api.py > logs/api.log 2>&1 &
    API_PID=$!
    echo "API server started with PID: $API_PID"

    # Wait for API to be ready
    echo "Waiting for API to be ready..."
    for i in {1..30}; do
        if curl -s --connect-timeout 1 --max-time 2 http://localhost:8000/api/v1/health > /dev/null 2>&1; then
            echo ""
            echo "✅ API is ready"
            break
        fi
        sleep 1
        echo -n "."
    done
    echo ""
fi

# Start Streamlit dashboard
echo ""
echo "=============================================================================="
echo "Starting Streamlit Dashboard..."
echo "=============================================================================="
echo ""
echo "Dashboard will be available at: http://localhost:8501"
echo ""
echo "Features:"
echo "  - 📈 Real-time system monitoring"
echo "  - 🤖 Agent management and testing"
echo "  - 🔄 Workflow tracking"
echo "  - 📊 Analytics and insights"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "=============================================================================="
echo ""

# Run Streamlit
python scripts/run_dashboard.py
