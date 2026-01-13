#!/bin/bash

# Start Gradio UI for Enterprise Marketing AI Agents
# Usage: ./start_gradio.sh [--dev|--prod]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}Warning: No virtual environment found at .venv${NC}"
    echo -e "${YELLOW}Consider creating one with: python -m venv .venv${NC}"
fi

# Check if FastAPI is running
echo -e "${GREEN}Checking FastAPI server...${NC}"
if ! curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo -e "${RED}Error: FastAPI server is not running on port 8000${NC}"
    echo -e "${YELLOW}Please start it first with: ./start_api.sh or uvicorn api.main:app --reload${NC}"
    exit 1
fi

echo -e "${GREEN}FastAPI server is running ✓${NC}"

# Check if Gradio is installed
if ! python -c "import gradio" 2>/dev/null; then
    echo -e "${YELLOW}Gradio not found. Installing...${NC}"
    pip install gradio>=5.7.0
fi

# Set environment mode
MODE="${1:-dev}"
if [ "$MODE" = "--prod" ] || [ "$MODE" = "prod" ]; then
    export ENVIRONMENT="production"
    echo -e "${GREEN}Running in PRODUCTION mode${NC}"
elif [ "$MODE" = "--dev" ] || [ "$MODE" = "dev" ]; then
    export ENVIRONMENT="development"
    echo -e "${GREEN}Running in DEVELOPMENT mode${NC}"
else
    export ENVIRONMENT="development"
    echo -e "${GREEN}Running in DEVELOPMENT mode (default)${NC}"
fi

# Create necessary directories
mkdir -p data/conversations
mkdir -p logs

# Start Gradio UI
echo ""
echo "========================================================================"
echo "  🚀 Starting Gradio Conversational UI"
echo "========================================================================"
echo ""
echo "  📡 API Backend: http://localhost:8000"
echo "  🌐 Gradio UI:   http://localhost:7860"
echo ""
echo "  Press Ctrl+C to stop"
echo ""
echo "========================================================================"
echo ""

# Run Gradio
python ui/gradio_app.py
