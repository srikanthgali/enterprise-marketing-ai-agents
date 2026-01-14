#!/bin/bash

# Start All Services: API + Gradio + Streamlit
# Usage: ./start_all.sh [--dev|--prod]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# PID file locations
PIDS_DIR="$PROJECT_ROOT/.pids"
mkdir -p "$PIDS_DIR"
API_PID_FILE="$PIDS_DIR/api.pid"
GRADIO_PID_FILE="$PIDS_DIR/gradio.pid"
STREAMLIT_PID_FILE="$PIDS_DIR/streamlit.pid"

# Log directory
LOGS_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOGS_DIR"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"

    # Kill processes if PID files exist
    if [ -f "$API_PID_FILE" ]; then
        API_PID=$(cat "$API_PID_FILE")
        if kill -0 "$API_PID" 2>/dev/null; then
            echo -e "${YELLOW}Stopping API (PID: $API_PID)...${NC}"
            kill "$API_PID" 2>/dev/null || true
        fi
        rm -f "$API_PID_FILE"
    fi

    if [ -f "$GRADIO_PID_FILE" ]; then
        GRADIO_PID=$(cat "$GRADIO_PID_FILE")
        if kill -0 "$GRADIO_PID" 2>/dev/null; then
            echo -e "${YELLOW}Stopping Gradio (PID: $GRADIO_PID)...${NC}"
            kill "$GRADIO_PID" 2>/dev/null || true
        fi
        rm -f "$GRADIO_PID_FILE"
    fi

    if [ -f "$STREAMLIT_PID_FILE" ]; then
        STREAMLIT_PID=$(cat "$STREAMLIT_PID_FILE")
        if kill -0 "$STREAMLIT_PID" 2>/dev/null; then
            echo -e "${YELLOW}Stopping Streamlit (PID: $STREAMLIT_PID)...${NC}"
            kill "$STREAMLIT_PID" 2>/dev/null || true
        fi
        rm -f "$STREAMLIT_PID_FILE"
    fi

    echo -e "${GREEN}All services stopped${NC}"
    exit 0
}

# Register cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Print banner
echo -e "${BLUE}============================================================================${NC}"
echo -e "${CYAN}   Enterprise Marketing AI Agents - Full Stack Launcher${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}Warning: No virtual environment found at .venv${NC}"
    echo -e "${YELLOW}Consider creating one with: python -m venv .venv${NC}"
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

# Pre-flight check: Clean up existing ports
echo ""
echo -e "${YELLOW}Checking for existing processes on ports...${NC}"
for port in 8000 7860 8501; do
    PIDS=$(lsof -ti:$port 2>/dev/null || true)
    if [ ! -z "$PIDS" ]; then
        echo -e "${YELLOW}Freeing port $port (Killing PIDs: $(echo "$PIDS" | xargs))...${NC}"
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
    fi
done

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${CYAN}   Starting Services...${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# 1. Start FastAPI
echo -e "${CYAN}[1/3] Starting FastAPI server...${NC}"
python scripts/run_api.py --reload > "$LOGS_DIR/api.log" 2>&1 &
API_PID=$!
echo $API_PID > "$API_PID_FILE"
echo -e "${GREEN}✓ API started (PID: $API_PID)${NC}"

# Wait for API to be ready
echo -e "${YELLOW}Waiting for API to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is ready at http://localhost:8000${NC}"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Check if API is actually running
if ! curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo -e "${RED}✗ API failed to start. Check logs/api.log for details${NC}"
    exit 1
fi

echo ""

# 2. Start Gradio
echo -e "${CYAN}[2/3] Starting Gradio UI...${NC}"
python launch_gradio.py > "$LOGS_DIR/gradio.log" 2>&1 &
GRADIO_PID=$!
echo $GRADIO_PID > "$GRADIO_PID_FILE"
echo -e "${GREEN}✓ Gradio started (PID: $GRADIO_PID)${NC}"

# Wait a bit for Gradio to start
sleep 3
echo ""

# 3. Start Streamlit
echo -e "${CYAN}[3/3] Starting Streamlit Dashboard...${NC}"
python scripts/run_dashboard.py > "$LOGS_DIR/streamlit.log" 2>&1 &
STREAMLIT_PID=$!
echo $STREAMLIT_PID > "$STREAMLIT_PID_FILE"
echo -e "${GREEN}✓ Streamlit started (PID: $STREAMLIT_PID)${NC}"

# Wait a bit for Streamlit to start
sleep 3
echo ""

# Print status
echo -e "${BLUE}============================================================================${NC}"
echo -e "${CYAN}   All Services Running!${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${GREEN}API Server:${NC}        http://localhost:8000"
echo -e "  ${YELLOW}→ Swagger UI:${NC}    http://localhost:8000/api/docs"
echo -e "  ${YELLOW}→ ReDoc:${NC}         http://localhost:8000/api/redoc"
echo -e "  ${YELLOW}→ Health Check:${NC}  http://localhost:8000/api/v1/health"
echo ""
echo -e "${GREEN}Gradio UI:${NC}         http://localhost:7860"
echo ""
echo -e "${GREEN}Streamlit Dashboard:${NC} http://localhost:8501"
echo ""
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${YELLOW}Logs are available at:${NC}"
echo -e "  - API:       $LOGS_DIR/api.log"
echo -e "  - Gradio:    $LOGS_DIR/gradio.log"
echo -e "  - Streamlit: $LOGS_DIR/streamlit.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running and wait for signals
wait
