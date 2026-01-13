#!/bin/bash

# Stop All Services
# Usage: ./stop_all.sh

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_DIR="$PROJECT_ROOT/.pids"

echo -e "${YELLOW}Stopping all services...${NC}"
echo ""

# Function to stop a service
stop_service() {
    local service_name=$1
    local pid_file=$2

    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}Stopping $service_name (PID: $PID)...${NC}"
            kill "$PID" 2>/dev/null || true
            sleep 1
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID" 2>/dev/null || true
            fi
            echo -e "${GREEN}✓ $service_name stopped${NC}"
        else
            echo -e "${YELLOW}$service_name is not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}No PID file for $service_name${NC}"
    fi
}

# Stop each service
stop_service "Streamlit" "$PIDS_DIR/streamlit.pid"
stop_service "Gradio" "$PIDS_DIR/gradio.pid"
stop_service "API" "$PIDS_DIR/api.pid"

# Also kill any remaining processes on these ports
echo ""
echo -e "${YELLOW}Checking for processes on default ports...${NC}"

for port in 8000 7860 8501; do
    PID=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}Killing process on port $port (PID: $PID)${NC}"
        kill -9 "$PID" 2>/dev/null || true
    fi
done

echo ""
echo -e "${GREEN}✓ All services stopped${NC}"
