#!/bin/bash
# Quick test script for intent routing system

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=================================================="
echo "  Intent Routing System - Quick Test"
echo "=================================================="
echo ""

# Check if API is running
echo -n "Checking API status... "
if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API is running${NC}"
else
    echo -e "${RED}✗ API is not running${NC}"
    echo ""
    echo "Please start the API first:"
    echo "  python scripts/run_api.py"
    echo ""
    echo "Or use:"
    echo "  ./start_api.sh"
    exit 1
fi

echo ""
echo "Running test suite..."
echo ""

# Run the test script
python scripts/test_intent_routing.py

echo ""
echo "=================================================="
echo "  Test complete!"
echo "=================================================="
