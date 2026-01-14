#!/bin/bash
# Script to clear all test-related caches

echo "🧹 Clearing all test caches..."

# Clear pytest cache
echo "  Clearing pytest cache..."
rm -rf .pytest_cache

# Clear Python bytecode cache
echo "  Clearing Python bytecode cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

echo "✅ All caches cleared!"
echo ""
echo "Now:"
echo "  1. Close VS Code completely (Cmd+Q or Alt+F4)"
echo "  2. Wait 5 seconds"
echo "  3. Reopen VS Code"
echo "  4. Run tests from Test Explorer"
echo ""
echo "Expected result: 134 unit tests should pass"
