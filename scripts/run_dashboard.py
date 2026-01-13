#!/usr/bin/env python3
"""
Run Streamlit Dashboard for Enterprise Marketing AI Agents.

This script starts the Streamlit web interface for monitoring and managing
the multi-agent marketing system.
"""

import subprocess
import sys
import os
from pathlib import Path

# Ensure we're in the project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)


def main():
    """Run the Streamlit dashboard."""
    print("=" * 80)
    print("Enterprise Marketing AI Agents - Dashboard")
    print("=" * 80)
    print()
    print("Starting Streamlit dashboard...")
    print()
    print("Dashboard will be available at: http://localhost:8501")
    print("Make sure the API server is running at: http://localhost:8000")
    print()
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 80)
    print()

    # Run Streamlit
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_app.py",
                "--server.port=8501",
                "--server.address=localhost",
                "--browser.gatherUsageStats=false",
            ]
        )
    except KeyboardInterrupt:
        print("\n\nShutting down dashboard...")
    except Exception as e:
        print(f"\n❌ Error running dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
