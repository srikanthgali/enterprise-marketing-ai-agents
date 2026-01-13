#!/usr/bin/env python
"""
Simple launcher for Gradio UI.

This script can be run directly without shell scripts:
    python launch_gradio.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"  # No Color


def print_colored(message, color=NC):
    """Print colored message."""
    print(f"{color}{message}{NC}")


def check_virtualenv():
    """Check if virtual environment is activated."""
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        print_colored("✓ Virtual environment is activated", GREEN)
        return True
    else:
        print_colored("⚠ Warning: No virtual environment detected", YELLOW)
        venv_path = Path(".venv")
        if venv_path.exists():
            print_colored(f"  Virtual environment found at {venv_path}", YELLOW)
            print_colored("  Activate it with: source .venv/bin/activate", YELLOW)
        return False


def check_dependencies():
    """Check if required packages are installed."""
    # Map display name to actual import name
    required = {
        "gradio": "gradio",
        "httpx": "httpx",
        "pydantic": "pydantic",
        "pyyaml": "yaml",  # PyYAML imports as 'yaml'
    }
    missing = []

    for display_name, import_name in required.items():
        try:
            __import__(import_name)
            print_colored(f"✓ {display_name} is installed", GREEN)
        except ImportError:
            missing.append(display_name)
            print_colored(f"✗ {display_name} is not installed", RED)

    if missing:
        print_colored(f"\nInstall missing packages with:", YELLOW)
        print_colored(f"  pip install {' '.join(missing)}", YELLOW)
        return False

    return True


def check_fastapi():
    """Check if FastAPI server is running."""
    try:
        import httpx

        response = httpx.get("http://localhost:8000/api/v1/health", timeout=5.0)
        if response.status_code == 200:
            print_colored("✓ FastAPI server is running", GREEN)
            return True
        else:
            print_colored(f"⚠ FastAPI returned status {response.status_code}", YELLOW)
            return False
    except Exception:
        print_colored("✗ FastAPI server is not running", RED)
        print_colored("  Start it with one of:", YELLOW)
        print_colored("    ./start_api.sh", YELLOW)
        print_colored("    uvicorn api.main:app --reload --port 8000", YELLOW)
        return False


def create_directories():
    """Create necessary directories."""
    dirs = ["data/conversations", "logs"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print_colored("✓ Directories created", GREEN)


def launch_gradio():
    """Launch Gradio UI."""
    print("\n" + "=" * 70)
    print("  🚀 Starting Gradio Conversational UI")
    print("=" * 70)
    print(f"\n  📡 API Backend: http://localhost:8000")
    print(f"  🌐 Gradio UI:   http://localhost:7860")
    print(f"\n  Press Ctrl+C to stop")
    print("=" * 70 + "\n")

    # Run Gradio
    try:
        subprocess.run([sys.executable, "ui/gradio_app.py"])
    except KeyboardInterrupt:
        print_colored("\n\n✓ Gradio UI stopped", GREEN)
    except Exception as e:
        print_colored(f"\n✗ Error running Gradio: {e}", RED)
        return 1

    return 0


def main():
    """Main entry point."""
    print("=" * 70)
    print("Gradio UI Launcher")
    print("=" * 70)
    print()

    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    print_colored(f"Working directory: {project_root}", GREEN)
    print()

    # Checks
    print("Running pre-flight checks...\n")

    check_virtualenv()

    if not check_dependencies():
        print_colored("\n✗ Missing dependencies. Install them first.", RED)
        return 1

    fastapi_running = check_fastapi()

    create_directories()

    print()

    if not fastapi_running:
        response = input("\nFastAPI is not running. Continue anyway? (y/N): ")
        if response.lower() != "y":
            print_colored("Aborted. Start FastAPI first.", YELLOW)
            return 1

    # Launch
    return launch_gradio()


if __name__ == "__main__":
    sys.exit(main())
