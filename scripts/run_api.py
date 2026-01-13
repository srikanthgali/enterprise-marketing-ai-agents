"""
Python script to run the FastAPI server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
import argparse
from config.settings import get_settings


def main():
    """Run the FastAPI application."""
    parser = argparse.ArgumentParser(description="Run FastAPI server")
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (default: from settings)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from settings)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level",
    )

    args = parser.parse_args()
    settings = get_settings()

    # Use command line args or fall back to settings
    host = args.host or settings.fastapi.host
    port = args.port or settings.fastapi.port
    reload = args.reload or settings.fastapi.reload
    workers = args.workers or settings.fastapi.workers
    log_level = args.log_level or settings.fastapi.log_level

    print(f"Starting FastAPI server on {host}:{port}")
    print(f"Environment: {settings.system.environment}")
    print(f"Debug mode: {settings.system.debug}")
    print(f"Reload: {reload}")
    print(f"Workers: {workers}")
    print(f"Log level: {log_level}")
    print(f"\nAPI Documentation:")
    print(f"  - Swagger UI: http://{host}:{port}/api/docs")
    print(f"  - ReDoc: http://{host}:{port}/api/redoc")
    print(f"  - Health Check: http://{host}:{port}/api/v1/health")
    print(f"\nPress CTRL+C to stop the server\n")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # workers only in non-reload mode
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
