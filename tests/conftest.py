"""
Pytest configuration and fixtures for all tests.
"""

import sys
import os
import pytest
from pathlib import Path

# Prevent Python from writing bytecode during tests
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Force .env reload for VS Code test explorer
from dotenv import load_dotenv
load_dotenv(override=True)


@pytest.fixture(scope="session", autouse=True)
def debug_environment():
    """Print environment for debugging VS Code test explorer."""
    print(f"\n[DEBUG] Python: {sys.executable}")
    print(f"[DEBUG] CWD: {os.getcwd()}")
    print(f"[DEBUG] ALLOWED_ORIGINS: {os.getenv('ALLOWED_ORIGINS', 'NOT SET')}")
    yield


@pytest.fixture(scope="function", autouse=True)
def clear_settings_cache():
    """Clear the settings cache before each test to ensure fresh settings."""
    from config.settings import get_settings
    
    # Clear LRU cache before and after each test
    get_settings.cache_clear()
    
    yield
    
    get_settings.cache_clear()
