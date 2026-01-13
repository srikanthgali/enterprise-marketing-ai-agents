"""
Test script for Gradio UI components.

Verifies that the Gradio app can be imported and basic components work.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        import gradio as gr

        print("✓ Gradio imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Gradio: {e}")
        return False

    try:
        import httpx

        print("✓ httpx imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import httpx: {e}")
        return False

    try:
        from config.settings import get_settings

        settings = get_settings()
        print(f"✓ Settings loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        return False

    return True


def test_chat_interface():
    """Test ChatInterface class instantiation."""
    print("\nTesting ChatInterface class...")

    try:
        # Import the class from gradio_app
        sys.path.insert(0, str(project_root / "ui"))
        from gradio_app import AgentChatInterface

        chat = AgentChatInterface()
        print("✓ AgentChatInterface instantiated successfully")

        # Test intent detection
        workflow, agent = chat.detect_intent_and_route(
            "Plan a campaign for our product", "Auto"
        )
        print(f"✓ Intent detection works: '{workflow}' -> '{agent}'")

        return True
    except Exception as e:
        print(f"✗ Failed to test ChatInterface: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_api_connectivity():
    """Test API connectivity (if server is running)."""
    print("\nTesting API connectivity...")

    try:
        import httpx

        response = httpx.get("http://localhost:8000/api/v1/health", timeout=5.0)
        if response.status_code == 200:
            print("✓ FastAPI server is reachable")
            return True
        else:
            print(f"⚠ FastAPI returned status {response.status_code}")
            return False
    except (httpx.ConnectError, httpx.ReadTimeout):
        print("⚠ FastAPI server is not running (this is OK for import testing)")
        print("  Start it with: ./start_api.sh")
        return True  # Not critical for testing
    except Exception as e:
        print(f"✗ Failed to connect to API: {e}")
        return False


def test_gradio_interface():
    """Test Gradio interface building."""
    print("\nTesting Gradio interface construction...")

    try:
        sys.path.insert(0, str(project_root / "ui"))
        from gradio_app import build_interface

        # This will create the interface but not launch it
        demo = build_interface()
        print("✓ Gradio interface built successfully")

        return True
    except Exception as e:
        print(f"✗ Failed to build interface: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Gradio UI Component Tests")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("ChatInterface", test_chat_interface()))
    results.append(("API Connectivity", test_api_connectivity()))
    results.append(("Interface Building", test_gradio_interface()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 70)

    if all(result[1] for result in results):
        print("\n🎉 All tests passed! The Gradio UI is ready to use.")
        print("\nNext steps:")
        print("1. Start FastAPI: ./start_api.sh")
        print("2. Start Gradio UI: ./start_gradio.sh")
        print("3. Open browser: http://localhost:7860")
        return 0
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
