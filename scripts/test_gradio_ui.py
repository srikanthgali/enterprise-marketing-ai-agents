#!/usr/bin/env python3
"""
Quick test to verify Gradio UI can import and initialize properly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_gradio_imports():
    """Test that all Gradio imports work."""
    print("Testing Gradio UI imports...")

    try:
        # Import the Gradio app module
        from ui import gradio_app

        print("✅ Gradio app module imported successfully")

        # Check that key classes exist
        assert hasattr(gradio_app, "AgentChatInterface")
        print("✅ AgentChatInterface class exists")

        # Check that key functions exist
        assert hasattr(gradio_app, "build_interface")
        print("✅ build_interface function exists")

        assert hasattr(gradio_app, "handle_message")
        print("✅ handle_message function exists")

        # Try to instantiate the interface (without starting it)
        interface = gradio_app.AgentChatInterface()
        print("✅ AgentChatInterface instantiated successfully")

        # Check that new method exists
        assert hasattr(interface, "classify_intent")
        print("✅ classify_intent method exists")

        # Check that old methods are removed
        assert not hasattr(interface, "detect_intent_and_route")
        print("✅ Old detect_intent_and_route method removed")

        assert not hasattr(interface, "format_workflow_result")
        print("✅ Old format_workflow_result method removed")

        assert not hasattr(interface, "poll_workflow_status")
        print("✅ Old poll_workflow_status method removed")

        print("\n" + "=" * 60)
        print("✅ All Gradio UI tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gradio_imports()
    sys.exit(0 if success else 1)
