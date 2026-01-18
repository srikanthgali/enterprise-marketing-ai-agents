"""
Gradio Conversational UI for Enterprise Marketing AI Agents.

Interactive chat interface for multi-agent system with:
- Real-time agent conversations
- Agent handoff visualization
- Intermediate results display
- Conversation history management
- Export capabilities
"""

import gradio as gr
import httpx
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys
import time
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# API Configuration
API_BASE_URL = f"http://localhost:{settings.security.fastapi_host.split(':')[-1] if hasattr(settings.security, 'fastapi_host') else '8000'}/api/v1"
TIMEOUT = 300.0

# State management
conversation_states: Dict[str, Dict[str, Any]] = {}


class AgentChatInterface:
    """Main chat interface for agent interactions."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        self.current_workflow_id = None

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    def format_message_html(
        self, role: str, content: str, metadata: Optional[Dict] = None
    ) -> str:
        """Format message with syntax highlighting and metadata."""
        # Detect code blocks and JSON
        if "```" in content:
            # Already has code formatting
            formatted_content = content
        elif content.strip().startswith("{") or content.strip().startswith("["):
            # Try to parse as JSON
            try:
                parsed = json.loads(content)
                formatted_content = f"```json\n{json.dumps(parsed, indent=2)}\n```"
            except:
                formatted_content = content
        else:
            formatted_content = content

        # Add metadata if present
        if metadata:
            meta_str = "\n\n---\n"
            if "agent" in metadata:
                meta_str += f"**Agent:** {metadata['agent']}\n"
            if "timestamp" in metadata:
                meta_str += f"**Time:** {metadata['timestamp']}\n"
            if "workflow_id" in metadata:
                meta_str += f"**Workflow ID:** {metadata['workflow_id'][:8]}...\n"
            formatted_content += meta_str

        return formatted_content

    async def classify_intent(self, message: str) -> Dict[str, Any]:
        """
        Classify user intent using the new LLM-based chat endpoint.

        Returns:
            Dict with intent, confidence, agent routing info
        """
        try:
            response = await self.client.post(
                f"{API_BASE_URL}/chat", json={"message": message}
            )

            if response.status_code == 200:
                return response.json()
            else:
                # Fallback to default
                return {
                    "intent": "general_inquiry",
                    "agent": "customer_support",
                    "confidence": 0.5,
                    "message": "Unable to classify intent, routing to customer support.",
                }
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return {
                "intent": "general_inquiry",
                "agent": "customer_support",
                "confidence": 0.5,
                "message": f"Error during classification: {str(e)}",
            }

    async def stream_response(
        self,
        message: str,
        agent_choice: str,
        history: List[Dict[str, str]],
        conversation_id: str,
    ):
        """
        Stream agent response using the new unified chat endpoint.

        Yields:
            Tuple of (history, status, metrics)
        """
        try:
            # Add user message to history (Gradio 6.0 format)
            history = history + [{"role": "user", "content": message}]
            yield history, "🔄 Classifying intent...", {}

            await asyncio.sleep(0.3)

            # Use the new unified chat endpoint
            yield history, "⚙️ Processing with AI agents...", {}

            # Make API request to unified chat endpoint
            response = await self.client.post(
                f"{API_BASE_URL}/chat", json={"message": message}
            )

            if response.status_code == 200:
                result = response.json()

                # Extract response data
                response_message = result.get("message", "No response available")
                agent_name = result.get("agent", "unknown")
                intent = result.get("intent", "unknown")
                confidence = result.get("confidence", 0.0)
                workflow_id = result.get("workflow_id", "")
                agents_executed = result.get("agents_executed", [])
                handoffs = result.get("handoffs", [])

                # Determine primary agent (first in chain) vs final agent
                primary_agent = agents_executed[0] if agents_executed else agent_name
                final_agent = agents_executed[-1] if agents_executed else agent_name

                # Determine primary agent (first in chain) vs final agent
                primary_agent = agents_executed[0] if agents_executed else agent_name

                # Format the response
                formatted_response = f"{response_message}"

                # Add metadata section
                metadata_lines = [
                    "\n\n---",
                    f"**Intent:** {intent} (confidence: {confidence:.0%})",
                    f"**Primary Agent:** {primary_agent}",
                ]

                if len(agents_executed) > 1:
                    metadata_lines.append(
                        f"**Agents Executed:** {', '.join(agents_executed)}"
                    )

                if handoffs:
                    # Filter out self-handoffs (A → A)
                    valid_handoffs = [
                        h for h in handoffs if h.get("from") != h.get("to")
                    ]

                    if valid_handoffs:
                        handoff_text = "**Handoffs:** "
                        handoff_descriptions = []
                        for handoff in valid_handoffs:
                            from_agent = handoff.get("from", "Start")
                            to_agent = handoff.get("to", "End")
                            handoff_descriptions.append(f"{from_agent} → {to_agent}")
                        handoff_text += ", ".join(handoff_descriptions)
                        metadata_lines.append(handoff_text)

                if workflow_id:
                    metadata_lines.append(f"**Workflow ID:** {workflow_id[:12]}...")

                formatted_response += "\n".join(metadata_lines)

                # Update history with final response
                history = history + [
                    {"role": "assistant", "content": formatted_response}
                ]

                # Extract metrics
                metrics = {
                    "workflow_id": workflow_id,
                    "intent": intent,
                    "confidence": f"{confidence:.0%}",
                    "agent": agent_name,
                    "agents_executed": len(agents_executed),
                    "handoffs": len(handoffs),
                    "status": "completed",
                }

                yield history, "✅ Complete", metrics

            else:
                error_msg = f"API Error: {response.status_code}"
                try:
                    error_detail = response.json().get("detail", error_msg)
                    error_msg = error_detail
                except:
                    pass

                history = history + [
                    {"role": "assistant", "content": f"❌ {error_msg}"}
                ]
                yield history, "❌ Error", {"error": error_msg}

        except httpx.TimeoutException:
            error_msg = "Request timeout. The operation took too long."
            history = history + [{"role": "assistant", "content": f"⏱️ {error_msg}"}]
            yield history, "⏱️ Timeout", {"error": error_msg}

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Stream response error: {e}", exc_info=True)
            history = history + [{"role": "assistant", "content": f"❌ {error_msg}"}]
            yield history, "❌ Error", {"error": str(e)}

    def export_conversation(
        self, history: List[Dict[str, str]], conversation_id: str
    ) -> str:
        """Export conversation to JSON file."""
        export_data = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": [
                {**msg, "timestamp": datetime.now().isoformat()} for msg in history
            ],
        }

        export_path = (
            project_root
            / "data"
            / "conversations"
            / f"conversation_{conversation_id[:8]}.json"
        )
        export_path.parent.mkdir(parents=True, exist_ok=True)

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return str(export_path)


# Initialize chat interface
chat_interface = AgentChatInterface()


# Gradio event handlers
async def handle_message(
    message: str, agent_choice: str, history: List, conversation_id: str
):
    """Handle user message and stream response."""
    if not message.strip():
        # Use yield for async generator consistency
        yield history, "💬 Waiting for input", {}, conversation_id
        return

    # Generate conversation ID if needed
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    # Stream response
    async for updated_history, status, metrics in chat_interface.stream_response(
        message, agent_choice, history, conversation_id
    ):
        yield updated_history, status, metrics, conversation_id


def clear_conversation():
    """Clear conversation history."""
    return [], "", "💬 Ready", {}, str(uuid.uuid4())


def export_conversation_handler(history: List, conversation_id: str):
    """Export conversation to file."""
    if not history:
        return "No conversation to export", ""

    try:
        file_path = chat_interface.export_conversation(history, conversation_id)
        return f"✅ Exported to: {file_path}", file_path
    except Exception as e:
        return f"❌ Export failed: {str(e)}", ""


# Build Gradio Interface
def build_interface():
    """Build Gradio chat interface."""

    with gr.Blocks(title="Enterprise Marketing AI Agents") as demo:
        # State management
        conversation_id_state = gr.State(value=str(uuid.uuid4()))

        # Header
        gr.Markdown(
            """
            # 🤖 Enterprise Marketing AI Agents

            **Conversational interface for multi-agent marketing workflows**

            Ask me to:
            - 📱 Plan a campaign for your product
            - 📚 Search knowledge base (e.g., "How does Stripe handle subscriptions?")
            - 📊 Analyze campaign performance
            - 🎯 Get improvement recommendations
            """
        )

        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Agent Conversation",
                    height=500,
                    show_label=True,
                    avatar_images=(
                        None,
                        "https://api.dicebear.com/7.x/bottts/svg?seed=agent",
                    ),
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Ask me anything about marketing, campaigns, or product questions...",
                        lines=1,
                        max_lines=3,
                        scale=4,
                        container=False,
                    )

                with gr.Row():
                    submit = gr.Button("Send", variant="primary", scale=1)
                    clear = gr.Button("Clear", scale=1)
                    export = gr.Button("📥 Export", scale=1)

                # Export status
                export_status = gr.Textbox(label="Export Status", visible=False)
                export_path = gr.Textbox(label="Export Path", visible=False)

            # Sidebar
            with gr.Column(scale=1):
                gr.Markdown("### Settings")

                agent_selector = gr.Dropdown(
                    label="Route to Agent",
                    choices=[
                        "Auto",
                        "Marketing Strategy",
                        "Customer Support",
                        "Analytics",
                        "Feedback Learning",
                    ],
                    value="Auto",
                    info="Auto-routing analyzes your message",
                )

                status = gr.Textbox(
                    label="Status",
                    value="💬 Ready",
                    interactive=False,
                    elem_classes="status-box",
                )

                metrics = gr.JSON(
                    label="Current Metrics",
                    elem_classes="metrics-box",
                )

                gr.Markdown("### Example Queries")
                gr.Examples(
                    examples=[
                        ["Plan a campaign for our new payment API product"],
                        ["How does Stripe handle subscription billing?"],
                        ["Analyze our campaign performance from last month"],
                        ["What improvements can we make to agent performance?"],
                        ["Create a content strategy for developer outreach"],
                    ],
                    inputs=msg,
                )

                gr.Markdown(
                    """
                    ### Tips
                    - Use natural language
                    - Be specific about your needs
                    - Mention products/features for context
                    - Ask follow-up questions
                    """
                )

        # Event handlers
        msg.submit(
            fn=handle_message,
            inputs=[msg, agent_selector, chatbot, conversation_id_state],
            outputs=[chatbot, status, metrics, conversation_id_state],
            queue=True,
        ).then(fn=lambda: gr.update(value=""), inputs=None, outputs=[msg])

        submit.click(
            fn=handle_message,
            inputs=[msg, agent_selector, chatbot, conversation_id_state],
            outputs=[chatbot, status, metrics, conversation_id_state],
            queue=True,
        ).then(fn=lambda: gr.update(value=""), inputs=None, outputs=[msg])

        clear.click(
            fn=clear_conversation,
            inputs=None,
            outputs=[chatbot, msg, status, metrics, conversation_id_state],
            queue=False,
        )

        export.click(
            fn=export_conversation_handler,
            inputs=[chatbot, conversation_id_state],
            outputs=[export_status, export_path],
            queue=False,
        )

    return demo


# Main entry point
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting Enterprise Marketing AI Agents - Gradio Interface")
    print("=" * 80)
    print(f"\n📡 API Backend: {API_BASE_URL}")
    print(
        f"💡 Make sure FastAPI server is running on port {API_BASE_URL.split(':')[-1].split('/')[0]}"
    )
    print(f"\n🌐 Gradio UI will launch on: http://localhost:7860")
    print("=" * 80 + "\n")

    # Build and launch interface
    demo = build_interface()

    demo.queue(
        max_size=20,  # Max queue size
        default_concurrency_limit=5,  # Max concurrent requests
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public sharing
        # Gradio 6.0: theme and css moved to launch()
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .message-box {font-family: 'Inter', sans-serif;}
        .status-box {font-weight: 600; padding: 10px; border-radius: 5px;}
        .metrics-box {background: #f0f4f8; padding: 10px; border-radius: 5px;}
        """,
    )
