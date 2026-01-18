"""
Unified chat endpoint for multi-agent system.

Single endpoint that accepts raw user messages and returns agent responses.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import asyncio
import logging
from datetime import datetime, timedelta

from api.dependencies import get_orchestrator, get_memory_manager
from api.utils.response_formatter import format_agent_response
from config.settings import get_settings

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

# Import workflow_states for tracking workflows
from api.routes.workflows import workflow_states


class ChatRequest(BaseModel):
    """Simple chat request with just a message."""

    message: str = Field(..., description="User's natural language message")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation tracking"
    )
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for history"
    )
    agent_hint: Optional[str] = Field(
        None, description="Optional agent preference (for manual override)"
    )


class ChatResponse(BaseModel):
    """Chat response with agent reply."""

    message: str = Field(..., description="Agent's response message")
    agent: str = Field(..., description="Agent that handled the request")
    workflow_id: str = Field(..., description="Workflow execution ID")
    intent: Optional[str] = Field(None, description="Classified intent")
    confidence: Optional[float] = Field(
        None, description="Classification confidence (0-1)"
    )
    reasoning: Optional[str] = Field(None, description="Classification reasoning")
    agents_executed: list = Field(
        default_factory=list, description="List of agents that executed"
    )
    handoffs: list = Field(
        default_factory=list, description="List of handoffs that occurred"
    )


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    orchestrator=Depends(get_orchestrator),
    memory_manager=Depends(get_memory_manager),
):
    """
    Single unified chat endpoint - send a message, get a response.

    The orchestrator will:
    1. Classify intent using LLM
    2. Route to appropriate agent
    3. Handle any necessary handoffs
    4. Return final response

    Args:
        request: Chat request with user message

    Returns:
        Agent response with metadata
    """
    try:
        logger.info(f"Received chat request: {request.message[:100]}...")

        # Process message through orchestrator
        result = await orchestrator.process(
            {
                "message": request.message,
                "session_id": request.session_id,
                "conversation_id": request.conversation_id,
                "agent_hint": request.agent_hint,
            }
        )

        # Track workflow in workflow_states for Streamlit dashboard
        workflow_id = result.get("workflow_id", "unknown")
        if workflow_id != "unknown":
            intent_classification = result.get("intent_classification") or {}
            execution_summary = result.get("execution_summary") or {}

            # Get intent and clean up the name
            intent = intent_classification.get("intent", "unknown")
            intent_str = str(intent).replace("AgentIntent.", "").lower()

            # Calculate actual timestamps from duration
            completed_at = datetime.now()
            duration_seconds = result.get("duration", 0)
            started_at = completed_at - timedelta(seconds=duration_seconds)

            workflow_states[workflow_id] = {
                "workflow_id": workflow_id,
                "workflow_type": f"chat_{intent_str}",
                "status": "completed" if result.get("success") else "failed",
                "current_agent": (
                    execution_summary.get("agents_executed", [])[-1]
                    if execution_summary.get("agents_executed")
                    else None
                ),
                "progress": 1.0,
                "started_at": started_at,
                "completed_at": completed_at,
                "agents_executed": execution_summary.get("agents_executed", []),
                "state_transitions": [],
                "error": result.get("error") if not result.get("success") else None,
                "message": request.message[:100],
            }

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Processing failed"),
            )

        # Extract relevant info from result
        intent_classification = result.get("intent_classification") or {}
        execution_summary = result.get("execution_summary") or {}
        final_result = result.get("result") or {}

        # Determine which agent produced final result
        agents_executed = execution_summary.get("agents_executed", [])
        final_agent = agents_executed[-1] if agents_executed else "orchestrator"

        # Format the response message using the response formatter
        response_message = format_agent_response(final_result, final_agent)

        # Extract handoff information - only include records where is_handoff=True
        handoffs = []
        execution_history = execution_summary.get("execution_history", [])
        for record in execution_history:
            if record.get("is_handoff"):
                handoffs.append(
                    {
                        "from": record.get("handoff_from"),
                        "to": record.get("agent_id"),
                        "reason": record.get("handoff_reason", "handoff_requested"),
                    }
                )

        return ChatResponse(
            message=response_message,
            agent=final_agent,
            workflow_id=result.get("workflow_id", "unknown"),
            intent=intent_classification.get("intent"),
            confidence=intent_classification.get("confidence"),
            reasoning=intent_classification.get("reasoning"),
            agents_executed=agents_executed,
            handoffs=handoffs,
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    orchestrator=Depends(get_orchestrator),
):
    """
    Streaming chat endpoint for real-time agent updates.

    Returns Server-Sent Events (SSE) stream with intermediate agent updates.

    Args:
        request: Chat request with user message

    Returns:
        SSE stream with workflow progress
    """

    async def event_generator():
        try:
            # Process message
            result = await orchestrator.process(
                {
                    "message": request.message,
                    "session_id": request.session_id,
                }
            )

            workflow_id = result.get("workflow_id", "unknown")

            # Send intent classification
            if result.get("intent_classification"):
                yield f"data: {json.dumps({'type': 'intent', 'data': result['intent_classification']})}\n\n"

            # Stream execution updates
            execution_summary = result.get("execution_summary", {})

            # Send agent execution updates
            for record in execution_summary.get("execution_history", []):
                yield f"data: {json.dumps({'type': 'agent_update', 'data': record})}\n\n"
                await asyncio.sleep(0.1)  # Small delay for visual effect

            # Send final result
            yield f"data: {json.dumps({'type': 'complete', 'data': result})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
