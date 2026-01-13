"""
Customer Support Agent - Handles customer inquiries, tickets, and support requests.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import uuid
import asyncio

from langchain_core.messages import SystemMessage, HumanMessage

from ..core.base_agent import BaseAgent, AgentStatus, HandoffRequest
from ..tools.kb_search import KnowledgeBaseSearchTool
from ..tools.sentiment_analysis import SentimentAnalyzer


class CustomerSupportAgent(BaseAgent):
    """Specialized agent for customer support."""

    def __init__(
        self,
        config: Dict[str, Any],
        memory_manager=None,
        message_bus=None,
        prompt_manager=None,
    ):
        """Initialize customer support agent."""
        super().__init__(
            agent_id="customer_support",
            name="Customer Support Agent",
            description="Handles customer inquiries, tickets, and support requests",
            config=config,
            memory_manager=memory_manager,
            message_bus=message_bus,
            prompt_manager=prompt_manager,
        )
        self.active_tickets: Dict[str, Dict] = {}
        self.response_time_sla = 300  # 5 minutes

        # Initialize tools
        self.kb_search_tool = KnowledgeBaseSearchTool()
        self.sentiment_analyzer = SentimentAnalyzer()

        # SLA configurations (in hours)
        self.sla_hours = {
            "critical": 1,
            "high": 4,
            "medium": 24,
            "low": 72,
        }

    def _register_tools(self) -> None:
        """Register customer support tools."""
        self.register_tool("search_knowledge_base", self._search_knowledge_base)
        self.register_tool("create_ticket", self._create_ticket)
        self.register_tool("analyze_sentiment", self._analyze_sentiment)
        self.register_tool("generate_response", self._generate_response)
        self.register_tool("escalate_issue", self._escalate_issue)
        self.register_tool("collect_feedback", self._collect_feedback)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process customer support request with KB search and LLM response."""
        self.status = AgentStatus.PROCESSING
        start_time = datetime.utcnow()

        try:
            # Handle case where input_data might be a string
            if isinstance(input_data, str):
                input_data = {
                    "type": "inquiry",
                    "message": input_data,
                }
            elif not isinstance(input_data, dict):
                input_data = {"type": "inquiry", "raw_input": str(input_data)}

            request_type = input_data.get("type", "inquiry")
            # Support multiple key formats from different sources
            message = (
                input_data.get("message")
                or input_data.get("inquiry")  # From Gradio UI
                or input_data.get("raw_input")
                or ""
            )

            self.logger.info(f"Processing support request: {request_type}")
            self.logger.info(f"Query: {message[:100] if message else '(empty)'}...")

            # Step 1: Search knowledge base
            kb_results_data = await self._search_knowledge_base(query=message, top_k=5)
            kb_results = kb_results_data.get("results", [])
            confidence = kb_results_data.get("confidence", 0.0)

            self.logger.info(
                f"KB search returned {len(kb_results)} results (confidence: {confidence:.2f})"
            )

            # Step 2: Analyze sentiment
            sentiment = await self._analyze_sentiment(message)

            # Step 3: Generate response using LLM with KB context
            response_data = await self._generate_response(
                query=message, kb_results=kb_results, sentiment=sentiment
            )

            response_text = response_data.get("response", "No response generated")
            citations = response_data.get("citations", [])
            tone = response_data.get("tone", "professional")

            # Format final response
            result = {
                "request_type": request_type,
                "status": "completed",
                "response": response_text,
                "citations": citations,
                "tone": tone,
                "kb_confidence": confidence,
                "sources_count": len(kb_results),
                "sentiment": sentiment.get("label", "neutral"),
            }

            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Support request completed in {processing_time:.2f}s")

            self.status = AgentStatus.IDLE
            return {
                "success": True,
                "response": response_text,
                "details": result,
                "timestamp": datetime.utcnow().isoformat(),
                "is_final": True,
                "summary": f"Answered customer inquiry about: {message[:50]}...",
            }

        except Exception as e:
            self.logger.error(f"Support processing failed: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
            return {
                "success": False,
                "response": "I apologize, but I encountered an error processing your request. Please try again or contact support.",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "is_final": True,
                "summary": f"Support processing failed: {str(e)}",
            }

    # ==================== Tool Implementation Methods ====================

    async def _search_knowledge_base(
        self, query: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search Stripe documentation using FAISS vector store.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            Dictionary with results, confidence, and sources
        """
        try:
            self.logger.info(f"Searching knowledge base: {query}")

            # Use KB search tool
            search_result = await self.kb_search_tool.search(
                query=query,
                top_k=top_k,
                min_score=0.3,  # Filter low-relevance results
            )

            if not search_result.get("success"):
                return {
                    "results": [],
                    "confidence": 0.0,
                    "sources": [],
                    "error": search_result.get("error", "Search failed"),
                }

            results = search_result.get("results", [])

            # Calculate overall confidence (average of top results)
            if results:
                confidence = sum(r.get("score", 0) for r in results) / len(results)
            else:
                confidence = 0.0

            # Extract unique sources
            sources = list(set(r.get("source", "unknown") for r in results))

            # Format results
            formatted_results = [
                {
                    "content": r.get("content", ""),
                    "source": r.get("source", "unknown"),
                    "score": r.get("score", 0.0),
                    "metadata": r.get("metadata", {}),
                }
                for r in results
            ]

            self.logger.info(
                f"Found {len(formatted_results)} KB results "
                f"(confidence: {confidence:.2f})"
            )

            return {
                "results": formatted_results,
                "confidence": float(confidence),
                "sources": sources,
            }

        except Exception as e:
            self.logger.error(f"Knowledge base search failed: {e}")
            return {
                "results": [],
                "confidence": 0.0,
                "sources": [],
                "error": str(e),
            }

    async def _create_ticket(
        self,
        customer_id: str,
        issue_type: str,
        description: str,
        priority: str = "medium",
    ) -> Dict[str, Any]:
        """
        Create a support ticket and save to memory.

        Args:
            customer_id: Customer identifier
            issue_type: Type of issue (billing, technical, account, etc.)
            description: Detailed description of the issue
            priority: Priority level (low, medium, high, critical)

        Returns:
            Dictionary with ticket details
        """
        try:
            # Generate unique ticket ID
            ticket_id = f"TICKET-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

            # Calculate SLA deadline
            sla_hours = self.sla_hours.get(priority.lower(), 24)
            sla_deadline = datetime.utcnow() + timedelta(hours=sla_hours)

            # Create ticket record
            ticket = {
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "issue_type": issue_type,
                "description": description,
                "priority": priority.lower(),
                "status": "open",
                "created_at": datetime.utcnow().isoformat(),
                "sla_deadline": sla_deadline.isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }

            # Save to long-term memory
            if self.memory_manager:
                self.memory_manager.save(
                    agent_id=self.agent_id,
                    key=f"ticket_{ticket_id}",
                    value=ticket,
                    memory_type="long_term",
                )
                self.logger.info(f"Ticket saved to long-term memory: {ticket_id}")

            # Track in active tickets
            self.active_tickets[ticket_id] = ticket

            self.logger.info(
                f"Created ticket {ticket_id} for customer {customer_id} "
                f"(priority: {priority}, SLA: {sla_hours}h)"
            )

            # Publish event
            await self.publish_event(
                "ticket.created",
                {
                    "ticket_id": ticket_id,
                    "customer_id": customer_id,
                    "priority": priority,
                },
            )

            return {
                "ticket_id": ticket_id,
                "status": "open",
                "sla_deadline": sla_deadline.isoformat(),
                "priority": priority.lower(),
            }

        except Exception as e:
            self.logger.error(f"Ticket creation failed: {e}")
            return {
                "ticket_id": None,
                "status": "error",
                "error": str(e),
            }

    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of customer message using LLM.

        Args:
            text: Customer message text

        Returns:
            Dictionary with sentiment, confidence, emotions, and urgency
        """
        try:
            self.logger.info("Analyzing customer sentiment")

            # Use sentiment analyzer
            result = await self.sentiment_analyzer.analyze(text)

            self.logger.info(
                f"Sentiment: {result['sentiment']} "
                f"(confidence: {result['confidence']:.2f}, "
                f"urgency: {result['urgency_level']})"
            )

            return result

        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "emotions": [],
                "urgency_level": "medium",
                "error": str(e),
            }

    async def _generate_response(
        self,
        query: str,
        kb_results: List[Dict[str, Any]],
        sentiment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate contextual response using LLM.

        Args:
            query: Customer query
            kb_results: Results from knowledge base search
            sentiment: Sentiment analysis results

        Returns:
            Dictionary with response, citations, and tone
        """
        try:
            self.logger.info("Generating customer response")

            if not self.llm:
                return {
                    "response": "I apologize, but I'm unable to generate a response at this time.",
                    "citations": [],
                    "tone": "apologetic",
                    "error": "LLM not initialized",
                }

            # Get tone recommendation
            tone = self.sentiment_analyzer.get_tone_recommendation(sentiment)

            # Build context from KB results
            context_parts = []
            citations = []

            for i, result in enumerate(kb_results[:3], 1):  # Use top 3
                content = result.get("content", "")
                source = result.get("source", "documentation")

                context_parts.append(f"[Source {i}]: {content}")
                citations.append(
                    {
                        "number": i,
                        "source": source,
                        "relevance": result.get("score", 0.0),
                    }
                )

            context = (
                "\n\n".join(context_parts)
                if context_parts
                else "No specific documentation found."
            )

            # Tone guidelines
            tone_guidelines = {
                "empathetic_urgent": "Be extremely empathetic and acknowledge the urgency. Prioritize immediate solutions.",
                "empathetic_apologetic": "Start with a sincere apology. Show understanding of frustration.",
                "empathetic_supportive": "Be supportive and patient. Offer clear step-by-step guidance.",
                "empathetic_professional": "Be understanding yet professional. Maintain helpful tone.",
                "friendly_professional": "Be warm and friendly while staying professional.",
                "neutral_professional": "Be clear, concise, and professional.",
            }

            tone_instruction = tone_guidelines.get(
                tone, tone_guidelines["neutral_professional"]
            )

            # Build prompt
            system_prompt = f"""You are a Stripe customer support agent. {tone_instruction}

Guidelines:
- Provide accurate information based on the documentation
- Include specific citations [Source N] in your response
- Be concise but complete
- If unsure, acknowledge limitations
- Offer next steps or alternatives"""

            user_prompt = f"""Customer Query: {query}

Documentation:
{context}

Generate a helpful response that addresses the customer's query using the documentation provided. Include [Source N] citations where appropriate."""

            # Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = await self.llm.ainvoke(messages)
            response_text = response.content

            self.logger.info(
                f"Generated response with {len(citations)} citations (tone: {tone})"
            )

            return {
                "response": response_text,
                "citations": citations,
                "tone": tone,
            }

        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return {
                "response": "I apologize, but I'm having trouble generating a response. Please try rephrasing your question or contact support directly.",
                "citations": [],
                "tone": "apologetic",
                "error": str(e),
            }

    async def _escalate_issue(
        self,
        ticket_id: str,
        reason: str,
        target: str = "marketing_strategy",
    ) -> Dict[str, Any]:
        """
        Escalate issue to another agent.

        Args:
            ticket_id: Ticket identifier
            reason: Reason for escalation
            target: Target agent ID

        Returns:
            Dictionary with escalation details
        """
        try:
            escalation_id = f"ESC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

            # Get ticket details
            ticket = self.active_tickets.get(ticket_id)
            if not ticket and self.memory_manager:
                ticket = self.memory_manager.retrieve(
                    agent_id=self.agent_id,
                    key=f"ticket_{ticket_id}",
                    memory_type="long_term",
                )

            if not ticket:
                return {
                    "escalation_id": None,
                    "target_agent": target,
                    "status": "error",
                    "error": f"Ticket {ticket_id} not found",
                }

            # Create escalation record
            escalation = {
                "escalation_id": escalation_id,
                "ticket_id": ticket_id,
                "from_agent": self.agent_id,
                "target_agent": target,
                "reason": reason,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
                "ticket_data": ticket,
            }

            # Save escalation to memory
            if self.memory_manager:
                self.memory_manager.save(
                    agent_id=self.agent_id,
                    key=f"escalation_{escalation_id}",
                    value=escalation,
                    memory_type="long_term",
                )

            # Create handoff request
            handoff = HandoffRequest(
                from_agent=self.agent_id,
                to_agent=target,
                reason=reason,
                context={
                    "escalation_id": escalation_id,
                    "ticket_id": ticket_id,
                    "ticket_data": ticket,
                },
                priority="high",
            )

            # Request handoff
            await self.request_handoff(handoff)

            self.logger.info(
                f"Escalated ticket {ticket_id} to {target} (escalation: {escalation_id})"
            )

            # Publish event
            await self.publish_event(
                "ticket.escalated",
                {
                    "escalation_id": escalation_id,
                    "ticket_id": ticket_id,
                    "target_agent": target,
                },
            )

            return {
                "escalation_id": escalation_id,
                "target_agent": target,
                "status": "pending",
            }

        except Exception as e:
            self.logger.error(f"Escalation failed: {e}")
            return {
                "escalation_id": None,
                "target_agent": target,
                "status": "error",
                "error": str(e),
            }

    async def _collect_feedback(
        self,
        ticket_id: str,
        rating: int,
        comments: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Collect and store customer feedback.

        Args:
            ticket_id: Ticket identifier
            rating: Customer satisfaction rating (1-5)
            comments: Optional feedback comments

        Returns:
            Dictionary with feedback details
        """
        try:
            feedback_id = f"FEEDBACK-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

            # Validate rating
            rating = max(1, min(5, rating))  # Clamp to 1-5

            # Create feedback record
            feedback = {
                "feedback_id": feedback_id,
                "ticket_id": ticket_id,
                "rating": rating,
                "comments": comments or "",
                "created_at": datetime.utcnow().isoformat(),
            }

            # Save to memory
            stored = False
            if self.memory_manager:
                self.memory_manager.save(
                    agent_id=self.agent_id,
                    key=f"feedback_{feedback_id}",
                    value=feedback,
                    memory_type="long_term",
                )
                stored = True
                self.logger.info(f"Feedback saved: {feedback_id}")

            # Calculate satisfaction score (normalized 0-1)
            satisfaction_score = rating / 5.0

            # Publish event for analytics
            await self.publish_event(
                "feedback.collected",
                {
                    "feedback_id": feedback_id,
                    "ticket_id": ticket_id,
                    "rating": rating,
                    "satisfaction_score": satisfaction_score,
                },
            )

            self.logger.info(
                f"Collected feedback for ticket {ticket_id}: "
                f"{rating}/5 (satisfaction: {satisfaction_score:.2f})"
            )

            return {
                "feedback_id": feedback_id,
                "satisfaction_score": satisfaction_score,
                "stored": stored,
            }

        except Exception as e:
            self.logger.error(f"Feedback collection failed: {e}")
            return {
                "feedback_id": None,
                "satisfaction_score": 0.0,
                "stored": False,
                "error": str(e),
            }
