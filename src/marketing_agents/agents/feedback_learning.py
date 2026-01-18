"""
Feedback & Learning Agent - Continuously improves the system.

Collects feedback from all agents, fine-tunes models, optimizes workflows,
and implements systematic improvements to enhance overall performance.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import statistics

from ..core.base_agent import BaseAgent, AgentStatus
from ..core.handoff_detector import HandoffDetector
from ..tools.pattern_detector import PatternDetector


class FeedbackLearningAgent(BaseAgent):
    """Specialized agent for system improvement through learning."""

    def __init__(
        self,
        config: Dict[str, Any],
        memory_manager=None,
        message_bus=None,
        prompt_manager=None,
    ):
        """Initialize feedback learning agent."""
        # Initialize our attributes BEFORE calling super().__init__
        # because _register_tools() is called from BaseAgent.__init__
        self.feedback_queue: List[Dict] = []
        self.learning_history: List[Dict] = []
        self.experiment_store: Dict[str, List[Dict]] = defaultdict(list)
        self.pattern_detector = PatternDetector()
        self.config_path = Path("config/agents_config.yaml")

        # Performance baselines from config
        self.performance_baselines = {
            "response_time": 30.0,
            "success_rate": 0.85,
            "handoff_accuracy": 0.90,
            "user_satisfaction": 0.80,
        }

        super().__init__(
            agent_id="feedback_learning",
            name="Feedback & Learning Agent",
            description="Continuously improves system through learning",
            config=config,
            memory_manager=memory_manager,
            message_bus=message_bus,
            prompt_manager=prompt_manager,
        )

        # Initialize handoff detector
        handoff_config = self.config.get("agents", {}).get("orchestrator", {}).get("handoff_detector")
        self.handoff_detector = HandoffDetector(llm=self.llm, config=handoff_config)

    def _register_tools(self) -> None:
        """Register tools for the feedback learning agent."""
        # Register pattern detection tool
        self.tools["pattern_detector"] = self.pattern_detector.detect_patterns

        # Register aggregation tool
        self.tools["aggregate_feedback"] = self._aggregate_feedback

        # Register evaluation tool
        self.tools["evaluate_performance"] = self._evaluate_agent_performance

        # Register optimization tool
        self.tools["optimize_prompts"] = self._optimize_prompts

        # Register experiment tracking
        self.tools["track_experiment"] = self._track_experiment

        self.logger.info(
            f"Registered {len(self.tools)} tools for feedback learning agent"
        )

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process learning and optimization request."""
        self.status = AgentStatus.PROCESSING
        start_time = datetime.utcnow()

        try:
            # Handle case where input_data might be a string
            if isinstance(input_data, str):
                input_data = {
                    "request_type": "analyze_feedback",
                    "message": input_data,
                }
            elif not isinstance(input_data, dict):
                input_data = {
                    "request_type": "analyze_feedback",
                    "raw_input": str(input_data),
                }

            # Support both 'type' and 'request_type' keys
            request_type = input_data.get("request_type") or input_data.get(
                "type", "analyze_feedback"
            )
            self.logger.info(f"Processing learning request: {request_type}")

            # Extract message for analysis
            message = input_data.get("message", "")

            # Process based on request type
            if request_type == "analyze_feedback":
                # Analyze the message content for specific feedback
                result = self._analyze_user_feedback(
                    message, input_data.get("time_range", "last_7_days")
                )
            elif request_type == "optimize_agent":
                agent_id = input_data.get("agent_id", "marketing_strategy")
                metrics = input_data.get("metrics", {})
                performance = self._evaluate_agent_performance(agent_id, metrics)
                optimizations = self._optimize_prompts(agent_id, performance)
                result = {
                    "request_type": request_type,
                    "agent_id": agent_id,
                    "performance_evaluation": performance,
                    "optimizations": optimizations,
                }
            elif request_type == "track_experiment":
                experiment_name = input_data.get(
                    "experiment_name", "default_experiment"
                )
                variant = input_data.get("variant", "control")
                metrics = input_data.get("metrics", {})
                tracking_result = self._track_experiment(
                    experiment_name, variant, metrics
                )
                result = {
                    "request_type": request_type,
                    "experiment_tracking": tracking_result,
                }
            elif request_type == "detect_patterns":
                # Analyze message for recurring issues or patterns
                result = self._detect_issue_patterns(
                    message, input_data.get("time_range", "last_7_days")
                )
            elif request_type == "prediction_improvement":
                # Generate prediction improvement recommendations
                context = input_data.get("context", {})
                metrics = context.get("metrics", input_data.get("metrics", {}))
                result = self._generate_prediction_improvement_plan(message, metrics)
            else:
                # Handle handoff requests or analyze for prediction improvements
                # Check if this is about prediction/forecast improvement
                if message and any(
                    word in message.lower()
                    for word in [
                        "prediction",
                        "predictions",
                        "forecast",
                        "accuracy",
                        "accurate",
                    ]
                ):
                    # Extract metrics from context if available
                    context = input_data.get("context", {})
                    metrics = context.get("metrics", input_data.get("metrics", {}))

                    result = self._generate_prediction_improvement_plan(
                        message, metrics
                    )
                else:
                    # Generic learning analysis
                    result = {
                        "request_type": request_type,
                        "improvements": [
                            "Analyzed system performance patterns",
                            "Identified optimization opportunities",
                        ],
                    }

            # Check if handoff is needed
            handoff_info = await self._detect_handoff_need(message, result)

            self.status = AgentStatus.IDLE

            # Determine if this is final based on handoff
            is_final = not handoff_info.get("handoff_required", False)

            # Generate user-friendly summary
            if result.get("request_type") == "prediction_improvement":
                num_recs = len(result.get("recommendations", []))
                confidence = result.get("analysis", {}).get(
                    "prediction_confidence", "Unknown"
                )
                summary = f"Prediction Improvement Analysis Complete: Generated {num_recs} recommendations (Confidence: {confidence})"
            elif handoff_info.get("handoff_required"):
                summary = f"Routing to {handoff_info.get('target_agent', '')} agent for specialized assistance."
            else:
                summary = f"Learning analysis completed: {request_type}"

            response_dict = {
                "success": True,
                "learning_results": result,
                "timestamp": datetime.utcnow().isoformat(),
                "is_final": is_final,
                "summary": summary,
            }

            # Add handoff information if needed
            if handoff_info.get("handoff_required"):
                response_dict.update(handoff_info)

            return response_dict

        except Exception as e:
            self.logger.error(f"Learning processing failed: {e}")
            self.status = AgentStatus.ERROR
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "is_final": True,
                "summary": f"Learning processing failed: {str(e)}",
            }

    def _aggregate_feedback(
        self, source: str = "all", time_range: str = "last_7_days"
    ) -> dict:
        """Aggregate feedback from all agents' execution history.

        Args:
            source: Filter by source ('all', 'agents', 'users', 'system')
            time_range: Time range for feedback ('last_24h', 'last_7_days', 'last_30_days', 'all')

        Returns:
            dict: {
                agent_feedback: dict of agent_id -> feedback metrics,
                user_feedback: list of user ratings/comments,
                error_patterns: dict of error types and frequencies,
                total_items: int
            }
        """
        self.logger.info(f"Aggregating feedback from {source} for {time_range}")

        # Parse time range
        now = datetime.utcnow()
        time_filters = {
            "last_24h": now - timedelta(hours=24),
            "last_7_days": now - timedelta(days=7),
            "last_30_days": now - timedelta(days=30),
            "all": datetime.min,
        }
        cutoff_time = time_filters.get(time_range, time_filters["last_7_days"])

        agent_feedback = defaultdict(
            lambda: {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "avg_duration": 0.0,
                "total_duration": 0.0,
                "errors": [],
            }
        )

        user_feedback = []
        error_patterns = defaultdict(int)
        total_items = 0

        # Collect from agent execution history (in-memory)
        all_executions = []

        # Get from own history
        all_executions.extend(self.execution_history)

        # Try to get from memory manager if available
        if hasattr(self, "memory_manager") and self.memory_manager:
            try:
                stored_history = self.memory_manager.get(
                    key="execution_history", namespace="system"
                )
                if stored_history:
                    all_executions.extend(stored_history)
            except Exception as e:
                self.logger.warning(f"Could not retrieve from memory manager: {e}")

        # Process executions
        for record in all_executions:
            try:
                # Parse timestamp
                if isinstance(record.get("timestamp"), str):
                    record_time = datetime.fromisoformat(
                        record["timestamp"].replace("Z", "+00:00")
                    )
                elif isinstance(record.get("timestamp"), datetime):
                    record_time = record["timestamp"]
                else:
                    continue

                # Filter by time range
                if record_time < cutoff_time:
                    continue

                agent_id = record.get("agent_id", "unknown")
                success = record.get("success", False)
                duration = record.get("duration", 0.0)

                # Update agent feedback
                agent_feedback[agent_id]["executions"] += 1
                if success:
                    agent_feedback[agent_id]["successes"] += 1
                else:
                    agent_feedback[agent_id]["failures"] += 1

                agent_feedback[agent_id]["total_duration"] += duration

                # Track errors
                if not success and "error" in record:
                    error_msg = record["error"]
                    agent_feedback[agent_id]["errors"].append(error_msg)
                    error_patterns[self._categorize_error(error_msg)] += 1

                # Extract user feedback if present
                if "user_rating" in record:
                    user_feedback.append(
                        {
                            "rating": record["user_rating"],
                            "comment": record.get("user_comment", ""),
                            "agent_id": agent_id,
                            "timestamp": record_time.isoformat(),
                        }
                    )

                total_items += 1

            except Exception as e:
                self.logger.warning(f"Error processing record: {e}")
                continue

        # Calculate averages
        for agent_id, metrics in agent_feedback.items():
            if metrics["executions"] > 0:
                metrics["avg_duration"] = (
                    metrics["total_duration"] / metrics["executions"]
                )
                metrics["success_rate"] = metrics["successes"] / metrics["executions"]

        return {
            "agent_feedback": dict(agent_feedback),
            "user_feedback": user_feedback,
            "error_patterns": dict(error_patterns),
            "total_items": total_items,
            "time_range": time_range,
            "collected_at": now.isoformat(),
        }

    def _categorize_error(self, error_msg: str) -> str:
        """Categorize error messages into patterns."""
        error_msg_lower = str(error_msg).lower()

        if "timeout" in error_msg_lower:
            return "timeout_error"
        elif "connection" in error_msg_lower or "network" in error_msg_lower:
            return "network_error"
        elif "rate limit" in error_msg_lower:
            return "rate_limit_error"
        elif "authentication" in error_msg_lower or "unauthorized" in error_msg_lower:
            return "auth_error"
        elif "validation" in error_msg_lower or "invalid" in error_msg_lower:
            return "validation_error"
        elif "not found" in error_msg_lower or "404" in error_msg_lower:
            return "not_found_error"
        else:
            return "unknown_error"

    def _evaluate_agent_performance(self, agent_id: str, metrics: dict) -> dict:
        """Evaluate agent performance against baselines and identify trends.

        Args:
            agent_id: ID of the agent to evaluate
            metrics: Current performance metrics

        Returns:
            dict: {
                score: float (0-1),
                trend: str ('improving', 'declining', 'stable'),
                areas_for_improvement: list,
                strengths: list
            }
        """
        self.logger.info(f"Evaluating performance for agent: {agent_id}")

        areas_for_improvement = []
        strengths = []
        scores = []

        # Success rate evaluation
        success_rate = metrics.get("success_rate", 0.0)
        baseline_success = self.performance_baselines.get("success_rate", 0.85)

        if success_rate >= baseline_success:
            strengths.append(f"High success rate: {success_rate:.2%}")
            scores.append(1.0)
        elif success_rate >= baseline_success * 0.8:
            scores.append(0.7)
        else:
            areas_for_improvement.append(
                f"Success rate ({success_rate:.2%}) below baseline ({baseline_success:.2%})"
            )
            scores.append(0.4)

        # Response time evaluation
        avg_duration = metrics.get("avg_duration", 0.0)
        baseline_time = self.performance_baselines.get("response_time", 30.0)

        if avg_duration <= baseline_time:
            strengths.append(f"Fast response time: {avg_duration:.2f}s")
            scores.append(1.0)
        elif avg_duration <= baseline_time * 1.5:
            scores.append(0.7)
        else:
            areas_for_improvement.append(
                f"Response time ({avg_duration:.2f}s) exceeds baseline ({baseline_time}s)"
            )
            scores.append(0.4)

        # Error rate evaluation
        executions = metrics.get("executions", 0)
        failures = metrics.get("failures", 0)
        error_rate = failures / executions if executions > 0 else 0

        if error_rate <= 0.05:
            strengths.append(f"Low error rate: {error_rate:.2%}")
            scores.append(1.0)
        elif error_rate <= 0.15:
            scores.append(0.7)
        else:
            areas_for_improvement.append(f"High error rate: {error_rate:.2%}")
            scores.append(0.4)

        # Calculate overall performance score
        performance_score = statistics.mean(scores) if scores else 0.5

        # Determine trend by comparing with historical data
        trend = self._calculate_trend(agent_id, performance_score)

        return {
            "score": round(performance_score, 3),
            "trend": trend,
            "areas_for_improvement": areas_for_improvement,
            "strengths": strengths,
            "metrics": {
                "success_rate": success_rate,
                "avg_duration": avg_duration,
                "error_rate": error_rate,
                "total_executions": executions,
            },
        }

    def _calculate_trend(self, agent_id: str, current_score: float) -> str:
        """Calculate performance trend based on historical data."""
        # Look for historical scores in learning history
        recent_scores = []
        for entry in self.learning_history[-10:]:  # Last 10 evaluations
            if entry.get("agent_id") == agent_id and "score" in entry:
                recent_scores.append(entry["score"])

        if len(recent_scores) < 2:
            return "stable"  # Not enough data

        avg_historical = statistics.mean(recent_scores)

        if current_score > avg_historical * 1.1:
            return "improving"
        elif current_score < avg_historical * 0.9:
            return "declining"
        else:
            return "stable"

    def _generate_improvement_recommendations(
        self, feedback_data: dict, agent_evaluations: dict
    ) -> list:
        """Generate improvement recommendations based on feedback and evaluations.

        Args:
            feedback_data: Aggregated feedback data
            agent_evaluations: Performance evaluations for each agent

        Returns:
            List of actionable improvement recommendations
        """
        recommendations = []

        # Analyze error patterns
        error_patterns = feedback_data.get("error_patterns", {})
        if error_patterns:
            top_errors = sorted(
                error_patterns.items(), key=lambda x: x[1], reverse=True
            )[:3]
            for error_type, count in top_errors:
                recommendations.append(
                    f"Address {error_type} errors (occurred {count} times)"
                )

        # Analyze agent performance
        for agent_id, evaluation in agent_evaluations.items():
            score = evaluation.get("score", 0)
            areas = evaluation.get("areas_for_improvement", [])

            if score < 0.7 and areas:
                recommendations.append(f"Improve {agent_id} performance: {areas[0]}")

        # Add general recommendations if specific ones are limited
        if len(recommendations) < 2:
            recommendations.extend(
                [
                    "Continue monitoring agent performance metrics",
                    "Maintain current optimization strategies",
                ]
            )

        return recommendations[:5]  # Return top 5 recommendations

    def _analyze_user_feedback(
        self, message: str, time_range: str = "last_7_days"
    ) -> dict:
        """Analyze user feedback message and generate appropriate response.

        Args:
            message: User feedback message
            time_range: Time range for historical data

        Returns:
            dict: Analysis results with recommendations
        """
        message_lower = message.lower()

        # Check if this is an investigation request for specific issues
        if any(
            phrase in message_lower
            for phrase in [
                "investigate",
                "investigate and recommend",
                "can you investigate",
                "multiple agents reporting",
                "agents are reporting",
                "look into",
                "what's wrong with",
                "issues with",
                "problems with",
            ]
        ):
            # Extract the issue being reported
            issue = self._extract_issue_from_query(message)
            return self._investigate_and_recommend(message, issue, time_range)

        # Check if this is an agent performance evaluation
        if any(
            phrase in message_lower
            for phrase in [
                "how well is",
                "how is the",
                "agent performing",
                "agent performance",
                "performance of",
                "evaluate agent",
            ]
        ):
            # Extract agent name if mentioned
            agent_name = "system"
            if "customer support" in message_lower or "support agent" in message_lower:
                agent_name = "customer_support"
            elif "marketing" in message_lower or "strategy" in message_lower:
                agent_name = "marketing_strategy"
            elif "analytics" in message_lower:
                agent_name = "analytics_evaluation"
            elif "feedback" in message_lower or "learning" in message_lower:
                agent_name = "feedback_learning"

            return self._generate_performance_evaluation(agent_name, time_range)

        # Check if this is a "what should we learn" scenario
        if any(
            phrase in message_lower
            for phrase in [
                "what should we learn",
                "what can we learn",
                "learn from this",
                "learn from",
            ]
        ):
            return self._analyze_success_pattern(message)

        # Check if this is a rating/quality feedback
        if any(
            word in message_lower
            for word in ["rate", "rating", "stars", "/5", "quality"]
        ):
            # Extract rating if present
            rating = None
            for word in message.split():
                if "/5" in word:
                    try:
                        rating = int(word.split("/")[0])
                    except:
                        pass

            # Determine sentiment
            sentiment = (
                "negative"
                if rating and rating < 3
                else "mixed" if rating == 3 else "positive"
            )
            if not rating:
                sentiment = (
                    "negative"
                    if any(
                        word in message_lower
                        for word in ["poor", "bad", "generic", "terrible"]
                    )
                    else "mixed"
                )

            # Extract what was being rated
            subject = "system"
            if "campaign" in message_lower or "strategy" in message_lower:
                subject = "marketing_strategy"
            elif "support" in message_lower:
                subject = "customer_support"
            elif "analytics" in message_lower or "report" in message_lower:
                subject = "analytics_evaluation"

            return {
                "request_type": "analyze_feedback",
                "feedback_type": "user_rating",
                "analysis": {
                    "rating": rating,
                    "sentiment": sentiment,
                    "subject": subject,
                    "issues_mentioned": self._extract_issues(message),
                },
                "recommendations": self._generate_rating_recommendations(
                    rating, message, subject
                ),
                "summary": (
                    f"Analyzed user rating: {rating}/5 stars for {subject}"
                    if rating
                    else "Analyzed user feedback"
                ),
            }
        else:
            # Generic feedback analysis
            time_range = time_range or "last_7_days"
            feedback_data = self._aggregate_feedback(
                source="all", time_range=time_range
            )

            agent_evaluations = {}
            for agent_id, metrics in feedback_data.get("agent_feedback", {}).items():
                if metrics.get("executions", 0) > 0:
                    evaluation = self._evaluate_agent_performance(agent_id, metrics)
                    agent_evaluations[agent_id] = evaluation

            return {
                "request_type": "analyze_feedback",
                "feedback_type": "system_analysis",
                "feedback_summary": {
                    "total_items": feedback_data.get("total_items", 0),
                    "time_range": time_range,
                    "agents_analyzed": len(agent_evaluations),
                },
                "agent_evaluations": agent_evaluations,
                "improvements": self._generate_improvement_recommendations(
                    feedback_data, agent_evaluations
                ),
            }

    def _analyze_success_pattern(self, message: str) -> dict:
        """Analyze successful outcomes to extract learnings and best practices.

        Args:
            message: Message describing a successful outcome

        Returns:
            dict: Analysis with key learnings and recommendations
        """
        message_lower = message.lower()

        # Extract metrics if mentioned
        metrics = {}
        percentages = []
        for word in message.split():
            if "%" in word:
                try:
                    percentages.append(float(word.replace("%", "")))
                except:
                    pass

        # Determine what type of success
        success_type = "general"
        if "email" in message_lower or "open rate" in message_lower:
            success_type = "email_campaign"
        elif "conversion" in message_lower:
            success_type = "conversion_optimization"
        elif "engagement" in message_lower:
            success_type = "engagement"
        elif "click" in message_lower or "ctr" in message_lower:
            success_type = "click_through"

        # Generate context-aware learnings
        key_learnings = []
        recommendations = []

        if success_type == "email_campaign":
            if percentages and len(percentages) >= 2:
                # We have both the rate and the increase
                rate = percentages[0]
                increase = percentages[1] if len(percentages) > 1 else 0

                key_learnings = [
                    f"Email achieved {rate}% open rate, significantly above baseline",
                    f"Performance improved by {increase}% compared to typical campaigns",
                    "Indicates strong subject line effectiveness and audience targeting",
                ]

                recommendations = [
                    {
                        "priority": "High",
                        "category": "Replication",
                        "action": "Analyze and replicate successful elements",
                        "details": f"Document the specific elements that contributed to the {rate}% open rate: subject line format, send time, audience segment, content preview, and sender name. Create a playbook for future campaigns.",
                        "expected_impact": "Consistently achieve 15-25% higher open rates",
                    },
                    {
                        "priority": "High",
                        "category": "Testing",
                        "action": "A/B test individual success factors",
                        "details": "Isolate each winning element (subject line style, send time, personalization) and test variations to understand which factors had the most impact.",
                        "expected_impact": "Identify top 2-3 drivers of performance",
                    },
                    {
                        "priority": "Medium",
                        "category": "Segmentation",
                        "action": "Analyze audience segment characteristics",
                        "details": "Profile the recipients who opened the email to identify common characteristics. Use this to refine audience segmentation for future campaigns.",
                        "expected_impact": "Improve targeting precision by 30%",
                    },
                    {
                        "priority": "Medium",
                        "category": "Optimization",
                        "action": "Extend analysis to click-through and conversion",
                        "details": "High open rates are valuable, but track the full funnel: CTR, landing page engagement, and conversions to understand complete campaign effectiveness.",
                        "expected_impact": "Optimize entire email funnel",
                    },
                ]
            else:
                key_learnings = [
                    "Email campaign showed strong performance",
                    "Success indicators warrant deeper analysis",
                ]
                recommendations = [
                    {
                        "priority": "Medium",
                        "category": "Analysis",
                        "action": "Conduct detailed campaign post-mortem",
                        "details": "Gather comprehensive metrics and identify success factors that can be replicated in future campaigns.",
                        "expected_impact": "Build repeatable success patterns",
                    }
                ]
        else:
            # Generic success analysis
            key_learnings = [
                "Positive performance outcome detected",
                "Opportunity to capture and replicate success factors",
            ]
            recommendations = [
                {
                    "priority": "Medium",
                    "category": "Learning",
                    "action": "Document success factors for future use",
                    "details": "Capture what worked well in this scenario to build organizational knowledge and improve future outcomes.",
                    "expected_impact": "Build institutional knowledge",
                }
            ]

        return {
            "request_type": "analyze_feedback",
            "feedback_type": "success_analysis",
            "analysis": {
                "success_type": success_type,
                "metrics_mentioned": percentages,
                "key_learnings": key_learnings,
            },
            "recommendations": recommendations,
            "next_steps": [
                "Document successful elements in a reusable playbook",
                "Share learnings with relevant team members",
                "Plan follow-up campaigns to validate findings",
                "Monitor ongoing performance against new baseline",
            ],
            "summary": f"Analyzed successful {success_type.replace('_', ' ')} to extract learnings and best practices",
        }

    def _detect_issue_patterns(
        self, message: str, time_range: str = "last_7_days"
    ) -> dict:
        """Detect patterns in reported issues and generate recommendations.

        Args:
            message: User message describing issues
            time_range: Time range for pattern detection

        Returns:
            dict: Pattern analysis with recommendations
        """
        message_lower = message.lower()

        # Extract the reported issue
        issues = self._extract_issues(message)

        # Determine affected area
        affected_area = "unknown"
        if "checkout" in message_lower:
            affected_area = "checkout_flow"
        elif "mobile" in message_lower:
            affected_area = "mobile_experience"
        elif "payment" in message_lower:
            affected_area = "payment_processing"
        elif "email" in message_lower:
            affected_area = "email_delivery"

        # Check if multiple agents/sources mentioned
        multiple_reports = any(
            word in message_lower
            for word in ["multiple", "several", "many", "recurring"]
        )

        # Generate contextual recommendations
        recommendations = []

        if "checkout" in message_lower and "mobile" in message_lower:
            recommendations = [
                {
                    "priority": "High",
                    "category": "User Experience",
                    "action": "Audit mobile checkout flow for responsive design issues",
                    "details": "Multiple reports suggest mobile checkout has usability problems. Conduct a comprehensive audit of the mobile checkout experience, focusing on touch targets, form fields, and payment integration.",
                    "expected_impact": "Could improve mobile conversion rates by 15-25%",
                },
                {
                    "priority": "High",
                    "category": "Technical",
                    "action": "Review mobile checkout error logs and analytics",
                    "details": "Analyze error logs, abandonment rates, and user session recordings to identify specific failure points in the mobile checkout process.",
                    "expected_impact": "Identify root causes of checkout failures",
                },
                {
                    "priority": "Medium",
                    "category": "Testing",
                    "action": "Implement automated mobile checkout testing",
                    "details": "Set up automated end-to-end tests across multiple mobile devices and browsers to catch checkout issues before they reach production.",
                    "expected_impact": "Reduce checkout-related bugs by 40%",
                },
                {
                    "priority": "Medium",
                    "category": "User Research",
                    "action": "Conduct user testing sessions on mobile checkout",
                    "details": "Run moderated user testing with 8-10 participants to observe real-world mobile checkout behavior and gather qualitative feedback.",
                    "expected_impact": "Uncover hidden UX friction points",
                },
            ]
        else:
            # Generic issue recommendations
            recommendations = [
                {
                    "priority": "High" if multiple_reports else "Medium",
                    "category": "Investigation",
                    "action": f"Investigate reported {affected_area} issues",
                    "details": f"The issue with {affected_area} has been reported{' by multiple sources' if multiple_reports else ''}. Conduct a thorough investigation including logs, metrics, and user feedback.",
                    "expected_impact": "Identify root cause and scope of issue",
                },
                {
                    "priority": "Medium",
                    "category": "Monitoring",
                    "action": f"Enhance monitoring for {affected_area}",
                    "details": "Set up additional monitoring and alerting to catch similar issues early and track resolution effectiveness.",
                    "expected_impact": "Faster issue detection and response",
                },
            ]

        return {
            "request_type": "detect_patterns",
            "feedback_type": (
                "recurring_issue" if multiple_reports else "reported_issue"
            ),
            "pattern_type": "recurring_issue" if multiple_reports else "reported_issue",
            "analysis": {
                "affected_area": affected_area,
                "issues_identified": issues,
                "severity": "high" if multiple_reports else "medium",
                "multiple_reports": multiple_reports,
            },
            "recommendations": recommendations,
            "next_steps": [
                "Gather additional data from affected users",
                "Prioritize fixes based on impact and frequency",
                "Implement monitoring to track issue resolution",
                "Follow up with users to verify fixes",
            ],
            "summary": f"Detected {'recurring' if multiple_reports else 'reported'} issue with {affected_area}: {len(recommendations)} recommendations generated",
        }

    def _extract_issues(self, message: str) -> list:
        """Extract specific issues mentioned in the message.

        Args:
            message: User message

        Returns:
            list: List of identified issues
        """
        message_lower = message.lower()
        issues = []

        issue_keywords = {
            "too generic": "Content lacks specificity",
            "generic": "Generic content",
            "slow": "Performance issues",
            "error": "Errors occurring",
            "crash": "System crashes",
            "not working": "Functionality broken",
            "broken": "Feature not functioning",
            "confusing": "User experience unclear",
            "complicated": "Overly complex",
        }

        for keyword, issue in issue_keywords.items():
            if keyword in message_lower:
                issues.append(issue)

        return issues if issues else ["Unspecified issue"]

    def _generate_performance_evaluation(
        self, agent_name: str, time_range: str = "last_7_days"
    ) -> dict:
        """Generate performance evaluation for a specific agent.

        Args:
            agent_name: Name of agent to evaluate
            time_range: Time range for evaluation

        Returns:
            dict: Performance evaluation with metrics and recommendations
        """
        # Aggregate feedback for this agent
        feedback_data = self._aggregate_feedback(source="all", time_range=time_range)

        # Get agent-specific metrics
        agent_metrics = feedback_data.get("agent_feedback", {}).get(agent_name, {})

        if not agent_metrics or agent_metrics.get("executions", 0) == 0:
            # No data available - provide guidance on what to measure
            return {
                "request_type": "analyze_feedback",
                "feedback_type": "performance_evaluation",
                "agent_name": agent_name,
                "analysis": {
                    "status": "Insufficient data",
                    "message": f"No execution data available for {agent_name} in the {time_range} period.",
                    "metrics": {
                        "executions": 0,
                        "avg_response_time": "N/A",
                        "success_rate": "N/A",
                        "errors": 0,
                    },
                },
                "recommendations": [
                    {
                        "priority": "High",
                        "category": "Data Collection",
                        "action": "Enable comprehensive agent monitoring",
                        "details": f"Set up metrics collection for {agent_name} to track: execution count, response times, success rates, error patterns, and user satisfaction scores.",
                        "expected_impact": "Establish performance baseline for ongoing optimization",
                    },
                    {
                        "priority": "High",
                        "category": "Testing",
                        "action": "Execute test workflows",
                        "details": f"Run sample workflows through {agent_name} to generate initial performance data and identify any immediate issues.",
                        "expected_impact": "Generate baseline metrics for evaluation",
                    },
                    {
                        "priority": "Medium",
                        "category": "KPI Definition",
                        "action": "Define success metrics",
                        "details": "Establish clear KPIs for agent performance: target response time (<30s), success rate (>85%), user satisfaction (>4/5), and handoff accuracy (>90%).",
                        "expected_impact": "Enable objective performance assessment",
                    },
                    {
                        "priority": "Medium",
                        "category": "Monitoring",
                        "action": "Implement continuous monitoring",
                        "details": "Set up dashboards and alerts to track agent performance in real-time, with automated notifications for performance degradation.",
                        "expected_impact": "Proactive identification of performance issues",
                    },
                ],
                "summary": f"Performance evaluation for {agent_name}: Insufficient data for comprehensive analysis. Implement monitoring and testing to establish baseline metrics.",
            }

        # Evaluate performance with available data
        evaluation = self._evaluate_agent_performance(agent_name, agent_metrics)

        # Generate specific recommendations based on performance
        recommendations = []

        # Check response time
        avg_time = agent_metrics.get("avg_response_time", 0)
        if avg_time > self.performance_baselines["response_time"]:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Performance",
                    "action": "Optimize response time",
                    "details": f"Average response time ({avg_time:.1f}s) exceeds target ({self.performance_baselines['response_time']}s). Review prompt complexity, reduce unnecessary tool calls, and implement caching for common queries.",
                    "expected_impact": f"Reduce response time to <{self.performance_baselines['response_time']}s",
                }
            )
        else:
            recommendations.append(
                {
                    "priority": "Low",
                    "category": "Performance",
                    "action": "Maintain response time performance",
                    "details": f"Response time ({avg_time:.1f}s) is within target range. Continue monitoring to ensure consistency.",
                    "expected_impact": "Sustained fast response times",
                }
            )

        # Check success rate
        success_rate = agent_metrics.get("success_rate", 0)
        if success_rate < self.performance_baselines["success_rate"]:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Reliability",
                    "action": "Improve success rate",
                    "details": f"Success rate ({success_rate:.1%}) is below target ({self.performance_baselines['success_rate']:.0%}). Analyze error patterns, improve error handling, and add input validation.",
                    "expected_impact": f"Increase success rate to >{self.performance_baselines['success_rate']:.0%}",
                }
            )
        else:
            recommendations.append(
                {
                    "priority": "Low",
                    "category": "Reliability",
                    "action": "Maintain high success rate",
                    "details": f"Success rate ({success_rate:.1%}) meets or exceeds target. Continue monitoring and address errors proactively.",
                    "expected_impact": "Sustained high reliability",
                }
            )

        # Check error patterns
        errors = agent_metrics.get("errors", 0)
        if errors > 0:
            recommendations.append(
                {
                    "priority": "Medium",
                    "category": "Error Handling",
                    "action": "Analyze and reduce errors",
                    "details": f"Detected {errors} errors in {time_range}. Review error logs to identify root causes, implement fixes, and add preventive measures.",
                    "expected_impact": "Reduce error rate by 50%",
                }
            )

        return {
            "request_type": "analyze_feedback",
            "feedback_type": "performance_evaluation",
            "agent_name": agent_name,
            "time_range": time_range,
            "analysis": {
                "status": "Data available",
                "metrics": {
                    "executions": agent_metrics.get("executions", 0),
                    "avg_response_time": f"{avg_time:.2f}s",
                    "success_rate": f"{success_rate:.1%}",
                    "errors": errors,
                },
                "performance_score": evaluation.get("performance_score", 0),
                "trend": evaluation.get("trend", "stable"),
            },
            "recommendations": recommendations,
            "summary": f"Performance evaluation for {agent_name}: {len(agent_metrics.get('executions', 0))} executions analyzed. Overall performance score: {evaluation.get('performance_score', 0):.2f}/1.0 ({evaluation.get('trend', 'stable')} trend).",
        }

    def _generate_rating_recommendations(
        self, rating: Optional[int], message: str, subject: str
    ) -> list:
        """Generate recommendations based on user rating.

        Args:
            rating: Numeric rating (1-5)
            message: Feedback message
            subject: What was being rated

        Returns:
            list: Recommendations for improvement
        """
        recommendations = []
        issues = self._extract_issues(message)

        if rating and rating < 3:  # Poor rating
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Quality Improvement",
                    "action": f"Address quality concerns with {subject}",
                    "details": f"User rated {subject} as {rating}/5 stars. Specific issues: {', '.join(issues)}. Review and enhance output quality, add more specific context, and improve personalization.",
                    "expected_impact": "Improve user satisfaction by 20-30%",
                }
            )

            recommendations.append(
                {
                    "priority": "High",
                    "category": "Training",
                    "action": f"Enhance {subject} prompts and examples",
                    "details": "Update system prompts to generate more specific, actionable outputs. Include better examples and constraints to prevent generic responses.",
                    "expected_impact": "Increase output specificity and relevance",
                }
            )

            if "generic" in message.lower():
                recommendations.append(
                    {
                        "priority": "Medium",
                        "category": "Personalization",
                        "action": "Incorporate more context into responses",
                        "details": "Ensure the agent has access to sufficient context (user history, preferences, industry specifics) to generate personalized, non-generic responses.",
                        "expected_impact": "Reduce generic output by 40%",
                    }
                )
        elif rating and rating == 3:  # Average rating
            recommendations.append(
                {
                    "priority": "Medium",
                    "category": "Optimization",
                    "action": f"Optimize {subject} for better results",
                    "details": f"User gave a neutral rating of {rating}/5. While functional, there's room for improvement. Focus on enhancing response quality and relevance.",
                    "expected_impact": "Move ratings from 3★ to 4-5★",
                }
            )

        return recommendations

    def _generate_prediction_improvement_plan(
        self, message: str, metrics: dict
    ) -> dict:
        """Generate a detailed plan for improving prediction accuracy.

        Args:
            message: User's query about prediction improvement
            metrics: Current performance metrics

        Returns:
            dict: Detailed improvement plan with recommendations
        """
        message_lower = message.lower()

        # Extract campaign metrics if available
        campaign_metrics = metrics.get("campaign_metrics", {})
        conversion_rate = campaign_metrics.get("conversion_rate", 0)
        impressions = campaign_metrics.get("total_impressions", 0)
        conversions = campaign_metrics.get("total_conversions", 0)

        # Determine confidence level based on data volume
        if impressions > 10000 and conversions > 100:
            confidence = "High"
            data_quality = "Excellent"
        elif impressions > 5000 and conversions > 50:
            confidence = "Medium"
            data_quality = "Good"
        else:
            confidence = "Low"
            data_quality = "Limited - More data needed"

        # Generate specific recommendations
        recommendations = []

        # Data collection recommendations
        if impressions < 10000:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Data Collection",
                    "action": "Increase sample size",
                    "details": f"Current: {impressions:,} impressions. Target: 10,000+ for reliable predictions.",
                    "expected_impact": "Significantly improve prediction confidence by reducing variance",
                }
            )

        # Segmentation recommendations
        recommendations.append(
            {
                "priority": "High",
                "category": "Segmentation",
                "action": "Implement multi-dimensional segmentation",
                "details": "Break down conversions by channel, audience demographics, time periods, and device types",
                "expected_impact": "Enable more accurate predictions for specific user segments",
            }
        )

        # Model improvement recommendations
        recommendations.append(
            {
                "priority": "Medium",
                "category": "Model Enhancement",
                "action": "Implement time-series forecasting",
                "details": "Use historical data patterns to account for seasonality and trends",
                "expected_impact": "Capture temporal patterns that affect conversion rates",
            }
        )

        # Testing recommendations
        recommendations.append(
            {
                "priority": "Medium",
                "category": "Validation",
                "action": "Set up A/B testing framework",
                "details": "Run controlled experiments to validate prediction models against actual results",
                "expected_impact": "Measure and improve prediction accuracy systematically",
            }
        )

        # External factors
        recommendations.append(
            {
                "priority": "Low",
                "category": "Context Integration",
                "action": "Incorporate external factors",
                "details": "Account for market conditions, competitor activity, and seasonal events",
                "expected_impact": "Reduce prediction errors from external variables",
            }
        )

        # Model maintenance
        recommendations.append(
            {
                "priority": "Medium",
                "category": "Maintenance",
                "action": "Establish quarterly model refresh cycle",
                "details": "Update baseline metrics and retrain models with recent data every 90 days",
                "expected_impact": "Prevent model drift and maintain accuracy over time",
            }
        )

        return {
            "request_type": "prediction_improvement",
            "analysis": {
                "query": message,
                "current_metrics": {
                    "conversion_rate": f"{conversion_rate:.2f}%",
                    "impressions": f"{impressions:,}",
                    "conversions": f"{conversions:,}",
                },
                "prediction_confidence": confidence,
                "data_quality": data_quality,
            },
            "recommendations": recommendations,
            "summary": f"Generated {len(recommendations)} actionable recommendations to improve prediction accuracy",
            "next_steps": [
                "Prioritize high-impact recommendations",
                "Establish baseline metrics for measuring improvement",
                "Implement recommendations incrementally",
                "Monitor prediction accuracy after each change",
                "Document learnings and update models accordingly",
            ],
        }

    def _detect_patterns(self, execution_data: list) -> dict:
        """Detect success and failure patterns in execution data.

        Args:
            execution_data: List of execution records

        Returns:
            dict: {
                success_patterns: list[dict],
                failure_patterns: list[dict],
                best_practices: list
            }
        """
        self.logger.info(f"Detecting patterns in {len(execution_data)} executions")

        if not execution_data:
            return {
                "success_patterns": [],
                "failure_patterns": [],
                "best_practices": [],
            }

        # Use pattern detector tool
        patterns = self.pattern_detector.detect_patterns(execution_data)

        # Extract success patterns
        success_patterns = patterns.get("success_patterns", [])
        failure_patterns = patterns.get("failure_patterns", [])

        # Extract best practices from patterns
        best_practices = self._extract_best_practices(
            success_patterns, failure_patterns
        )

        return {
            "success_patterns": success_patterns,
            "failure_patterns": failure_patterns,
            "best_practices": best_practices,
        }

    def _extract_best_practices(
        self, success_patterns: list, failure_patterns: list
    ) -> list:
        """Extract actionable best practices from patterns."""
        best_practices = []

        # From success patterns
        for pattern in success_patterns:
            if pattern.get("type") == "optimal_duration":
                best_practices.append(
                    "Optimize agent execution time to match successful patterns"
                )
            elif pattern.get("type") == "high_performing_agent":
                best_practices.append(
                    f"Study and replicate strategies from {pattern.get('agent_id', 'top')} agent"
                )

        # From failure patterns
        for pattern in failure_patterns:
            pattern_type = pattern.get("type", "")
            error_type = pattern.get("error_type", "")

            if "timeout" in pattern_type or "timeout" in error_type:
                best_practices.append(
                    "Implement retry logic with exponential backoff for timeout errors"
                )
            elif "network" in pattern_type or "network" in error_type:
                best_practices.append(
                    "Add connection pooling and circuit breakers for network reliability"
                )
            elif "rate_limit" in pattern_type or "rate_limit" in error_type:
                best_practices.append("Implement rate limiting and request throttling")
            elif "validation" in pattern_type or "validation" in error_type:
                best_practices.append("Strengthen input validation and schema checking")

        return list(set(best_practices))  # Remove duplicates

    def _optimize_prompts(self, agent_id: str, performance_data: dict) -> dict:
        """Analyze prompt performance and suggest optimizations.

        Args:
            agent_id: ID of the agent
            performance_data: Current performance metrics

        Returns:
            dict: {
                current_config: dict,
                suggested_changes: dict,
                expected_improvement: str
            }
        """
        self.logger.info(f"Optimizing prompts for agent: {agent_id}")

        # Get current config
        current_config = self._get_agent_config(agent_id)
        suggested_changes = {}
        expected_improvement = "moderate"

        if not current_config:
            return {
                "current_config": {},
                "suggested_changes": {},
                "expected_improvement": "unknown",
                "error": "Agent config not found",
            }

        model_config = current_config.get("model", {})
        performance_score = performance_data.get("score", 0.5)
        trend = performance_data.get("trend", "stable")

        # Temperature optimization
        current_temp = model_config.get("temperature", 0.7)
        if performance_score < 0.6:
            # Low performance - try reducing temperature for more deterministic outputs
            if current_temp > 0.3:
                suggested_changes["temperature"] = max(0.2, current_temp - 0.2)
                expected_improvement = "significant"
        elif trend == "improving" and performance_score > 0.8:
            # Good performance - might try slightly higher creativity
            if current_temp < 0.5:
                suggested_changes["temperature"] = min(0.7, current_temp + 0.1)
                expected_improvement = "minor"

        # Max tokens optimization
        current_max_tokens = model_config.get("max_tokens", 2000)
        avg_duration = performance_data.get("metrics", {}).get("avg_duration", 0)

        if avg_duration > 25:  # Slow responses
            if current_max_tokens > 1500:
                suggested_changes["max_tokens"] = 1500
                expected_improvement = "significant"

        # Error rate optimization
        error_rate = performance_data.get("metrics", {}).get("error_rate", 0)
        if error_rate > 0.15:
            # High error rate - suggest more structured output
            suggested_changes["system_message_adjustment"] = (
                "Add explicit error handling instructions"
            )
            suggested_changes["response_format"] = "json_object"
            expected_improvement = "significant"

        # Model optimization
        if performance_score < 0.5 and model_config.get("name") == "gpt-3.5-turbo":
            suggested_changes["model_name"] = "gpt-4-turbo-preview"
            expected_improvement = "significant"

        return {
            "current_config": model_config,
            "suggested_changes": suggested_changes,
            "expected_improvement": expected_improvement,
            "reasoning": self._explain_optimizations(
                suggested_changes, performance_data
            ),
        }

    def _explain_optimizations(self, changes: dict, performance_data: dict) -> list:
        """Provide reasoning for suggested optimizations."""
        explanations = []

        if "temperature" in changes:
            explanations.append(
                f"Temperature adjustment to {changes['temperature']:.2f} "
                f"based on current performance score of {performance_data.get('score', 0):.2f}"
            )

        if "max_tokens" in changes:
            explanations.append(
                f"Reducing max_tokens to {changes['max_tokens']} "
                f"to improve response time (current: {performance_data.get('metrics', {}).get('avg_duration', 0):.2f}s)"
            )

        if "model_name" in changes:
            explanations.append(
                f"Upgrading to {changes['model_name']} "
                f"due to low performance score ({performance_data.get('score', 0):.2f})"
            )

        return explanations

    def _get_agent_config(self, agent_id: str) -> Optional[dict]:
        """Load agent configuration from YAML."""
        try:
            if not self.config_path.exists():
                self.logger.warning(f"Config file not found: {self.config_path}")
                return None

            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            agents_config = config.get("agents", {})
            return agents_config.get(agent_id)

        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return None

    def _update_agent_config(self, agent_id: str, new_config: dict) -> dict:
        """Update agent configuration in memory and optionally write to YAML.

        Args:
            agent_id: ID of the agent
            new_config: New configuration parameters

        Returns:
            dict: {
                updated: bool,
                agent_id: str,
                changes: dict,
                timestamp: datetime
            }
        """
        self.logger.info(f"Updating config for agent: {agent_id}")
        timestamp = datetime.utcnow()

        try:
            # Validate configuration
            validation_result = self._validate_config(new_config)
            if not validation_result["valid"]:
                return {
                    "updated": False,
                    "agent_id": agent_id,
                    "changes": {},
                    "timestamp": timestamp,
                    "error": validation_result["errors"],
                }

            # Load current config
            if not self.config_path.exists():
                self.logger.error(f"Config file not found: {self.config_path}")
                return {
                    "updated": False,
                    "agent_id": agent_id,
                    "changes": {},
                    "timestamp": timestamp,
                    "error": "Config file not found",
                }

            with open(self.config_path, "r") as f:
                full_config = yaml.safe_load(f)

            # Get current agent config
            if agent_id not in full_config.get("agents", {}):
                return {
                    "updated": False,
                    "agent_id": agent_id,
                    "changes": {},
                    "timestamp": timestamp,
                    "error": f"Agent {agent_id} not found in config",
                }

            agent_config = full_config["agents"][agent_id]

            # Track changes
            changes = {}
            for key, value in new_config.items():
                if key in agent_config:
                    old_value = agent_config[key]
                    if old_value != value:
                        changes[key] = {"old": old_value, "new": value}
                        agent_config[key] = value
                elif key in agent_config.get("model", {}):
                    old_value = agent_config["model"][key]
                    if old_value != value:
                        changes[f"model.{key}"] = {"old": old_value, "new": value}
                        agent_config["model"][key] = value

            # Write back to file (optional - can be disabled for safety)
            # Uncomment to enable auto-save:
            # with open(self.config_path, 'w') as f:
            #     yaml.dump(full_config, f, default_flow_style=False)

            # Log configuration change
            log_entry = {
                "timestamp": timestamp.isoformat(),
                "agent_id": agent_id,
                "changes": changes,
                "source": "feedback_learning_agent",
            }
            self.learning_history.append(log_entry)

            return {
                "updated": True,
                "agent_id": agent_id,
                "changes": changes,
                "timestamp": timestamp,
                "note": "Config updated in memory only. Uncomment save code to persist to YAML.",
            }

        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
            return {
                "updated": False,
                "agent_id": agent_id,
                "changes": {},
                "timestamp": timestamp,
                "error": str(e),
            }

    def _validate_config(self, config: dict) -> dict:
        """Validate configuration changes."""
        errors = []

        # Temperature validation
        if "temperature" in config:
            temp = config["temperature"]
            if not isinstance(temp, (int, float)):
                errors.append("temperature must be a number")
            elif temp < 0 or temp > 2:
                errors.append("temperature must be between 0 and 2")

        # Max tokens validation
        if "max_tokens" in config:
            tokens = config["max_tokens"]
            if not isinstance(tokens, int):
                errors.append("max_tokens must be an integer")
            elif tokens < 100 or tokens > 8000:
                errors.append("max_tokens must be between 100 and 8000")

        # Model name validation
        if "model_name" in config:
            valid_models = [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo-preview",
                "gpt-4o",
                "gpt-4o-mini",
            ]
            if config["model_name"] not in valid_models:
                errors.append(f"model_name must be one of {valid_models}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    def _track_experiment(
        self, experiment_name: str, variant: str, metrics: dict
    ) -> dict:
        """Track A/B test or experiment results.

        Args:
            experiment_name: Name of the experiment
            variant: Variant identifier (e.g., 'control', 'variant_a', 'variant_b')
            metrics: Performance metrics for this variant

        Returns:
            dict: {
                experiment_id: str,
                variant: str,
                metrics: dict,
                stored: bool
            }
        """
        self.logger.info(f"Tracking experiment: {experiment_name} - {variant}")

        experiment_id = f"{experiment_name}_{datetime.utcnow().strftime('%Y%m%d')}"

        experiment_record = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "variant": variant,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Store in memory
        self.experiment_store[experiment_name].append(experiment_record)

        # Keep only last 100 experiments per name
        if len(self.experiment_store[experiment_name]) > 100:
            self.experiment_store[experiment_name] = self.experiment_store[
                experiment_name
            ][-100:]

        return {
            "experiment_id": experiment_id,
            "variant": variant,
            "metrics": metrics,
            "stored": True,
            "total_variants": len(
                [
                    e
                    for e in self.experiment_store[experiment_name]
                    if e["experiment_id"] == experiment_id
                ]
            ),
        }

    def get_experiment_results(self, experiment_name: str) -> dict:
        """Get aggregated results for an experiment."""
        if experiment_name not in self.experiment_store:
            return {
                "experiment_name": experiment_name,
                "found": False,
                "error": "Experiment not found",
            }

        records = self.experiment_store[experiment_name]

        # Group by variant
        variants = defaultdict(list)
        for record in records:
            variants[record["variant"]].append(record["metrics"])

        # Calculate statistics per variant
        variant_stats = {}
        for variant, metrics_list in variants.items():
            variant_stats[variant] = self._calculate_variant_stats(metrics_list)

        return {
            "experiment_name": experiment_name,
            "found": True,
            "total_records": len(records),
            "variants": variant_stats,
            "recommendation": self._recommend_variant(variant_stats),
        }

    def _calculate_variant_stats(self, metrics_list: list) -> dict:
        """Calculate statistics for a variant."""
        if not metrics_list:
            return {}

        # Extract common metrics
        success_rates = [
            m.get("success_rate", 0) for m in metrics_list if "success_rate" in m
        ]
        durations = [
            m.get("avg_duration", 0) for m in metrics_list if "avg_duration" in m
        ]

        stats = {
            "sample_size": len(metrics_list),
        }

        if success_rates:
            stats["avg_success_rate"] = statistics.mean(success_rates)
            stats["std_success_rate"] = (
                statistics.stdev(success_rates) if len(success_rates) > 1 else 0
            )

        if durations:
            stats["avg_duration"] = statistics.mean(durations)
            stats["std_duration"] = (
                statistics.stdev(durations) if len(durations) > 1 else 0
            )

        return stats

    def _recommend_variant(self, variant_stats: dict) -> str:
        """Recommend best performing variant."""
        if not variant_stats:
            return "No data available"

        best_variant = None
        best_score = -1

        for variant, stats in variant_stats.items():
            # Simple scoring: higher success rate and lower duration is better
            success_rate = stats.get("avg_success_rate", 0)
            duration = stats.get("avg_duration", 100)

            # Normalize duration (inverse, lower is better)
            normalized_duration = 1.0 / (1.0 + duration / 10.0)

            # Combined score
            score = (success_rate * 0.7) + (normalized_duration * 0.3)

            if score > best_score:
                best_score = score
                best_variant = variant

        return f"Recommended variant: {best_variant} (score: {best_score:.3f})"

    def _extract_issue_from_query(self, message: str) -> str:
        """Extract the specific issue being mentioned in the query.

        Args:
            message: User message text

        Returns:
            str: Extracted issue description
        """
        message_lower = message.lower()

        # Look for common patterns
        issue_indicators = [
            "issues with",
            "problems with",
            "trouble with",
            "bug in",
            "error in",
            "failing",
            "broken",
        ]

        for indicator in issue_indicators:
            if indicator in message_lower:
                # Extract what comes after the indicator
                parts = message_lower.split(indicator, 1)
                if len(parts) > 1:
                    issue_text = parts[1].split(".")[0].strip()
                    return issue_text

        # If no specific indicator, try to extract key phrases
        keywords = [
            "checkout",
            "payment",
            "mobile",
            "cart",
            "conversion",
            "form",
            "api",
            "integration",
        ]
        found_keywords = [kw for kw in keywords if kw in message_lower]

        if found_keywords:
            return " ".join(found_keywords)

        return "general system issues"

    def _investigate_and_recommend(
        self, message: str, issue: str, time_range: str
    ) -> dict:
        """Investigate reported issues and provide recommendations.

        Args:
            message: Original user message
            issue: Extracted issue description
            time_range: Time range for analysis

        Returns:
            dict: Investigation results with recommendations
        """
        # Analyze the issue context
        investigation = {
            "issue": issue,
            "severity": "medium",  # Default
            "affected_areas": [],
            "potential_causes": [],
            "recommendations": [],
        }

        message_lower = message.lower()

        # Determine severity based on keywords
        if any(
            word in message_lower
            for word in ["critical", "urgent", "multiple agents", "many users"]
        ):
            investigation["severity"] = "high"
        elif any(
            word in message_lower for word in ["minor", "occasional", "sometimes"]
        ):
            investigation["severity"] = "low"

        # Identify affected areas
        if "mobile" in message_lower:
            investigation["affected_areas"].append("Mobile experience")
        if "checkout" in message_lower or "payment" in message_lower:
            investigation["affected_areas"].append("Checkout/Payment flow")
        if "cart" in message_lower:
            investigation["affected_areas"].append("Shopping cart")
        if "form" in message_lower:
            investigation["affected_areas"].append("Form submission")

        # Generate context-specific recommendations based on the issue
        if "mobile" in message_lower and "checkout" in message_lower:
            investigation["potential_causes"] = [
                "Responsive design issues on smaller screens",
                "Touch target sizes too small for mobile interaction",
                "Payment form not optimized for mobile keyboards",
                "Page load times affecting mobile completion rates",
                "Third-party payment widgets not mobile-friendly",
            ]
            investigation["recommendations"] = [
                {
                    "priority": "high",
                    "action": "Conduct mobile UX audit",
                    "details": "Test checkout flow on various mobile devices to identify specific friction points",
                },
                {
                    "priority": "high",
                    "action": "Optimize touch targets",
                    "details": "Ensure all buttons and form fields meet minimum 44x44px touch target size",
                },
                {
                    "priority": "medium",
                    "action": "Review payment integration",
                    "details": "Verify payment provider's mobile SDK is properly implemented and up-to-date",
                },
                {
                    "priority": "medium",
                    "action": "A/B test simplified flow",
                    "details": "Test a streamlined mobile checkout with fewer fields and steps",
                },
                {
                    "priority": "low",
                    "action": "Monitor mobile analytics",
                    "details": "Set up detailed tracking for mobile checkout drop-off points",
                },
            ]
        elif "checkout" in message_lower or "payment" in message_lower:
            investigation["potential_causes"] = [
                "Complex checkout process with too many steps",
                "Payment gateway integration errors",
                "Lack of preferred payment methods",
                "Trust indicators missing",
                "Hidden costs revealed late in the process",
            ]
            investigation["recommendations"] = [
                {
                    "priority": "high",
                    "action": "Simplify checkout flow",
                    "details": "Reduce checkout to 2-3 steps maximum, enable guest checkout",
                },
                {
                    "priority": "high",
                    "action": "Review error handling",
                    "details": "Ensure payment errors provide clear, actionable messages to users",
                },
                {
                    "priority": "medium",
                    "action": "Add payment options",
                    "details": "Consider adding popular payment methods (Apple Pay, Google Pay, PayPal)",
                },
                {
                    "priority": "medium",
                    "action": "Display trust signals",
                    "details": "Add security badges, SSL indicators, and money-back guarantees",
                },
            ]
        else:
            # Generic recommendations for other issues
            investigation["potential_causes"] = [
                "User experience friction in the workflow",
                "Technical errors or integration issues",
                "Insufficient user guidance or documentation",
                "Performance or reliability problems",
            ]
            investigation["recommendations"] = [
                {
                    "priority": "high",
                    "action": "Gather detailed user feedback",
                    "details": f"Set up surveys or interviews to understand specific pain points with {issue}",
                },
                {
                    "priority": "high",
                    "action": "Review error logs",
                    "details": "Check system logs for recurring errors or exceptions",
                },
                {
                    "priority": "medium",
                    "action": "Conduct usability testing",
                    "details": "Observe real users attempting to complete the affected workflow",
                },
                {
                    "priority": "medium",
                    "action": "Benchmark against competitors",
                    "details": "Compare your implementation with industry best practices",
                },
            ]

        return {
            "request_type": "investigate_issues",
            "issue_summary": {
                "description": issue,
                "severity": investigation["severity"],
                "affected_areas": investigation["affected_areas"],
                "agents_reporting": (
                    "Multiple"
                    if "multiple agents" in message_lower
                    else "User reported"
                ),
            },
            "investigation": investigation,
            "recommendations": investigation["recommendations"],
            "next_steps": [
                "Prioritize recommendations by severity and impact",
                "Create implementation plan with timeline",
                "Set up monitoring to track improvement metrics",
                "Schedule follow-up review in 2-4 weeks",
            ],
            "summary": f"Investigation complete: {len(investigation['recommendations'])} recommendations for {issue}",
        }

    async def _detect_handoff_need(
        self, message: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect if feedback & learning request warrants a handoff to another agent using LLM reasoning.

        Args:
            message: User message text
            result: Learning result dictionary

        Returns:
            Dictionary with handoff information if needed, empty dict otherwise
        """
        try:
            # Use LLM-driven handoff detection
            handoff_info = await self.handoff_detector.detect_handoff(
                current_agent="feedback_learning",
                user_message=message,
                agent_analysis=result,
            )

            return handoff_info

        except Exception as e:
            self.logger.error(f"Handoff detection failed: {e}", exc_info=True)
            return {}
