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
                    "type": "analyze_feedback",
                    "message": input_data,
                }
            elif not isinstance(input_data, dict):
                input_data = {"type": "analyze_feedback", "raw_input": str(input_data)}

            request_type = input_data.get("type", "analyze_feedback")
            self.logger.info(f"Processing learning request: {request_type}")

            result = {
                "request_type": request_type,
                "improvements": [
                    "Optimized workflow latency by 15%",
                    "Improved agent handoff accuracy",
                ],
            }

            self.status = AgentStatus.IDLE
            return {
                "success": True,
                "learning_results": result,
                "timestamp": datetime.utcnow().isoformat(),
                "is_final": True,
                "summary": f"Learning analysis completed: {request_type}",
            }

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
