"""
Analytics & Evaluation Agent - Monitors performance and generates insights.

Tracks campaign performance, evaluates metrics, generates reports,
and provides data-driven insights for optimization.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import statistics
from scipy import stats
import numpy as np

from ..core.base_agent import BaseAgent, AgentStatus
from ..tools.metrics_calculator import (
    calculate_campaign_metrics,
    calculate_agent_metrics,
    calculate_system_metrics,
    calculate_time_series_metrics,
    calculate_comparison_metrics,
)
from ..tools.visualization import (
    generate_campaign_dashboard,
    generate_agent_dashboard,
    generate_system_dashboard,
    create_line_chart,
    create_funnel_chart,
)


class AnalyticsEvaluationAgent(BaseAgent):
    """Specialized agent for analytics and performance evaluation."""

    def __init__(self, agent_id: str, config: Optional[Dict] = None):
        """Initialize analytics agent."""
        super().__init__(agent_id, config)
        self.metrics_history: List[Dict] = []
        self.kpi_definitions = {}

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process analytics request."""
        self.status = AgentStatus.PROCESSING
        start_time = datetime.utcnow()

        try:
            request_type = input_data.get("type", "performance_report")
            self.logger.info(f"Processing analytics request: {request_type}")

            # Route to appropriate handler
            if request_type == "calculate_metrics":
                result = await self._handle_calculate_metrics(input_data)
            elif request_type == "generate_report":
                result = await self._handle_generate_report(input_data)
            elif request_type == "forecast_performance":
                result = await self._handle_forecast(input_data)
            elif request_type == "detect_anomalies":
                result = await self._handle_anomaly_detection(input_data)
            elif request_type == "analyze_ab_test":
                result = await self._handle_ab_test(input_data)
            else:
                # Default: comprehensive performance report
                metrics = self._calculate_metrics(
                    time_range=input_data.get("time_range", "24h")
                )
                report = self._generate_report(metrics)
                result = {
                    "request_type": request_type,
                    "metrics": metrics,
                    "report": report,
                }

            self.status = AgentStatus.IDLE
            return {
                "success": True,
                "analytics": result,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Analytics processing failed: {e}")
            self.status = AgentStatus.ERROR
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _handle_calculate_metrics(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle metrics calculation request."""
        time_range = input_data.get("time_range", "24h")
        metric_types = input_data.get("metric_types")

        metrics = self._calculate_metrics(time_range, metric_types)
        return {"metrics": metrics}

    async def _handle_generate_report(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle report generation request."""
        metrics = input_data.get("metrics")
        if not metrics:
            time_range = input_data.get("time_range", "24h")
            metrics = self._calculate_metrics(time_range)

        report_format = input_data.get("format", "markdown")
        report = self._generate_report(metrics, report_format)
        return {"report": report}

    async def _handle_forecast(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle forecasting request."""
        historical_data = input_data.get("historical_data")
        if not historical_data:
            # Fetch historical data
            time_range = input_data.get("time_range", "7d")
            historical_data = self._calculate_metrics(time_range)

        periods_ahead = input_data.get("periods_ahead", 7)
        forecast = self._forecast_performance(historical_data, periods_ahead)
        return {"forecast": forecast}

    async def _handle_anomaly_detection(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle anomaly detection request."""
        metrics_history = input_data.get("metrics_history", self.metrics_history)
        threshold = input_data.get("threshold", 2.0)

        anomalies = self._detect_anomalies(metrics_history, threshold)
        return {"anomalies": anomalies}

    async def _handle_ab_test(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle A/B test analysis request."""
        test_id = input_data.get("test_id", "unknown")
        variant_a = input_data.get("variant_a", {})
        variant_b = input_data.get("variant_b", {})

        analysis = self._analyze_ab_test(test_id, variant_a, variant_b)
        return {"ab_test_analysis": analysis}

    def _calculate_metrics(self, time_range: str, metric_types: list = None) -> dict:
        """
        Calculate comprehensive metrics from execution history.

        Args:
            time_range: Time range for analysis (e.g., '1h', '24h', '7d', '30d')
            metric_types: Optional list of metric types to calculate
                         ['campaign', 'agent', 'system'] (default: all)

        Returns:
            Dictionary containing:
            - campaign_metrics: Campaign performance metrics
            - agent_metrics: Agent performance metrics
            - system_metrics: System performance metrics
            - time_range: Time range analyzed
        """
        # Parse time range
        duration = self._parse_time_range(time_range)
        cutoff_time = datetime.utcnow() - duration

        # Query execution history from memory manager
        execution_data = []
        if self.memory_manager:
            try:
                # Retrieve execution history from memory
                stored_history = self.memory_manager.retrieve(
                    key="execution_history", namespace="system"
                )
                if stored_history:
                    execution_data = stored_history
            except Exception as e:
                self.logger.warning(f"Could not retrieve execution history: {e}")

        # Fallback to agent's local execution history
        if not execution_data:
            execution_data = self.execution_history

        # Filter by time range
        filtered_data = [
            record
            for record in execution_data
            if self._is_within_time_range(record, cutoff_time)
        ]

        # Determine which metrics to calculate
        if metric_types is None:
            metric_types = ["campaign", "agent", "system"]

        metrics = {
            "time_range": time_range,
            "data_points": len(filtered_data),
            "period_start": cutoff_time.isoformat(),
            "period_end": datetime.utcnow().isoformat(),
        }

        # Calculate requested metrics
        if "campaign" in metric_types:
            metrics["campaign_metrics"] = calculate_campaign_metrics(filtered_data)

        if "agent" in metric_types:
            metrics["agent_metrics"] = calculate_agent_metrics(filtered_data)

        if "system" in metric_types:
            metrics["system_metrics"] = calculate_system_metrics(filtered_data)

        # Store in history for trend analysis
        self.metrics_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics,
            }
        )

        # Keep only recent history (last 100 records)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

        return metrics

    def _generate_report(self, metrics: dict, format: str = "markdown") -> dict:
        """
        Generate structured analytics report.

        Args:
            metrics: Metrics dictionary from _calculate_metrics
            format: Report format ('markdown', 'json', 'html')

        Returns:
            Dictionary containing:
            - report_content: Formatted report string
            - visualizations: List of chart configurations
            - insights: List of key insights
            - format: Report format
        """
        campaign_metrics = metrics.get("campaign_metrics", {})
        agent_metrics = metrics.get("agent_metrics", {})
        system_metrics = metrics.get("system_metrics", {})

        # Generate insights
        insights = self._generate_insights(metrics)

        # Generate visualizations
        visualizations = []
        visualizations.extend(generate_campaign_dashboard(metrics))
        visualizations.extend(generate_agent_dashboard(metrics))
        visualizations.extend(generate_system_dashboard(metrics))

        # Generate report content based on format
        if format == "markdown":
            report_content = self._generate_markdown_report(metrics, insights)
        elif format == "json":
            report_content = {
                "metrics": metrics,
                "insights": insights,
            }
        elif format == "html":
            report_content = self._generate_html_report(metrics, insights)
        else:
            report_content = str(metrics)

        # Get comparison with previous period if available
        comparison = None
        if len(self.metrics_history) >= 2:
            try:
                previous_metrics = self.metrics_history[-2]["metrics"]
                comparison = self._compare_periods(metrics, previous_metrics)
            except Exception as e:
                self.logger.warning(f"Could not generate comparison: {e}")

        return {
            "report_content": report_content,
            "visualizations": visualizations,
            "insights": insights,
            "format": format,
            "comparison": comparison,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _forecast_performance(
        self, historical_data: dict, periods_ahead: int = 7
    ) -> dict:
        """
        Forecast future performance using time series analysis.

        Args:
            historical_data: Historical metrics data
            periods_ahead: Number of periods to forecast

        Returns:
            Dictionary containing:
            - forecasts: Predicted values for each metric
            - confidence_intervals: Upper and lower bounds
            - method: Forecasting method used
        """
        forecasts = {}
        confidence_intervals = {}

        # Extract time series data from metrics history
        if not self.metrics_history or len(self.metrics_history) < 3:
            # Not enough data for forecasting
            return {
                "forecasts": {},
                "confidence_intervals": {},
                "method": "insufficient_data",
                "message": "Need at least 3 historical data points for forecasting",
            }

        # Key metrics to forecast
        metrics_to_forecast = [
            ("campaign_metrics", "ctr"),
            ("campaign_metrics", "conversion_rate"),
            ("campaign_metrics", "roi"),
            ("agent_metrics", "success_rate"),
            ("agent_metrics", "avg_response_time"),
            ("system_metrics", "throughput"),
        ]

        for category, metric_name in metrics_to_forecast:
            try:
                # Extract historical values
                values = []
                for record in self.metrics_history[-30:]:  # Last 30 periods
                    metric_category = record.get("metrics", {}).get(category, {})
                    value = metric_category.get(metric_name)
                    if value is not None:
                        values.append(float(value))

                if len(values) < 3:
                    continue

                # Use simple moving average for forecasting
                window_size = min(7, len(values))
                forecast_value = statistics.mean(values[-window_size:])

                # Calculate standard deviation for confidence intervals
                std_dev = statistics.stdev(values) if len(values) > 1 else 0

                key = f"{category}.{metric_name}"
                forecasts[key] = round(forecast_value, 2)
                confidence_intervals[key] = {
                    "lower": round(forecast_value - 1.96 * std_dev, 2),
                    "upper": round(forecast_value + 1.96 * std_dev, 2),
                }

            except Exception as e:
                self.logger.warning(f"Could not forecast {category}.{metric_name}: {e}")

        return {
            "forecasts": forecasts,
            "confidence_intervals": confidence_intervals,
            "method": "moving_average",
            "periods_ahead": periods_ahead,
            "forecast_date": (
                datetime.utcnow() + timedelta(days=periods_ahead)
            ).isoformat(),
        }

    def _detect_anomalies(self, metrics_history: list, threshold: float = 2.0) -> dict:
        """
        Detect anomalies in metrics using statistical methods.

        Args:
            metrics_history: List of historical metrics
            threshold: Z-score threshold for anomaly detection (default: 2.0)

        Returns:
            Dictionary containing:
            - anomalies: List of detected anomalies
            - severity_counts: Count by severity level
            - alerts: List of actionable alerts
        """
        if not metrics_history or len(metrics_history) < 10:
            return {
                "anomalies": [],
                "severity_counts": {"warning": 0, "critical": 0},
                "alerts": [],
                "message": "Insufficient data for anomaly detection (need at least 10 data points)",
            }

        anomalies = []
        alerts = []

        # Key metrics to monitor
        metrics_to_monitor = [
            ("campaign_metrics", "ctr", "lower_is_bad"),
            ("campaign_metrics", "conversion_rate", "lower_is_bad"),
            ("campaign_metrics", "roi", "lower_is_bad"),
            ("agent_metrics", "success_rate", "lower_is_bad"),
            ("agent_metrics", "error_rate", "higher_is_bad"),
            ("system_metrics", "error_rate", "higher_is_bad"),
            ("system_metrics", "latency_p99", "higher_is_bad"),
        ]

        for category, metric_name, direction in metrics_to_monitor:
            try:
                # Extract values
                values = []
                timestamps = []
                for record in metrics_history:
                    metrics = record.get("metrics", {})
                    metric_category = metrics.get(category, {})
                    value = metric_category.get(metric_name)
                    if value is not None:
                        values.append(float(value))
                        timestamps.append(record.get("timestamp", ""))

                if len(values) < 10:
                    continue

                # Calculate z-scores
                mean_value = statistics.mean(values)
                std_dev = statistics.stdev(values) if len(values) > 1 else 0

                if std_dev == 0:
                    continue

                # Check last value for anomaly
                last_value = values[-1]
                z_score = (last_value - mean_value) / std_dev

                # Determine if anomalous based on direction
                is_anomaly = False
                severity = None

                if direction == "lower_is_bad" and z_score < -threshold:
                    is_anomaly = True
                    severity = "critical" if z_score < -3.0 else "warning"
                elif direction == "higher_is_bad" and z_score > threshold:
                    is_anomaly = True
                    severity = "critical" if z_score > 3.0 else "warning"

                if is_anomaly:
                    anomaly_record = {
                        "metric": f"{category}.{metric_name}",
                        "value": last_value,
                        "z_score": round(z_score, 2),
                        "mean": round(mean_value, 2),
                        "std_dev": round(std_dev, 2),
                        "severity": severity,
                        "timestamp": timestamps[-1] if timestamps else "",
                        "direction": direction,
                    }
                    anomalies.append(anomaly_record)

                    # Generate alert
                    alert_msg = (
                        f"{severity.upper()}: {metric_name} is {'below' if direction == 'lower_is_bad' else 'above'} "
                        f"normal range. Current: {last_value:.2f}, Expected: {mean_value:.2f} ± {std_dev:.2f}"
                    )
                    alerts.append(
                        {
                            "severity": severity,
                            "message": alert_msg,
                            "metric": f"{category}.{metric_name}",
                        }
                    )

            except Exception as e:
                self.logger.warning(
                    f"Could not detect anomalies for {category}.{metric_name}: {e}"
                )

        # Count by severity
        severity_counts = {
            "warning": sum(1 for a in anomalies if a["severity"] == "warning"),
            "critical": sum(1 for a in anomalies if a["severity"] == "critical"),
        }

        return {
            "anomalies": anomalies,
            "severity_counts": severity_counts,
            "alerts": alerts,
            "threshold": threshold,
            "analyzed_at": datetime.utcnow().isoformat(),
        }

    def _analyze_ab_test(self, test_id: str, variant_a: dict, variant_b: dict) -> dict:
        """
        Analyze A/B test results for statistical significance.

        Args:
            test_id: Unique test identifier
            variant_a: Metrics for variant A (control)
            variant_b: Metrics for variant B (treatment)

        Returns:
            Dictionary containing:
            - winner: 'A', 'B', or 'no_significant_difference'
            - confidence: Confidence level (0-1)
            - p_value: Statistical p-value
            - recommendation: Actionable recommendation
        """
        try:
            # Extract conversion data
            conversions_a = variant_a.get("conversions", 0)
            trials_a = variant_a.get("trials", variant_a.get("impressions", 0))

            conversions_b = variant_b.get("conversions", 0)
            trials_b = variant_b.get("trials", variant_b.get("impressions", 0))

            if trials_a == 0 or trials_b == 0:
                return {
                    "test_id": test_id,
                    "winner": "insufficient_data",
                    "confidence": 0.0,
                    "p_value": 1.0,
                    "recommendation": "Collect more data before making a decision",
                    "error": "Insufficient sample size",
                }

            # Calculate conversion rates
            rate_a = conversions_a / trials_a
            rate_b = conversions_b / trials_b

            # Perform chi-square test
            observed = np.array(
                [
                    [conversions_a, trials_a - conversions_a],
                    [conversions_b, trials_b - conversions_b],
                ]
            )

            chi2, p_value, dof, expected = stats.chi2_contingency(observed)

            # Determine winner
            alpha = 0.05  # 95% confidence level
            confidence = 1 - p_value

            if p_value < alpha:
                winner = "B" if rate_b > rate_a else "A"
                is_significant = True
            else:
                winner = "no_significant_difference"
                is_significant = False

            # Calculate lift
            lift = ((rate_b - rate_a) / rate_a * 100) if rate_a > 0 else 0

            # Generate recommendation
            if is_significant and winner == "B":
                recommendation = (
                    f"Implement Variant B. It shows a {abs(lift):.2f}% "
                    f"{'increase' if lift > 0 else 'decrease'} in conversion rate with "
                    f"{confidence*100:.1f}% confidence."
                )
            elif is_significant and winner == "A":
                recommendation = (
                    f"Keep Variant A (control). Variant B shows a {abs(lift):.2f}% "
                    f"decrease in conversion rate with {confidence*100:.1f}% confidence."
                )
            else:
                recommendation = (
                    "Continue testing. No statistically significant difference detected. "
                    f"Current p-value: {p_value:.4f}. Need p-value < {alpha}."
                )

            return {
                "test_id": test_id,
                "winner": winner,
                "confidence": round(confidence, 4),
                "p_value": round(p_value, 4),
                "is_significant": is_significant,
                "recommendation": recommendation,
                "metrics": {
                    "variant_a": {
                        "conversion_rate": round(rate_a * 100, 2),
                        "conversions": conversions_a,
                        "trials": trials_a,
                    },
                    "variant_b": {
                        "conversion_rate": round(rate_b * 100, 2),
                        "conversions": conversions_b,
                        "trials": trials_b,
                    },
                    "lift": round(lift, 2),
                },
                "statistical_test": "chi_square",
                "alpha": alpha,
            }

        except Exception as e:
            self.logger.error(f"A/B test analysis failed: {e}")
            return {
                "test_id": test_id,
                "winner": "error",
                "confidence": 0.0,
                "p_value": 1.0,
                "recommendation": "Error occurred during analysis",
                "error": str(e),
            }

    # Helper methods

    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta."""
        unit = time_range[-1]
        value = int(time_range[:-1])

        if unit == "h":
            return timedelta(hours=value)
        elif unit == "d":
            return timedelta(days=value)
        elif unit == "w":
            return timedelta(weeks=value)
        else:
            return timedelta(hours=24)  # Default to 24 hours

    def _is_within_time_range(self, record: dict, cutoff_time: datetime) -> bool:
        """Check if record is within time range."""
        timestamp = record.get("started_at")
        if not timestamp:
            return False

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        return timestamp >= cutoff_time

    def _generate_insights(self, metrics: dict) -> List[str]:
        """Generate actionable insights from metrics."""
        insights = []

        campaign = metrics.get("campaign_metrics", {})
        agent = metrics.get("agent_metrics", {})
        system = metrics.get("system_metrics", {})

        # Campaign insights
        if campaign.get("ctr", 0) < 2.0:
            insights.append(
                "CTR is below industry average (2%). Consider improving ad copy or targeting."
            )

        if campaign.get("conversion_rate", 0) < 2.0:
            insights.append("Conversion rate is low. Optimize landing page and CTA.")

        if campaign.get("roi", 0) > 200:
            insights.append("Excellent ROI! Consider scaling this campaign.")

        # Agent insights
        if agent.get("error_rate", 0) > 5:
            insights.append(
                "Agent error rate is elevated. Review recent changes and logs."
            )

        if agent.get("success_rate", 0) > 95:
            insights.append("Agents are performing excellently with high success rate.")

        # System insights
        if system.get("latency_p99", 0) > 5.0:
            insights.append("P99 latency is high. Investigate performance bottlenecks.")

        return insights

    def _generate_markdown_report(self, metrics: dict, insights: List[str]) -> str:
        """Generate markdown-formatted report."""
        campaign = metrics.get("campaign_metrics", {})
        agent = metrics.get("agent_metrics", {})
        system = metrics.get("system_metrics", {})

        report = f"""# Analytics Report
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
Time Range: {metrics.get('time_range', 'N/A')}

## Executive Summary

### Campaign Performance
- **CTR**: {campaign.get('ctr', 0):.2f}%
- **Conversion Rate**: {campaign.get('conversion_rate', 0):.2f}%
- **ROI**: {campaign.get('roi', 0):.2f}%
- **Total Impressions**: {campaign.get('total_impressions', 0):,}
- **Total Conversions**: {campaign.get('total_conversions', 0):,}

### Agent Performance
- **Success Rate**: {agent.get('success_rate', 0):.2f}%
- **Avg Response Time**: {agent.get('avg_response_time', 0):.2f}s
- **Total Executions**: {agent.get('total_executions', 0):,}

### System Performance
- **Throughput**: {system.get('throughput', 0):.2f} executions/hour
- **Error Rate**: {system.get('error_rate', 0):.2f}%
- **P99 Latency**: {system.get('latency_p99', 0):.3f}s

## Key Insights

"""
        for i, insight in enumerate(insights, 1):
            report += f"{i}. {insight}\n"

        report += "\n## Recommendations\n\n"
        report += "- Monitor trends and adjust strategies accordingly\n"
        report += "- Focus on metrics showing declining performance\n"
        report += "- Scale successful campaigns and optimize underperforming ones\n"

        return report

    def _generate_html_report(self, metrics: dict, insights: List[str]) -> str:
        """Generate HTML-formatted report."""
        # Simple HTML wrapper around markdown
        markdown_content = self._generate_markdown_report(metrics, insights)
        html = f"""
        <html>
        <head><title>Analytics Report</title></head>
        <body>
        <pre>{markdown_content}</pre>
        </body>
        </html>
        """
        return html

    def _compare_periods(self, current: dict, previous: dict) -> dict:
        """Compare current period with previous period."""
        comparison = {}

        # Helper function
        def calc_change(curr_val, prev_val):
            if prev_val == 0:
                return 100.0 if curr_val > 0 else 0.0
            return (curr_val - prev_val) / prev_val * 100

        # Campaign comparisons
        curr_campaign = current.get("campaign_metrics", {})
        prev_campaign = previous.get("campaign_metrics", {})

        comparison["campaign"] = {
            "ctr_change": calc_change(
                curr_campaign.get("ctr", 0), prev_campaign.get("ctr", 0)
            ),
            "conversion_rate_change": calc_change(
                curr_campaign.get("conversion_rate", 0),
                prev_campaign.get("conversion_rate", 0),
            ),
            "roi_change": calc_change(
                curr_campaign.get("roi", 0), prev_campaign.get("roi", 0)
            ),
        }

        return comparison
