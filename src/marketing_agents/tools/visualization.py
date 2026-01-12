"""
Visualization - Chart data generation for analytics dashboards.

Generates JSON-formatted chart configurations for Streamlit and other UI frameworks.
All visualizations are returned as data structures ready for rendering.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


def create_line_chart(
    data: List[Dict[str, Any]],
    x_key: str,
    y_keys: List[str],
    title: str,
    x_label: str = "",
    y_label: str = "",
) -> Dict[str, Any]:
    """
    Create a line chart configuration.

    Args:
        data: List of data points with x and y values
        x_key: Key for x-axis values
        y_keys: List of keys for y-axis values (supports multiple lines)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label

    Returns:
        Dictionary with chart configuration
    """
    return {
        "type": "line",
        "title": title,
        "data": data,
        "config": {
            "x_key": x_key,
            "y_keys": y_keys,
            "x_label": x_label or x_key,
            "y_label": y_label or ", ".join(y_keys),
        },
    }


def create_bar_chart(
    data: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    x_label: str = "",
    y_label: str = "",
    orientation: str = "vertical",
) -> Dict[str, Any]:
    """
    Create a bar chart configuration.

    Args:
        data: List of data points
        x_key: Key for x-axis (categories)
        y_key: Key for y-axis (values)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        orientation: 'vertical' or 'horizontal'

    Returns:
        Dictionary with chart configuration
    """
    return {
        "type": "bar",
        "title": title,
        "data": data,
        "config": {
            "x_key": x_key,
            "y_key": y_key,
            "x_label": x_label or x_key,
            "y_label": y_label or y_key,
            "orientation": orientation,
        },
    }


def create_pie_chart(
    data: List[Dict[str, Any]],
    label_key: str,
    value_key: str,
    title: str,
) -> Dict[str, Any]:
    """
    Create a pie chart configuration.

    Args:
        data: List of data points
        label_key: Key for labels
        value_key: Key for values
        title: Chart title

    Returns:
        Dictionary with chart configuration
    """
    return {
        "type": "pie",
        "title": title,
        "data": data,
        "config": {
            "label_key": label_key,
            "value_key": value_key,
        },
    }


def create_metric_card(
    value: float,
    label: str,
    delta: Optional[float] = None,
    delta_label: str = "vs previous period",
    format_string: str = "{:.2f}",
) -> Dict[str, Any]:
    """
    Create a metric card configuration for KPI display.

    Args:
        value: Primary metric value
        label: Metric label
        delta: Change value (can be positive or negative)
        delta_label: Label for the delta
        format_string: Format string for the value

    Returns:
        Dictionary with metric card configuration
    """
    return {
        "type": "metric_card",
        "value": value,
        "label": label,
        "formatted_value": format_string.format(value),
        "delta": delta,
        "delta_label": delta_label if delta is not None else None,
        "trend": (
            "up"
            if delta and delta > 0
            else "down" if delta and delta < 0 else "neutral"
        ),
    }


def create_table(
    data: List[Dict[str, Any]],
    columns: List[str],
    title: str = "",
    sortable: bool = True,
) -> Dict[str, Any]:
    """
    Create a table configuration.

    Args:
        data: List of row dictionaries
        columns: List of column keys to display
        title: Table title
        sortable: Whether columns should be sortable

    Returns:
        Dictionary with table configuration
    """
    return {
        "type": "table",
        "title": title,
        "data": data,
        "config": {
            "columns": columns,
            "sortable": sortable,
        },
    }


def create_heatmap(
    data: List[List[float]],
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    color_scale: str = "viridis",
) -> Dict[str, Any]:
    """
    Create a heatmap configuration.

    Args:
        data: 2D array of values
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        title: Chart title
        color_scale: Color scale name

    Returns:
        Dictionary with heatmap configuration
    """
    return {
        "type": "heatmap",
        "title": title,
        "data": data,
        "config": {
            "x_labels": x_labels,
            "y_labels": y_labels,
            "color_scale": color_scale,
        },
    }


def create_scatter_plot(
    data: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    size_key: Optional[str] = None,
    color_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a scatter plot configuration.

    Args:
        data: List of data points
        x_key: Key for x-axis values
        y_key: Key for y-axis values
        title: Chart title
        size_key: Optional key for point sizes
        color_key: Optional key for point colors

    Returns:
        Dictionary with scatter plot configuration
    """
    return {
        "type": "scatter",
        "title": title,
        "data": data,
        "config": {
            "x_key": x_key,
            "y_key": y_key,
            "size_key": size_key,
            "color_key": color_key,
        },
    }


def create_funnel_chart(
    stages: List[Dict[str, Any]],
    title: str,
) -> Dict[str, Any]:
    """
    Create a funnel chart configuration for conversion tracking.

    Args:
        stages: List of stages with 'name' and 'value' keys
        title: Chart title

    Returns:
        Dictionary with funnel chart configuration

    Example:
        >>> stages = [
        ...     {"name": "Impressions", "value": 10000},
        ...     {"name": "Clicks", "value": 500},
        ...     {"name": "Conversions", "value": 50}
        ... ]
    """
    # Calculate conversion rates between stages
    enriched_stages = []
    for i, stage in enumerate(stages):
        stage_data = stage.copy()
        if i > 0:
            previous_value = stages[i - 1]["value"]
            conversion_rate = (
                (stage["value"] / previous_value * 100) if previous_value > 0 else 0
            )
            stage_data["conversion_rate"] = round(conversion_rate, 2)
        enriched_stages.append(stage_data)

    return {
        "type": "funnel",
        "title": title,
        "data": enriched_stages,
        "config": {
            "stages": [s["name"] for s in stages],
        },
    }


def create_gauge_chart(
    value: float,
    min_value: float,
    max_value: float,
    title: str,
    thresholds: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Create a gauge chart configuration for KPI visualization.

    Args:
        value: Current value
        min_value: Minimum value on gauge
        max_value: Maximum value on gauge
        title: Chart title
        thresholds: List of threshold dicts with 'value' and 'color' keys

    Returns:
        Dictionary with gauge chart configuration
    """
    if thresholds is None:
        # Default thresholds (red, yellow, green)
        thresholds = [
            {"value": min_value + (max_value - min_value) * 0.33, "color": "red"},
            {"value": min_value + (max_value - min_value) * 0.66, "color": "yellow"},
            {"value": max_value, "color": "green"},
        ]

    return {
        "type": "gauge",
        "title": title,
        "value": value,
        "config": {
            "min_value": min_value,
            "max_value": max_value,
            "thresholds": thresholds,
        },
    }


def generate_campaign_dashboard(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate a complete dashboard layout for campaign metrics.

    Args:
        metrics: Dictionary containing campaign metrics

    Returns:
        List of visualization configurations
    """
    visualizations = []

    campaign_metrics = metrics.get("campaign_metrics", {})

    # Metric cards for key KPIs
    visualizations.append(
        create_metric_card(
            value=campaign_metrics.get("ctr", 0),
            label="Click-Through Rate",
            delta=metrics.get("comparisons", {}).get("ctr_change"),
            format_string="{:.2f}%",
        )
    )

    visualizations.append(
        create_metric_card(
            value=campaign_metrics.get("conversion_rate", 0),
            label="Conversion Rate",
            delta=metrics.get("comparisons", {}).get("conversion_rate_change"),
            format_string="{:.2f}%",
        )
    )

    visualizations.append(
        create_metric_card(
            value=campaign_metrics.get("roi", 0),
            label="ROI",
            delta=metrics.get("comparisons", {}).get("roi_change"),
            format_string="{:.2f}%",
        )
    )

    # Funnel chart for conversion path
    if "funnel_data" in metrics:
        visualizations.append(
            create_funnel_chart(
                stages=metrics["funnel_data"],
                title="Conversion Funnel",
            )
        )

    # Time series for trends
    if "time_series" in metrics:
        visualizations.append(
            create_line_chart(
                data=metrics["time_series"],
                x_key="timestamp",
                y_keys=["clicks", "conversions"],
                title="Performance Over Time",
                x_label="Date",
                y_label="Count",
            )
        )

    return visualizations


def generate_agent_dashboard(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate a complete dashboard layout for agent metrics.

    Args:
        metrics: Dictionary containing agent metrics

    Returns:
        List of visualization configurations
    """
    visualizations = []

    agent_metrics = metrics.get("agent_metrics", {})

    # Metric cards
    visualizations.append(
        create_metric_card(
            value=agent_metrics.get("success_rate", 0),
            label="Success Rate",
            format_string="{:.2f}%",
        )
    )

    visualizations.append(
        create_metric_card(
            value=agent_metrics.get("avg_response_time", 0),
            label="Avg Response Time",
            format_string="{:.2f}s",
        )
    )

    # Agent breakdown bar chart
    agent_breakdown = agent_metrics.get("agent_breakdown", {})
    if agent_breakdown:
        breakdown_data = [
            {"agent": agent_id, "executions": stats["executions"]}
            for agent_id, stats in agent_breakdown.items()
        ]
        visualizations.append(
            create_bar_chart(
                data=breakdown_data,
                x_key="agent",
                y_key="executions",
                title="Executions by Agent",
                x_label="Agent",
                y_label="Number of Executions",
            )
        )

    return visualizations


def generate_system_dashboard(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate a complete dashboard layout for system metrics.

    Args:
        metrics: Dictionary containing system metrics

    Returns:
        List of visualization configurations
    """
    visualizations = []

    system_metrics = metrics.get("system_metrics", {})

    # Metric cards
    visualizations.append(
        create_metric_card(
            value=system_metrics.get("throughput", 0),
            label="Throughput (per hour)",
            format_string="{:.2f}",
        )
    )

    visualizations.append(
        create_metric_card(
            value=system_metrics.get("error_rate", 0),
            label="Error Rate",
            format_string="{:.2f}%",
        )
    )

    # Latency distribution
    latency_data = [
        {"percentile": "P50", "latency": system_metrics.get("latency_p50", 0)},
        {"percentile": "P95", "latency": system_metrics.get("latency_p95", 0)},
        {"percentile": "P99", "latency": system_metrics.get("latency_p99", 0)},
    ]
    visualizations.append(
        create_bar_chart(
            data=latency_data,
            x_key="percentile",
            y_key="latency",
            title="Latency Distribution",
            x_label="Percentile",
            y_label="Latency (seconds)",
        )
    )

    return visualizations
