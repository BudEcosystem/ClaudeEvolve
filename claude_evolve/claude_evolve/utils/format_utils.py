"""
Utility functions for formatting output.

Extracted and adapted from OpenEvolve's format_utils module for the
claude_evolve artifact-evolution pipeline.
"""

from typing import Any, Dict


def format_metrics_safe(metrics: Dict[str, Any]) -> str:
    """
    Safely format metrics dictionary for logging, handling both numeric and
    string values.

    Numeric values are formatted to four decimal places; non-numeric values
    are converted with ``str()``.

    Args:
        metrics: Dictionary of metric names to values.

    Returns:
        Formatted string representation of metrics, or ``""`` if empty.
    """
    if not metrics:
        return ""

    formatted_parts: list[str] = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            try:
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={value}")
        else:
            formatted_parts.append(f"{name}={value}")

    return ", ".join(formatted_parts)


def format_improvement_safe(
    parent_metrics: Dict[str, Any], child_metrics: Dict[str, Any]
) -> str:
    """
    Safely format improvement metrics for logging.

    For each metric present in both parent and child, compute the difference
    and display with a leading sign (``+`` or ``-``).

    Args:
        parent_metrics: Parent artifact metrics.
        child_metrics: Child artifact metrics.

    Returns:
        Formatted string representation of improvements, or ``""`` if no
        comparable metrics.
    """
    if not parent_metrics or not child_metrics:
        return ""

    improvement_parts: list[str] = []
    for metric, child_value in child_metrics.items():
        if metric in parent_metrics:
            parent_value = parent_metrics[metric]
            if isinstance(child_value, (int, float)) and isinstance(
                parent_value, (int, float)
            ):
                try:
                    diff = child_value - parent_value
                    improvement_parts.append(f"{metric}={diff:+.4f}")
                except (ValueError, TypeError):
                    continue

    return ", ".join(improvement_parts)
