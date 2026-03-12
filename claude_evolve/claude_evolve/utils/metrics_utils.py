"""
Safe calculation utilities for metrics containing mixed types.

Extracted and adapted from OpenEvolve's metrics_utils module for the
claude_evolve artifact-evolution pipeline.
"""

from typing import Any, Dict, List, Optional


def safe_numeric_average(metrics: Dict[str, Any]) -> float:
    """
    Calculate the average of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    NaN values are excluded from the calculation.

    Args:
        metrics: Dictionary of metric names to values.

    Returns:
        Average of numeric values, or ``0.0`` if no valid numeric values found.
    """
    if not metrics:
        return 0.0

    numeric_values: list[float] = []
    for value in metrics.values():
        if isinstance(value, (int, float)):
            try:
                float_val = float(value)
                if not (float_val != float_val):  # NaN check (NaN != NaN is True)
                    numeric_values.append(float_val)
            except (ValueError, TypeError, OverflowError):
                continue

    if not numeric_values:
        return 0.0

    return sum(numeric_values) / len(numeric_values)


def safe_numeric_sum(metrics: Dict[str, Any]) -> float:
    """
    Calculate the sum of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    NaN values are excluded from the calculation.

    Args:
        metrics: Dictionary of metric names to values.

    Returns:
        Sum of numeric values, or ``0.0`` if no valid numeric values found.
    """
    if not metrics:
        return 0.0

    numeric_sum = 0.0
    for value in metrics.values():
        if isinstance(value, (int, float)):
            try:
                float_val = float(value)
                if not (float_val != float_val):  # NaN check
                    numeric_sum += float_val
            except (ValueError, TypeError, OverflowError):
                continue

    return numeric_sum


def get_fitness_score(
    metrics: Dict[str, Any], feature_dimensions: Optional[List[str]] = None
) -> float:
    """
    Calculate fitness score, excluding MAP-Elites feature dimensions.

    This ensures that MAP-Elites features don't pollute the fitness calculation
    when ``combined_score`` is not available.

    Priority:
      1. ``combined_score`` key if present and convertible to float.
      2. Average of non-feature numeric metrics.
      3. Fall back to ``safe_numeric_average`` of all metrics if filtering
         removes everything.

    Args:
        metrics: All metrics from evaluation.
        feature_dimensions: List of MAP-Elites dimensions to exclude from
            fitness.

    Returns:
        Fitness score.
    """
    if not metrics:
        return 0.0

    # Always prefer combined_score if available
    if "combined_score" in metrics:
        try:
            return float(metrics["combined_score"])
        except (ValueError, TypeError):
            pass

    # Otherwise, average only non-feature metrics
    feature_dimensions = feature_dimensions or []
    fitness_metrics: Dict[str, float] = {}

    for key, value in metrics.items():
        if key not in feature_dimensions:
            if isinstance(value, (int, float)):
                try:
                    float_val = float(value)
                    if not (float_val != float_val):  # NaN check
                        fitness_metrics[key] = float_val
                except (ValueError, TypeError, OverflowError):
                    continue

    # If no non-feature metrics, fall back to all metrics (backward compatibility)
    if not fitness_metrics:
        return safe_numeric_average(metrics)

    return safe_numeric_average(fitness_metrics)


def format_feature_coordinates(
    metrics: Dict[str, Any], feature_dimensions: List[str]
) -> str:
    """
    Format feature coordinates for display in prompts.

    Each feature dimension that exists in *metrics* is formatted as
    ``dim=value``. Numeric values are shown with two decimal places; NaN
    values and non-numeric values are rendered with their default ``str``
    representation.

    Args:
        metrics: All metrics from evaluation.
        feature_dimensions: List of MAP-Elites feature dimensions.

    Returns:
        Comma-separated string of feature coordinates, or ``""`` if none.
    """
    feature_values: list[str] = []
    for dim in feature_dimensions:
        if dim in metrics:
            value = metrics[dim]
            if isinstance(value, (int, float)):
                try:
                    float_val = float(value)
                    if not (float_val != float_val):  # NaN check
                        feature_values.append(f"{dim}={float_val:.2f}")
                    else:
                        feature_values.append(f"{dim}={value}")
                except (ValueError, TypeError, OverflowError):
                    feature_values.append(f"{dim}={value}")
            else:
                feature_values.append(f"{dim}={value}")

    if not feature_values:
        return ""

    return ", ".join(feature_values)
