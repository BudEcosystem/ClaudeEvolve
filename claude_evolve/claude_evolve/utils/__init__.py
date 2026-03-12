"""Utility modules for claude_evolve."""

from claude_evolve.utils.code_utils import (
    apply_diff,
    apply_diff_blocks,
    calculate_edit_distance,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_evolve_blocks,
    parse_full_rewrite,
    split_diffs_by_target,
)
from claude_evolve.utils.metrics_utils import (
    format_feature_coordinates,
    get_fitness_score,
    safe_numeric_average,
    safe_numeric_sum,
)
from claude_evolve.utils.format_utils import (
    format_metrics_safe,
    format_improvement_safe,
)

__all__ = [
    "apply_diff",
    "apply_diff_blocks",
    "calculate_edit_distance",
    "extract_code_language",
    "extract_diffs",
    "format_diff_summary",
    "format_feature_coordinates",
    "format_improvement_safe",
    "format_metrics_safe",
    "get_fitness_score",
    "parse_evolve_blocks",
    "parse_full_rewrite",
    "safe_numeric_average",
    "safe_numeric_sum",
    "split_diffs_by_target",
]
