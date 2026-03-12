"""
Evaluator for circle packing optimization.

Scores a candidate program based on how well it packs circles into a unit square.
The program must define a `pack_circles(n_circles=5)` function that returns
a list of (x, y, r) tuples.

Metrics:
  - total_radius: Sum of all radii (primary optimization target)
  - validity: Fraction of circles that are valid (inside bounds, no overlaps)
  - coverage: Area covered by circles / total area
  - combined_score: Weighted combination
"""

import importlib.util
import math
import sys
import json
import os
import tempfile


def load_candidate(candidate_path):
    """Load and execute the candidate program."""
    spec = importlib.util.spec_from_file_location("candidate", candidate_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_circle_validity(circles):
    """Check each circle for bounds and overlap violations."""
    n = len(circles)
    valid = [True] * n

    for i in range(n):
        x, y, r = circles[i]

        # Check bounds: circle must be fully inside [0,1]x[0,1]
        if r <= 0:
            valid[i] = False
            continue
        if x - r < -1e-9 or x + r > 1.0 + 1e-9:
            valid[i] = False
            continue
        if y - r < -1e-9 or y + r > 1.0 + 1e-9:
            valid[i] = False
            continue

        # Check overlaps with other circles
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((x - xj) ** 2 + (y - yj) ** 2)
            if dist < r + rj - 1e-9:  # Overlapping
                valid[i] = False
                valid[j] = False

    return valid


def evaluate(candidate_path):
    """Evaluate a circle packing candidate.

    Args:
        candidate_path: Path to the candidate Python file.

    Returns:
        dict with metric scores (all 0.0-1.0 range).
    """
    try:
        module = load_candidate(candidate_path)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "total_radius": 0.0,
            "validity": 0.0,
            "coverage": 0.0,
            "error": str(e),
        }

    try:
        circles = module.pack_circles(n_circles=5)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "total_radius": 0.0,
            "validity": 0.0,
            "coverage": 0.0,
            "error": f"pack_circles() failed: {e}",
        }

    if not isinstance(circles, list) or len(circles) == 0:
        return {
            "combined_score": 0.0,
            "total_radius": 0.0,
            "validity": 0.0,
            "coverage": 0.0,
            "error": "pack_circles() must return a non-empty list of (x, y, r) tuples",
        }

    # Validate circles
    valid = check_circle_validity(circles)
    validity = sum(valid) / len(valid)

    # Calculate total radius (only for valid circles)
    valid_circles = [c for c, v in zip(circles, valid) if v]
    total_radius_raw = sum(r for _, _, r in valid_circles)

    # Normalize total_radius to 0-1 range
    # Theoretical maximum for 5 circles in unit square is approximately 1.0
    # (e.g., one circle with r~0.5 gives total~0.5, five optimal circles ~1.0)
    max_theoretical_radius = 1.0
    total_radius_score = min(total_radius_raw / max_theoretical_radius, 1.0)

    # Calculate area coverage
    total_area = sum(math.pi * r ** 2 for _, _, r in valid_circles)
    coverage = min(total_area / 1.0, 1.0)  # 1.0 = area of unit square

    # Combined score: weighted average
    combined_score = (
        0.50 * total_radius_score
        + 0.30 * validity
        + 0.20 * coverage
    )

    return {
        "combined_score": round(combined_score, 4),
        "total_radius": round(total_radius_score, 4),
        "validity": round(validity, 4),
        "coverage": round(coverage, 4),
        "n_circles": len(circles),
        "n_valid": sum(valid),
        "raw_total_radius": round(total_radius_raw, 4),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <candidate.py>")
        sys.exit(1)

    result = evaluate(sys.argv[1])
    print(json.dumps(result, indent=2))
