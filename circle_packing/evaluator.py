"""
Evaluator for circle packing (n=26 circles in a unit square).

Adapted from OpenEvolve's circle_packing example.
Target: sum_radii >= 2.6360 (beyond FICO Xpress world record of 2.63593).

Scoring:
  - combined_score = (sum_radii / 2.6360) * validity
  - validity: 1.0 if all constraints met, else 0.0
  - Constraints: no overlaps, all circles inside [0,1]x[0,1]
"""

import importlib.util
import json
import sys
import time
import traceback

import numpy as np


TARGET_VALUE = 2.6360  # Beyond FICO Xpress world record (2.63593) for n=26


def load_candidate(candidate_path):
    """Load candidate module."""
    spec = importlib.util.spec_from_file_location("candidate", candidate_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_packing(centers, radii):
    """Validate that circles don't overlap and are inside the unit square."""
    n = centers.shape[0]

    # Check for NaN
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN values detected"

    # Check non-negative radii
    if np.any(radii < 0):
        return False, f"Negative radius found"

    # Check circles inside unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            return False, f"Circle {i} at ({x:.4f},{y:.4f}) r={r:.4f} outside square"

    # Check no overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:
                return False, f"Circles {i},{j} overlap: dist={dist:.6f} < r1+r2={radii[i]+radii[j]:.6f}"

    return True, "valid"


def evaluate(candidate_path):
    """Evaluate a circle packing candidate."""
    start_time = time.time()

    try:
        module = load_candidate(candidate_path)
    except Exception as e:
        return {
            "combined_score": 0.0, "sum_radii": 0.0,
            "target_ratio": 0.0, "validity": 0.0,
            "error": f"Import failed: {e}",
        }

    try:
        if hasattr(module, "run_packing"):
            centers, radii, reported_sum = module.run_packing()
        elif hasattr(module, "construct_packing"):
            centers, radii, reported_sum = module.construct_packing()
        else:
            return {
                "combined_score": 0.0, "sum_radii": 0.0,
                "target_ratio": 0.0, "validity": 0.0,
                "error": "No run_packing() or construct_packing() found",
            }
    except Exception as e:
        return {
            "combined_score": 0.0, "sum_radii": 0.0,
            "target_ratio": 0.0, "validity": 0.0,
            "error": f"Execution failed: {e}",
        }

    # Convert to numpy arrays
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    # Validate shapes
    if centers.shape != (26, 2) or radii.shape != (26,):
        return {
            "combined_score": 0.0, "sum_radii": 0.0,
            "target_ratio": 0.0, "validity": 0.0,
            "error": f"Bad shapes: centers={centers.shape}, radii={radii.shape}, expected (26,2) and (26,)",
        }

    # Validate packing constraints
    valid, msg = validate_packing(centers, radii)
    eval_time = time.time() - start_time

    if not valid:
        return {
            "combined_score": 0.0, "sum_radii": 0.0,
            "target_ratio": 0.0, "validity": 0.0,
            "eval_time": round(eval_time, 3),
            "error": msg,
        }

    sum_radii = float(np.sum(radii))
    target_ratio = sum_radii / TARGET_VALUE
    combined_score = min(target_ratio, 1.0)  # Cap at 1.0

    return {
        "combined_score": round(combined_score, 12),
        "sum_radii": round(sum_radii, 12),
        "target_ratio": round(target_ratio, 12),
        "validity": 1.0,
        "eval_time": round(eval_time, 3),
        "n_circles": int(centers.shape[0]),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <candidate.py>")
        sys.exit(1)
    result = evaluate(sys.argv[1])
    print(json.dumps(result, indent=2))
