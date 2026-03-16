#!/usr/bin/env python3
"""
Evaluator for Ramsey number R(5,5) lower bound improvement.

OPEN PROBLEM: R(5,5) — the diagonal Ramsey number for K_5.
Current bounds: 43 ≤ R(5,5) ≤ 48

  R(5,5) ≥ 43 means there EXISTS a 2-edge-coloring of K_42 with no
  monochromatic K_5. This was established by Exoo (1989).

  R(5,5) ≤ 48 is the best known upper bound.

TARGET: Find a 2-edge-coloring of K_43 (or larger) with NO monochromatic K_5.
  - This would prove R(5,5) ≥ 44, improving a bound that has stood since 1989.
  - This would be a publishable mathematical result.

CANDIDATE INTERFACE:
  The candidate must define:
    generate_coloring(n) -> numpy array of shape (n, n), dtype convertible to bool
      where True/1 = "red" edge, False/0 = "blue" edge
      Must be symmetric with zeros on diagonal.

SCORING:
  - Primary: largest n with 0 monochromatic K_5 subgraphs (max_valid_n)
  - Secondary: partial credit at frontier (fewer mono-K_5s = better)
  - combined_score = (max_valid_n + frontier_credit) / TARGET_N
  - TARGET_N = 43 (achieving this with 0 mono-K_5s gives score 1.0)

VERIFICATION:
  For each n, exhaustively checks ALL C(n,5) subsets of 5 vertices for
  monochromatic K_5 in both colors. This is exact — no sampling or heuristics.
"""

import sys
import os
import json
import time
import importlib.util
import traceback
import numpy as np
from itertools import combinations
from math import comb


# ============================================================================
# CONSTANTS
# ============================================================================

TARGET_N = 43           # Improving R(5,5) lower bound requires a valid K_43 coloring
KNOWN_LOWER = 42        # R(5,5) ≥ 43 means K_42 is achievable
MAX_TEST_N = 50         # Don't test beyond this (diminishing returns + time)
EVAL_TIMEOUT = 180      # Per-n timeout in seconds (generous for large n)
TOTAL_TIMEOUT = 600     # Total evaluation timeout

# Test schedule: which n values to check
# Dense near the frontier (42-45), sparse elsewhere
TEST_SCHEDULE = [5, 10, 15, 20, 25, 30, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]


# ============================================================================
# CORE VERIFICATION: Count monochromatic K_5 subgraphs
# ============================================================================

def count_monochromatic_k5(adj, n):
    """
    Exhaustively count the number of monochromatic K_5 subgraphs in a
    2-edge-coloring of K_n.

    adj: (n, n) boolean numpy array. True = red edge, False = blue edge.
         Must be symmetric with False on diagonal.

    Returns: (red_k5_count, blue_k5_count, total_mono_k5, checked_count)
    """
    red_count = 0
    blue_count = 0
    checked = 0

    # For each 5-subset of vertices, check if all 10 edges are same color
    for combo in combinations(range(n), 5):
        i, j, k, l, m = combo
        checked += 1

        # Extract all 10 edges of K_5
        edges = [
            adj[i, j], adj[i, k], adj[i, l], adj[i, m],
            adj[j, k], adj[j, l], adj[j, m],
            adj[k, l], adj[k, m],
            adj[l, m],
        ]

        # Check monochromatic: all True (red K_5) or all False (blue K_5)
        if all(edges):
            red_count += 1
        elif not any(edges):
            blue_count += 1

    return red_count, blue_count, red_count + blue_count, checked


def count_monochromatic_k5_fast(adj, n):
    """
    Optimized monochromatic K_5 counter using numpy vectorization.
    Falls back to iterative method for small n or if vectorized fails.

    For n=43: C(43,5) = 962,598 five-subsets to check.
    With numpy vectorization, this should take < 10 seconds.
    """
    if n < 10:
        return count_monochromatic_k5(adj, n)

    try:
        red_count = 0
        blue_count = 0
        total_combos = comb(n, 5)

        # Process in batches to manage memory
        batch_size = 100000
        all_combos = combinations(range(n), 5)
        checked = 0

        batch = []
        for combo in all_combos:
            batch.append(combo)
            if len(batch) >= batch_size:
                rc, bc = _check_batch(adj, batch)
                red_count += rc
                blue_count += bc
                checked += len(batch)
                batch = []

        # Process remaining
        if batch:
            rc, bc = _check_batch(adj, batch)
            red_count += rc
            blue_count += bc
            checked += len(batch)

        return red_count, blue_count, red_count + blue_count, checked

    except Exception:
        # Fallback to simple iterative method
        return count_monochromatic_k5(adj, n)


def _check_batch(adj, combos):
    """Check a batch of 5-tuples for monochromatic K_5."""
    if not combos:
        return 0, 0

    combos_arr = np.array(combos, dtype=np.int32)  # shape: (batch, 5)

    # For each combo, extract all 10 edges
    # Edge pairs within a 5-clique: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
    pair_indices = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

    # Build index arrays for all edges across all combos
    edges = np.zeros((len(combos), 10), dtype=bool)
    for idx, (a, b) in enumerate(pair_indices):
        rows = combos_arr[:, a]
        cols = combos_arr[:, b]
        edges[:, idx] = adj[rows, cols]

    # Red K_5: all 10 edges are True
    red_mask = np.all(edges, axis=1)
    red_count = int(np.sum(red_mask))

    # Blue K_5: all 10 edges are False
    blue_mask = ~np.any(edges, axis=1)
    blue_count = int(np.sum(blue_mask))

    return red_count, blue_count


# ============================================================================
# VALIDATION: Ensure the coloring matrix is well-formed
# ============================================================================

def validate_coloring(adj, n):
    """
    Validate that adj is a proper 2-edge-coloring of K_n.

    Returns: (is_valid, error_message)
    """
    # Type check
    if not isinstance(adj, np.ndarray):
        try:
            adj = np.array(adj)
        except Exception as e:
            return False, f"Cannot convert to numpy array: {e}"

    # Shape check
    if adj.ndim != 2:
        return False, f"Expected 2D array, got {adj.ndim}D"
    if adj.shape != (n, n):
        return False, f"Expected shape ({n}, {n}), got {adj.shape}"

    # Convert to bool if needed
    try:
        adj_bool = adj.astype(bool)
    except Exception as e:
        return False, f"Cannot convert to boolean: {e}"

    # Symmetry check
    if not np.array_equal(adj_bool, adj_bool.T):
        asymmetric_count = int(np.sum(adj_bool != adj_bool.T))
        return False, f"Matrix is not symmetric ({asymmetric_count} asymmetric entries)"

    # Diagonal check (no self-loops)
    diag = np.diag(adj_bool)
    if np.any(diag):
        nonzero_diag = int(np.sum(diag))
        return False, f"Diagonal has {nonzero_diag} non-zero entries (should all be 0)"

    return True, "OK"


# ============================================================================
# SCORING
# ============================================================================

def compute_score(max_valid_n, frontier_n, frontier_mono_k5, frontier_total_combos):
    """
    Compute the combined fitness score.

    max_valid_n: largest n with 0 monochromatic K_5
    frontier_n: first n where monochromatic K_5 appears (max_valid_n + 1 in test schedule)
    frontier_mono_k5: number of monochromatic K_5 at frontier_n
    frontier_total_combos: total C(frontier_n, 5) for normalization

    Returns: combined_score in [0.0, 1.0]
    """
    if max_valid_n <= 0:
        # No valid n found — give tiny credit based on frontier performance
        if frontier_n > 0 and frontier_total_combos > 0 and frontier_mono_k5 < frontier_total_combos:
            # Fraction of non-monochromatic combos at smallest tested n
            frac_clean = 1.0 - (frontier_mono_k5 / frontier_total_combos)
            return round(max(frac_clean * 0.01, 0.0), 6)  # Tiny credit
        return 0.0

    # Base score: max_valid_n / TARGET_N
    base = max_valid_n / TARGET_N

    # Frontier credit: smooth bonus for partial success at the next n
    frontier_credit = 0.0
    if frontier_n is not None and frontier_mono_k5 is not None and frontier_mono_k5 > 0:
        # Use log scale: fewer mono-K_5 = more credit
        # credit = 1 / (1 + log2(1 + mono_k5_count))
        # Normalized to contribute at most 1/TARGET_N to the score
        import math
        raw_credit = 1.0 / (1.0 + math.log2(1.0 + frontier_mono_k5))
        frontier_credit = raw_credit / TARGET_N
    elif frontier_n is not None and frontier_mono_k5 == 0:
        # frontier_n is also valid! (shouldn't happen if max_valid_n is correct)
        frontier_credit = 1.0 / TARGET_N

    combined = base + frontier_credit
    return round(min(combined, 1.0), 6)


# ============================================================================
# MAIN EVALUATOR
# ============================================================================

def evaluate(candidate_path):
    """
    Evaluate a candidate R(5,5) construction.

    Returns dict with metrics:
      - combined_score: float in [0, 1] (primary fitness)
      - max_valid_n: largest n with 0 monochromatic K_5
      - frontier_n: first failing n
      - frontier_mono_k5: count of mono-K_5 at frontier
      - frontier_red_k5: red mono-K_5 count at frontier
      - frontier_blue_k5: blue mono-K_5 count at frontier
      - validity: 1.0 if code runs, 0.0 on crash
      - eval_time: total evaluation time in seconds
      - n_tested: number of n values tested
      - details: per-n results summary
    """
    start_time = time.time()
    results = {
        "combined_score": 0.0,
        "max_valid_n": 0,
        "frontier_n": None,
        "frontier_mono_k5": None,
        "frontier_red_k5": None,
        "frontier_blue_k5": None,
        "validity": 0.0,
        "eval_time": 0.0,
        "n_tested": 0,
        "details": "",
    }

    # Load candidate module
    try:
        spec = importlib.util.spec_from_file_location("candidate", candidate_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        results["error"] = f"Failed to load candidate: {e}\n{traceback.format_exc()}"
        results["eval_time"] = round(time.time() - start_time, 4)
        return results

    # Check that generate_coloring exists
    if not hasattr(module, "generate_coloring"):
        results["error"] = "Candidate must define generate_coloring(n) function"
        results["eval_time"] = round(time.time() - start_time, 4)
        return results

    generate_coloring = module.generate_coloring
    results["validity"] = 1.0

    # Test increasing n values
    max_valid_n = 0
    frontier_n = None
    frontier_mono_k5 = None
    frontier_red_k5 = None
    frontier_blue_k5 = None
    frontier_total_combos = 0
    details = []
    n_tested = 0
    consecutive_failures = 0

    for n in TEST_SCHEDULE:
        elapsed = time.time() - start_time
        if elapsed > TOTAL_TIMEOUT:
            details.append(f"n={n}: SKIPPED (total timeout {TOTAL_TIMEOUT}s exceeded)")
            break

        n_tested += 1

        # Generate coloring
        try:
            gen_start = time.time()
            adj = generate_coloring(n)
            gen_time = time.time() - gen_start
        except Exception as e:
            details.append(f"n={n}: CRASH in generate_coloring: {e}")
            consecutive_failures += 1
            if consecutive_failures >= 3:
                details.append(f"  -> Stopping: {consecutive_failures} consecutive failures")
                break
            continue

        # Convert to numpy bool array
        try:
            adj = np.asarray(adj, dtype=bool)
        except Exception as e:
            details.append(f"n={n}: Cannot convert to bool array: {e}")
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
            continue

        # Validate
        is_valid, err_msg = validate_coloring(adj, n)
        if not is_valid:
            details.append(f"n={n}: INVALID coloring: {err_msg}")
            # Try to fix common issues
            if "not symmetric" in err_msg:
                adj = np.triu(adj, k=1)
                adj = adj | adj.T
                np.fill_diagonal(adj, False)
                is_valid2, _ = validate_coloring(adj, n)
                if not is_valid2:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break
                    continue
                details[-1] += " (auto-symmetrized)"
            elif "non-zero" in err_msg:
                np.fill_diagonal(adj, False)
                is_valid2, _ = validate_coloring(adj, n)
                if not is_valid2:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break
                    continue
                details[-1] += " (diagonal zeroed)"
            else:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
                continue

        # Count monochromatic K_5 subgraphs
        try:
            check_start = time.time()
            red_k5, blue_k5, total_k5, checked = count_monochromatic_k5_fast(adj, n)
            check_time = time.time() - check_start
        except Exception as e:
            details.append(f"n={n}: CRASH in K_5 counting: {e}")
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
            continue

        expected_combos = comb(n, 5)
        if checked != expected_combos:
            details.append(f"n={n}: WARNING checked {checked} != expected {expected_combos}")

        if total_k5 == 0:
            max_valid_n = n
            consecutive_failures = 0
            # Reset frontier to track the NEXT failure after this success
            frontier_n = None
            frontier_mono_k5 = None
            frontier_red_k5 = None
            frontier_blue_k5 = None
            frontier_total_combos = 0
            details.append(
                f"n={n}: VALID (0 mono-K_5, checked {checked} subsets, "
                f"gen={gen_time:.3f}s, verify={check_time:.3f}s)"
            )
        else:
            consecutive_failures += 1
            details.append(
                f"n={n}: FAILED ({total_k5} mono-K_5: {red_k5} red + {blue_k5} blue, "
                f"checked {checked}, gen={gen_time:.3f}s, verify={check_time:.3f}s)"
            )
            # Track frontier: first failure AFTER the current max_valid_n
            if frontier_n is None:
                frontier_n = n
                frontier_mono_k5 = total_k5
                frontier_red_k5 = red_k5
                frontier_blue_k5 = blue_k5
                frontier_total_combos = expected_combos

            # Stop if too many failures in a row at large n
            if consecutive_failures >= 5 and n > max_valid_n + 10:
                details.append(f"  -> Stopping: {consecutive_failures} consecutive failures beyond max_valid_n")
                break

    # Compute final score
    combined = compute_score(max_valid_n, frontier_n, frontier_mono_k5, frontier_total_combos)

    results["combined_score"] = combined
    results["max_valid_n"] = max_valid_n
    results["frontier_n"] = frontier_n
    results["frontier_mono_k5"] = frontier_mono_k5 if frontier_mono_k5 is not None else 0
    results["frontier_red_k5"] = frontier_red_k5 if frontier_red_k5 is not None else 0
    results["frontier_blue_k5"] = frontier_blue_k5 if frontier_blue_k5 is not None else 0
    results["validity"] = 1.0
    results["eval_time"] = round(time.time() - start_time, 4)
    results["n_tested"] = n_tested
    results["details"] = "\n".join(details)

    return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <candidate_path>")
        print()
        print("Evaluates a candidate R(5,5) construction.")
        print("The candidate must define: generate_coloring(n) -> (n,n) bool array")
        sys.exit(1)

    candidate_path = sys.argv[1]
    if not os.path.isfile(candidate_path):
        print(f"Error: File not found: {candidate_path}")
        sys.exit(1)

    result = evaluate(candidate_path)
    print(json.dumps(result, indent=2, default=str))
