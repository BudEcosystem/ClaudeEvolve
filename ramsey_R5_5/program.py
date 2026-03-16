#!/usr/bin/env python3
"""
Ramsey R(5,5) lower bound construction.

GOAL: Find 2-edge-coloring of K_n with NO monochromatic K_5, for largest possible n.
Current world record: n=42 (R(5,5) ≥ 43, Exoo 1989).
Target: n=43 (would prove R(5,5) ≥ 44).
"""

import numpy as np


# EVOLVE-BLOCK-START

def generate_coloring(n):
    """
    Generate a 2-edge-coloring of K_n with no monochromatic K_5.

    Approach: Circulant graph using quadratic residues modulo the
    nearest prime to n. Edge (i,j) is red iff the circular distance
    min(|i-j|, n-|i-j|) is in the connection set derived from QRs.

    Returns: numpy array of shape (n, n), dtype bool
    """
    half = n // 2

    # Find a good prime for quadratic residue construction
    p = n
    while not _is_prime(p):
        p += 1

    # Compute quadratic residues mod p, mapped to circulant distances
    connection_set = set()
    for x in range(1, p):
        r = (x * x) % p
        if 0 < r <= half:
            connection_set.add(r)

    # Build the circulant adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    for d in connection_set:
        for i in range(n):
            j = (i + d) % n
            adj[i, j] = True
            adj[j, i] = True
    np.fill_diagonal(adj, False)

    return adj


def _is_prime(num):
    """Primality test."""
    if num < 2:
        return False
    if num < 4:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import sys
    from itertools import combinations

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    adj = generate_coloring(n)
    print(f"Generated coloring for K_{n}")
    print(f"Symmetric: {np.array_equal(adj, adj.T)}")
    red_edges = int(np.sum(adj)) // 2
    total_edges = n * (n - 1) // 2
    print(f"Red edges: {red_edges}/{total_edges} ({100*red_edges/total_edges:.1f}%)")

    mono = 0
    for combo in combinations(range(n), 5):
        i, j, k, l, m = combo
        e = [adj[i,j], adj[i,k], adj[i,l], adj[i,m],
             adj[j,k], adj[j,l], adj[j,m], adj[k,l], adj[k,m], adj[l,m]]
        if all(e) or not any(e):
            mono += 1
    print(f"Monochromatic K_5: {mono}")
