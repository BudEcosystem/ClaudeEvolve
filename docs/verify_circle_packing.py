#!/usr/bin/env python3
"""
Independent verification of Claude Evolve circle packing result.
No dependency on Claude Evolve code — uses only numpy.

Usage: python verify_circle_packing.py
"""
import numpy as np

# Solution coordinates from the paper (Section 8)
SOLUTION = np.array([
    [0.501331924483, 0.529963419454, 0.137010423754],
    [0.728370146230, 0.597634795411, 0.099898344594],
    [0.596641215431, 0.742417047077, 0.095842319787],
    [0.500571630855, 0.906072658662, 0.093927331338],
    [0.404780268012, 0.742049441040, 0.096018969798],
    [0.273094287957, 0.596042700948, 0.100600361813],
    [0.297390398327, 0.381665845636, 0.115148874009],
    [0.504468239281, 0.275342619018, 0.117629681870],
    [0.705253938899, 0.386923554540, 0.112077082829],
    [0.904267666650, 0.683258533140, 0.095732323350],
    [0.760289469470, 0.763673566747, 0.069180670665],
    [0.686884188161, 0.907608444353, 0.092391545647],
    [0.314056979873, 0.907407900974, 0.092592089026],
    [0.240647601033, 0.762958861024, 0.069440188017],
    [0.096151338084, 0.682080041112, 0.096151328084],
    [0.103467237323, 0.482595582385, 0.103467227323],
    [0.105182564217, 0.273952841884, 0.105182554217],
    [0.297690476934, 0.133258576438, 0.133258566438],
    [0.705390509164, 0.130221104763, 0.130221094763],
    [0.893209851439, 0.274783285603, 0.106790138561],
    [0.896939475889, 0.484600802806, 0.103060514111],
    [0.084926266606, 0.084926266606, 0.084926256606],
    [0.915360495151, 0.084639504849, 0.084639494849],
    [0.111156183299, 0.888843816701, 0.111156173299],
    [0.889220983317, 0.889220983317, 0.110779006683],
    [0.502715553769, 0.078860377127, 0.078860367127],
])

centers = SOLUTION[:, :2]
radii = SOLUTION[:, 2]
n = len(radii)

print("=" * 60)
print("  INDEPENDENT VERIFICATION OF CIRCLE PACKING RESULT")
print("=" * 60)
print()

errors = []

# 1. Shape check
assert centers.shape == (26, 2), f"Bad centers shape: {centers.shape}"
assert radii.shape == (26,), f"Bad radii shape: {radii.shape}"
print(f"  Circles: {n}")
print(f"  Sum of radii: {np.sum(radii):.10f}")
print()

# 2. Positivity
if np.any(radii <= 0):
    errors.append(f"Non-positive radius: min = {np.min(radii)}")
print(f"  Min radius: {np.min(radii):.10f}  (must be > 0)")

# 3. Boundary containment (strict)
min_border_gap = float("inf")
for i in range(n):
    x, y, r = centers[i, 0], centers[i, 1], radii[i]
    gaps = [x - r, y - r, 1 - x - r, 1 - y - r]
    for g in gaps:
        if g < min_border_gap:
            min_border_gap = g
        if g < 0:
            errors.append(f"Circle {i} protrudes: gap = {g:.3e}")

print(f"  Tightest border gap: {min_border_gap:+.3e}  (must be >= 0)")

# 4. Non-overlap (strict)
min_pair_gap = float("inf")
for i in range(n):
    for j in range(i + 1, n):
        dist = np.linalg.norm(centers[i] - centers[j])
        gap = dist - radii[i] - radii[j]
        if gap < min_pair_gap:
            min_pair_gap = gap
        if gap < 0:
            errors.append(f"Circles {i},{j} overlap: gap = {gap:.3e}")

print(f"  Tightest pair gap:   {min_pair_gap:+.3e}  (must be >= 0)")
print()

# 5. Verdict
if errors:
    print(f"  ERRORS FOUND: {len(errors)}")
    for e in errors:
        print(f"    - {e}")
    print()
    print("  VERDICT: INVALID")
else:
    print("  VERDICT: VALID")
    print("    - All radii positive")
    print("    - All circles inside unit square")
    print("    - No overlaps (strict)")
    print(f"    - Sum of radii: {np.sum(radii):.10f}")

print()

# 6. Comparison with published results
print("  COMPARISON:")
published = [
    ("AlphaEvolve (DeepMind)", 2.635863),
    ("FICO Xpress (ZIB/MODAL)", 2.635916),
    ("OpenEvolve community", 2.635977),
]
our = np.sum(radii)
for name, val in published:
    margin = our - val
    print(f"    vs {name}: {margin:+.8f}")

print()
print("=" * 60)
