"""
Circle packing optimization - pack circles into a unit square.

Maximize the sum of radii of non-overlapping circles placed inside [0,1]x[0,1].
Each circle must be fully contained within the square and must not overlap other circles.

EVOLVE-BLOCK-START
"""

import math


def pack_circles(n_circles=5):
    """Place n_circles non-overlapping circles in the unit square [0,1]x[0,1].

    Returns a list of (x, y, r) tuples.
    """
    circles = []
    # Simple greedy approach: place circles in a grid pattern
    grid_size = math.ceil(math.sqrt(n_circles))
    cell_size = 1.0 / grid_size
    max_radius = cell_size / 2.0 * 0.9  # 90% of half-cell to avoid touching

    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= n_circles:
                break
            cx = (i + 0.5) * cell_size
            cy = (j + 0.5) * cell_size
            circles.append((cx, cy, max_radius))
            count += 1

    return circles


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    result = pack_circles()
    total_radius = sum(r for _, _, r in result)
    print(f"Circles: {len(result)}")
    for i, (x, y, r) in enumerate(result):
        print(f"  Circle {i+1}: center=({x:.3f}, {y:.3f}), radius={r:.4f}")
    print(f"Total sum of radii: {total_radius:.4f}")
