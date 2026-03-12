"""
Circle packing optimization - improved candidate.
Uses a slightly better placement strategy.
"""

import math


def pack_circles(n_circles=5):
    """Place n_circles non-overlapping circles in the unit square [0,1]x[0,1].

    Returns a list of (x, y, r) tuples.
    """
    circles = []

    if n_circles == 1:
        return [(0.5, 0.5, 0.5)]

    # Better approach: use a hexagonal-ish arrangement
    # Place one large circle in the center, smaller ones around it
    if n_circles <= 5:
        # Central circle
        center_r = 0.25
        circles.append((0.5, 0.5, center_r))

        # Corner circles - maximize radius while avoiding center and bounds
        remaining = n_circles - 1
        angles = [math.pi / 4 + i * math.pi / 2 for i in range(remaining)]

        for angle in angles:
            # Place at distance from center
            dist = 0.5  # distance from center
            cx = 0.5 + dist * math.cos(angle) * 0.55
            cy = 0.5 + dist * math.sin(angle) * 0.55

            # Calculate max radius: must not overlap center circle
            dist_to_center = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
            max_r_center = dist_to_center - center_r

            # Must stay within bounds
            max_r_bounds = min(cx, 1 - cx, cy, 1 - cy)

            # Must not overlap other corner circles
            max_r_others = float("inf")
            for ox, oy, or_ in circles[1:]:
                d = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
                max_r_others = min(max_r_others, d - or_)

            r = min(max_r_center, max_r_bounds, max_r_others) * 0.99
            if r > 0.01:
                circles.append((cx, cy, r))

    return circles


if __name__ == "__main__":
    result = pack_circles()
    total_radius = sum(r for _, _, r in result)
    print(f"Circles: {len(result)}")
    for i, (x, y, r) in enumerate(result):
        print(f"  Circle {i+1}: center=({x:.3f}, {y:.3f}), radius={r:.4f}")
    print(f"Total sum of radii: {total_radius:.4f}")
