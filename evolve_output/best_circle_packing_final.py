# EVOLVE-BLOCK-START
"""Circle packing n=26: STRICT constraints + 100 multi-start perturbations.

Uses strict zero-tolerance non-overlap (SAFETY=1e-8).
Maximizes perturbation restarts within time budget for maximum coverage.
Saves best result to warm cache for multi-iteration accumulation.
"""
import numpy as np
import os
import time
from scipy.optimize import minimize, linprog

STRICT_TOL = 0.0
SAFETY = 1e-8


def construct_packing():
    n = 26
    t0 = time.time()
    best_c, best_r, best_sum = None, None, 0.0

    # Load warm cache if available
    cache_path = '.claude/evolve-state/warm_cache/items/best_packing.npz'
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            cached_c, cached_r = data['centers'], data['radii']
            if cached_c.shape == (n, 2) and cached_r.shape == (n,):
                s = np.sum(cached_r)
                if s > best_sum:
                    best_sum, best_c, best_r = s, cached_c.copy(), cached_r.copy()
        except Exception:
            pass

    # Cold start: ring topology
    if best_c is None:
        c, r = _pat_ring()
        c, r = _opt_strict(c, r)
        s = np.sum(r)
        if s > best_sum:
            best_sum, best_c, best_r = s, c.copy(), r.copy()

    # Massive perturbation campaign — use ALL remaining time
    seed = 0
    iter_offset = 0
    iter_path = '.claude/evolve-state/warm_cache/items/packing_iter.json'
    if os.path.exists(iter_path):
        try:
            import json
            with open(iter_path) as f:
                iter_offset = json.load(f).get('offset', 0)
        except Exception:
            pass

    base_seed = iter_offset * 100  # Different seeds each evolution iteration
    while time.time() - t0 < 250:  # Stay within eval budget
        rng = np.random.RandomState(base_seed + seed)
        mag = 0.0005 + (seed % 20) * 0.0007
        cp = best_c + rng.uniform(-mag, mag, (n, 2))
        cp = np.clip(cp, 0.005, 0.995)
        rp = _heur(cp)
        try:
            cp, rp = _opt_strict(cp, rp)
            if np.sum(rp) > best_sum:
                best_sum = np.sum(rp)
                best_c, best_r = cp.copy(), rp.copy()
        except Exception:
            pass
        seed += 1

    # Save to warm cache
    try:
        cache_dir = '.claude/evolve-state/warm_cache/items'
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(os.path.join(cache_dir, 'best_packing.npz'),
                 centers=best_c, radii=best_r)
        import json
        with open(os.path.join(cache_dir, 'packing_iter.json'), 'w') as f:
            json.dump({'offset': iter_offset + 1, 'seeds_tried': seed,
                       'best_sum': float(best_sum)}, f)
    except Exception:
        pass

    return best_c, best_r, best_sum


def _pat_ring():
    n = 26
    c, r = np.zeros((n, 2)), np.zeros(n)
    c[0] = [0.5, 0.5]; r[0] = 0.133
    for i in range(8):
        a = 2*np.pi*i/8; c[1+i] = [0.5+0.231*np.cos(a), 0.5+0.231*np.sin(a)]; r[1+i] = 0.098
    for i in range(12):
        a = 2*np.pi*i/12+np.pi/12; c[9+i] = [0.5+0.417*np.cos(a), 0.5+0.417*np.sin(a)]; r[9+i] = 0.088
    cr = 0.093; c[21]=[cr,cr]; c[22]=[1-cr,cr]; c[23]=[cr,1-cr]; c[24]=[1-cr,1-cr]; r[21:25]=cr
    c[25]=[0.5,0.15]; r[25]=0.083
    return c, r


def _heur(centers):
    n = centers.shape[0]
    r = np.minimum(np.minimum(centers[:,0], centers[:,1]),
                   np.minimum(1-centers[:,0], 1-centers[:,1]))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i]-centers[j]); h = d/2
            r[i] = min(r[i], h); r[j] = min(r[j], h)
    return np.maximum(r, 0)


def _opt_strict(centers, radii):
    n = len(centers)

    # Stage 1: Penalty relaxation
    def pen(x_flat):
        cc = x_flat.reshape(n, 2); p = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(cc[i]-cc[j])
                ov = radii[i]+radii[j]-d
                if ov > 0: p += 500*ov**2*(radii[i]+radii[j])
            x, y = cc[i]
            for v in [radii[i]-x, x+radii[i]-1, radii[i]-y, y+radii[i]-1]:
                if v > 0: p += 500*v**2
        return p
    res1 = minimize(pen, centers.flatten(), method='L-BFGS-B',
                    bounds=[(0,1)]*(2*n), options={'maxiter':300, 'ftol':1e-10})
    centers = res1.x.reshape(n, 2)

    # Stage 2: LP radii (strict)
    A, b = [], []
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n); row[i]=1; row[j]=1; A.append(row)
            b.append(np.linalg.norm(centers[i]-centers[j]) - SAFETY)
        x, y = centers[i]
        for v in [x-SAFETY, y-SAFETY, 1-x-SAFETY, 1-y-SAFETY]:
            row = np.zeros(n); row[i]=1; A.append(row); b.append(v)
    res_lp = linprog(-np.ones(n), A_ub=np.array(A), b_ub=np.array(b),
                     bounds=[(0, None)]*n, method='highs')
    if res_lp.success: radii = res_lp.x

    # Stage 3: Joint SLSQP (strict)
    def con(x):
        cc, rr = x[:2*n].reshape(n,2), x[2*n:]
        cs = []
        for i in range(n):
            for j in range(i+1, n):
                cs.append(np.linalg.norm(cc[i]-cc[j]) - rr[i] - rr[j] - SAFETY)
            cs.extend([cc[i,0]-rr[i]-SAFETY, 1-cc[i,0]-rr[i]-SAFETY,
                        cc[i,1]-rr[i]-SAFETY, 1-cc[i,1]-rr[i]-SAFETY])
        return np.array(cs)
    x0 = np.concatenate([centers.flatten(), radii])
    res = minimize(lambda x: -np.sum(x[2*n:]), x0, method='SLSQP',
                   constraints={'type':'ineq','fun':con},
                   bounds=[(0.001,0.999)]*(2*n)+[(0.001,0.5)]*n,
                   options={'maxiter':2000, 'ftol':1e-13})
    fc, fr = res.x[:2*n].reshape(n,2), res.x[2*n:]
    _enforce_strict(fc, fr, n)
    return fc, fr


def _enforce_strict(co, ro, n):
    for i in range(n):
        ro[i] = min(ro[i], co[i,0]-SAFETY, co[i,1]-SAFETY,
                    1-co[i,0]-SAFETY, 1-co[i,1]-SAFETY)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(co[i]-co[j])
            if ro[i]+ro[j] > d - SAFETY:
                ex = (ro[i]+ro[j]-d+SAFETY)/2 + 1e-12
                ro[i] -= ex; ro[j] -= ex
    ro[:] = np.maximum(ro, 0)


def compute_max_radii(centers):
    return _heur(centers)


# EVOLVE-BLOCK-END


def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(True)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(center, radius, alpha=0.5))
        ax.text(center[0], center[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    visualize(centers, radii)
