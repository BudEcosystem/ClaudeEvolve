# EVOLVE-BLOCK-START
"""Circle packing n=26: hard-coded optimized layout + multi-round SLSQP refinement"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    n = 26

    # Pre-optimized centers from LP + SLSQP (sum_radii=2.609)
    centers = np.array([
        [0.123844646486, 0.123844646486],
        [0.336832347260, 0.091573923395],
        [0.515930003957, 0.087568516902],
        [0.691301931967, 0.087803568629],
        [0.888866383479, 0.111133616522],
        [0.277908986343, 0.243028032111],
        [0.433624470879, 0.239048790132],
        [0.603185824766, 0.236171524658],
        [0.755074293708, 0.229180687467],
        [0.129114129388, 0.376748531077],
        [0.344782595444, 0.385829152724],
        [0.521062040183, 0.389957909818],
        [0.705793104366, 0.384063400494],
        [0.896183460497, 0.325959196839],
        [0.266221163743, 0.523182491941],
        [0.426436990099, 0.542449141129],
        [0.612899288883, 0.552183191509],
        [0.855822162305, 0.570647113848],
        [0.109822610667, 0.614905208610],
        [0.306460989281, 0.690900950993],
        [0.503880220781, 0.713293176003],
        [0.683195742975, 0.716235333880],
        [0.857410277015, 0.857410277015],
        [0.138464895831, 0.861535104169],
        [0.387623342236, 0.887913952770],
        [0.608398863232, 0.891284794416],
    ])

    radii = np.array([
        0.123844646486, 0.091573922395, 0.087568515902, 0.087803567629,
        0.111133616521, 0.070938565920, 0.084827752223, 0.084758009886,
        0.067291252506, 0.129114128388, 0.086745418718, 0.089582368562,
        0.095242712460, 0.103816538503, 0.071488072402, 0.089882042986,
        0.096834158395, 0.144177836695, 0.109822609667, 0.100990112816,
        0.097694974476, 0.081644681137, 0.142589722985, 0.138464894831,
        0.112086047230, 0.108715204584,
    ])

    # Multi-round SLSQP refinement with perturbation between rounds
    for round_idx in range(4):
        c_new, r_new = _slsqp_refine(centers, radii, n, maxiter=1000)
        if np.sum(r_new) > np.sum(radii):
            centers = c_new
            radii = r_new

        # Perturb for next round (small perturbation to explore nearby basins)
        if round_idx < 3:
            rng = np.random.RandomState(round_idx * 7 + 13)
            centers_p = centers + rng.uniform(-0.005, 0.005, (n, 2))
            centers_p = np.clip(centers_p, 0.003, 0.997)
            # Recompute radii for perturbed centers
            r_p = _heuristic_radii(centers_p)
            c_p2, r_p2 = _slsqp_refine(centers_p, r_p, n, maxiter=800)
            if np.sum(r_p2) > np.sum(radii):
                centers = c_p2
                radii = r_p2

    radii *= 0.9999999
    return centers, radii, float(np.sum(radii))


def _slsqp_refine(centers, radii, n, maxiter=500):
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]

    cons = []
    for i in range(n):
        a, b, c = 3*i, 3*i+1, 3*i+2
        cons.append({'type': 'ineq', 'fun': lambda x, a=a, c=c: x[a] - x[c]})
        cons.append({'type': 'ineq', 'fun': lambda x, b=b, c=c: x[b] - x[c]})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a, c=c: 1 - x[a] - x[c]})
        cons.append({'type': 'ineq', 'fun': lambda x, b=b, c=c: 1 - x[b] - x[c]})
    for i in range(n):
        for j in range(i + 1, n):
            def mk(ii, jj):
                a, b, c = 3*ii, 3*ii+1, 3*ii+2
                d, e, f = 3*jj, 3*jj+1, 3*jj+2
                return lambda x: np.sqrt((x[a]-x[d])**2 + (x[b]-x[e])**2) - x[c] - x[f]
            cons.append({'type': 'ineq', 'fun': mk(i, j)})

    bnds = []
    for i in range(n):
        bnds.extend([(0.003, 0.997), (0.003, 0.997), (0.0001, 0.5)])

    jac = np.zeros(3 * n)
    jac[2::3] = -1.0

    res = minimize(lambda x: -np.sum(x[2::3]), x0, method='SLSQP',
                   jac=lambda x: jac, bounds=bnds, constraints=cons,
                   options={'maxiter': maxiter, 'ftol': 1e-14})

    co = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
    ro = np.array([max(res.x[3*i+2], 0) for i in range(n)])
    _enforce(co, ro, n)
    return co, ro


def _heuristic_radii(centers):
    n = centers.shape[0]
    radii = np.minimum(np.minimum(centers[:, 0], centers[:, 1]),
                       np.minimum(1 - centers[:, 0], 1 - centers[:, 1]))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            h = d / 2.0
            radii[i] = min(radii[i], h)
            radii[j] = min(radii[j], h)
    return np.maximum(radii, 0)


def _enforce(centers, radii, n):
    for i in range(n):
        radii[i] = min(radii[i], centers[i, 0], centers[i, 1],
                       1 - centers[i, 0], 1 - centers[i, 1])
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                ex = (radii[i] + radii[j] - d) / 2 + 1e-9
                radii[i] -= ex
                radii[j] -= ex
    radii[:] = np.maximum(radii, 0)


def compute_max_radii(centers):
    return _heuristic_radii(centers)


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
