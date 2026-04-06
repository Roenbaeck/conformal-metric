"""Reproduce Table 2 from the paper: arithmetic-mean QP vs. geometric-mean
L-BFGS-B across three meshes of increasing size.

Downloads ~6 MB of meshes on first run (cached in meshes/).

Expected output (timings are hardware-dependent):

  Mesh               Verts    Edges   Arith (s)  Geom (s)  Speedup
  ─────────────────────────────────────────────────────────────────
  spot               2,930    8,784      0.002      0.04      17×
  stanford-bunny    34,834  104,288      0.019      7.79     410×
  nefertiti         49,971  149,907      0.033     96.48    3324×
"""

import time
import numpy as np
from scipy.optimize import minimize

from conformal_metric import solve, load_mesh, seam_gt_3d

N_REPEATS = 5  # median of 5 for arithmetic; fewer for large geometric


def run_geometric(vertices, edges, w_star):
    """Geometric-mean inverse via L-BFGS-B (the classical approach)."""
    u, v = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
    N = len(vertices)

    def objective(s):
        w = l0 * np.exp(0.5 * (s[u] + s[v]))
        r = w - w_star
        f = 0.5 * np.dot(r, r)
        grad = np.zeros(N)
        rw = r * w * 0.5
        np.add.at(grad, u, rw)
        np.add.at(grad, v, rw)
        return f, grad

    t0 = time.perf_counter()
    minimize(objective, np.zeros(N), method="L-BFGS-B", jac=True,
             options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    return time.perf_counter() - t0


MESHES = ["spot", "stanford-bunny", "nefertiti"]

print(f"\n{'Mesh':<20s} {'Verts':>7s} {'Edges':>8s}  "
      f"{'Arith (s)':>9s} {'Geom (s)':>9s}  {'Speedup':>7s}")
print("─" * 68)

for name in MESHES:
    vertices, faces, edges = load_mesh(name)
    N, E = len(vertices), len(edges)
    u, v = edges[:, 0], edges[:, 1]

    s_gt = seam_gt_3d(vertices)
    l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
    X_gt = np.exp(s_gt)
    w_gt = l0 * (X_gt[u] + X_gt[v]) / 2.0

    rng = np.random.default_rng(42)
    w_star = w_gt * (1.0 + 0.10 * rng.standard_normal(E))
    w_star = np.clip(w_star, 1e-6, None)

    # Geometric-mean target weights for L-BFGS-B comparison
    w_gt_g = l0 * np.exp(0.5 * (s_gt[u] + s_gt[v]))
    w_star_g = w_gt_g * (1.0 + 0.10 * rng.standard_normal(E))
    w_star_g = np.clip(w_star_g, 1e-6, None)

    # Arithmetic CG (median of N_REPEATS runs)
    arith_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        solve(vertices, edges, w_star)
        arith_times.append(time.perf_counter() - t0)
    dt_a = np.median(arith_times)

    # Geometric L-BFGS-B (single run for large meshes)
    dt_g = run_geometric(vertices, edges, w_star_g)

    print(f"  {name:<18s} {N:>7,d} {E:>8,d}  "
          f"{dt_a:>9.3f} {dt_g:>9.2f}  {dt_g/dt_a:>6.0f}×")
