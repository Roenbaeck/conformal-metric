"""Noise robustness sweep on the Stanford Bunny.

Runs the inverse-design QP at noise levels σ ∈ [0.01, 0.30] and reports
edge-weight error and field correlation (R) for both base and regularised
variants.  Corresponds to the noise-sweep analysis in the paper (§6).

Expected output:

    σ      err (μ=0)  R (μ=0)   err (μ=1e-4)  R (μ=1e-4)
  ─────────────────────────────────────────────────────────
  0.01     0.0069    0.9992      0.0065       0.9995
  0.05     0.0309    0.9838      0.0105       0.9989
  0.10     0.0616    0.9161      0.0137       0.9970
  0.20     0.1219    0.7750      0.0189       0.9936
  0.30     0.1853    0.5709      0.0242       0.9895
"""

import numpy as np
from conformal_metric import solve, load_mesh, seam_gt_3d

# --- Setup ---
vertices, faces, edges = load_mesh("stanford-bunny")
N, E = len(vertices), len(edges)
u, v = edges[:, 0], edges[:, 1]

s_gt = seam_gt_3d(vertices)
l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
X_gt = np.exp(s_gt)
w_gt = l0 * (X_gt[u] + X_gt[v]) / 2.0

SIGMAS = [0.01, 0.05, 0.10, 0.20, 0.30]

print(f"\n  {'σ':>5s}   {'err (μ=0)':>9s} {'R (μ=0)':>8s}   "
      f"{'err (μ=1e-4)':>12s} {'R (μ=1e-4)':>10s}")
print("  " + "─" * 57)

for sigma in SIGMAS:
    rng = np.random.default_rng(42)
    w_star = w_gt * (1.0 + sigma * rng.standard_normal(E))
    w_star = np.clip(w_star, 1e-6, None)

    # Baseline
    s_b, _, w_b, _ = solve(vertices, edges, w_star, mu=0.0)
    err_b = np.linalg.norm(w_b - w_gt) / np.linalg.norm(w_gt)
    r_b = float(np.corrcoef(s_gt, s_b)[0, 1])

    # Regularised
    s_r, _, w_r, _ = solve(vertices, edges, w_star, mu=1e-4)
    err_r = np.linalg.norm(w_r - w_gt) / np.linalg.norm(w_gt)
    r_r = float(np.corrcoef(s_gt, s_r)[0, 1])

    print(f"  {sigma:>5.2f}   {err_b:>9.4f} {r_b:>8.4f}   "
          f"{err_r:>12.4f} {r_r:>10.4f}")
