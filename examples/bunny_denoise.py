"""Reproduce Table 1 from the paper: baseline vs. regularised QP on the
Stanford Bunny (34,834 vertices, 104,288 edges).

The script downloads the mesh on first run (~2 MB, cached in meshes/).

Expected output (timings may vary):

  Stanford Bunny: 34,834 verts, 104,288 edges
  ──────────────────────────────────────────────────────────────
    Baseline QP  (μ=0):     err=0.0616  R²=0.859  solve 0.01s
    Regularised QP (μ=1e-4): err=0.0137  R²=0.997  solve 0.04s
    Error reduction: 4.5×
"""

import numpy as np
from conformal_metric import solve, load_mesh, seam_gt_3d

# --- Load mesh ---
vertices, faces, edges = load_mesh("stanford-bunny")
N, E = len(vertices), len(edges)
u, v = edges[:, 0], edges[:, 1]

# --- Ground-truth field and target weights ---
s_gt = seam_gt_3d(vertices)
l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
X_gt = np.exp(s_gt)
w_gt = l0 * (X_gt[u] + X_gt[v]) / 2.0

# Add 10% multiplicative noise (fixed seed for reproducibility)
rng = np.random.default_rng(42)
w_star = w_gt * (1.0 + 0.10 * rng.standard_normal(E))
w_star = np.clip(w_star, 1e-6, None)

# --- Baseline QP (unregularised) ---
s_b, X_b, w_b, dt_b = solve(vertices, edges, w_star, mu=0.0)
err_b = np.linalg.norm(w_b - w_gt) / np.linalg.norm(w_gt)
r2_b = float(np.corrcoef(s_gt, s_b)[0, 1]) ** 2

# --- Regularised QP ---
s_r, X_r, w_r, dt_r = solve(vertices, edges, w_star, mu=1e-4)
err_r = np.linalg.norm(w_r - w_gt) / np.linalg.norm(w_gt)
r2_r = float(np.corrcoef(s_gt, s_r)[0, 1]) ** 2

# --- Report (cf. Table 1) ---
print(f"\nStanford Bunny: {N:,} verts, {E:,} edges")
print("─" * 62)
print(f"  Baseline QP   (μ=0):    err={err_b:.4f}  "
      f"R²={r2_b:.3f}  solve {dt_b:.2f}s")
print(f"  Regularised QP (μ=1e-4): err={err_r:.4f}  "
      f"R²={r2_r:.3f}  solve {dt_r:.2f}s")
print(f"  Error reduction: {err_b / err_r:.1f}×")
