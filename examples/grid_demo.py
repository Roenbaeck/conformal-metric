"""Minimal example: recover a scalar field on a small triangulated grid."""

import numpy as np
from conformal_metric import solve

# --- Build a small triangulated 10x10 grid ---
nx, ny = 10, 10
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xx, yy = np.meshgrid(x, y)
vertices = np.column_stack([xx.ravel(), yy.ravel()])
N = len(vertices)

# Triangulate: each quad → 2 triangles
faces = []
for i in range(ny - 1):
    for j in range(nx - 1):
        v0 = i * nx + j
        v1 = v0 + 1
        v2 = v0 + nx
        v3 = v2 + 1
        faces.append([v0, v1, v2])
        faces.append([v1, v3, v2])
faces = np.array(faces)

# Extract unique edges
edge_set = set()
for f in faces:
    for a, b in [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])]:
        edge_set.add((min(a, b), max(a, b)))
edges = np.array(sorted(edge_set))
E = len(edges)

# --- Ground-truth scalar field: smooth bump ---
cx, cy = 0.5, 0.5
s_gt = 0.5 * np.exp(-((vertices[:, 0] - cx)**2 + (vertices[:, 1] - cy)**2) / 0.1)

# --- Generate target weights under arithmetic-mean rule, add noise ---
u, v = edges[:, 0], edges[:, 1]
l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
X_gt = np.exp(s_gt)
w_gt = l0 * (X_gt[u] + X_gt[v]) / 2.0

rng = np.random.default_rng(42)
sigma = 0.05
w_star = w_gt * (1.0 + sigma * rng.standard_normal(E))
w_star = np.clip(w_star, 1e-6, None)

# --- Solve ---
s_rec, X_rec, w_rec, dt = solve(vertices, edges, w_star, mu=1e-3)

# --- Report ---
w_err = np.linalg.norm(w_rec - w_gt) / np.linalg.norm(w_gt)
r = np.corrcoef(s_gt, s_rec)[0, 1]

print(f"Grid: {N} vertices, {E} edges")
print(f"Solve time:       {dt:.4f} s")
print(f"Weight error:     {w_err:.4f}")
print(f"Field correlation: {r:.4f}")
