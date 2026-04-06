# conformal-metric

Recover a smooth scalar field from noisy edge weights on a mesh or graph,
via a strictly convex quadratic program solved in O(E) time.

The algorithm uses the **arithmetic-mean conformal edge rule**:

    w(u,v) = ℓ₀(u,v) · (e^{s(u)} + e^{s(v)}) / 2

which makes the inverse-design Hessian a signless graph Laplacian —
symmetric positive definite on non-bipartite graphs (e.g. triangulations).
The resulting SPD system is solved by conjugate gradient with a Jacobi
preconditioner, achieving 400–3,500× speedup over L-BFGS-B for the
classical geometric-mean rule on meshes with 35–50k vertices.

## Install

```bash
pip install .
```

## Quick start

```python
import numpy as np
from conformal_metric import solve

# vertices: (N, 3) vertex positions
# edges:    (E, 2) edge endpoint indices
# w_star:   (E,)   target edge weights

s, X, w_recovered, dt = solve(vertices, edges, w_star, mu=1e-4)

print(f"Solved in {dt:.4f}s")
print(f"Scalar field range: [{s.min():.3f}, {s.max():.3f}]")
```

## Examples

The `examples/` directory contains scripts that reproduce the paper's
key experiments.  Standard test meshes are downloaded automatically on
first run (~2 MB each, cached in `meshes/`).

| Script | Paper reference | What it does |
|--------|-----------------|--------------|
| `grid_demo.py` | — | Minimal demo on a synthetic 10×10 grid (no download) |
| `bunny_denoise.py` | Table 1 | Baseline vs. regularised QP on Stanford Bunny |
| `multi_mesh_benchmark.py` | Table 2 | Speedup comparison across Spot / Bunny / Nefertiti |
| `noise_sweep.py` | §6 | Noise robustness at σ ∈ [0.01, 0.30] |

Run any example with:

```bash
python examples/bunny_denoise.py
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vertices` | — | (N, d) vertex positions |
| `edges` | — | (E, 2) edge endpoint indices |
| `w_star` | — | (E,) target edge weights (positive) |
| `mu` | 0.0 | Laplacian regularization strength (use mu > 0 for quad meshes) |
| `cg_tol` | 1e-10 | CG relative tolerance |
| `cg_maxiter` | 1000 | Maximum CG iterations |

### Returns

| Value | Description |
|-------|-------------|
| `s` | (N,) recovered scalar field (log-scale) |
| `X` | (N,) conformal factors X = e^s |
| `w_rec` | (E,) reconstructed edge weights |
| `solve_time` | Wall-clock seconds for the CG solve |

## Reference

L. Rönnbäck, "Discrete-to-Continuum Metrics from Scalar Fields",
*Computer Aided Geometric Design*, 2026.

## License

MIT
