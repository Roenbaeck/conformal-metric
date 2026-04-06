"""Conformal metric inverse design via arithmetic-mean edge rule.

Recovers a scalar field s from noisy target edge weights w* on a mesh
or graph, by solving a strictly convex quadratic program in X = e^s.

The arithmetic-mean conformal rule is:

    w(u,v) = ℓ₀(u,v) · (X_u + X_v) / 2

The Hessian of the least-squares fitting objective is proportional to
the signless graph Laplacian, which is positive definite on non-bipartite
graphs (e.g. triangulations).  The resulting SPD system is solved by
conjugate gradient with a Jacobi (diagonal) preconditioner.

Reference:
    L. Rönnbäck, "Discrete-to-Continuum Metrics from Scalar Fields",
    Computer Aided Geometric Design, 2026.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg


def solve(
    vertices: np.ndarray,
    edges: np.ndarray,
    w_star: np.ndarray,
    mu: float = 0.0,
    cg_tol: float = 1e-10,
    cg_maxiter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Recover a scalar field from target edge weights.

    Parameters
    ----------
    vertices : (N, d) array
        Vertex positions (any dimension).
    edges : (E, 2) array of int
        Edge endpoint indices.
    w_star : (E,) array
        Target edge weights (positive).
    mu : float, optional
        Laplacian regularization strength.  Default 0 (unregularized).
        Must be >= 0.  For quad meshes (bipartite graphs), use mu > 0.
    cg_tol : float, optional
        Relative tolerance for conjugate gradient.
    cg_maxiter : int, optional
        Maximum CG iterations.

    Returns
    -------
    s : (N,) array
        Recovered scalar field (log-scale).
    X : (N,) array
        Recovered conformal factors X = e^s.
    w_rec : (E,) array
        Reconstructed edge weights from the recovered field.
    solve_time : float
        Wall-clock time for the CG solve (seconds).
    """
    import time

    N = len(vertices)
    E = len(edges)
    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)

    # Background edge lengths
    l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
    alpha = l0 / 2.0
    alpha_sq = alpha ** 2

    # Signless Laplacian Hessian: H = D + A (weighted)
    I = np.concatenate([u, v])
    J = np.concatenate([v, u])
    V_off = np.concatenate([alpha_sq, alpha_sq])
    H = sp.coo_matrix((V_off, (I, J)), shape=(N, N))
    H_diag = (np.bincount(u, weights=alpha_sq, minlength=N)
              + np.bincount(v, weights=alpha_sq, minlength=N))
    H = (H + sp.diags(H_diag)).tocsr()

    # Right-hand side
    b_edge = alpha * w_star
    b = (np.bincount(u, weights=b_edge, minlength=N)
         + np.bincount(v, weights=b_edge, minlength=N))

    # Laplacian regularization
    if mu > 0:
        ones_E = np.ones(E)
        A_adj = sp.coo_matrix(
            (np.concatenate([ones_E, ones_E]),
             (np.concatenate([u, v]), np.concatenate([v, u]))),
            shape=(N, N),
        )
        deg = np.bincount(u, minlength=N) + np.bincount(v, minlength=N)
        L = sp.diags(deg.astype(float)) - A_adj.tocsr()
        H = H + mu * L.tocsr()

    # Tiny ridge for numerical stability
    ridge = 1e-12 * max(1.0, float(H.diagonal().mean()))
    H = H + sp.eye(N, format="csr") * ridge

    # CG with Jacobi preconditioner
    M_inv = sp.diags(1.0 / H.diagonal())

    t0 = time.perf_counter()
    X, info = cg(H, b, rtol=cg_tol, maxiter=cg_maxiter, M=M_inv)
    solve_time = time.perf_counter() - t0

    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")

    X = np.clip(X, 1e-12, None)
    s = np.log(X)
    w_rec = l0 * (X[u] + X[v]) / 2.0

    return s, X, w_rec, solve_time
