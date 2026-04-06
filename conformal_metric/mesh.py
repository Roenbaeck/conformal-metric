"""Mesh loading, OBJ parsing, and standard test model downloads."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# Standard test meshes from alecjacobson/common-3d-test-models
MESH_URLS: dict[str, str] = {
    "stanford-bunny": (
        "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/"
        "master/data/stanford-bunny.obj"
    ),
    "spot": (
        "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/"
        "master/data/spot.obj"
    ),
    "nefertiti": (
        "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/"
        "master/data/nefertiti.obj"
    ),
}

_CACHE_DIR = Path(__file__).resolve().parent.parent / "meshes"


def download_mesh(name: str, cache_dir: Path | None = None) -> Path:
    """Download a standard test mesh (cached after first call).

    Parameters
    ----------
    name : str
        One of: ``"stanford-bunny"``, ``"spot"``, ``"nefertiti"``.
    cache_dir : Path, optional
        Directory for cached OBJ files.  Defaults to ``meshes/`` next to
        the package root.

    Returns
    -------
    Path to the downloaded OBJ file.
    """
    if name not in MESH_URLS:
        raise ValueError(
            f"Unknown mesh {name!r}. Available: {sorted(MESH_URLS)}"
        )
    d = cache_dir or _CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.obj"
    if not path.exists():
        print(f"Downloading {name}.obj …")
        urllib.request.urlretrieve(MESH_URLS[name], path)  # noqa: S310
    return path


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal OBJ parser.

    Returns
    -------
    vertices : (N, 3) float array
    faces : (F, 3) int64 array (fan-triangulated if needed)
    """
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) >= 4:
                    for k in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[k], idx[k + 1]])
    return np.array(vertices, dtype=float), np.array(faces, dtype=np.int64)


def extract_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges from a triangle array.  Returns (E, 2)."""
    edge_set: set[tuple[int, int]] = set()
    for i, j, k in faces:
        i, j, k = int(i), int(j), int(k)
        edge_set.add((min(i, j), max(i, j)))
        edge_set.add((min(j, k), max(j, k)))
        edge_set.add((min(k, i), max(k, i)))
    return np.array(sorted(edge_set), dtype=np.int64)


def largest_component(
    vertices: np.ndarray, faces: np.ndarray, edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only the largest connected component."""
    N = len(vertices)
    u, v = edges[:, 0], edges[:, 1]
    A = sp.coo_matrix(
        (np.ones(2 * len(edges)), (np.r_[u, v], np.r_[v, u])),
        shape=(N, N),
    )
    n_comp, labels = connected_components(A, directed=False)
    if n_comp == 1:
        return vertices, faces, edges

    comp_sizes = np.bincount(labels)
    main_label = int(np.argmax(comp_sizes))
    keep = labels == main_label

    old_to_new = -np.ones(N, dtype=np.int64)
    old_to_new[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    new_verts = vertices[keep]

    face_keep = keep[faces].all(axis=1)
    new_faces = old_to_new[faces[face_keep]]

    edge_keep = keep[edges].all(axis=1)
    new_edges = old_to_new[edges[edge_keep]]

    n_removed = N - int(keep.sum())
    if n_removed > 0:
        print(f"  Removed {n_removed:,} orphan vertices "
              f"({n_comp} components → 1).")
    return new_verts, new_faces, new_edges


def load_mesh(
    name: str, cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download (if needed), parse, and clean a standard test mesh.

    Returns
    -------
    vertices : (N, 3)
    faces : (F, 3)
    edges : (E, 2)
    """
    path = download_mesh(name, cache_dir=cache_dir)
    vertices, faces = load_obj(path)
    edges = extract_edges(faces)
    return largest_component(vertices, faces, edges)


def seam_gt_3d(vertices: np.ndarray) -> np.ndarray:
    """Smooth ground-truth scalar field on a 3D surface.

    This is the test field used throughout the paper's experiments:
    a height-modulated sinusoid on normalised vertex coordinates.
    """
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    scale = v_max - v_min
    scale[scale < 1e-12] = 1.0
    v_n = (vertices - v_min) / scale
    return (
        0.5 * np.sin(np.pi * v_n[:, 1])
        + 0.25 * np.cos(np.pi * v_n[:, 0])
    )
