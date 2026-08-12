from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def validate_box_size(box_size: np.ndarray | tuple[float, float]) -> np.ndarray:
    """Validate and return a rectangular 2D periodic box vector."""
    box = np.asarray(box_size, dtype=float)
    if box.shape != (2,):
        raise ValueError("box_size must contain exactly (Lx, Ly)")
    if not np.all(np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box_size values must be finite and positive")
    return box


def minimum_image_displacement(
    x_i: np.ndarray,
    x_j: np.ndarray,
    box_size: np.ndarray | tuple[float, float],
) -> np.ndarray:
    """Return displacement(s) from ``j`` to ``i`` under minimum images."""
    box = validate_box_size(box_size)
    dvec = np.asarray(x_i, dtype=float) - np.asarray(x_j, dtype=float)
    return dvec - box * np.round(dvec / box)


def candidate_pairs_periodic(
    x: np.ndarray,
    r: float,
    box_size: np.ndarray | tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return periodic cKDTree candidate pairs as two index arrays."""
    points = np.asarray(x, dtype=float)
    box = validate_box_size(box_size)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("x must have shape (N, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("x must contain only finite values")
    if np.any(points < 0.0) or np.any(points >= box):
        raise ValueError("periodic positions must lie inside [0, Lx) x [0, Ly)")
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("r must be finite and positive")

    pairs = cKDTree(points, boxsize=box).query_pairs(r=r, output_type="ndarray")
    if pairs.size == 0:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty.copy()
    return pairs[:, 0].astype(np.int32), pairs[:, 1].astype(np.int32)
