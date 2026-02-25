from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def interaction_radius(behavior_R: np.ndarray, buffer: float) -> float:
    """Compute neighbor query radius from max cell radius and buffer."""
    if buffer < 0:
        raise ValueError("buffer must be non-negative")
    r_max = float(np.max(behavior_R))
    sigma_max = 2.0 * r_max
    return sigma_max * (1.0 + buffer)


def candidate_pairs_ckdtree(x: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate pairs as index arrays using a cKDTree radius query."""
    if r <= 0:
        raise ValueError("r must be positive")
    tree = cKDTree(x)
    pairs = tree.query_pairs(r=r, output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return pairs[:, 0].astype(np.int32), pairs[:, 1].astype(np.int32)
