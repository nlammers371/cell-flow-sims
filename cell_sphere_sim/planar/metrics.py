from __future__ import annotations

import numpy as np


def polarization_magnitude(p: np.ndarray) -> float:
    """Magnitude of the population-mean unit polarity vector."""
    if p.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(np.mean(p, axis=0)))


def nematic_order_2d(p: np.ndarray) -> float:
    """Magnitude of ``mean(exp(2j*theta))`` for planar polarities."""
    if p.shape[0] == 0:
        return 0.0
    q_x = np.mean(p[:, 0] ** 2 - p[:, 1] ** 2)
    q_y = np.mean(2.0 * p[:, 0] * p[:, 1])
    return float(np.hypot(q_x, q_y))


def largest_cluster_fraction(
    n_cells: int,
    contact_i: np.ndarray,
    contact_j: np.ndarray,
) -> float:
    """Fraction of all cells in the largest connected contact component."""
    if n_cells == 0:
        return 0.0
    parent = np.arange(n_cells, dtype=np.int64)
    size = np.ones(n_cells, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for left, right in zip(contact_i, contact_j):
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left == root_right:
            continue
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    return float(np.max(size) / n_cells)
