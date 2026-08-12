from __future__ import annotations

import numpy as np

from ..state import StateTable
from .neighbors import minimum_image_displacement, validate_box_size


def init_random_periodic(
    N: int,
    box_size: np.ndarray | tuple[float, float],
    state_id: np.ndarray,
    state_table: StateTable,
    rng: np.random.Generator,
    initial_min_separation_factor: float = 0.9,
    max_attempts_per_cell: int = 5000,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize heterogeneous cells uniformly in a periodic 2D box.

    ``initial_min_separation_factor`` controls packing clearance only. It is
    deliberately independent of the force law's ``alpha_dmin`` regularizer.
    """
    if not isinstance(N, (int, np.integer)) or N < 0:
        raise ValueError("N must be a non-negative integer")
    box = validate_box_size(box_size)
    states = np.asarray(state_id, dtype=np.int32)
    if states.shape != (N,):
        raise ValueError("state_id must have shape (N,)")
    if states.size and (np.any(states < 0) or np.any(states >= len(state_table.R))):
        raise ValueError("state_id contains an index outside the state table")
    if not np.isfinite(initial_min_separation_factor) or initial_min_separation_factor <= 0.0:
        raise ValueError("initial_min_separation_factor must be finite and positive")
    if not isinstance(max_attempts_per_cell, (int, np.integer)) or max_attempts_per_cell <= 0:
        raise ValueError("max_attempts_per_cell must be a positive integer")
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("eps must be finite and non-negative")

    radii = np.asarray(state_table.R, dtype=float)
    if radii.ndim != 1 or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("state_table.R must contain finite positive radii")

    x = np.empty((N, 2), dtype=float)
    for placed in range(N):
        radius = radii[states[placed]]
        for _ in range(max_attempts_per_cell):
            candidate = rng.uniform(np.zeros(2), box)
            if placed == 0:
                x[placed] = candidate
                break
            dvec = minimum_image_displacement(candidate, x[:placed], box)
            distances = np.linalg.norm(dvec, axis=1)
            required = initial_min_separation_factor * (radius + radii[states[:placed]])
            if np.all(distances >= required - eps):
                x[placed] = candidate
                break
        else:
            raise ValueError(
                "Failed to place periodic cells at the requested initial clearance; "
                "reduce N or initial_min_separation_factor, increase box_size, or "
                "increase max_attempts_per_cell"
            )

    angles = rng.uniform(0.0, 2.0 * np.pi, size=N)
    p = np.column_stack((np.cos(angles), np.sin(angles)))
    return x, p
