"""Model-independent diagnostics for periodic scalar fields."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage


def structure_factor_length(field: np.ndarray, grid) -> float:
    """Return 2π times the inverse first moment of the structure factor."""

    fluctuations = field - np.mean(field)
    spectrum = np.abs(grid.fft(fluctuations)) ** 2
    wave_number = np.sqrt(grid.k2)
    valid = wave_number > 0.0
    total = float(np.sum(spectrum[valid]))
    weighted = float(np.sum(wave_number[valid] * spectrum[valid]))
    if total <= 0.0 or weighted <= 0.0:
        return float("nan")
    return float(2.0 * np.pi * total / weighted)


class _UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count + 1))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        if first == 0 or second == 0:
            return
        root_a, root_b = self.find(first), self.find(second)
        if root_a != root_b:
            self.parent[root_b] = root_a


def periodic_clusters(field: np.ndarray, threshold: float) -> Tuple[int, float]:
    """Count four-connected threshold components with periodic edge merging."""

    occupied = np.asarray(field > threshold, dtype=bool)
    if not np.any(occupied):
        return 0, 0.0
    labels, count = ndimage.label(occupied, structure=ndimage.generate_binary_structure(2, 1))
    union = _UnionFind(int(count))
    rows, cols = occupied.shape
    for column in range(cols):
        if occupied[0, column] and occupied[-1, column]:
            union.union(int(labels[0, column]), int(labels[-1, column]))
    for row in range(rows):
        if occupied[row, 0] and occupied[row, -1]:
            union.union(int(labels[row, 0]), int(labels[row, -1]))

    component_sizes: Dict[int, int] = {}
    for label in labels[occupied]:
        root = union.find(int(label))
        component_sizes[root] = component_sizes.get(root, 0) + 1
    largest = max(component_sizes.values()) / occupied.size
    return len(component_sizes), float(largest)


def scalar_diagnostics(field, grid, initial_mass: float, threshold: float) -> Dict[str, float]:
    mass = float(np.sum(field) * grid.dx * grid.dx)
    scale = max(abs(initial_mass), grid.length * grid.length, np.finfo(float).eps)
    count, largest = periodic_clusters(field, threshold)
    return {
        "mass": mass,
        "mass_error": float((mass - initial_mass) / scale),
        "variance": float(np.var(field)),
        "minimum": float(np.min(field)),
        "maximum": float(np.max(field)),
        "cluster_count": int(count),
        "largest_cluster": largest,
        "length_scale": structure_factor_length(field, grid),
        "cluster_threshold": float(threshold),
    }
