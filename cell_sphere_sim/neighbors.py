from __future__ import annotations

import numpy as np


def build_spatial_hash(x: np.ndarray, cell_size: float) -> dict[tuple[int, int, int], np.ndarray]:
    """Build a 3D spatial hash map of cell indices.

    cell_size should be at least the max interaction range to avoid missing contacts.
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    coords = np.floor(x / cell_size).astype(int)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, c in enumerate(coords):
        key = (int(c[0]), int(c[1]), int(c[2]))
        buckets.setdefault(key, []).append(i)
    return {k: np.asarray(v, dtype=int) for k, v in buckets.items()}


def candidate_pairs_from_hash(
    x: np.ndarray,
    hash_map: dict[tuple[int, int, int], np.ndarray],
    cell_size: float,
) -> list[tuple[int, int]]:
    """Generate candidate pairs by checking adjacent hash cells.

    This guarantees no missed contacts when cell_size >= max contact range.
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    coords = np.floor(x / cell_size).astype(int)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    pairs: set[tuple[int, int]] = set()
    for i, c in enumerate(coords):
        base = (int(c[0]), int(c[1]), int(c[2]))
        for dx, dy, dz in offsets:
            key = (base[0] + dx, base[1] + dy, base[2] + dz)
            if key not in hash_map:
                continue
            for j in hash_map[key]:
                if j > i:
                    pairs.add((i, int(j)))
    return list(pairs)
