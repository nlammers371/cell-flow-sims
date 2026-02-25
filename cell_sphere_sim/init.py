from __future__ import annotations

from typing import Iterable
import numpy as np
from scipy.spatial import cKDTree

from .state import StateTable


def sample_state_ids(N: int, composition: dict[int, float] | Iterable[int], rng: np.random.Generator) -> np.ndarray:
    """Sample state IDs from fractions or explicit counts."""
    if isinstance(composition, dict):
        keys = list(sorted(composition.keys()))
        values = np.array([composition[k] for k in keys], dtype=float)
        if np.all(values.astype(int) == values) and int(np.sum(values)) == N:
            counts = values.astype(int)
            state_id = np.concatenate([np.full(c, k, dtype=np.int32) for k, c in zip(keys, counts)])
            rng.shuffle(state_id)
            return state_id
        probs = values / np.sum(values)
        return rng.choice(keys, size=N, p=probs).astype(np.int32)
    counts = np.asarray(list(composition), dtype=int)
    if int(np.sum(counts)) != N:
        raise ValueError("composition counts must sum to N")
    state_id = np.concatenate([np.full(c, i, dtype=np.int32) for i, c in enumerate(counts)])
    rng.shuffle(state_id)
    return state_id


def _random_point_on_sphere(rng: np.random.Generator) -> np.ndarray:
    g = rng.normal(size=3)
    g = g / np.linalg.norm(g)
    return g


def _bias_weight(n: np.ndarray, mode: str, strength: float, axis: np.ndarray) -> float:
    if strength <= 0.0 or mode == "uniform":
        return 1.0
    axis = axis / np.linalg.norm(axis)
    ndot = float(np.dot(n, axis))
    if mode == "polar_cap":
        return (1.0 - strength) + strength * ((ndot + 1.0) / 2.0) ** 2
    if mode == "equatorial_band":
        return (1.0 - strength) + strength * (1.0 - abs(ndot))
    if mode == "axial_gradient":
        return (1.0 - strength) + strength * ((ndot + 1.0) / 2.0)
    raise ValueError(f"Unknown pos_mode: {mode}")


def _sample_position(
    rng: np.random.Generator,
    R_E: float,
    pos_mode: str,
    pos_strength: float,
    axis: np.ndarray,
) -> np.ndarray:
    while True:
        n = _random_point_on_sphere(rng)
        if rng.random() <= _bias_weight(n, pos_mode, pos_strength, axis):
            return R_E * n


def _random_tangent_polarity(rng: np.random.Generator, n: np.ndarray) -> np.ndarray:
    g = rng.normal(size=3)
    u = g - np.dot(g, n) * n
    norm = np.linalg.norm(u)
    if norm < 1e-12:
        u = np.array([1.0, 0.0, 0.0]) - n[0] * n
        norm = np.linalg.norm(u)
    return u / norm


def _biased_heading(n: np.ndarray, axis: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    p_rand = _random_tangent_polarity(rng, n)
    if strength <= 0.0:
        return p_rand
    axis = axis / np.linalg.norm(axis)
    b = axis - np.dot(axis, n) * n
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-12:
        return p_rand
    b_hat = b / b_norm
    p = (1.0 - strength) * p_rand + strength * b_hat
    return p / np.linalg.norm(p)


def init_random_on_sphere(
    N: int,
    R_E: float,
    state_id: np.ndarray,
    state_table: StateTable,
    alpha_dmin: float,
    eps: float,
    rng: np.random.Generator,
    pos_mode: str = "uniform",
    pos_strength: float = 0.0,
    mixing: float = 1.0,
    heading_mode: str = "isotropic",
    heading_strength: float = 0.0,
    max_attempts_per_cell: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize positions and headings with overlap avoidance."""
    if N <= 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
    if mixing < 0.0 or mixing > 1.0:
        raise ValueError("mixing must be in [0, 1]")
    if pos_strength < 0.0 or pos_strength > 1.0:
        raise ValueError("pos_strength must be in [0, 1]")
    if heading_strength < 0.0 or heading_strength > 1.0:
        raise ValueError("heading_strength must be in [0, 1]")

    R = state_table.R
    state_id = np.asarray(state_id, dtype=np.int32)
    if state_id.shape[0] != N:
        raise ValueError("state_id must have length N")

    unique_states = int(np.max(state_id)) + 1 if state_id.size else 1
    if unique_states <= 1:
        axes = [np.array([0.0, 0.0, 1.0])]
    else:
        z_vals = np.linspace(1.0, -1.0, unique_states)
        axes = [np.array([np.sqrt(1.0 - z**2), 0.0, z]) for z in z_vals]

    x = np.zeros((N, 3), dtype=float)
    p = np.zeros((N, 3), dtype=float)
    placed = 0

    while placed < N:
        attempts = 0
        axis = axes[state_id[placed]]
        use_axis = rng.random() > mixing
        pos_axis = axis if use_axis else np.array([0.0, 0.0, 1.0])
        effective_strength = pos_strength
        if use_axis:
            effective_strength = max(pos_strength, 1.0 - mixing)

        while attempts < max_attempts_per_cell:
            candidate = _sample_position(rng, R_E, pos_mode, effective_strength, pos_axis)
            if placed == 0:
                x[placed] = candidate
                break
            tree = cKDTree(x[:placed])
            r_query = alpha_dmin * (float(R[state_id[placed]]) + float(np.max(R)))
            neighbors = tree.query_ball_point(candidate, r=r_query)
            ok = True
            if neighbors:
                dvec = x[neighbors] - candidate
                d = np.linalg.norm(dvec, axis=1)
                sigma = R[state_id[placed]] + R[state_id[neighbors]]
                d_min = alpha_dmin * sigma
                if np.any(d < d_min - eps):
                    ok = False
            if ok:
                x[placed] = candidate
                break
            attempts += 1
        if attempts >= max_attempts_per_cell:
            raise ValueError("Failed to place non-overlapping points; reduce N or increase R_E")

        n = x[placed] / R_E
        if heading_mode == "isotropic":
            p[placed] = _random_tangent_polarity(rng, n)
        elif heading_mode == "axial_bias":
            p[placed] = _biased_heading(n, axis, heading_strength, rng)
        else:
            raise ValueError(f"Unknown heading_mode: {heading_mode}")

        placed += 1

    return x, p
